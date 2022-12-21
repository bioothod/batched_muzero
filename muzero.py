from typing import Dict, List

import itertools
import logging
import pickle
import os
import time

from collections import defaultdict
from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True

import connectx_impl
from evaluate_score import EvaluationDataset
from hparams import Hparams as GenericHparams
from inference import NetworkOutput
from logger import setup_logger
import mcts
import muzero_server
import networks
import simulation

class Hparams(GenericHparams):
    init_lr = 1e-4
    min_lr = 1e-5

    num_training_steps = 1000

class Sample:
    num_steps: int
    player_id: int
    initial_game_state: torch.Tensor

    children_visits: torch.Tensor
    values: torch.Tensor
    last_rewards: torch.Tensor

    actions: torch.Tensor

class SampleDataset:
    def __init__(self, hparams: Hparams):
        self.samples = []

        self.num_unroll_steps = hparams.num_unroll_steps

    def __len__(self):
        return len(self.samples)

    def append(self, sample_dict: Dict[str, torch.Tensor]):
        game_states = sample_dict['game_states']
        actions = sample_dict['actions']
        episode_len = sample_dict['actions']
        values = sample_dict['values']
        last_rewards = sample_dict['last_rewards']
        children_visits = sample_dict['children_visits']
        player_ids = sample_dict['player_ids']

        for sample_idx in range(len(game_states)):
            s = Sample()

            s.num_steps = episode_len[sample_idx]
            s.player_id = player_ids[sample_idx]
            s.initial_game_state = game_states[sample_idx]

            s.children_visits = children_visits[sample_idx]
            s.values = values[sample_idx]
            s.last_rewards = last_rewards[sample_idx]
            s.actions = actions[sample_idx]

            self.samples.append(s)

    def __getitem__(self, index):
        s = self.samples[index]

def scale_gradient(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return tensor * scale + tensor.detach() * (1 - scale)

class Trainer:
    def __init__(self, hparams: Hparams, logger: logging.Logger, eval_ds: EvaluationDataset):
        self.hparams = hparams
        self.logger = logger
        self.eval_ds = eval_ds

        self.max_best_score = None

        self.grpc_server, self.muzero_server = muzero_server.start_server(hparams, logger)

        tensorboard_log_dir = os.path.join(hparams.checkpoints_dir, 'tensorboard_logs')
        first_run = True
        if os.path.exists(tensorboard_log_dir) and len(os.listdir(tensorboard_log_dir)) > 0:
            first_run = False

        self.summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.global_step = 0

        self.inference = networks.Inference(hparams, logger)

        self.representation_opt = torch.optim.Adam(self.inference.representation.parameters(), lr=hparams.init_lr)
        self.prediction_opt = torch.optim.Adam(self.inference.prediction.parameters(), lr=hparams.init_lr)
        self.dynamic_opt = torch.optim.Adam(self.inference.dynamic.parameters(), lr=hparams.init_lr)
        self.optimizers = [self.representation_opt, self.prediction_opt, self.dynamic_opt]

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.scalar_loss = nn.MSELoss(reduction='none')

        self.all_games: Dict[int, List[simulation.GameStats]] = defaultdict(list)

        self.try_load()
        self.save_muzero_server_weights()

    def close(self):
        self.grpc_server.wait_for_termination()

    def training_forward_one_game(self, epoch: int, game_stat: simulation.GameStats):
        start_pos = []
        for i, max_episode_len in enumerate(game_stat.episode_len):
            max_start_pos = max_episode_len - self.hparams.num_unroll_steps
            if max_start_pos < 0:
                pos = 0
            else:
                pos = torch.randint(low=0, high=max_start_pos.item(), size=(1,))

            start_pos.append(pos)

        start_pos = torch.cat(start_pos).to(self.hparams.device)

        self.summary_writer.add_scalars('train/episode_len', {
            'mean': game_stat.episode_len.float().mean(),
            'median': game_stat.episode_len.float().median(),
            'min': game_stat.episode_len.min(),
            'max': game_stat.episode_len.max(),
        }, self.global_step)

        sample_dict = game_stat.make_target(start_pos)

        game_states = sample_dict['game_states']
        actions = sample_dict['actions']
        sample_len = sample_dict['sample_len']
        values = sample_dict['values']
        last_rewards = sample_dict['last_rewards']
        children_visits = sample_dict['children_visits']
        player_ids = sample_dict['player_ids']

        invalid_sample_len_index = sample_len == 0
        if invalid_sample_len_index.sum() > 0:
            self.logger.error(f'{epoch}: invalid sample_len: number of zeros: {invalid_sample_len_index.sum()}, sample_len:\n{sample_len}\nstart_pos:{start_pos}')
            self.logger.info(f'game_states: {game_states.shape}, player_ids: {player_ids.shape}, actions: {actions.shape}, children_visits: {children_visits.shape}')
            exit(-1)

        train_examples = len(game_states)

        out = self.inference.initial(player_ids[:, 0], game_states)
        policy_loss = self.ce_loss(out.policy_logits, children_visits[:, :, 0])
        value_loss = self.scalar_loss(out.value, values[:, 0])
        reward_loss = self.scalar_loss(out.reward, out.reward)

        iteration_loss = policy_loss + value_loss + reward_loss
        iteration_loss = scale_gradient(iteration_loss, 1/sample_len)
        total_loss = torch.mean(iteration_loss)

        self.summary_writer.add_scalars('train/initial_losses', {
                'policy': policy_loss.mean(),
                'value': value_loss.mean(),
                'total': total_loss,
        }, self.global_step)

        #policy_softmax = F.log_softmax(out.policy_logits, 1)

        #self.logger.info(f'children_visits:\n{children_visits[:5, :, 0]}, policy_logits:\n{out.policy_logits[:5]}\npolicy_softmax:\n{policy_softmax[:5]}\npolicy_loss:\n{policy_loss[:5]}\nactions:\n{actions[:5, 0]}')
        #logger.info(f'values:\n{values[:5, 0]}\nout.value:\n{out.value[:5]}\nvalue_loss: {value_loss[:5]}')

        if False:
            self.logger.info(f'{epoch:3d}: '
            f'policy_loss: {policy_loss.mean().item():.4f}, '
            f'value_loss: {value_loss.mean().item():.4f}, '
            f'reward_loss: {reward_loss.mean().item():.4f}, '
            f'total_loss: {total_loss.item():.4f}')


        for step_idx in range(1, sample_len.max()):
            batch_index = step_idx < sample_len
            sample_len = sample_len[batch_index]
            actions = actions[batch_index]
            values = values[batch_index]
            children_visits = children_visits[batch_index]
            last_rewards = last_rewards[batch_index]

            scale = torch.ones(len(last_rewards), device=self.hparams.device)*0.5
            scale = scale.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            hidden_states = scale_gradient(out.hidden_state[batch_index], scale)

            out = self.inference.recurrent(hidden_states, actions[:, step_idx-1])

            policy_loss = self.ce_loss(out.policy_logits, children_visits[:, :, step_idx])
            value_loss = self.scalar_loss(out.value, values[:, step_idx])
            reward_loss = self.scalar_loss(out.reward, last_rewards[:, step_idx])

            iteration_loss = policy_loss + value_loss + reward_loss
            iteration_loss = scale_gradient(iteration_loss, 1/sample_len)

            total_loss += torch.mean(iteration_loss)

        return total_loss, train_examples

    def training_step(self, epoch: int):
        self.inference.train(True)
        self.inference.zero_grad()
        for opt in self.optimizers:
            opt.zero_grad()

        all_games = []
        for gen, games in self.all_games.items():
            all_games += games

        game_stat = np.random.choice(all_games)
        total_loss, total_train_examples = self.training_forward_one_game(epoch, game_stat)

        total_loss.backward()

        for opt in self.optimizers:
            opt.step()

        self.summary_writer.add_scalar('train/total_loss', total_loss, self.global_step)
        self.summary_writer.add_scalar('train/samples', total_train_examples, self.global_step)


        self.global_step += 1

        self.save_muzero_server_weights()
        self.move_games()

    def move_games(self):
        new_games = self.muzero_server.move_games()
        if len(new_games) > 0:
            for gen, games in new_games.items():
                self.all_games[gen] += games

        num_games = 0
        for gen, games in self.all_games.items():
            num_games += len(games)

        if num_games > self.hparams.max_training_games:
            to_remove = num_games - self.hparams.max_training_games

            all_keys = sorted(list(self.all_games.keys()))
            for gen in all_keys:
                games = self.all_games[gen]
                if to_remove >= len(games):
                    to_remove -= len(games)
                    num_games -= len(games)
                    del self.all_games[gen]
                else:
                    self.all_games[gen] = games[to_remove:]
                    num_games -= to_remove
                    to_remove = 0
                    break

        if num_games > 0:
            self.summary_writer.add_scalar('train/num_games', num_games, self.global_step)


    def save_muzero_server_weights(self):
        save_dict = {
            'representation_state_dict': self.inference.representation.state_dict(),
            'prediction_state_dict': self.inference.prediction.state_dict(),
            'dynamic_state_dict': self.inference.dynamic.state_dict(),
        }
        meta = pickle.dumps(save_dict)

        self.muzero_server.update_weights(self.global_step, meta)

    def run_training(self, epoch: int):
        self.summary_writer.add_scalar('train/epoch', epoch, self.global_step)

        self.move_games()

        while len(self.all_games) == 0:
            time.sleep(1)
            self.move_games()

        for train_idx in range(self.hparams.num_training_steps):
            self.training_step(epoch)

            if train_idx % 10 == 0:
                best_score, good_score = self.run_evaluation(save_if_best=True)

        best_score, good_score = self.run_evaluation(save_if_best=True)

    def run_evaluation(self, save_if_best: bool):
        hparams = deepcopy(self.hparams)
        hparams.batch_size = len(self.eval_ds.game_states)
        train = simulation.Train(hparams, self.inference, self.logger)
        with torch.no_grad():
            game_states = torch.zeros(hparams.batch_size, *hparams.state_shape).float().to(hparams.device)
            active_games_index = torch.arange(hparams.batch_size).long().to(hparams.device)

            active_game_states = game_states[active_games_index]
            active_player_ids = self.eval_ds.game_player_ids[active_games_index]
            pred_actions, children_visits, root_values = train.run_simulations(active_player_ids, active_game_states)

        best_score, good_score = self.eval_ds.evaluate(pred_actions, debug=False)

        self.summary_writer.add_scalars('eval/ref_moves_score', {
            'good': good_score,
            'best': best_score,
        }, self.global_step)

        if save_if_best and (self.max_best_score is None or best_score > self.max_best_score):
            self.max_best_score = best_score
            checkpoint_path = os.path.join(self.hparams.checkpoints_dir, f'muzero_{best_score:.1f}.ckpt')
            self.save(checkpoint_path)
            self.logger.info(f'stored checkpoint: best_score: {best_score:.1f}, checkpoint: {checkpoint_path}')

        return best_score, good_score

    def save(self, checkpoint_path):
        torch.save({
            'representation_state_dict': self.inference.representation.state_dict(),
            'representation_optimizer_state_dict': self.representation_opt.state_dict(),
            'prediction_state_dict': self.inference.prediction.state_dict(),
            'prediction_optimizer_state_dict': self.prediction_opt.state_dict(),
            'dynamic_state_dict': self.inference.dynamic.state_dict(),
            'dynamic_optimizer_state_dict': self.dynamic_opt.state_dict(),
            'global_step': self.global_step,
            'max_best_score': self.max_best_score,
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        self.inference.representation.load_state_dict(checkpoint['representation_state_dict'])
        self.representation_opt.load_state_dict(checkpoint['representation_optimizer_state_dict'])
        self.inference.prediction.load_state_dict(checkpoint['prediction_state_dict'])
        self.prediction_opt.load_state_dict(checkpoint['prediction_optimizer_state_dict'])
        self.inference.dynamic.load_state_dict(checkpoint['dynamic_state_dict'])
        self.dynamic_opt.load_state_dict(checkpoint['dynamic_optimizer_state_dict'])

        self.global_step = checkpoint['global_step']
        self.max_best_score = checkpoint['max_best_score']

        self.save_muzero_server_weights()

        self.logger.info(f'loaded checkpoint {checkpoint_path}')

    def try_load(self):
        max_score = None
        max_score_fn = None
        for fn in os.listdir(self.hparams.checkpoints_dir):
            if not fn.endswith('.ckpt'):
                continue

            score = float(fn.split('.')[0][7:])
            if max_score is None or score > max_score:
                max_score = score
                max_score_fn = fn

        if max_score_fn is not None:
            checkpoint_path = os.path.join(self.hparams.checkpoints_dir, max_score_fn)
            self.load(checkpoint_path)


def main():
    hparams = Hparams()

    epoch = 0
    hparams.batch_size = 128
    hparams.num_simulations = 800
    hparams.num_training_steps = 30
    hparams.device = torch.device('cuda:0')

    logfile = os.path.join(hparams.checkpoints_dir, 'muzero.log')
    os.makedirs(hparams.checkpoints_dir, exist_ok=True)
    logger = setup_logger('muzero', logfile, hparams.log_to_stdout)

    refmoves_fn = 'refmoves1k_kaggle'
    eval_ds = EvaluationDataset(refmoves_fn, hparams, logger)

    trainer = Trainer(hparams, logger, eval_ds)

    for epoch in itertools.count():
        trainer.run_training(epoch)

if __name__ == '__main__':
    main()
