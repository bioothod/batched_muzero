from typing import Dict, List, Optional

import argparse
import dataclasses
import itertools
import logging
import pickle
import os
import time

from collections import defaultdict
from copy import deepcopy
from time import perf_counter

import numpy as np
from replay_buffer import ReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True

from evaluate_score import EvaluationDataset
from hparams import GenericHparams as Hparams
from logger import setup_logger
import module_loader
import muzero_server
import networks
import simulation

def scale_gradient(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return tensor * scale + tensor.detach() * (1 - scale)
    #return tensor

def train_element_collate_fn(samples: List[simulation.TrainElement]):
    collated_dict = defaultdict(list)
    for sample in samples:
        sample = dataclasses.asdict(sample)
        for key, value in sample.items():
            collated_dict[key].append(value)

    converted_dict = {}
    for key, list_value in collated_dict.items():
        converted_dict[key] = torch.stack(list_value, 0)

    return simulation.TrainElement(**converted_dict)

class Trainer:
    def __init__(self, game_ctl: module_loader.GameModule, logger: logging.Logger, eval_ds: Optional[EvaluationDataset]):
        self.game_ctl = game_ctl
        self.hparams = game_ctl.hparams
        self.logger = logger
        self.eval_ds = eval_ds

        self.max_best_score = None

        self.replay_buffer = ReplayBuffer(self.hparams)

        self.grpc_server, self.muzero_server = muzero_server.start_server(self.hparams, self.replay_buffer, logger)

        tensorboard_log_dir = os.path.join(self.hparams.checkpoints_dir, 'tensorboard_logs')
        first_run = True
        if os.path.exists(tensorboard_log_dir) and len(os.listdir(tensorboard_log_dir)) > 0:
            first_run = False

        self.summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.global_step = 0

        self.inference = networks.Inference(self.game_ctl, logger)

        self.representation_opt = torch.optim.Adam(self.inference.representation.parameters(), lr=self.hparams.init_lr)
        self.prediction_opt = torch.optim.Adam(self.inference.prediction.parameters(), lr=self.hparams.init_lr)
        self.dynamic_opt = torch.optim.Adam(self.inference.dynamic.parameters(), lr=self.hparams.init_lr)
        self.optimizers = [self.representation_opt, self.prediction_opt, self.dynamic_opt]

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.scalar_loss = nn.MSELoss(reduction='none')

        self.all_games: Dict[int, List[simulation.GameStats]] = defaultdict(list)

        self.try_load()
        self.save_muzero_server_weights()

    def save_muzero_server_weights(self):
        save_dict = {
            'representation_state_dict': self.inference.representation.state_dict(),
            'prediction_state_dict': self.inference.prediction.state_dict(),
            'dynamic_state_dict': self.inference.dynamic.state_dict(),
        }
        meta = pickle.dumps(save_dict)

        self.muzero_server.update_weights(self.global_step, meta)

    def close(self):
        self.grpc_server.wait_for_termination()

    def training_step(self, epoch: int, sample):
        out = self.inference.initial(sample.player_ids[:, 0], sample.game_states)
        policy_loss = self.ce_loss(out.policy_logits, sample.children_visits[:, :, 0])
        value_loss = self.scalar_loss(out.value, sample.values[:, 0])
        reward_loss = self.scalar_loss(out.reward, out.reward)

        iteration_loss = policy_loss + value_loss + reward_loss
        iteration_loss = scale_gradient(iteration_loss, 1/sample.sample_len)
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


        for step_idx in range(1, sample.sample_len.max()):
            batch_index = step_idx < sample.sample_len
            sample_len = sample.sample_len[batch_index]
            actions = sample.actions[batch_index]
            values = sample.values[batch_index]
            children_visits = sample.children_visits[batch_index]
            last_rewards = sample.last_rewards[batch_index]

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

        return total_loss

    def run_training(self, epoch: int):
        while self.replay_buffer.num_games() == 0:
            time.sleep(1)

        self.inference.train(True)

        samples = self.replay_buffer.sample(batch_size=self.hparams.batch_size*self.hparams.num_training_steps)
        data_loader = torch.utils.data.DataLoader(samples, batch_size=self.hparams.batch_size, shuffle=True, drop_last=False, collate_fn=train_element_collate_fn)

        for sample in data_loader:
            self.inference.zero_grad()
            for opt in self.optimizers:
                opt.zero_grad()

            total_loss = self.training_step(epoch, sample)

            total_loss.backward()

            for opt in self.optimizers:
                opt.step()

            self.summary_writer.add_scalar('train/total_loss', total_loss, self.global_step)

            self.global_step += 1
            self.save_muzero_server_weights()

        self.run_evaluation(save_if_best=True)

    def run_evaluation(self, save_if_best: bool):
        if self.eval_ds is None:
            return

        hparams = deepcopy(self.hparams)
        hparams.batch_size = len(self.eval_ds.game_states)

        train = simulation.Train(self.game_ctl, self.inference, self.logger, self.summary_writer, 'eval')
        with torch.no_grad():
            game_states = torch.zeros(hparams.batch_size, *hparams.state_shape, dtype=torch.uint8).to(hparams.device)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=400, help='Number of steps')
    parser.add_argument('--num_training_steps', type=int, default=40, help='Number of training steps before evaluation')
    parser.add_argument('--game', type=str, required=True, help='Name of the game')
    FLAGS = parser.parse_args()

    module = module_loader.GameModule(FLAGS.game, load=True)

    epoch = 0
    module.hparams.num_simulations = FLAGS.num_steps
    module.hparams.num_training_steps = FLAGS.num_training_steps
    module.hparams.device = torch.device('cuda:0')

    logfile = os.path.join(module.hparams.checkpoints_dir, 'muzero.log')
    os.makedirs(module.hparams.checkpoints_dir, exist_ok=True)
    logger = setup_logger('muzero', logfile, module.hparams.log_to_stdout)

    refmoves_fn = 'refmoves1k_kaggle'
    if FLAGS.game == 'connectx':
        eval_ds = EvaluationDataset(refmoves_fn, module.hparams, logger)
    else:
        eval_ds = None

    trainer = Trainer(module, logger, eval_ds)

    for epoch in itertools.count():
        trainer.run_training(epoch)

if __name__ == '__main__':
    main()
