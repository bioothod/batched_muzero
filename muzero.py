from typing import Dict, List, Optional

import argparse
import dataclasses
import itertools
import logging
import pickle
import os
import random
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


from matplotlib.lines import Line2D

torch.backends.cuda.matmul.allow_tf32 = True

import checkpoints
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

def action_selection_fn(children_visit_counts: torch.Tensor, episode_len: torch.Tensor):
    actions = torch.argmax(children_visit_counts, 1)
    return actions

class Trainer:
    def __init__(self, game_ctl: module_loader.GameModule, logger: logging.Logger, eval_ds: Optional[EvaluationDataset]):
        self.game_ctl = game_ctl
        self.hparams = game_ctl.hparams
        self.logger = logger
        self.eval_ds = eval_ds

        self.max_best_score = 0.
        self.max_good_score = 0.

        self.replay_buffer = ReplayBuffer(self.hparams)

        self.grpc_server, self.muzero_server = muzero_server.start_server(self.hparams, self.replay_buffer, logger)

        tensorboard_log_dir = os.path.join(self.hparams.checkpoints_dir, 'tensorboard_logs')
        first_run = True
        if os.path.exists(tensorboard_log_dir) and len(os.listdir(tensorboard_log_dir)) > 0:
            first_run = False

        self.summary_writer = SummaryWriter(log_dir=tensorboard_log_dir, flush_secs=1)
        self.global_step = 0

        self.inference = networks.Inference(self.game_ctl, logger)

        self.opt = torch.optim.Adam(self.inference.parameters(), lr=self.hparams.init_lr)

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.scalar_loss = nn.MSELoss(reduction='none')
        #self.scalar_loss = nn.HuberLoss(delta=1, reduction='none')

        self.all_games: Dict[int, List[simulation.GameStats]] = defaultdict(list)

        self.try_load()
        self.save_muzero_server_weights()

        self.start_training = time.time()

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

    def policy_loss(self, policy_logits: torch.Tensor, children_visit_counts_for_step: torch.Tensor) -> torch.Tensor:
        # children_visit_counts: [B, Nactions]
        children_visits_sum = children_visit_counts_for_step.sum(1, keepdim=True)
        action_probs = children_visit_counts_for_step / children_visits_sum
        loss = self.ce_loss(policy_logits, action_probs)
        return loss

        self.logger.info(f'policy_logits: {policy_logits[:10]}')

        pred_probs = F.softmax(policy_logits, 1)
        action_log_probs = action_probs.log()
        pred_log_probs = pred_probs.log()

        loss_raw = (pred_log_probs - action_log_probs).exp()
        loss_clipped = loss_raw.clamp(0.9, 1.1)
        loss = torch.min(loss_raw, loss_clipped)
        loss = loss.mean(1)

        # max_debug = 10
        # self.logger.info(f'action_probs:\n{action_probs[:max_debug]}\n'
        #                  f'pred_probs:\n{pred_probs[:max_debug]}\n'
        #                  f'action_log_probs:\n{action_log_probs[:max_debug]}\n'
        #                  f'pred_log_probs:\n{pred_log_probs[:max_debug]}\n'
        #                  f'loss_raw:\n{loss_raw[:max_debug]}\n'
        #                  f'loss_clipped:\n{loss_clipped[:max_debug]}')
        return loss

    def training_step(self, sample: simulation.TrainElement):
        sample_len_max = sample.sample_len.max().item()
        sample.sample_len = sample.sample_len.float()

        out = self.inference.initial(sample.initial_game_state)

        policy_loss = self.policy_loss(out.policy_logits, sample.children_visits[:, :, 0])
        value_loss = self.scalar_loss(out.value.squeeze(1), sample.values[:, 0])

        iteration_loss = policy_loss + value_loss
        total_loss_mean = torch.mean(iteration_loss)

        policy_loss_mean = policy_loss.mean()
        value_loss_mean = value_loss.mean()
        reward_loss_mean = 0

        self.summary_writer.add_scalars('train/initial_losses', {
                'policy': policy_loss_mean,
                'value': value_loss_mean,
                'total': total_loss_mean,
        }, self.global_step)

        for player_id in self.hparams.player_ids:
            player_idx = sample.player_ids[:, 0] == player_id

            if player_idx.sum() > 0:
                pred_values = out.value[player_idx].detach().cpu().numpy()
                true_values = sample.values[player_idx, 0].detach().cpu().numpy()
                value_loss_local = value_loss[player_idx]

                self.summary_writer.add_scalars(f'train/initial_values{player_id}', {
                    f'pred': pred_values.mean(),
                    f'true_target': true_values.mean(),
                    f'loss': value_loss_local.mean(),
                }, self.global_step)

        batch_index = torch.arange(len(sample))
        for step_idx in range(1, sample_len_max):
            len_idx = step_idx < sample.sample_len[batch_index]
            batch_index = batch_index[len_idx]
            sample_len = sample.sample_len[batch_index]
            actions = sample.actions[batch_index]
            values = sample.values[batch_index]
            children_visits = sample.children_visits[batch_index]
            rewards = sample.rewards[batch_index]

            hidden_states = out.hidden_state[len_idx]

            last_actions = actions[:, step_idx-1]
            out = self.inference.recurrent(hidden_states, last_actions)

            scale = torch.ones_like(out.hidden_state, device=out.hidden_state.device) * 0.8
            out.hidden_state = scale_gradient(out.hidden_state, scale)

            policy_loss = self.policy_loss(out.policy_logits, children_visits[:, :, step_idx])
            value_loss = self.scalar_loss(out.value.squeeze(1), values[:, step_idx])
            reward_loss = self.scalar_loss(out.reward.squeeze(1), rewards[:, step_idx-1])

            iteration_loss = policy_loss + value_loss + reward_loss
            iteration_loss = scale_gradient(iteration_loss, 1/sample_len)

            total_loss_mean += torch.mean(iteration_loss)
            policy_loss_mean += policy_loss.mean()
            value_loss_mean += value_loss.mean()
            reward_loss_mean += reward_loss.mean()

        self.summary_writer.add_scalars('train/final_losses', {
                'policy': policy_loss_mean,
                'reward': reward_loss_mean,
                'value': value_loss_mean,
                'total': total_loss_mean,
        }, self.global_step)

        return total_loss_mean

    def plot_grad_flow(self, model):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

        ave_grads = []
        max_grads= []
        layers = []
        for n, p in model.named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


    def plot_grad_tensorboard(self, model):
        #... your learning loop
        _limits = np.array([float(i) for i in range(len(gradmean))])
        _num = len(gradmean)
        writer.add_histogram_raw(tag=netname+"/abs_mean", min=0.0, max=0.3, num=_num,
                                 sum=gradmean.sum(), sum_squares=np.power(gradmean, 2).sum(), bucket_limits=_limits,
                                 bucket_counts=gradmean, global_step=self.global_step)
        # where gradmean is np.abs(p.grad.clone().detach().cpu().numpy()).mean()
        # _limits is the x axis, the layers
        # and
        _mean = {}
        for i, name in enumerate(layers):
            _mean[name] = gradmean[i]
        self.writer.add_scalars(netname+"/abs_mean", _mean, global_step=self.global_step)

    def run_training_online(self):
        self.replay_buffer.truncate(max_generation=self.global_step)
        while self.replay_buffer.num_games() < 10:
            self.replay_buffer.truncate(max_generation=self.global_step)

            time.sleep(1)

        self.inference.train(True)

        all_games = self.replay_buffer.flatten_games()

        for _ in range(self.hparams.num_training_steps):
            start_time = perf_counter()

            # do not need to call optimizers zero_grad() because we are settig grads to zero in every model
            self.inference.zero_grad()

            total_losses = []
            total_batch_size = 0
            for _ in range(self.hparams.num_gradient_accumulation_steps):
                sample = self.replay_buffer.sample(batch_size=self.hparams.batch_size, all_games=all_games)[:self.hparams.batch_size]
                random.shuffle(sample)
                sample = train_element_collate_fn(sample)
                sample = sample.to(self.hparams.device)
                total_batch_size += len(sample)

                total_loss = self.training_step(sample)
                total_loss.backward()

                total_losses.append(total_loss.item())

            total_loss_mean = sum(total_losses) / len(total_losses)

            nn.utils.clip_grad_norm_(self.inference.representation.parameters(), 1)
            nn.utils.clip_grad_norm_(self.inference.prediction.parameters(), 1)
            nn.utils.clip_grad_norm_(self.inference.dynamic.parameters(), 1)

            self.opt.step()

            train_step_time = perf_counter() - start_time

            self.summary_writer.add_scalar('train/one_step_time', train_step_time, self.global_step)
            self.summary_writer.add_scalar('train/batch_size', total_batch_size, self.global_step)

            self.summary_writer.add_scalar('train/total_loss', total_loss_mean, self.global_step)
            self.summary_writer.add_scalars('train/num_games', {
                'current': self.replay_buffer.num_games(),
                'current_max': self.replay_buffer.max_num_games,
                'config_max': self.hparams.max_training_games,
            }, self.global_step)
            self.summary_writer.add_scalar('train/games_received', self.replay_buffer.num_games_received, self.global_step)

            self.global_step += 1
            self.save_muzero_server_weights()

        self.run_evaluation(try_saving=True)

    def run_training_offline(self):
        while self.replay_buffer.num_games() == 0:
            time.sleep(1)

        self.inference.train(True)

        for _ in range(self.hparams.num_training_steps):
            start_time = perf_counter()

            # do not need to call optimizers zero_grad() because we are settig grads to zero in every model
            self.inference.zero_grad()

            total_losses = []
            total_batch_size = 0
            sample_start = []
            sample_len = []
            if self.global_step < 100:
                num_gradient_accumulation_steps = 1
            else:
                num_gradient_accumulation_steps = self.hparams.num_gradient_accumulation_steps

            for _ in range(num_gradient_accumulation_steps):
                with torch.no_grad():
                    sample = self.replay_buffer.sample(batch_size=self.hparams.batch_size)[:self.hparams.batch_size]
                    sample = train_element_collate_fn(sample)
                    sample = sample.to(self.hparams.device)
                    total_batch_size += len(sample)
                    sample_start.append(sample.start_index)
                    sample_len.append(sample.sample_len)

                total_loss = self.training_step(sample)
                total_loss.backward()

                # do not hold gpu memory
                sample.to('cpu')

                total_losses.append(total_loss.item())

            total_loss_mean = sum(total_losses) / len(total_losses)
            sample_start = torch.cat(sample_start, 0)
            sample_len = torch.cat(sample_len, 0)

            nn.utils.clip_grad_norm_(self.inference.representation.parameters(), self.hparams.max_gradient_norm)
            nn.utils.clip_grad_norm_(self.inference.prediction.parameters(), self.hparams.max_gradient_norm)
            nn.utils.clip_grad_norm_(self.inference.dynamic.parameters(), self.hparams.max_gradient_norm)

            self.opt.step()

            train_step_time = perf_counter() - start_time

            self.summary_writer.add_scalar('train/one_step_time', train_step_time, self.global_step)
            self.summary_writer.add_scalar('train/batch_size', total_batch_size, self.global_step)

            self.summary_writer.add_scalars('train/start_index', {
                'mean': sample_start.float().mean(),
                'min': sample_start.min(),
                'max': sample_start.max(),
            }, self.global_step)
            self.summary_writer.add_scalars('train/sample_len', {
                'mean': sample_len.float().mean(),
                'min': sample_len.min(),
                'max': sample_len.max(),
            }, self.global_step)

            self.summary_writer.add_scalar('train/total_loss', total_loss_mean, self.global_step)
            self.summary_writer.add_scalars('train/num_games', {
                'current': self.replay_buffer.num_games(),
                'current_max': self.replay_buffer.max_num_games,
                'config_max': self.hparams.max_training_games,
            }, self.global_step)
            self.summary_writer.add_scalar('train/games_received', self.replay_buffer.num_games_received, self.global_step)

            self.global_step += 1
            self.save_muzero_server_weights()

        self.run_evaluation(try_saving=True)

    def run_evaluation(self, try_saving: bool):
        if try_saving and self.hparams.save_latest:
            checkpoint_path = os.path.join(self.hparams.checkpoints_dir, f'muzero_latest.ckpt')
            self.save(checkpoint_path)

        if self.eval_ds is None:
            return

        if time.time() < self.start_training + self.hparams.save_best_after_seconds:
            return

        if self.global_step < self.hparams.save_best_after_training_steps:
            return

        start_time = perf_counter()
        hparams = deepcopy(self.hparams)
        hparams.batch_size = len(self.eval_ds.game_states)

        train = simulation.Train(self.game_ctl, self.inference, self.logger, self.summary_writer, 'eval', action_selection_fn)
        game_state_stack = networks.GameState(hparams.batch_size, hparams, train.game_ctl.network_hparams)

        with torch.no_grad():
            active_game_states = self.eval_ds.game_states
            active_player_ids = self.eval_ds.game_player_ids
            invalid_actions_mask = train.game_ctl.invalid_actions_mask(train.game_ctl.game_hparams, active_game_states)

            game_state_stack.push_game(active_player_ids, active_game_states)
            game_states = game_state_stack.create_state()

            pred_actions, children_visits, root_values, out_initial = train.run_simulations(active_player_ids, game_states, invalid_actions_mask)

        best_score, good_score, total_best_score, total_good_score = self.eval_ds.evaluate(pred_actions)

        eval_time = perf_counter() - start_time
        for player_id in self.hparams.player_ids:
            self.summary_writer.add_scalars(f'eval/ref_moves_score{player_id}', {
                'good': good_score[player_id],
                'best': best_score[player_id],
            }, self.global_step)

        self.summary_writer.add_scalar('eval/time', eval_time, self.global_step)
        self.summary_writer.add_scalars(f'eval/ref_moves_score_total', {
            'max_good': self.max_good_score,
            'good': total_good_score,
            'max_best': self.max_best_score,
            'best': total_best_score,
        }, self.global_step)

        if try_saving and (total_best_score >= self.max_best_score):
            self.max_best_score = total_best_score
            checkpoint_path = os.path.join(self.hparams.checkpoints_dir, f'muzero_best_{total_best_score:.1f}.ckpt')
            self.save(checkpoint_path)
            self.logger.info(f'stored checkpoint: generation: {self.global_step}, best_score: {total_best_score:.1f}, checkpoint: {checkpoint_path}')

        if try_saving and (total_good_score >= self.max_good_score):
            self.max_good_score = total_good_score
            checkpoint_path = os.path.join(self.hparams.checkpoints_dir, f'muzero_good_{total_good_score:.1f}.ckpt')
            self.save(checkpoint_path)
            self.logger.info(f'stored checkpoint: generation: {self.global_step}, good_score: {total_good_score:.1f}, checkpoint: {checkpoint_path}')


    def save(self, checkpoint_path):
        torch.save({
            'state_dict': self.inference.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'global_step': self.global_step,
            'max_best_score': self.max_best_score,
            'max_good_score': self.max_good_score,
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        self.inference.load_state_dict(checkpoint['state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])

        self.global_step = int(checkpoint['global_step'])
        self.max_best_score = float(checkpoint['max_best_score'])
        self.max_good_score = float(checkpoint['max_good_score'])

        self.save_muzero_server_weights()

        self.logger.info(f'loaded checkpoint {checkpoint_path}')

    def try_load(self):
        checkpoint_path = checkpoints.find_checkpoint(self.hparams.checkpoints_dir, self.hparams.load_latest)
        if checkpoint_path is not None:
            self.load(checkpoint_path)


def main():
    #torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--num_eval_simulations', type=int, default=400, help='Number of evaluation simulations')
    parser.add_argument('--num_training_steps', type=int, default=40, help='Number of training steps before evaluation')
    parser.add_argument('--num_gradient_accumulation_steps', type=int, default=1, help='Number of accumulating gradient steps before running backward propagation of the error')
    parser.add_argument('--checkpoints_dir', type=str, required=True, help='Checkpoints directory')
    parser.add_argument('--game', type=str, required=True, help='Name of the game')
    parser.add_argument('--save_latest', action='store_true', default=False, help='Save a checkpoint after each training epoch')
    parser.add_argument('--load_latest', action='store_true', default=False, help='Whether to load the latest checkpoint instead of the best metrics')
    parser.add_argument('--save_best_after_seconds', type=int, default=0, help='Start saving best checkpoints only after this number of seconds has passed after the start')
    parser.add_argument('--save_best_after_training_steps', type=int, default=0, help='Start saving best checkpoints only after this number of training steps has passed')
    parser.add_argument('--online', action='store_true', help='Run online training, i.e. waiting for number of episodes made with the latest model and then training with them')
    FLAGS = parser.parse_args()

    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    module = module_loader.GameModule(FLAGS.game, load=True)

    module.hparams.num_simulations = FLAGS.num_eval_simulations
    module.hparams.num_training_steps = FLAGS.num_training_steps
    module.hparams.num_gradient_accumulation_steps = FLAGS.num_gradient_accumulation_steps
    module.hparams.checkpoints_dir = FLAGS.checkpoints_dir
    if FLAGS.batch_size:
        module.hparams.batch_size = FLAGS.batch_size
    module.hparams.device = torch.device('cuda:0')
    module.hparams.save_best_after_training_steps = FLAGS.save_best_after_training_steps
    module.hparams.save_best_after_seconds = FLAGS.save_best_after_seconds
    module.hparams.save_latest = FLAGS.save_latest
    module.hparams.load_latest = FLAGS.load_latest

    logfile = os.path.join(module.hparams.checkpoints_dir, 'muzero.log')
    os.makedirs(module.hparams.checkpoints_dir, exist_ok=True)
    logger = setup_logger('muzero', logfile, module.hparams.log_to_stdout)

    refmoves_fn = 'refmoves1k_kaggle'
    if FLAGS.game == 'connectx':
        eval_ds = EvaluationDataset(refmoves_fn, module.hparams, logger)
    else:
        eval_ds = None

    trainer = Trainer(module, logger, eval_ds)
    if FLAGS.online:
        trainer.replay_buffer.max_num_games = trainer.hparams.max_training_games

    while True:
        if FLAGS.online:
            trainer.run_training_online()
        else:
            trainer.run_training_offline()

if __name__ == '__main__':
    main()
