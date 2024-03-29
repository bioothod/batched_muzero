from typing import Dict, Optional

import argparse
import datetime
import grpc
import io
import logging
import os
import pickle
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from time import perf_counter

import muzero_pb2
import muzero_pb2_grpc

from hparams import GenericHparams as Hparams
from logger import setup_logger
import module_loader
from networks import GameState, Inference
import simulation

def fix(map_loc):
    # Closure rather than a lambda to preserve map_loc
    return lambda b: torch.load(io.BytesIO(b), map_location=map_loc)

class MappedUnpickler(pickle.Unpickler):
    # https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219

    def __init__(self, *args, map_location='cpu', **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return fix(self._map_location)
        else:
            return super().find_class(module, name)

def mapped_loads(s, map_location='cpu'):
    bs = io.BytesIO(s)
    unpickler = MappedUnpickler(bs, map_location=map_location)
    return unpickler.load()

class MuzeroCollectionClient:
    @torch.no_grad()
    def __init__(self, client_id: str, game_ctl: module_loader.GameModule, logger: logging.Logger, write_summary=False):
        self.logger = logger
        self.game_ctl = game_ctl
        self.client_id = client_id

        options = (
            ('grpc.max_send_message_length', -1),
            ('grpc.max_receive_message_length', -1),
        )
        self.channel = grpc.insecure_channel(target=f'localhost:{self.game_ctl.hparams.server_port}', options=options)
        self.stub = muzero_pb2_grpc.MuzeroStub(self.channel)

        self.generation = -1

        self.inference = Inference(self.game_ctl, logger)

        self.summary_writer: Optional[SummaryWriter] = None
        self.write_summary = write_summary
        if write_summary:
            tensorboard_log_dir = os.path.join(self.game_ctl.hparams.checkpoints_dir, 'tensorboard_logs')
            self.summary_writer = SummaryWriter(log_dir=tensorboard_log_dir, flush_secs=1)

    def action_selection_fn(self, children_visit_counts: torch.Tensor, episode_len: torch.Tensor) -> torch.Tensor:
        actions_argmax = torch.argmax(children_visit_counts, 1)

        dist = torch.pow(children_visit_counts.float(), 1. / self.game_ctl.hparams.action_selection_temperature)
        actions_dist = torch.multinomial(dist, 1).squeeze(1)

        actions = torch.where(episode_len >= self.game_ctl.hparams.num_steps_to_argmax_action_selection, actions_argmax, actions_dist)
        return actions

    def update_weights(self):
        resp = self.stub.WeightUpdateRequest(muzero_pb2.WeightRequest(
            generation=self.generation,
        ))

        if len(resp.weights) > 0:
            tensors_loaded, num_params = self.load_weights(resp.weights)

            self.logger.info(f'received weight update: generation: {self.generation} -> {resp.generation}, '
                             f'weights: {len(resp.weights)}, '
                             f'tensors_loaded: {tensors_loaded}, '
                             f'num_params: {num_params/1_000_000:.3f}m')

            self.generation = resp.generation

    def load_weights(self, weights):
        full_state = pickle.loads(weights)
        state = full_state['state_dict']

        tensors_loaded = 0
        num_params = 0
        for key, value in self.inference.state_dict().items():
            src = state[key]
            src = torch.from_numpy(src).to(self.game_ctl.hparams.device)
            value.copy_(src)
            tensors_loaded += 1
            num_params += np.prod(src.shape)

        self.inference.to(self.game_ctl.hparams.device)
        return tensors_loaded, num_params

    def send_game_stats(self, game_stats: simulation.GameStats, collection_time: float):
        meta = pickle.dumps([game_stats.to('cpu')])

        resp = self.stub.SendGameStats(muzero_pb2.GameStats(
            generation=self.generation,
            stats=meta,
        ))

        collection_time_str = str(datetime.timedelta(seconds=collection_time))
        self.logger.info(f'game stats updated: {self.generation} -> {resp.generation}, simulations: {self.game_ctl.hparams.num_simulations}, collection_time: {collection_time_str}')

    @torch.no_grad()
    def collect_episode(self):
        self.update_weights()
        if self.generation < 0:
            return

        start_time = perf_counter()

        self.inference.train(False)

        sim = simulation.Simulation(self.game_ctl, self.inference, self.action_selection_fn, self.logger, self.summary_writer, f'{self.client_id}_simulation', self.generation)
        game_stats = sim.run_single_game_and_collect_stats(self.game_ctl.hparams)

        collection_time = perf_counter() - start_time

        self.send_game_stats(game_stats, collection_time)
        if self.generation == 0:
            time.sleep(1)

        if self.write_summary and self.summary_writer is not None:
            prefix = f'{self.client_id}'

            for i in range(0, 8, 1):
                valid_index = game_stats.episode_len > i

                if valid_index.sum() > 0:
                    children_visits = game_stats.children_visits[valid_index, :, i].float()
                    children_visits = children_visits / children_visits.sum(1, keepdim=True)
                    actions = range(children_visits.shape[-1])
                    children_visits = {str(action):children_visits[:, action].mean(0) for action in actions}
                    #initial_policy_probs = {str(action):game_stat.initial_policy_probs[valid_index, action, i].mean(0) for action in actions}

                    self.summary_writer.add_scalars(f'{prefix}/children_visits{i}', children_visits, self.generation)
                    #self.summary_writer.add_scalars(f'{prefix}/pred_policy_probs{i}', initial_policy_probs, self.generation)

            root_values = {}
            for i in range(3):
                root_values[f'mcts{i}'] = game_stats.root_values[:, i].float().mean()
                root_values[f'initial_pred{i}'] = game_stats.initial_values[:, i].float().mean()
            self.summary_writer.add_scalars(f'{prefix}/root_values', root_values, self.generation)

            self.summary_writer.add_scalars(f'{prefix}/train_steps', {
                'min': game_stats.episode_len.min(),
                'max': game_stats.episode_len.max(),
                'mean': game_stats.episode_len.float().mean(),
                'median': game_stats.episode_len.float().median(),
                'max_reward_step_mean': torch.argmax(game_stats.rewards, 1).float().mean(0)
            }, self.generation)


            episode_index = (game_stats.episode_len - 1).unsqueeze(1)
            win_player_id = game_stats.player_ids.gather(1, episode_index).squeeze(1)
            win_rewards = game_stats.rewards.gather(1, episode_index).squeeze(1)
            episode_rewards = {}
            episode_rewards_mean = {}
            for player_id in self.game_ctl.hparams.player_ids:
                player_index = win_player_id == player_id
                rewards = win_rewards[player_index]
                wins = (rewards > 0).sum()

                episode_rewards[f'wins{player_id}'] = wins / len(win_rewards)
                episode_rewards_mean[f'{player_id}'] = rewards.sum() / len(win_rewards)

            episode_rewards[f'draws'] = (win_rewards == 0).sum() / len(win_rewards)

            self.summary_writer.add_scalars(f'{self.client_id}/results', episode_rewards, self.generation)
            self.summary_writer.add_scalars(f'{self.client_id}/mean_reward', episode_rewards_mean, self.generation)
            self.summary_writer.add_scalar(f'{self.client_id}/collection_time', collection_time, self.generation)

def run_process(client_id: str, module: module_loader.GameModule, write_summary: bool):
    logfile = os.path.join(module.hparams.checkpoints_dir, f'{client_id}.log')
    logger = setup_logger(client_id, logfile, True)

    if module.hparams.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    module.load()
    client = MuzeroCollectionClient(client_id, module, logger, write_summary)
    while True:
        client.collect_episode()

        if client.generation < 0:
            time.sleep(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_id', type=int, default=0, help='Initial id for the clients')
    parser.add_argument('--num_clients', type=int, default=2, help='Number of clients to start')
    parser.add_argument('--num_simulations', type=int, default=400, help='Number of simulations per step')
    parser.add_argument('--batch_size', type=int, default=1024, help='Simulation batch size')
    parser.add_argument('--checkpoints_dir', type=str, required=True, help='Checkpoints directory and base dir for logs and stats')
    parser.add_argument('--game', type=str, required=True, help='Name of the game')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run episode collection')
    parser.add_argument('--no_tensorboard', action='store_true', help='Do not store tensorboard statistics')
    FLAGS = parser.parse_args()

    module = module_loader.GameModule(FLAGS.game, load=False)
    module.hparams.checkpoints_dir = FLAGS.checkpoints_dir
    module.hparams.batch_size = FLAGS.batch_size
    module.hparams.num_simulations = FLAGS.num_simulations
    module.hparams.device = FLAGS.device

    os.makedirs(module.hparams.checkpoints_dir, exist_ok=True)


    mp.set_start_method('spawn')

    processes = []
    for cid in range(FLAGS.start_id, FLAGS.start_id+FLAGS.num_clients):
        client_id = f'client{cid}'
        p = mp.Process(target=run_process, args=(client_id, module, cid==FLAGS.start_id and not FLAGS.no_tensorboard))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
