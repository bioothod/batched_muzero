import argparse
import grpc
import io
import logging
import os
import pickle
import time
from tictactoe_impl import step_games

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from time import perf_counter

import muzero_pb2
import muzero_pb2_grpc

from hparams import GenericHparams as Hparams
from hparams import *
from logger import setup_logger
from networks import Inference
import simulation
import tictactoe_impl

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
    def __init__(self, client_id: str, hparams: Hparams, logger: logging.Logger):
        self.logger = logger
        self.hparams = hparams
        self.client_id = client_id

        options = (
            ('grpc.max_send_message_length', -1),
            ('grpc.max_receive_message_length', -1),
        )
        self.channel = grpc.insecure_channel(target=f'localhost:{hparams.server_port}', options=options)
        self.stub = muzero_pb2_grpc.MuzeroStub(self.channel)

        self.generation = -1

        self.inference = Inference(hparams, logger)
        tensorboard_log_dir = os.path.join(hparams.checkpoints_dir, 'tensorboard_logs')
        self.summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)


    def update_weights(self):
        resp = self.stub.WeightUpdateRequest(muzero_pb2.WeightRequest(
            generation=self.generation,
        ))

        self.logger.info(f'received weight update: generation: {self.generation} -> {resp.generation}, weights: {len(resp.weights)}')

        if len(resp.weights) > 0:
            self.load_weights(resp.weights)
            self.generation = resp.generation

    def load_weights(self, weights):
        checkpoint = mapped_loads(weights, map_location=self.hparams.device)

        self.inference.representation.load_state_dict(checkpoint['representation_state_dict'])
        self.inference.prediction.load_state_dict(checkpoint['prediction_state_dict'])
        self.inference.dynamic.load_state_dict(checkpoint['dynamic_state_dict'])

    def send_game_stats(self, game_stats: simulation.GameStats):
        game_stats = [game_stats]
        meta = pickle.dumps(game_stats)

        resp = self.stub.SendGameStats(muzero_pb2.GameStats(
            generation=self.generation,
            stats=meta,
        ))

        self.logger.info(f'game stats updated: {self.generation} -> {resp.generation}')

    def collect_episode(self):
        with torch.no_grad():
            self.update_weights()
            if self.generation < 0:
                return

            start_time = perf_counter()

            self.inference.train(False)

            step_games = tictactoe_impl.step_games
            invalid_actions_mask = tictactoe_impl.invalid_actions_mask
            game_hparams = tictactoe_impl.Hparams()

            train = simulation.Train(self.hparams, self.inference, self.logger, self.summary_writer, f'simulation/{self.client_id}', step_games, invalid_actions_mask, game_hparams)
            game_stats = simulation.run_single_game(self.hparams, train, num_steps=-1)

            collection_time = perf_counter() - start_time
            self.summary_writer.add_scalar(f'collect/{self.client_id}/time', collection_time, self.generation)

            for i in range(4):
                valid_index = game_stats.episode_len > i
                children_visits = game_stats.children_visits[valid_index, :, i].float()
                actions = game_stats.actions[valid_index, i]

                if len(children_visits) > 0:
                    self.summary_writer.add_histogram(f'collect/{self.client_id}/children_visits{i}', children_visits, self.generation)
                    self.summary_writer.add_histogram(f'collect/{self.client_id}/actions{i}', actions, self.generation)

            self.summary_writer.add_scalar(f'collect/{self.client_id}/root_values', game_stats.root_values[:, 0].float().mean(), self.generation)

            episode_rewards = game_stats.rewards.float().sum(1)
            self.summary_writer.add_histogram(f'collect/{self.client_id}/rewards', episode_rewards, self.generation, bins=3)
            self.summary_writer.add_scalar(f'collect/{self.client_id}/mean_reward', episode_rewards.mean(), self.generation)
            self.summary_writer.add_scalars(f'collect/{self.client_id}/results', {
                'wins': (episode_rewards > 0).sum() / len(episode_rewards),
                'looses': (episode_rewards < 0).sum() / len(episode_rewards),
                'draws': (episode_rewards == 0).sum() / len(episode_rewards),
            }, self.generation)

            self.summary_writer.add_histogram(f'collect/{self.client_id}/train_steps', game_stats.episode_len, self.generation)
            self.summary_writer.add_scalars(f'collect/{self.client_id}/train_steps', {
                'min': game_stats.episode_len.min(),
                'max': game_stats.episode_len.max(),
                'mean': game_stats.episode_len.float().mean(),
                'median': game_stats.episode_len.float().median(),
            }, self.generation)

            self.send_game_stats(game_stats)

def run_process(client_id: str, hparams: Hparams):
    logfile = os.path.join(hparams.checkpoints_dir, f'{client_id}.log')
    logger = setup_logger(client_id, logfile, True)

    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    client = MuzeroCollectionClient(client_id, hparams, logger)
    while True:
        client.collect_episode()

        if client.generation < 0:
            time.sleep(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_id', type=int, default=0, help='Initial id for the clients')
    parser.add_argument('--num_clients', type=int, default=2, help='Number of clients to start')
    FLAGS = parser.parse_args()

    hparams = TicTacToeHparams()
    hparams.batch_size = 1024*4
    hparams.num_simulations = 200
    hparams.device = 'cuda:0'

    mp.set_start_method('spawn')

    processes = []
    for cid in range(FLAGS.start_id, FLAGS.start_id+FLAGS.num_clients):
        client_id = f'client{cid}'
        p = mp.Process(target=run_process, args=(client_id, hparams))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
