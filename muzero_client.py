import grpc
import io
import logging
import os
import pickle
import time

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from time import perf_counter

import muzero_pb2
import muzero_pb2_grpc

from hparams import Hparams
from logger import setup_logger
from networks import Inference
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
        self.global_step = 0


    def update_weights(self):
        resp = self.stub.WeightUpdateRequest(muzero_pb2.WeightRequest(
            generation=self.generation,
        ))

        self.logger.info(f'received weight update: generation: {self.generation} -> {resp.generation}, weights: {len(resp.weights)}')

        if len(resp.weights) > 0:
            self.load_weights(resp.weights)
            self.generation = resp.generation
            self.summary_writer.add_scalar(f'collect/{self.client_id}/generation', self.generation, self.global_step)

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

            train = simulation.Train(self.hparams, self.inference, self.logger)
            game_stats = simulation.run_single_game(self.hparams, train, num_steps=-1)

            collection_time = perf_counter() - start_time
            self.summary_writer.add_scalar(f'collect/{self.client_id}/time', collection_time, self.global_step)

            for i in range(4):
                self.summary_writer.add_histogram(f'collect/{self.client_id}/children_visits{i}', game_stats.children_visits[:, :, i], self.global_step)
                self.summary_writer.add_histogram(f'collect/{self.client_id}/actions{i}', game_stats.actions[:, i], self.global_step)

            self.summary_writer.add_scalar(f'collect/{self.client_id}/root_values', game_stats.root_values[:, 0].mean(), self.global_step)
            self.summary_writer.add_scalar(f'collect/{self.client_id}/rewards', game_stats.rewards.mean(), self.global_step)
            self.summary_writer.add_scalar(f'collect/{self.client_id}/train_steps', train.num_train_steps, self.global_step)

            self.send_game_stats(game_stats)

            self.global_step += 1


def run_process(client_id: str, hparams: Hparams):
    logfile = os.path.join(hparams.checkpoints_dir, f'{client_id}.log')
    logger = setup_logger(client_id, logfile, True)

    client = MuzeroCollectionClient(client_id, hparams, logger)
    while True:
        client.collect_episode()

        if client.generation < 0:
            time.sleep(1)

def main():
    hparams = Hparams()
    hparams.batch_size = 512
    hparams.num_simulations = 800
    hparams.device = 'cuda:0'

    mp.set_start_method('spawn')

    processes = []
    for cid in range(2):
        client_id = f'client{cid}'
        p = mp.Process(target=run_process, args=(client_id, hparams))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
