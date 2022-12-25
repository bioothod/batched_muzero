from collections import defaultdict
from concurrent import futures
from typing import List, Dict

import grpc
import logging
import pickle

import muzero_pb2
import muzero_pb2_grpc

from hparams import Hparams
from simulation import GameStats

class MuzeroServer(muzero_pb2_grpc.MuzeroServicer):
    generation: int

    def __init__(self, hparams: Hparams, logger: logging.Logger):
        self.hparams = hparams
        self.logger = logger

        self.generation = 0
        self.latest_serialized_weights = None

        self.all_games = defaultdict(list)

    def update_weights(self, generation: int, weights: bytes):
        self.generation = generation
        self.latest_serialized_weights = weights

    def move_games(self) -> Dict[int, List[GameStats]]:
        all_games = self.all_games
        self.all_games = defaultdict(list)
        return all_games

    def WeightUpdateRequest(self, request, context) -> muzero_pb2.WeightResponse:
        self.logger.debug(f'server: weights update request: generation: r{request.generation}, s{self.generation}')

        if request.generation >= self.generation:
            return muzero_pb2.WeightResponse(
                generation=self.generation,
            )

        return muzero_pb2.WeightResponse(
            generation=self.generation,
            weights=self.latest_serialized_weights,
        )

    def SendGameStats(self, request, context) -> muzero_pb2.Status:
        games = pickle.loads(request.stats)
        dst_list = self.all_games[request.generation]
        for game in games:
            dst_list.append(game.move(self.hparams.device))

        self.logger.debug(f'server: game stats update: '
                         f'generation: r{request.generation}, s{self.generation}, '
                         f'recv_games: {len(games)}, '
                         f'all_games_for_gen{request.generation}: {len(self.all_games[request.generation])}')

        return muzero_pb2.Status(
            generation=self.generation,
            status=0,
            message=f'Ok',
        )

def start_server(hparams: Hparams, logger: logging.Logger):
    options = (
        ('grpc.max_send_message_length', -1),
        ('grpc.max_receive_message_length', -1),
    )
    server = grpc.server(
        thread_pool=futures.ThreadPoolExecutor(max_workers=hparams.num_server_workers),
        options=options,
    )
    muzero_server = MuzeroServer(hparams, logger)
    muzero_pb2_grpc.add_MuzeroServicer_to_server(muzero_server, server)
    server.add_insecure_port(f'[::]:{hparams.server_port}')
    server.start()

    logger.info(f'server started to listen on port {hparams.server_port}')
    return server, muzero_server
