from concurrent import futures
import logging

import grpc

from hparams import Hparams

import muzero_pb2
import muzero_pb2_grpc

class MuzeroServer(muzero_pb2_grpc.Muzero):
    generation: int

    def __init__(self, hparams: Hparams, logger: logging.Logger):
        self.hparams = hparams
        self.logger = logger

        self.generation = 0
        self.latest_serialized_weights = None

    def update_weights(self, generation: int, weights: bytes):
        self.logger.info(f'server: weights updated: generation: {self.generation} -> {generation}, weights: {len(weights)} bytes')

        self.generation = generation
        self.latest_serialized_weights = weights

    def WeightUpdateRequest(self, request, context) -> muzero_pb2.WeightResponse:
        self.logger.info(f'server: weights update request: generation: {request.generation} -> {self.generation}')
        return muzero_pb2.WeightResponse(
            generation=self.generation,
            weights=self.latest_serialized_weights,
        )

    def SendGameStats(self, request, context) -> muzero_pb2.Status:
        self.logger.info(f'server: game stats update: generation: {request.generation} -> {self.generation}')
        return muzero_pb2.Status(
            generation=self.generation,
            status=0,
            message=f'Ok',
        )

def start_server(hparams: Hparams, logger: logging.Logger):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=hparams.num_server_workers))
    muzero_server = MuzeroServer(hparams, logger)
    muzero_pb2_grpc.add_MuzeroServicer_to_server(muzero_server, server)
    server.add_insecure_port(f'[::]:{hparams.server_port}')
    server.start()

    logger.info(f'server started to listen on port {hparams.server_port}')
    return server, muzero_server
