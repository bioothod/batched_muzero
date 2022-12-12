import logging
import os

import grpc
import muzero_pb2
import muzero_pb2_grpc

from hparams import Hparams
from logger import setup_logger

class MuzeroCollectionClient:
    def __init__(self, client_id: str, hparams: Hparams, logger: logging.Logger):
        self.logger = logger
        self.hparams = hparams
        self.client_id = client_id

        self.channel = grpc.insecure_channel(f'localhost:{hparams.server_port}')
        self.stub = muzero_pb2_grpc.MuzeroStub(self.channel)

        self.generation = 0

        resp = self.stub.WeightUpdateRequest(muzero_pb2.WeightRequest(
            generation=self.generation,
        ))

        self.logger.info(f'received weight update: generation: {self.generation} -> {resp.generation}, weights: {len(resp.weights)}')

def main():
    hparams = Hparams()

    client_id = 'client0'
    checkpoints_dir = 'checkpoints_1'
    logfile = os.path.join(checkpoints_dir, f'client.log')
    logger = setup_logger(client_id, logfile, True)

    client = MuzeroCollectionClient(client_id, hparams, logger)


if __name__ == '__main__':
    main()
