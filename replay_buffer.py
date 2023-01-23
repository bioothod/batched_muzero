from typing import List, Dict, NamedTuple

import random

from collections import defaultdict

import torch

from hparams import GenericHparams
from simulation import GameStats, TrainElement

class ReplayBuffer:
    def __init__(self, hparams: GenericHparams):
        self.hparams = hparams
        self.games = defaultdict(list)


    def add_game(self, generation: int, game: GameStats):
        self.games[generation].append(game)
        self.truncate()

    def truncate(self):
        flatten_games = self.flatten_games()
        num_games = len(flatten_games)

        if num_games > self.hparams.max_training_games:
            to_remove = num_games - self.hparams.max_training_games

            all_keys = sorted(list(self.games.keys()))
            for gen in all_keys:
                games = self.games[gen]
                if to_remove >= len(games):
                    to_remove -= len(games)
                    num_games -= len(games)
                    del self.games[gen]
                else:
                    self.games[gen] = games[to_remove:]
                    num_games -= to_remove
                    to_remove = 0
                    break

    def num_games(self) -> int:
        return len(self.flatten_games())

    def flatten_games(self) -> List[GameStats]:
        all_keys = sorted(list(self.games.keys()))
        all_games = []
        for key in all_keys:
            all_games += self.games[key]

        return all_games

    def sample(self, batch_size: int) -> List[TrainElement]:
        all_games = self.flatten_games()

        samples = []
        while len(samples) < batch_size:
            game_stat = random.choice(all_games)

            start_pos = torch.randint(low=0, high=game_stat.episode_len.max(), size=(len(game_stat.episode_len),)).to(self.hparams.device)
            start_pos -= self.hparams.num_unroll_steps
            start_pos = torch.where(start_pos <= 0, 0, start_pos)
            start_pos = torch.where(start_pos+self.hparams.num_unroll_steps>=game_stat.episode_len, 0, start_pos)
            #self.logger.info(f'epoch: {epoch}: start_pos: {start_pos.shape}: {start_pos[:10]}, episode_len: {game_stat.episode_len[:10]}')

            elms = game_stat.make_target(start_pos)
            samples += elms

        return samples
