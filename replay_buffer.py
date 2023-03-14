from typing import List, Dict, NamedTuple, Optional

import random

from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch

from hparams import GenericHparams
from simulation import GameStats, TrainElement

class ReplayBuffer:
    def __init__(self, hparams: GenericHparams):
        self.hparams = hparams
        self.games = defaultdict(list)
        self.max_num_games = 1
        self.num_games_received = 0

    def add_game(self, generation: int, game: GameStats):
        self.games[generation].append(game)
        self.num_games_received += 1
        self.truncate()

    def truncate(self, max_generation=0):
        keys = deepcopy(list(self.games.keys()))
        for key in keys:
            if key < max_generation:
                del self.games[key]

        flatten_games = self.flatten_games()
        num_games = len(flatten_games)

        if num_games > self.max_num_games:
            to_remove = num_games - self.max_num_games

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

            if self.max_num_games < self.hparams.max_training_games:
                self.max_num_games += 1

    def num_games(self) -> int:
        return len(self.flatten_games())

    def flatten_games(self, min_key: int = 0) -> List[GameStats]:
        all_keys = sorted(list(self.games.keys()))
        all_games = []
        for key in all_keys:
            if key >= min_key:
                all_games += self.games[key]

        return all_games

    def sample(self, batch_size: int, all_games=[]) -> List[TrainElement]:
        if len(all_games) == 0:
            all_games = self.flatten_games()

        samples = []
        num_iterations = 0
        while len(samples) < batch_size and num_iterations < 10:
            game_stat = random.choice(all_games)

            random_init = np.random.randint(0, 2, size=len(game_stat.episode_len))
            high = np.maximum(random_init, game_stat.episode_len.cpu().numpy()-self.hparams.num_unroll_steps)
            start_pos = np.random.randint(0, high, len(game_stat.episode_len))
            start_pos = torch.from_numpy(start_pos).to(self.hparams.device)

            elms = game_stat.make_target(start_pos)
            samples += elms
            #samples.update(elms)
            num_iterations += 1

        return list(samples)
