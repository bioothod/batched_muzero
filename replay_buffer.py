from typing import List, Dict, NamedTuple

import random

from collections import defaultdict
from copy import deepcopy

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

    def flatten_games(self) -> List[GameStats]:
        all_keys = sorted(list(self.games.keys()))
        all_games = []
        for key in all_keys:
            all_games += self.games[key]

        return all_games

    def sample(self, batch_size: int, all_games=[]) -> List[TrainElement]:
        if len(all_games) == 0:
            all_games = self.flatten_games()

        samples = []
        num_iterations = 0
        while len(samples) < batch_size and num_iterations < 200:
            game_stat = random.choice(all_games)
            random_index = random.sample(range(len(game_stat.episode_len)), 32)
            game_stat = game_stat.index(random_index)


            max_start_pos = game_stat.episode_len.max() - self.hparams.num_unroll_steps
            if max_start_pos > 0:
                start_pos = torch.randint(low=0, high=max_start_pos, size=(len(game_stat.episode_len),)).to(self.hparams.device)
            else:
                start_pos = torch.zeros_like(game_stat.episode_len)

            start_pos = torch.where(start_pos+self.hparams.num_unroll_steps>=game_stat.episode_len, 0, game_stat.episode_len-self.hparams.num_unroll_steps-1)
            start_pos = torch.maximum(start_pos, torch.zeros_like(start_pos))

            elms = game_stat.make_target(start_pos)
            samples += elms
            num_iterations += 1

        return list(samples)
