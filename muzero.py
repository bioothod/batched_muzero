from typing import Dict, List

import os
import logging

from time import perf_counter
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import connectx_impl
from logger import setup_logger
import mcts
from mcts import NetworkOutput
import networks

class Hparams(mcts.Hparams):
    checkpoints_dir = 'checkpoints_1'
    log_to_stdout = True
    max_num_actions = 1024*16

    td_steps = 42
    num_unroll_steps = 5

class Inference(mcts.Inference):
    def __init__(self, hparams: Hparams, logger: logging.Logger):
        self.logger = logger
        self.hparams = hparams

        net_hparams = networks.NetworkParams(observation_shape=hparams.state_shape)
        self.logger.info(f'inference: network_params:\n{net_hparams}')

        self.representation = networks.Representation(net_hparams).to(hparams.device)
        self.prediction = networks.Prediction(net_hparams).to(hparams.device)
        self.dynamic = networks.Dynamic(net_hparams).to(hparams.device)

    def create_states(self, player_id: torch.Tensor, game_states: torch.Tensor):
        batch_size = len(game_states)
        input_shape = list(game_states.shape[1:]) # remove batch size

        states = torch.zeros(1 + len(self.hparams.player_ids), batch_size, *input_shape[1:]).to(self.hparams.device)

        player_id_exp = player_id.unsqueeze(1)
        player_id_exp = player_id_exp.tile([1, np.prod(input_shape)]).view([batch_size] + input_shape[1:]).to(self.hparams.device)
        states[0, ...] = player_id_exp
        for player_index, local_player_id in enumerate(self.hparams.player_ids):
            index = game_states == local_player_id
            index = index.squeeze(1)
            states[1 + player_index, index] = 1

        states = torch.transpose(states, 1, 0)
        return states

    def initial(self, player_id: torch.Tensor, game_states: torch.Tensor) -> NetworkOutput:
        with torch.no_grad():
            batch_size = game_states.shape[0]
            states = self.create_states(player_id, game_states)
            hidden_states = self.representation(states)
            policy_logits, values = self.prediction(hidden_states)
            rewards = torch.zeros(batch_size).float().to(self.hparams.device)
            #self.logger.info(f'inference: initial: game_states: {game_states.shape}: hidden_states: {hidden_states.shape}, policy_logits: {policy_logits.shape}, rewards: {rewards.shape}')
            return NetworkOutput(reward=rewards, hidden_state=hidden_states, policy_logits=policy_logits, value=values)

    def recurrent(self, hidden_states: torch.Tensor, actions: torch.Tensor) -> NetworkOutput:
        with torch.no_grad():
            policy_logits, values = self.prediction(hidden_states)
            actions_exp = F.one_hot(actions, self.hparams.num_actions)
            actions_exp = actions_exp.unsqueeze(1)
            fill = hidden_states.shape[2]
            actions_exp = actions_exp.tile([1, fill, 1])
            actions_exp = actions_exp.unsqueeze(1)
            inputs = torch.cat([hidden_states, actions_exp], 1)
            new_hidden_states, rewards = self.dynamic(inputs)
            #self.logger.info(f'inference: recurrent: hidden_states: {hidden_states.shape}, policy_logits: {policy_logits.shape}, rewards: {rewards.shape}, values: {values.shape}')
            return NetworkOutput(reward=rewards, hidden_state=new_hidden_states, policy_logits=policy_logits, value=values)

class GameStats:
    def __init__(self, hparams: Hparams, logger: logging.Logger):
        self.logger = logger
        self.hparams = hparams

        self.episode_len = torch.zeros(hparams.batch_size, dtype=torch.int64, device=hparams.device)
        self.rewards = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=hparams.dtype, device=hparams.device)
        self.root_values = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=hparams.dtype, device=hparams.device)
        self.child_visits = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=hparams.dtype, device=hparams.device)
        self.actions = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.int64, device=hparams.device)
        self.player_ids = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.int64, device=hparams.device)
        self.dones = torch.zeros(hparams.batch_size, dtype=torch.bool, device=hparams.device)
        self.game_states = torch.zeros([hparams.batch_size, hparams.max_episode_len] + hparams.state_shape, dtype=hparams.dtype, device=hparams.device)

    def __len__(self):
        return self.episode_len.sum().item()

    def append(self, tensors_dict: Dict[str, torch.Tensor]):
        for key, value in tensors_dict.items():
            if key == 'rewards':
                self.rewards[:, self.episode_len] = value.detach().clone()
                continue
            if key == 'root_values':
                self.root_values[:, self.episode_len] = value.detach().clone()
                continue
            if key == 'child_visits':
                self.child_visits[:, self.episode_len] = value.detach().clone()
                continue
            if key == 'actions':
                self.actions[:, self.episode_len] = value.detach().clone()
                continue
            if key == 'game_state':
                self.game_states[:, self.episode_len] = value.detach().clone()
                continue
            if key == 'player_id':
                self.player_ids[:, self.episode_len] = value.detach().clone()
                continue
            if key == 'dones':
                self.episode_len[self.dones != True] += 1
                self.dones = value.detach().clone()
                continue

            msg = f'invalid key: {key}, tensor shape: {value.shape}'
            self.logger.critical(msg)
            raise ValueError(msg)

    def make_target(self, start_index: torch.Tensor):
        target_values = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=self.hparams.device)
        target_last_rewards = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=self.hparams.device)
        target_child_visits = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=self.hparams.device)
        taken_actions = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=torch.int64, device=self.hparams.device)

        episode_len = self.episode_len - start_index
        game_states = self.game_states[:, start_index]

        for unroll_index in range(self.hparams.num_unroll_steps + 1):
            current_index = start_index + unroll_index
            bootstrap_index = current_index + self.hparams.td_steps

            bootstrap_update_index = bootstrap_index < self.episode_len
            batch_index = torch.arange(len(self.root_values), dtype=torch.int64, device=self.root_values.device)
            batch_index = batch_index[bootstrap_update_index]
            values = torch.zeros(len(self.root_values), dtype=self.root_values.dtype, device=self.root_values.device)
            values[bootstrap_update_index] = self.root_values[batch_index, bootstrap_index] * self.hparams.discount ** self.hparams.td_steps

            discount_mult = torch.logspace(0, self.hparams.td_steps, self.hparams.td_steps+1, base=self.hparams.discount)
            discount_mult = discount_mult.unsqueeze(0)
            discount_mult = discount_mult.tile([len(values), 1])

            last_index = torch.minimum(bootstrap_index, self.episode_len)
            values += self.rewards[:, current_index:last_index] * discount_mult[:, :last_index]

            valid_index = current_index < self.episode_len

            target_values[valid_index, unroll_index] = values[valid_index]
            target_last_rewards[valid_index, unroll_index] = self.rewards[valid_index, current_index]
            target_child_visits[valid_index, unroll_index] = self.child_visits[valid_index, current_index]
            taken_actions[valid_index, unroll_index] = self.actions[valid_index, current_index]

        return {
            'values': target_values,
            'last_rewards': target_last_rewards,
            'child_visits': target_child_visits,
            'game_states': game_states,
            'actions': taken_actions,
            'episode_len': episode_len,
        }

class Train:
    def __init__(self, hparams: Hparams, inference: Inference, logger: logging.Logger):
        self.hparams = hparams
        self.logger = logger
        self.inference = inference

        self.num_train_steps = 0
        self.game_stats = GameStats(hparams, self.logger)

    def run_simulations(self, initial_player_id: torch.Tensor, initial_game_states: torch.Tensor):
        tree = mcts.Tree(self.hparams, self.inference, self.logger)
        tree.player_id[:, 0] = initial_player_id

        batch_size = len(initial_game_states)
        batch_index = torch.arange(batch_size).long()
        node_index = torch.zeros(batch_size).long().to(self.hparams.device)

        out = self.inference.initial(initial_player_id, initial_game_states)

        episode_len = torch.ones(len(node_index)).long().to(self.hparams.device)
        search_path = torch.zeros(len(node_index), 1).long().to(self.hparams.device)

        tree.store_states(search_path, episode_len, out.hidden_state)

        tree.expand(initial_player_id, batch_index, node_index, out.policy_logits)

        start_simulation_time = perf_counter()

        for _ in range(self.hparams.num_simulations):
            search_path, episode_len = tree.run_one()
        simulation_time = perf_counter() - start_simulation_time
        one_sim_ms = int(simulation_time / self.hparams.num_simulations * 1000)
        self.logger.info(f'train: batch_size: {batch_size}, num_simulations: {self.hparams.num_simulations}, time: {simulation_time:.3f} sec, avg: {one_sim_ms} ms')

        actions = self.select_actions(tree)

        children_index = tree.children_index(batch_index, node_index)
        children_visit_counts = tree.visit_count[batch_index].gather(1, children_index)
        children_sum_visits = children_visit_counts.sum(1)
        children_visits = children_visit_counts / children_sum_visits.unsqueeze(1)
        root_values = tree.value(batch_index, node_index.unsqueeze(1))

        self.game_stats.append({
            'child_visits': children_visits,
            'root_values': root_values,
            'game_states': initial_game_states,
        })

        return actions

    def update_train_steps(self):
        self.num_train_steps += 1

    def select_actions(self, tree: mcts.Tree):
        batch_index = torch.arange(self.hparams.batch_size, device=self.hparams.device).long()
        node_index = torch.zeros(self.hparams.batch_size, device=self.hparams.device).long()
        children_index = tree.children_index(batch_index, node_index)
        visit_counts = tree.visit_count[batch_index].gather(1, children_index)

        if self.num_train_steps >= 30:
            actions = torch.argmax(visit_counts, 1)
            return actions

        temperature = 1.0 # play according to softmax distribution

        dist = torch.pow(visit_counts.float(), 1 / temperature)
        actions = torch.multinomial(dist, 1)
        return actions.squeeze(1)

def run_single_game(hparams: Hparams, train: Train):
    game_hparams = connectx_impl.Hparams()
    game_states = torch.zeros(hparams.batch_size, *hparams.state_shape).float().to(hparams.device)
    player_id = torch.ones(hparams.batch_size, device=hparams.device).long() * hparams.player_ids[0]

    while True:
        actions = train.run_simulations(player_id, game_states)
        game_states, rewards, dones = connectx_impl.step_games(game_hparams, game_states, player_id, actions)
        #print(game_states[0, 0].long())

        train.game_stats.append({
            'rewards': rewards,
            'dones': dones,
            'player_id': player_id,
        })
        train.update_train_steps()

        if dones.sum() == len(dones):
            break

        player_id = mcts.player_id_change(hparams, player_id)

    return train.game_stats

def train_network(inference: Inference, player_ids: torch.Tensor, actions: torch.Tensor, target_dict: Dict[str, List[torch.Tensor]]):
    pass

def main():
    hparams = Hparams()

    hparams.batch_size = 1024
    hparams.num_simulations = 400
    hparams.device = torch.device('cuda:0')

    logfile = os.path.join(hparams.checkpoints_dir, 'muzero.log')
    os.makedirs(hparams.checkpoints_dir, exist_ok=True)
    logger = setup_logger('muzero', logfile, hparams.log_to_stdout)

    inference = Inference(hparams, logger)

    all_games_window_size = 4
    all_games: List[GameStats] = []
    while True:
        train = Train(hparams, inference, logger)
        game_stats = run_single_game(hparams, train)
        all_games.append(game_stats)

        if len(all_games) > all_games_window_size:
            all_games.pop(0)

        all_actions = []
        all_player_ids = []
        target_dict = defaultdict(list)
        for game_stat in all_games:
            max_episode_len = game_stat.episode_len.max()
            if max_episode_len <= hparams.num_unroll_steps:
                start_pos = torch.zeros(hparams.batch_size, dtype=torch.int64, device=hparams.device)
            else:
                start_pos = torch.randn(0, max_episode_len, (hparams.batch_size)).long().to(hparams.device)

            target_sample = game_stat.make_target(start_pos)
            for key, value in target_sample.items():
                target_dict[key].append(value)

            actions = game_stat.actions[:, start_pos]
            all_actions.append(actions)
            player_ids = game_stat.player_ids[:, start_pos]
            all_player_ids.append(player_ids)


        all_actions = torch.cat(all_actions, 0)
        all_player_ids = torch.cat(all_player_ids, 0)
        train_network(inference, all_player_ids, all_actions, target_dict)


if __name__ == '__main__':
    main()
