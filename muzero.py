from typing import Dict, List, NamedTuple, Union

import os
import logging

from collections import defaultdict
from copy import deepcopy
from time import perf_counter

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

    init_lr = 1e-4
    min_lr = 1e-5

    train_batch_size = 4
    training_games_window_size = 4

def roll_by_gather(mat,dim, shifts: torch.LongTensor):
    # assumes 2D array
    n_rows, n_cols = mat.shape

    if dim==0:
        #print(mat)
        arange1 = torch.arange(n_rows, device=shifts.device).view((n_rows, 1)).repeat((1, n_cols))
        #print(arange1)
        arange2 = (arange1 - shifts) % n_rows
        #print(arange2)
        return torch.gather(mat, 0, arange2)
    elif dim==1:
        arange1 = torch.arange(n_cols, device=shifts.device).view(( 1,n_cols)).repeat((n_rows,1))
        #print(arange1)
        arange2 = (arange1 - shifts) % n_cols
        #print(arange2)
        return torch.gather(mat, 1, arange2)

class Inference(mcts.Inference):
    def __init__(self, hparams: Hparams, logger: logging.Logger):
        self.logger = logger
        self.hparams = hparams

        net_hparams = networks.NetworkParams(observation_shape=hparams.state_shape)
        self.logger.info(f'inference: network_params:\n{net_hparams}')

        self.representation = networks.Representation(net_hparams).to(hparams.device)
        self.prediction = networks.Prediction(net_hparams).to(hparams.device)
        self.dynamic = networks.Dynamic(net_hparams).to(hparams.device)

        self.models = [self.representation, self.prediction, self.dynamic]

    def train(self, mode: bool):
        for model in self.models:
            model.train(mode)

    def zero_grad(self):
        for model in self.models:
            model.zero_grad()

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

class Sample:
    num_steps: int
    player_id: int
    initial_game_state: torch.Tensor

    child_visits: torch.Tensor
    values: torch.Tensor
    last_rewards: torch.Tensor

    actions: torch.Tensor

class SampleDataset:
    def __init__(self, hparams: Hparams):
        self.samples = []

        self.num_unroll_steps = hparams.num_unroll_steps

    def __len__(self):
        return len(self.samples)

    def append(self, sample_dict: Dict[str, torch.Tensor]):
        game_states = sample_dict['game_states']
        actions = sample_dict['actions']
        episode_len = sample_dict['actions']
        values = sample_dict['values']
        last_rewards = sample_dict['last_rewards']
        child_visits = sample_dict['child_visits']
        player_ids = sample_dict['player_ids']

        for sample_idx in range(len(game_states)):
            s = Sample()

            s.num_steps = episode_len[sample_idx]
            s.player_id = player_ids[sample_idx]
            s.initial_game_state = game_states[sample_idx]

            s.child_visits = child_visits[sample_idx]
            s.values = values[sample_idx]
            s.last_rewards = last_rewards[sample_idx]
            s.actions = actions[sample_idx]

            self.samples.append(s)

    def __getitem__(self, index):
        s = self.samples[index]

class GameStats:
    def __init__(self, hparams: Hparams, logger: logging.Logger):
        self.logger = logger
        self.hparams = hparams

        self.episode_len = torch.zeros(hparams.batch_size, dtype=torch.int64, device=hparams.device)
        self.rewards = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=hparams.dtype, device=hparams.device)
        self.root_values = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=hparams.dtype, device=hparams.device)
        self.child_visits = torch.zeros(hparams.batch_size, hparams.num_actions, hparams.max_episode_len, dtype=hparams.dtype, device=hparams.device)
        self.actions = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.int64, device=hparams.device)
        self.player_ids = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.int64, device=hparams.device)
        self.dones = torch.zeros(hparams.batch_size, dtype=torch.bool, device=hparams.device)
        self.game_states = torch.zeros([hparams.batch_size, hparams.max_episode_len] + hparams.state_shape, dtype=hparams.dtype, device=hparams.device)

        self.stored_tensors = {
            'rewards': self.rewards,
            'root_values': self.root_values,
            'child_visits': self.child_visits,
            'actions': self.actions,
            'player_ids': self.player_ids,
            'dones': self.dones,
            'game_states': self.game_states,
        }

    def __len__(self):
        return self.episode_len.sum().item()

    def append(self, index: torch.Tensor, tensors_dict: Dict[str, torch.Tensor]):
        episode_len = self.episode_len[index]
        for key, value in tensors_dict.items():
            if not key in self.stored_tensors:
                msg = f'invalid key: {key}, tensor shape: {value.shape}, available keys: {list(self.stored_tensors.keys())}'
                self.logger.critical(msg)
                raise ValueError(msg)

            tensor = self.stored_tensors[key]
            if key == 'child_visits':
                tensor[index, :, episode_len] = value.detach().clone()
            elif key == 'dones':
                tensor[index] = value.detach().clone()
            else:
                tensor[index, episode_len] = value.detach().clone()

        self.episode_len[index] += 1

    def make_target(self, start_index: torch.Tensor):
        target_values = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=self.hparams.device)
        target_last_rewards = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=self.hparams.device)
        target_child_visits = torch.zeros(len(start_index), self.hparams.num_actions, self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=self.hparams.device)
        taken_actions = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=torch.int64, device=self.hparams.device)
        player_ids = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=torch.int64, device=self.hparams.device)

        sample_len = torch.zeros(len(start_index), dtype=torch.int64, device=self.hparams.device)

        batch_index = torch.arange(len(start_index), dtype=torch.int64, device=self.root_values.device)
        game_states = self.game_states[batch_index, start_index]

        for unroll_index in range(0, self.hparams.num_unroll_steps+1):
            current_index = start_index + unroll_index
            bootstrap_index = current_index + self.hparams.td_steps

            bootstrap_update_index = bootstrap_index < self.episode_len
            valid_batch_index = batch_index[bootstrap_update_index]

            if False:
                self.logger.info(f'{unroll_index}: '
                             f'start_index: {start_index.cpu().numpy()}, '
                             f'current_index: {current_index.cpu().numpy()}, '
                             f'bootstrap_index: {bootstrap_index.cpu().numpy()}, '
                             f'bootstrap_update_index: {bootstrap_update_index.cpu().numpy()}/{bootstrap_update_index.shape}, '
                             f'valid_batch_index: {valid_batch_index.cpu().numpy()}'
                             )
            values = torch.zeros(len(self.root_values), dtype=self.root_values.dtype, device=self.root_values.device)
            if bootstrap_update_index.sum() > 0:
                values[bootstrap_update_index] = self.root_values[valid_batch_index, bootstrap_index] * self.hparams.discount ** self.hparams.td_steps

            discount_mult = torch.logspace(0, self.hparams.td_steps, self.rewards.shape[1], base=self.hparams.discount).to(self.hparams.device)
            discount_mult = discount_mult.unsqueeze(0)
            discount_mult = discount_mult.tile([len(values), 1])

            last_index = torch.minimum(bootstrap_index, self.episode_len)
            all_rewards_index = torch.arange(self.rewards.shape[1], device=self.hparams.device).long().unsqueeze(0).tile([len(start_index), 1])

            discount_mult = roll_by_gather(discount_mult, 1, start_index.unsqueeze(1))
            discount_mult = torch.where(all_rewards_index < current_index.unsqueeze(1), 0, discount_mult)
            discount_mult = torch.where(all_rewards_index >= last_index.unsqueeze(1), 0, discount_mult)

            masked_rewards = torch.where(all_rewards_index < current_index.unsqueeze(1), 0, self.rewards)
            masked_rewards = torch.where(all_rewards_index >= last_index.unsqueeze(1), 0, masked_rewards)

            discounted_rewards = self.rewards * discount_mult
            discounted_rewards = discounted_rewards.sum(1)
            values += discounted_rewards

            valid_index = current_index < self.episode_len

            current_valid_index = current_index[valid_index]
            target_values[valid_index, unroll_index] = values[valid_index]
            target_child_visits[valid_index, :, unroll_index] = self.child_visits[valid_index, :, current_valid_index]

            if unroll_index > 0:
                target_last_rewards[valid_index, unroll_index] = self.rewards[valid_index, current_valid_index-1]
            taken_actions[valid_index, unroll_index] = self.actions[valid_index, current_valid_index]
            player_ids[valid_index, unroll_index] = self.player_ids[valid_index, current_valid_index]
            sample_len[valid_index] += 1

        return {
            'values': target_values,
            'last_rewards': target_last_rewards,
            'child_visits': target_child_visits,
            'game_states': game_states,
            'actions': taken_actions,
            'sample_len': sample_len,
            'player_ids': player_ids,
        }

class Train:
    def __init__(self, hparams: Hparams, inference: Inference, logger: logging.Logger):
        self.hparams = hparams
        self.logger = logger
        self.inference = inference

        self.num_train_steps = 0
        self.game_stats = GameStats(hparams, self.logger)

    def run_simulations(self, initial_player_id: torch.Tensor, initial_game_states: torch.Tensor):
        local_hparams = deepcopy(self.hparams)
        local_hparams.batch_size = len(initial_game_states)
        tree = mcts.Tree(local_hparams, self.inference, self.logger)
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
        self.logger.info(f'train: {self.num_train_steps:2d}: '
                         f'batch_size: {batch_size}, '
                         f'num_simulations: {self.hparams.num_simulations}, '
                         f'time: {simulation_time:.3f} sec, '
                         f'one_sim: {one_sim_ms:3d} ms')

        actions = self.select_actions(batch_size, tree)

        children_index = tree.children_index(batch_index, node_index)
        children_visit_counts = tree.visit_count[batch_index].gather(1, children_index)
        children_sum_visits = children_visit_counts.sum(1)
        children_visits = children_visit_counts / children_sum_visits.unsqueeze(1)
        root_values = tree.value(batch_index, node_index.unsqueeze(1)).squeeze(1)

        return actions, children_visits, root_values

    def update_train_steps(self):
        self.num_train_steps += 1

    def select_actions(self, batch_size: int, tree: mcts.Tree):
        batch_index = torch.arange(batch_size, device=self.hparams.device).long()
        node_index = torch.zeros(batch_size, device=self.hparams.device).long()
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
    player_ids = torch.ones(hparams.batch_size, device=hparams.device).long() * hparams.player_ids[0]

    active_games_index = torch.arange(hparams.batch_size).long().to(hparams.device)

    while True:
        active_game_states = game_states[active_games_index]
        active_player_ids = player_ids[active_games_index]
        actions, children_visits, root_values = train.run_simulations(active_player_ids, active_game_states)
        new_game_states, rewards, dones = connectx_impl.step_games(game_hparams, active_game_states, active_player_ids, actions)
        game_states[active_games_index] = new_game_states

        train.game_stats.append(active_games_index, {
            'child_visits': children_visits,
            'root_values': root_values,
            'game_states': active_game_states,
            'rewards': rewards,
            'dones': dones,
            'player_ids': active_player_ids,
        })
        train.update_train_steps()

        if dones.sum() == len(dones):
            break

        player_ids = mcts.player_id_change(hparams, player_ids)
        active_games_index = active_games_index[dones != True]

    return train.game_stats

def main():
    hparams = Hparams()

    hparams.batch_size = 2
    hparams.num_simulations = 64
    hparams.training_games_window_size = 4
    hparams.device = torch.device('cuda:0')

    logfile = os.path.join(hparams.checkpoints_dir, 'muzero.log')
    os.makedirs(hparams.checkpoints_dir, exist_ok=True)
    logger = setup_logger('muzero', logfile, hparams.log_to_stdout)

    inference = Inference(hparams, logger)
    representation_opt = torch.optim.Adam(inference.representation.parameters(), lr=hparams.init_lr)
    prediction_opt = torch.optim.Adam(inference.prediction.parameters(), lr=hparams.init_lr)
    dynamic_opt = torch.optim.Adam(inference.dynamic.parameters(), lr=hparams.init_lr)
    optimizers = [representation_opt, prediction_opt, dynamic_opt]

    ce_loss = nn.CrossEntropyLoss()
    scalar_loss = nn.MSELoss()

    all_games: List[GameStats] = []
    while True:
        inference.train(False)
        train = Train(hparams, inference, logger)
        game_stats = run_single_game(hparams, train)
        all_games.append(game_stats)

        if len(all_games) > hparams.training_games_window_size:
            all_games.pop(0)

        inference.train(True)
        for game_stat in all_games:
            max_episode_len = game_stat.episode_len.max().item() - hparams.num_unroll_steps
            max_episode_len = max(0, max_episode_len)

            for _ in range(max_episode_len + 1):
                if max_episode_len == 0:
                    start_pos = torch.zeros(hparams.batch_size, dtype=torch.int64, device=hparams.device)
                else:
                    start_pos = torch.randint(0, max_episode_len, (hparams.batch_size,)).long().to(hparams.device)


                sample_dict = game_stat.make_target(start_pos)

                game_states = sample_dict['game_states']
                actions = sample_dict['actions']
                sample_len = sample_dict['sample_len']
                values = sample_dict['values']
                last_rewards = sample_dict['last_rewards']
                child_visits = sample_dict['child_visits']
                player_ids = sample_dict['player_ids']

                inference.zero_grad()
                policy_loss = torch.zeros(1, requires_grad=True).float().to(game_states.device)
                value_loss = torch.zeros(1, requires_grad=True).float().to(game_states.device)
                reward_loss = torch.zeros(1, requires_grad=True).float().to(game_states.device)

                logger.info(f'game_states: {game_states.shape}, player_ids: {player_ids.shape}, actions: {actions.shape}, child_visits: {child_visits.shape}')
                out = inference.initial(player_ids[:, 0], game_states)
                policy_loss += ce_loss(out.policy_logits, child_visits[:, :, 0])
                value_loss += scalar_loss(out.value, values[:, 0])

                batch_index = torch.arange(len(game_states), dtype=torch.int64, device=game_states.device)
                for step_idx in range(1, sample_len.max()):
                    valid_states_index = batch_index[step_idx < sample_len]

                    active_hidden_states = out.hidden_state[valid_states_index]
                    active_actions = actions[valid_states_index, step_idx-1]
                    out = inference.recurrent(active_hidden_states, active_actions)

                    active_last_rewards = last_rewards[valid_states_index, step_idx]

                    active_values = values[valid_states_index, step_idx]
                    active_child_visits = child_visits[valid_states_index, :, step_idx]

                    policy_loss += ce_loss(out.policy_logits, active_child_visits)
                    value_loss += scalar_loss(out.value, active_values)
                    if step_idx > 0:
                        reward_loss += scalar_loss(out.reward, active_last_rewards)

                loss = policy_loss + value_loss + reward_loss
                loss.backward()

                for opt in optimizers:
                    opt.step()

if __name__ == '__main__':
    main()
