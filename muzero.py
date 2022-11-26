import os
import logging

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

class VisitStats:
    def __init__(self):
        self.children_visits = []
        self.root_values = []

    def append(self, children_visits: torch.Tensor, root_values: torch.Tensor):
        self.children_visits.append(children_visits)
        self.root_values.append(root_values)

class Train:
    def __init__(self, hparams: Hparams):
        self.hparams = hparams

        self.num_train_steps = 0

        logfile = os.path.join(hparams.checkpoints_dir, 'muzero.log')
        os.makedirs(hparams.checkpoints_dir, exist_ok=True)
        self.logger = setup_logger('muzero', logfile, hparams.log_to_stdout)

        self.inference = Inference(hparams, self.logger)
        self.visit_stats = VisitStats()

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
        self.visit_stats.append(children_visits, root_values)

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

class ActionHistory:
    def __init__(self):
        pass

    def __len__(self):
        return 0

def main():
    hparams = Hparams()

    hparams.batch_size = 1024
    hparams.num_simulations = 400
    hparams.device = torch.device('cuda:0')
    train = Train(hparams)
    action_history = ActionHistory()

    game_hparams = connectx_impl.Hparams()
    game_states = torch.zeros(hparams.batch_size, *hparams.state_shape).float().to(hparams.device)
    player_id = torch.ones(hparams.batch_size, device=hparams.device).long() * hparams.player_ids[0]

    while len(action_history) < hparams.max_num_actions:
        actions = train.run_simulations(player_id, game_states)
        game_states, rewards, dones = connectx_impl.step_games(game_hparams, game_states, player_id, actions)
        print(game_states[0, 0].long())

        player_id = mcts.player_id_change(hparams, player_id)

if __name__ == '__main__':
    main()
