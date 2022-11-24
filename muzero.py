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

class Inference:
    def __init__(self, hparams: Hparams, logger: logging.Logger):
        self.num_actions = hparams.num_actions
        self.logger = logger
        self.default_reward = hparams.default_reward
        self.player_ids = hparams.player_ids
        self.num_actions = hparams.num_actions

        net_hparams = networks.NetworkParams(observation_shape=hparams.state_shape)
        self.logger.info(f'inference: network_params:\n{net_hparams}')

        self.representation = networks.Representation(net_hparams)
        self.prediction = networks.Prediction(net_hparams)
        self.dynamic = networks.Dynamic(net_hparams)

    def create_states(self, player_id: torch.Tensor, game_states: torch.Tensor):
        batch_size = len(game_states)
        input_shape = list(game_states.shape[1:]) # remove batch size

        states = torch.zeros(1 + len(self.player_ids), batch_size, *input_shape[1:])

        player_id_exp = player_id.unsqueeze(1)
        player_id_exp = player_id_exp.tile([1, np.prod(input_shape)]).view([batch_size] + input_shape[1:])
        states[0, ...] = player_id_exp
        for player_index, local_player_id in enumerate(self.player_ids):
            index = game_states == local_player_id
            index = index.squeeze(1)
            states[1 + player_index, index] = 1

        states = torch.transpose(states, 1, 0)
        return states

    def initial(self, player_id: torch.Tensor, game_states: torch.Tensor) -> NetworkOutput:
        batch_size = game_states.shape[0]
        states = self.create_states(player_id, game_states)
        hidden_states = self.representation(states)
        policy_logits, values = self.prediction(hidden_states)
        rewards = torch.zeros(batch_size).float()
        #self.logger.info(f'inference: initial: game_states: {game_states.shape}: hidden_states: {hidden_states.shape}, policy_logits: {policy_logits.shape}, rewards: {rewards.shape}')
        return NetworkOutput(reward=rewards, hidden_state=hidden_states, policy_logits=policy_logits, value=values)

    def recurrent(self, hidden_states: torch.Tensor, actions: torch.Tensor) -> NetworkOutput:
        batch_size = hidden_states.shape[0]

        policy_logits, values = self.prediction(hidden_states)
        actions_exp = F.one_hot(actions, self.num_actions)
        actions_exp = actions_exp.unsqueeze(1)
        fill = hidden_states.shape[2]
        actions_exp = actions_exp.tile([1, fill, 1])
        actions_exp = actions_exp.unsqueeze(1)
        inputs = torch.cat([hidden_states, actions_exp], 1)
        new_hidden_states, rewards = self.dynamic(inputs)
        #self.logger.info(f'inference: recurrent: hidden_states: {hidden_states.shape}, policy_logits: {policy_logits.shape}, rewards: {rewards.shape}, values: {values.shape}')
        return NetworkOutput(reward=rewards, hidden_state=new_hidden_states, policy_logits=policy_logits, value=values)

class Train:
    def __init__(self, hparams: Hparams):
        logfile = os.path.join(hparams.checkpoints_dir, 'muzero.log')
        os.makedirs(hparams.checkpoints_dir, exist_ok=True)
        self.logger = setup_logger('muzero', logfile, hparams.log_to_stdout)

        self.game_states = torch.zeros([hparams.batch_size, 1, 6, 7]).float().cpu()

        self.inference = Inference(hparams, self.logger)

        tree = mcts.Tree(hparams, self.inference, self.logger)
        tree.player_id[:, 0] = 1

        batch_index = torch.arange(hparams.batch_size).long()
        node_index = torch.zeros(hparams.batch_size).long()

        out = self.inference.initial(tree.player_id[:, 0], self.game_states)

        episode_len = torch.ones(len(node_index)).long()
        search_path = torch.zeros(len(node_index), 1).long()

        tree.store_states(search_path, episode_len, out.hidden_state)

        player_id = tree.player_id[:, 0]
        tree.expand(player_id, batch_index, node_index, out.policy_logits)

        start_simulation_time = perf_counter()
        for _ in range(hparams.num_simulations):
            search_path, episode_len = tree.run_one()
        simulation_time = perf_counter() - start_simulation_time
        one_sim_ms = int(simulation_time / hparams.num_simulations * 1000)
        self.logger.info(f'train: batch_size: {hparams.batch_size}, num_simulations: {hparams.num_simulations}, time: {simulation_time:.3f} sec, avg: {one_sim_ms} ms')

def main():
    hparams = Hparams()
    hparams.batch_size = 1024
    hparams.num_simulations = 400
    train = Train(hparams)

if __name__ == '__main__':
    main()
