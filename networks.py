from collections import deque
from typing import Any, List, Dict, Optional

import logging

import numpy as np
from hparams import GenericHparams
import torch
import torch.nn as nn
import torch.nn.functional as F

import module_loader
from network_params import NetworkParams

class NetworkOutput:
    reward: torch.Tensor
    hidden_state: torch.Tensor
    policy_logits: torch.Tensor
    value: torch.Tensor

    def __init__(self, reward: torch.Tensor, hidden_state: torch.Tensor, policy_logits: torch.Tensor, value: torch.Tensor):
        self.reward = reward
        self.hidden_state = hidden_state
        self.policy_logits = policy_logits
        self.value = value

class ResidualBlock(nn.Module):
    def __init__(self, num_features, kernel_size, activation):
        super().__init__()

        self.activation = activation()

        self.conv0 = nn.Conv2d(num_features,
                               num_features,
                               kernel_size,
                               padding='same')
        self.b0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features,
                               num_features,
                               kernel_size,
                               padding='same')
        self.b1 = nn.BatchNorm2d(num_features)

    def forward(self, inputs):
        x = self.conv0(inputs)
        x = self.b0(x)
        x = self.activation(x)

        x = self.conv1(x)
        x = self.b1(x)
        x = x + inputs
        x = self.activation(x)

        return x

class LinearPrediction(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_dims,
                 output_size,
                 activation,
                 output_activation=None) -> None:
        super().__init__()

        blocks = [nn.Flatten()]
        prev_hidden_size = input_size
        for hidden_size in hidden_dims:
            blocks.append(nn.Linear(prev_hidden_size, hidden_size))
            blocks.append(activation())
            prev_hidden_size = hidden_size

        blocks.append(nn.Linear(prev_hidden_size, output_size))
        if output_activation is not None:
            blocks.append(output_activation())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.blocks(inputs)

class Representation(nn.Module):
    def __init__(self, hparams: NetworkParams) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv2d(2*hparams.num_stacked_states,
                      hparams.conv_res_num_features,
                      3,
                      padding='same'),
        )

        res_blocks = []
        for _ in range(hparams.repr_conv_num_blocks):
            block = ResidualBlock(
                num_features=hparams.conv_res_num_features,
                kernel_size=hparams.kernel_size,
                activation=hparams.activation)
            res_blocks.append(block)

        self.conv_blocks = nn.Sequential(
            *res_blocks,
        )

    def forward(self, inputs):
        x = self.input_proj(inputs)
        x = self.conv_blocks(x)
        return x

class Prediction(nn.Module):
    def __init__(self, hparams: NetworkParams) -> None:
        super().__init__()

        res_blocks = []
        for _ in range(hparams.pred_conv_num_blocks):
            block = ResidualBlock(
                num_features=hparams.conv_res_num_features,
                kernel_size=hparams.kernel_size,
                activation=hparams.activation)
            res_blocks.append(block)

        self.conv_blocks = nn.Sequential(*res_blocks)

        num_input_features = hparams.flat_projection_num_features * np.prod(hparams.observation_shape)
        self.output_policy_logits = nn.Sequential(
            nn.Conv2d(hparams.conv_res_num_features, hparams.flat_projection_num_features, 1),
            nn.BatchNorm2d(hparams.flat_projection_num_features),
            hparams.activation(),

            nn.Flatten(),

            LinearPrediction(num_input_features,
                             hparams.pred_hidden_linear_layers,
                             hparams.num_actions,
                             hparams.activation,
                             output_activation=None)
        )
        self.output_value = nn.Sequential(
            nn.Conv2d(hparams.conv_res_num_features, hparams.flat_projection_num_features, 1, padding='same'),
            nn.BatchNorm2d(hparams.flat_projection_num_features),
            hparams.activation(),

            nn.Flatten(),

            LinearPrediction(num_input_features,
                             hparams.pred_hidden_linear_layers,
                             1,
                             hparams.activation,
                             output_activation=None)
        )

    def forward(self, inputs):
        x = self.conv_blocks(inputs)
        p = self.output_policy_logits(x)
        v = self.output_value(x)
        return p, v

class Dynamic(nn.Module):
    def __init__(self, hparams: NetworkParams) -> None:
        super().__init__()

        num_input_features = hparams.conv_res_num_features + hparams.num_additional_planes
        self.input_proj = nn.Sequential(
            nn.Conv2d(num_input_features, hparams.conv_res_num_features, hparams.kernel_size, padding='same'),
            nn.BatchNorm2d(hparams.conv_res_num_features),
            hparams.activation(),
        )

        res_blocks = []
        for _ in range(hparams.repr_conv_num_blocks):
            block = ResidualBlock(
                num_features=hparams.conv_res_num_features,
                kernel_size=hparams.kernel_size,
                activation=hparams.activation)
            res_blocks.append(block)

        self.conv_blocks = nn.Sequential(*res_blocks)

        num_input_features = hparams.flat_projection_num_features * np.prod(hparams.observation_shape)
        self.output_reward = nn.Sequential(
            nn.Conv2d(hparams.conv_res_num_features, hparams.flat_projection_num_features, 1, padding='same'),
            nn.BatchNorm2d(hparams.flat_projection_num_features),
            hparams.activation(),

            nn.Flatten(),

            LinearPrediction(num_input_features,
                             hparams.dyn_reward_linear_layers,
                             1,
                             hparams.activation,
                             output_activation=None)
            )

    def forward(self, inputs):
        x = self.input_proj(inputs)
        x = self.conv_blocks(x)

        reward = self.output_reward(x)
        return x, reward

class GameState:
    def __init__(self, batch_size: int, hparams: GenericHparams, network_hparams: NetworkParams):
        self.device = hparams.device
        self.batch_size = batch_size
        self.player_ids = hparams.player_ids

        self.num_games = network_hparams.num_stacked_states
        self.state_shape = network_hparams.observation_shape
        self.num_additional_planes = network_hparams.num_additional_planes
        self.num_hidden_state_raw_planes = len(self.player_ids) * self.num_games

        self.game_stack = deque(maxlen=network_hparams.num_stacked_states)
        self.game_player_ids = deque(maxlen=network_hparams.num_stacked_states)
        self.reset()

    def to(self, device) -> 'GameState':
        for state in self.game_stack:
            state.to(device)
        for player_id in self.game_player_ids:
            player_id.to(device)

        return self

    def for_hash(self):
        states = tuple([state.detach().cpu().numpy().tostring() for state in self.game_stack])
        player_ids = tuple([pid.detach().cpu().numpy().tostring() for pid in self.game_player_ids])
        return [states, player_ids]

    def revert_state(self, pov_player_id: int, local_player_id: int, game_state: torch.Tensor) -> torch.Tensor:
        new_state = torch.zeros_like(game_state)
        new_state[game_state == pov_player_id] = local_player_id
        new_state[game_state == local_player_id] = pov_player_id
        return new_state

    def push_game(self, player_id: torch.Tensor, game_state: torch.Tensor):
        # converted_game_state = torch.zeros_like(game_state)

        # pov_player_id = self.player_ids[0]
        # for local_player_id in self.player_ids:
        #     index = player_id == local_player_id

        #     if local_player_id == pov_player_id:
        #         converted_game_state[index] = game_state[index]
        #     else:
        #         converted_game_state[index] = self.revert_state(pov_player_id, local_player_id, game_state[index])

        converted_game_state = game_state.detach().clone()
        self.game_stack.append(converted_game_state)
        self.game_player_ids.append(player_id.detach().clone())

    def reset(self):
        self.game_stack.clear()
        self.game_player_ids.clear()

        for _ in range(self.num_games):
            state = torch.zeros(self.batch_size, *self.state_shape)
            self.game_stack.append(state)
            self.game_player_ids.append(torch.zeros(len(state)))

    def create_state(self):
        states = torch.zeros(self.num_games*len(self.player_ids), self.batch_size, *self.state_shape).to(self.device)

        for game_idx, (game_player_id, game_state) in enumerate(zip(self.game_player_ids, self.game_stack)):
            offset = game_idx * len(self.player_ids)

            for local_idx, local_player_id in enumerate(self.player_ids):
                state_player_idx = game_state == local_player_id
                state_player_idx = state_player_idx.squeeze(1)
                states[offset + local_idx, state_player_idx] = 1

        states = torch.transpose(states, 1, 0)
        return states


class Inference(nn.Module):
    def __init__(self, game_ctl: module_loader.GameModule, logger: logging.Logger):
        super().__init__()

        self.logger = logger
        self.hparams = game_ctl.hparams
        self.game_ctl = game_ctl

        self.logger.info(f'inference: network_params:\n{game_ctl.network_hparams}')

        self.representation = Representation(game_ctl.network_hparams).to(self.hparams.device)
        self.prediction = Prediction(game_ctl.network_hparams).to(self.hparams.device)

        self.dynamic = Dynamic(game_ctl.network_hparams).to(self.hparams.device)

    def initial(self, state: torch.Tensor) -> NetworkOutput:
        hidden_state = self.representation(state)
        policy_logits, values = self.prediction(hidden_state)

        rewards = torch.zeros(len(state)).float().to(self.hparams.device)
        return NetworkOutput(reward=rewards, hidden_state=hidden_state, policy_logits=policy_logits, value=values)

    def recurrent(self, hidden_state: torch.Tensor, actions: torch.Tensor) -> NetworkOutput:
        actions_enc = self.game_ctl.encode_actions(self.game_ctl.game_hparams, actions)
        dyn_inputs = torch.cat([hidden_state, actions_enc], 1)
        new_hidden_states, rewards = self.dynamic(dyn_inputs)
        policy_logits, values = self.prediction(new_hidden_states)
        return NetworkOutput(reward=rewards, hidden_state=new_hidden_states, policy_logits=policy_logits, value=values)
