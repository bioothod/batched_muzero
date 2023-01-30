from typing import Any, List, Dict

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hparams import GenericHparams
from inference import GenericInference as GenericInference
from inference import NetworkOutput
import module_loader
from network_params import NetworkParams

class ResidualBlock(nn.Module):

    def __init__(self, num_features, kernel_size, activation):
        super().__init__()

        self.conv0 = nn.Conv2d(num_features,
                               num_features,
                               kernel_size,
                               padding='same')
        self.activation0 = activation()
        self.b0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features,
                               num_features,
                               kernel_size,
                               padding='same')
        self.activation1 = activation()
        self.b1 = nn.BatchNorm2d(num_features)

    def forward(self, inputs):
        x = self.conv0(inputs)
        x = self.b0(x)
        x = self.activation0(x)

        x = self.conv1(x)
        x = self.b1(x)
        x += inputs
        x = self.activation1(x)
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
            nn.Conv2d(2,
                      hparams.repr_conv_res_num_features,
                      hparams.kernel_size,
                      padding='same'),
            nn.BatchNorm2d(hparams.repr_conv_res_num_features),
            hparams.activation(),
        )

        res_blocks = []
        for _ in range(hparams.repr_conv_num_blocks):
            block = ResidualBlock(
                num_features=hparams.repr_conv_res_num_features,
                kernel_size=hparams.kernel_size,
                activation=hparams.activation)
            res_blocks.append(block)

        self.conv_blocks = nn.Sequential(*res_blocks)

        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(hparams.repr_features_dropout),
            nn.Linear(hparams.repr_conv_res_num_features*np.prod(hparams.observation_shape),
                      hparams.repr_linear_num_features),
            nn.LayerNorm([hparams.repr_linear_num_features]),
        )

    def forward(self, inputs):
        x = self.input_proj(inputs)
        x = self.conv_blocks(x)
        x = self.output_proj(x)
        return x

class Prediction(nn.Module):
    def __init__(self, hparams: NetworkParams) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Flatten(),
        )

        #num_input_features = hparams.repr_conv_res_num_features*np.prod(hparams.observation_shape)
        num_input_features = hparams.repr_linear_num_features

        self.output_policy_logits = LinearPrediction(num_input_features,
                                                     hparams.pred_hidden_linear_layers,
                                                     hparams.num_actions,
                                                     hparams.activation)
        self.output_value = LinearPrediction(num_input_features,
                                             hparams.pred_hidden_linear_layers,
                                             1,
                                             hparams.activation,
                                             output_activation=None)

    def forward(self, inputs):
        x = self.input_proj(inputs)
        p = self.output_policy_logits(x)
        v = self.output_value(x).squeeze(1)
        return p, v

class Dynamic(nn.Module):
    def __init__(self, hparams: NetworkParams, state_extension: int) -> None:
        super().__init__()

        num_input_features = hparams.repr_linear_num_features + state_extension
        self.output_state = nn.Sequential(
            nn.Dropout(hparams.dyn_state_dropout),
            LinearPrediction(num_input_features, hparams.dyn_state_layers, hparams.repr_linear_num_features, hparams.activation, hparams.activation),
            nn.LayerNorm([hparams.dyn_state_layers[-1]]),
        )

        self.output_reward = nn.Sequential(
            nn.Dropout(hparams.dyn_reward_dropout),
            LinearPrediction(num_input_features*np.prod(hparams.observation_shape),
                             hparams.dyn_reward_linear_layers,
                             1,
                             hparams.activation,
                             output_activation=None)
            )

    def forward(self, inputs):
        x = self.output_state(inputs)
        reward = self.output_reward(x).squeeze(1)
        return x, reward

class Inference(GenericInference):
    def __init__(self, game_ctl: module_loader.GameModule, logger: logging.Logger):
        self.logger = logger
        self.hparams = game_ctl.hparams
        self.game_ctl = game_ctl

        self.logger.info(f'inference: network_params:\n{game_ctl.network_hparams}')

        self.representation = Representation(game_ctl.network_hparams).to(self.hparams.device)
        self.prediction = Prediction(game_ctl.network_hparams).to(self.hparams.device)

        state_extension = game_ctl.hparams.num_actions
        self.dynamic = Dynamic(game_ctl.network_hparams, state_extension).to(self.hparams.device)

        self.models = [self.representation, self.prediction, self.dynamic]

    def train(self, mode: bool):
        for model in self.models:
            model.train(mode)

    def zero_grad(self):
        for model in self.models:
            model.zero_grad()

    def to(self, device):
        self.representation.to(device)
        self.prediction.to(device)
        self.dynamic.to(device)

    def create_states(self, player_id: torch.Tensor, game_states: torch.Tensor):
        batch_size = len(game_states)
        input_shape = list(game_states.shape[1:]) # remove batch size

        states = torch.zeros(1 + len(self.hparams.player_ids), batch_size, *input_shape).to(self.hparams.device)

        # states design:
        #  0: set 1 where current player has its marks
        #  1: set 1 where the other player has its marks

        def get_index_for_player_id(game_states, player_id):
            index = game_states == player_id
            index = index.squeeze(1)
            return index

        set_player_id_index = 0
        states[set_player_id_index, get_index_for_player_id(game_states, player_id)] = 1
        set_player_id_index += 1

        for local_player_id in self.hparams.player_ids:
            if local_player_id != player_id:
                states[set_player_id_index, get_index_for_player_id(game_states, local_player_id)] = 1
                set_player_id_index += 1

        states = torch.transpose(states, 1, 0)
        return states

    def initial(self, player_id: torch.Tensor, game_states: torch.Tensor) -> NetworkOutput:
        batch_size = game_states.shape[0]
        states = self.create_states(player_id, game_states)
        hidden_states = self.representation(states)
        policy_logits, values = self.prediction(hidden_states)
        rewards = torch.zeros(batch_size).float().to(self.hparams.device)
        #self.logger.info(f'inference: initial: game_states: {game_states.shape}: hidden_states: {hidden_states.shape}, policy_logits: {policy_logits.shape}, rewards: {rewards.shape}')
        return NetworkOutput(reward=rewards, hidden_state=hidden_states, policy_logits=policy_logits, value=values)

    def recurrent(self, hidden_states: torch.Tensor, actions: torch.Tensor) -> NetworkOutput:
        actions_exp = F.one_hot(actions, self.hparams.num_actions)

        ap = torch.cat([actions_exp, player_states], 1)
        ap = ap.unsqueeze(2).unsqueeze(3)
        ap = torch.tile(ap, [1, 1] + self.game_ctl.network_hparams.observation_shape)

        #self.logger.info(f'hidden_states: {hidden_states.shape}, player_states: {player_states.shape}, actions_exp: {actions_exp.shape}')
        dyn_inputs = torch.cat([hidden_states, ap], 1)
        new_hidden_states, rewards = self.dynamic(dyn_inputs)
        policy_logits, values = self.prediction(new_hidden_states)
        #self.logger.info(f'inference: recurrent: hidden_states: {hidden_states.shape}, policy_logits: {policy_logits.shape}, rewards: {rewards.shape}, values: {values.shape}')
        return NetworkOutput(reward=rewards, hidden_state=new_hidden_states, policy_logits=policy_logits, value=values)
