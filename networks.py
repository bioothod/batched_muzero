from typing import Any, List, Dict

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hparams import Hparams
from inference import Inference as GenericInference
from inference import NetworkOutput

class NetworkParams:
    observation_shape: List[int] = []

    kernel_size: int = 3
    hidden_size: int = 16

    repr_conv_res_num_features: int = 24
    repr_conv_num_blocks: int = 4

    pred_conv_res_num_features: int = 24
    pred_conv_num_blocks: int = 2
    pred_hidden_linear_layers: List[int] = [256, 256]
    num_actions: int = 7

    dyn_conv_res_num_features: int = 24
    dyn_conv_num_blocks: int = 4
    dyn_reward_linear_layers: List[int] = [256, 256]

    activation_str: str = 'LeakyReLU'
    activation_args: List = []
    activation_kwargs: Dict = {'negative_slope': 0.01, 'inplace': True}

    def __str__(self):
        ret = []
        for attr in dir(self):
            if attr.startswith('__') or callable(getattr(self, attr)):
                continue

            value = getattr(self, attr)
            ret.append(f'{str(attr):>32s} : {str(value)}')

        return '\n'.join(ret)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and type(args[0]) == dict:
            self.load_args(args[0])
            return

        self.load_args(kwargs)
        self.activation = lambda: getattr(nn, self.activation_str)(*self.activation_args, **self.activation_kwargs)

    def load_args(self, args_dict: Dict[Any, Any]):
        for k, v in args_dict.items():
            if self.__getattribute__(k) is not None:
                self.__setattr__(k, v)

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

class Representation(nn.Module):
    def __init__(self, hparams: NetworkParams) -> None:
        super().__init__()

        spatial_shape = hparams.observation_shape[-2:]

        self.input_proj = nn.Sequential(
            nn.Conv2d(3,
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

        self.hidden_size_proj = nn.Sequential(
            nn.Conv2d(hparams.repr_conv_res_num_features,
                      hparams.hidden_size,
                      kernel_size=1,
                      padding='same'),
            nn.LayerNorm([hparams.hidden_size, *spatial_shape]),
            hparams.activation(),
        )

    def forward(self, inputs):
        x = self.input_proj(inputs)
        x = self.conv_blocks(x)
        x = self.hidden_size_proj(x)
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


class Prediction(nn.Module):
    def __init__(self, hparams: NetworkParams) -> None:
        super().__init__()

        spatial_shape = hparams.observation_shape[-2:]

        conv_blocks = []

        if hparams.pred_conv_res_num_features != hparams.hidden_size:
            input_proj = nn.Sequential(
                nn.Conv2d(hparams.hidden_size,
                          hparams.pred_conv_res_num_features,
                          1,
                          padding='same'),
                nn.BatchNorm2d(hparams.pred_conv_res_num_features),
                hparams.activation(),
            )
            conv_blocks.append(input_proj)

        for _ in range(hparams.pred_conv_num_blocks):
            block = ResidualBlock(
                num_features=hparams.pred_conv_res_num_features,
                kernel_size=hparams.kernel_size,
                activation=hparams.activation)
            conv_blocks.append(block)

        self.hidden_size_proj = nn.Sequential(
            nn.Conv2d(hparams.pred_conv_res_num_features,
                      hparams.hidden_size,
                      kernel_size=1,
                      padding='same'),
            nn.LayerNorm([hparams.hidden_size, *spatial_shape]),
            hparams.activation(),
        )

        self.conv_blocks = nn.Sequential(*conv_blocks)

        hidden_output_size = hparams.hidden_size * np.prod(hparams.observation_shape)

        self.output_policy_logits = LinearPrediction(hidden_output_size,
                                                     hparams.pred_hidden_linear_layers,
                                                     hparams.num_actions,
                                                     hparams.activation)
        self.output_value = LinearPrediction(hidden_output_size,
                                             hparams.pred_hidden_linear_layers,
                                             1,
                                             hparams.activation,
                                             output_activation=None)

    def forward(self, inputs):
        x = self.conv_blocks(inputs)
        x = self.hidden_size_proj(x)

        p = self.output_policy_logits(x)
        v = self.output_value(x).squeeze(1)
        return p, v


class Dynamic(nn.Module):

    def __init__(self, hparams: NetworkParams) -> None:
        super().__init__()

        spatial_shape = hparams.observation_shape[-2:]

        conv_blocks = []

        if hparams.dyn_conv_res_num_features != hparams.hidden_size+1:
            input_proj = nn.Sequential(
                nn.Conv2d(hparams.hidden_size+1,
                          hparams.dyn_conv_res_num_features,
                          1,
                          padding='same'),
                nn.BatchNorm2d(hparams.dyn_conv_res_num_features),
                hparams.activation(),
            )
            conv_blocks.append(input_proj)

        for _ in range(hparams.dyn_conv_num_blocks):
            block = ResidualBlock(num_features=hparams.dyn_conv_res_num_features,
                                  kernel_size=hparams.kernel_size,
                                  activation=hparams.activation)
            conv_blocks.append(block)


        conv_blocks.append(nn.Sequential(
            nn.Conv2d(hparams.repr_conv_res_num_features,
                      hparams.hidden_size,
                      kernel_size=1,
                      padding='same'),
            nn.LayerNorm([hparams.hidden_size, *spatial_shape]),
            hparams.activation(),
        ))

        self.conv_blocks = nn.Sequential(*conv_blocks)

        hidden_output_size = hparams.hidden_size * np.prod(hparams.observation_shape)
        self.output_reward = LinearPrediction(hidden_output_size,
                                              hparams.dyn_reward_linear_layers,
                                              1,
                                              hparams.activation,
                                              output_activation=None)

    def forward(self, inputs):
        next_state = self.conv_blocks(inputs)
        reward = self.output_reward(next_state).squeeze(1)
        return next_state, reward

class Inference(GenericInference):
    def __init__(self, hparams: Hparams, logger: logging.Logger):
        self.logger = logger
        self.hparams = hparams

        net_hparams = NetworkParams(observation_shape=hparams.state_shape)
        self.logger.info(f'inference: network_params:\n{net_hparams}')

        self.representation = Representation(net_hparams).to(hparams.device)
        self.prediction = Prediction(net_hparams).to(hparams.device)
        self.dynamic = Dynamic(net_hparams).to(hparams.device)

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
        batch_size = game_states.shape[0]
        states = self.create_states(player_id, game_states)
        hidden_states = self.representation(states)
        policy_logits, values = self.prediction(hidden_states)
        rewards = torch.zeros(batch_size).float().to(self.hparams.device)
        #self.logger.info(f'inference: initial: game_states: {game_states.shape}: hidden_states: {hidden_states.shape}, policy_logits: {policy_logits.shape}, rewards: {rewards.shape}')
        return NetworkOutput(reward=rewards, hidden_state=hidden_states, policy_logits=policy_logits, value=values)

    def recurrent(self, hidden_states: torch.Tensor, actions: torch.Tensor) -> NetworkOutput:
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
