from typing import Callable, List, Dict

import numpy as np
import torch
import torch.nn as nn


class NetworkParams:
    observation_shape: List[int] = []

    kernel_size: int = 4
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

    def load_args(self, args_dict: dict):
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

        self.output_proj = nn.Sequential(
            nn.Conv2d(hparams.repr_conv_res_num_features,
                      hparams.hidden_size,
                      kernel_size=1,
                      padding='same'),
            nn.LayerNorm([hparams.hidden_size, *spatial_shape]),
            nn.Tanh(),
        )

    def forward(self, inputs):
        x = self.input_proj(inputs)
        x = self.conv_blocks(x)
        x = self.output_proj(x)
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

        conv_blocks.append(
            nn.Sequential(
                nn.Conv2d(hparams.pred_conv_res_num_features,
                          hparams.hidden_size,
                          kernel_size=1,
                          padding='same'),
                nn.LayerNorm([hparams.hidden_size, *spatial_shape]),
                nn.Tanh(),
            ))

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
                                             output_activation=nn.Tanh)

    def forward(self, inputs):
        x = self.conv_blocks(inputs)

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
            nn.Tanh(),
        ))

        self.conv_blocks = nn.Sequential(*conv_blocks)

        hidden_output_size = hparams.hidden_size * np.prod(hparams.observation_shape)
        self.output_reward = LinearPrediction(hidden_output_size,
                                              hparams.dyn_reward_linear_layers,
                                              1,
                                              hparams.activation,
                                              output_activation=nn.Tanh)

    def forward(self, inputs):
        next_state = self.conv_blocks(inputs)
        reward = self.output_reward(next_state).squeeze(1)
        return next_state, reward
