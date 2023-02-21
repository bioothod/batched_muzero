from typing import List, Dict, Any

import torch
import torch.nn as nn

class NetworkParams:
    observation_shape: List[int] = []

    kernel_size: int
    num_stacked_states: int
    num_additional_planes: int
    conv_res_num_features: int

    repr_features_dropout: float
    repr_conv_num_blocks: int

    pred_hidden_linear_layers: List[int]
    num_actions: int = 0

    dyn_reward_dropout: float
    dyn_conv_num_blocks: int
    dyn_reward_linear_layers: List[int]

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

class ConnectXParams(NetworkParams):
    num_stacked_states: int = 1
    num_additional_planes: int = 7
    kernel_size: int = 4
    conv_res_num_features: int = 64

    repr_features_dropout: float = 0.
    repr_conv_num_blocks: int = 8

    pred_hidden_linear_layers: List[int] = [512, 128]
    num_actions: int = 0

    dyn_reward_dropout: float = 0.
    dyn_conv_num_blocks: int = 8
    dyn_reward_linear_layers: List[int] = [128]

    activation_str: str = 'ReLU'
    #activation_str: str = 'LeakyReLU'
    activation_args: List = []
    #activation_kwargs: Dict = {'negative_slope': 0.01, 'inplace': True}
    activation_kwargs: Dict = {'inplace': True}

class TicTacToeParams(NetworkParams):
    kernel_size: int = 3
    hidden_size: int = 16

    repr_conv_res_num_features: int = 16
    repr_conv_num_blocks: int = 2

    pred_conv_res_num_features: int = 16
    pred_conv_num_blocks: int = 2
    pred_hidden_linear_layers: List[int] = [64, 64]
    num_actions: int = 9

    dyn_conv_res_num_features: int = 16
    dyn_conv_num_blocks: int = 2
    dyn_reward_linear_layers: List[int] = [64, 64]

    activation_str: str = 'LeakyReLU'
    activation_args: List = []
    activation_kwargs: Dict = {'negative_slope': 0.01, 'inplace': True}
