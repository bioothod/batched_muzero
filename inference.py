from typing import Optional

import logging

import numpy as np
import torch

import module_loader

class NetworkOutput:
    reward: torch.Tensor
    hidden_state: torch.Tensor
    policy_logits: torch.Tensor
    value: torch.Tensor

    def __init__(self, reward: torch.Tensor, hidden_state: torch.Tensor, policy_logits: torch.Tensor, value: Optional[torch.Tensor] = None):
        self.reward = reward
        self.hidden_state = hidden_state
        self.policy_logits = policy_logits

        if value is not None:
            self.value = value
        else:
            self.value = torch.zeros_like(reward)

class GenericInference:
    def __init__(self, game_ctl: module_loader.GameModule, logger: logging.Logger):
        self.logger = logger
        self.hparams = game_ctl.hparams

    def train(self, mode: bool):
        raise NotImplementedError(f'@train() method is not implemented')

    def zero_grad(self):
        raise NotImplementedError(f'@zer_grad() method is not implemented')

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
        raise NotImplementedError(f'@initial() method is not implemented')

    def recurrent(self, hidden_states: torch.Tensor, actions: torch.Tensor) -> NetworkOutput:
        raise NotImplementedError(f'@initial() method is not implemented')
