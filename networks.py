from typing import Any, List, Dict

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hparams import GenericHparams
from inference import Inference as GenericInference
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

        self.linear_encoder = nn.Sequential(
            nn.Flatten(),

            nn.Linear(hparams.repr_conv_res_num_features*np.prod(hparams.observation_shape), hparams.repr_linear_num_features),
            nn.LeakyReLU(),
            nn.LayerNorm(hparams.repr_linear_num_features),
        )

    def forward(self, inputs):
        x = self.input_proj(inputs)
        x = self.conv_blocks(x)
        x = self.linear_encoder(x)
        return x

class Prediction(nn.Module):
    def __init__(self, hparams: NetworkParams) -> None:
        super().__init__()

        self.output_policy_logits = LinearPrediction(hparams.repr_linear_num_features,
                                                     hparams.pred_hidden_linear_layers,
                                                     hparams.num_actions,
                                                     hparams.activation)
        self.output_value = LinearPrediction(hparams.repr_linear_num_features,
                                             hparams.pred_hidden_linear_layers,
                                             1,
                                             hparams.activation,
                                             output_activation=None)

    def forward(self, inputs):
        p = self.output_policy_logits(inputs)
        v = self.output_value(inputs).squeeze(1)
        return p, v

class MACDModelTransformer(nn.Module):
    def __init__(self, seq_len):
        super().__init__()

        input_dim = 1
        input_proj = 1+5


        self.decoder_projection = nn.Conv1d(input_proj, 1, 1)

        print_networks('MACDModelTransformer', self, True)

    def _generate_square_subsequent_mask(self, sz, shift=0):
        mask = torch.ones(sz, sz)
        mask = torch.triu(mask)
        if shift > 0:
            shifted_mask = torch.where(torch.triu(mask, diagonal=shift) == 1, 0, 1)
            mask = mask * shifted_mask
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, seq):
        # [B, L, Fin]
        batch_size, seq_len = seq.shape[:2]

        src = seq
        # [B, L, Fin] -> [L, B, Fin]
        src = src.permute((1, 0, 2))

        mask = self._generate_square_subsequent_mask(seq_len, 30).to(seq.device)
        #mask = self._generate_square_subsequent_mask(seq_len).to(seq.device)

        # [L, B, F] -> [L, B, F]
        out = self.encoder(src, mask)

        # [L, B, F] -> [B, F, L]
        out = out.permute((1, 2, 0))

        # [B, F, L] -> [B, 1, L]
        out = self.decoder_projection(out)

        # [B, 1, L] -> [B, L, 1]
        out = out.permute((0, 2, 1))

        return out

class Dynamic(nn.Module):
    def __init__(self, hparams: NetworkParams, state_extension: int) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv1d(hparams.repr_linear_num_features+state_extension, hparams.repr_linear_num_features, 1),
            nn.LeakyReLU(0.01),
            nn.LayerNorm([hparams.repr_linear_num_features, 1])
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hparams.repr_linear_num_features,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.2,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1)

        self.output_reward = LinearPrediction(hparams.repr_linear_num_features,
                                              hparams.dyn_reward_linear_layers,
                                              1,
                                              hparams.activation,
                                              output_activation=None)

    def forward(self, inputs):
        x = inputs.unsqueeze(2)
        x = self.input_proj(x)
        x = x.squeeze(2)
        next_state = self.encoder(x)
        reward = self.output_reward(next_state).squeeze(1)
        return next_state, reward

class Inference(GenericInference):
    def __init__(self, game_ctl: module_loader.GameModule, logger: logging.Logger):
        self.logger = logger
        self.hparams = game_ctl.hparams
        self.game_name = game_ctl.game_name

        self.logger.info(f'inference: network_params:\n{game_ctl.network_hparams}')

        self.representation = Representation(game_ctl.network_hparams).to(self.hparams.device)
        self.prediction = Prediction(game_ctl.network_hparams).to(self.hparams.device)

        state_extension = len(game_ctl.hparams.player_ids) + game_ctl.hparams.num_actions
        self.dynamic = Dynamic(game_ctl.network_hparams, state_extension).to(self.hparams.device)

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

        states = torch.zeros(1 + len(self.hparams.player_ids), batch_size, *input_shape).to(self.hparams.device)

        player_id_exp = player_id.unsqueeze(1)
        player_id_exp = player_id_exp.tile([1, np.prod(input_shape)]).view([batch_size] + input_shape).to(self.hparams.device)
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

    def recurrent(self, hidden_states: torch.Tensor, player_id: torch.Tensor, actions: torch.Tensor) -> NetworkOutput:
        actions_exp = F.one_hot(actions, self.hparams.num_actions)

        player_states = F.one_hot(player_id.long()-1, len(self.hparams.player_ids))

        #self.logger.info(f'hidden_states: {hidden_states.shape}, player_states: {player_states.shape}, actions_exp: {actions_exp.shape}')
        inputs = torch.cat([hidden_states, player_states, actions_exp], 1)
        new_hidden_states, rewards = self.dynamic(inputs)
        policy_logits, values = self.prediction(new_hidden_states)
        #self.logger.info(f'inference: recurrent: hidden_states: {hidden_states.shape}, policy_logits: {policy_logits.shape}, rewards: {rewards.shape}, values: {values.shape}')
        return NetworkOutput(reward=rewards, hidden_state=new_hidden_states, policy_logits=policy_logits, value=values)
