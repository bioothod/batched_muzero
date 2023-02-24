from copy import deepcopy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import connectx_dnn_utils as utils

class CombinedModel(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()

        self.num_actions = utils.default_config['columns']

        model_paths = [
            #('rl_agents_ppo9_multichannel.py', 'feature_model_ppo9_multichannel.py', 'submission_9_ppo86_multichannel_critic.ckpt', utils.config_ppo9_multichannel),
            ('rl_agents_ppo29.py', 'feature_model_ppo29.py', 'submission_29_ppo96_critic.ckpt', utils.config_ppo29),
        ]

        self.actors = []
        for rl_model_path, feature_model_path, checkpoint_path, config in model_paths:
            rl_model_path = os.path.join(model_dir, rl_model_path)
            feature_model_path = os.path.join(model_dir, feature_model_path)
            checkpoint_path = os.path.join(model_dir, checkpoint_path)

            actor, critic = utils.create_actor_critic(feature_model_path, rl_model_path, config, checkpoint_path, create_critic=False)
            self.actors.append(actor)

    def create_game_from_state(self, player_id, state):
        raise NotImplementedError(f'combined_mode::create_game_from_state: player_id: {player_id}, state: {state}')

    def forward(self, player_id, game_state):
        all_probs = torch.ones((len(game_state), self.num_actions), dtype=torch.float32)
        for actor in self.actors:
            state = actor.create_state(player_id, game_state)

            state_features = actor.state_features(state)
            logits = actor.features(state_features)
            probs = F.softmax(logits, 1)
            all_probs *= probs

        return all_probs
