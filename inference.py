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
