from typing import List

import torch

class Hparams:
    rows: int = 6
    columns: int = 7

    batch_size: int = 1024
    state_shape: List[int] = [1, 6, 7]
    num_actions: int = 7
    device: torch.device = torch.device('cpu')
    dtype: torch.dtype = torch.float32

    max_episode_len: int = 42
    num_simulations: int = 1000

    default_reward: float = 0.0

    discount: float = 0.99
    c1: float = 1.25
    c2: float = 19652
    add_exploration_noise: bool = True
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25

    player_ids: List[int] = [1, 2]

    server_port = 50051
    num_server_workers = 2
