from typing import List

import torch

class GenericHparams:
    checkpoints_dir: str
    log_to_stdout: bool

    rows: int
    columns: int

    batch_size: int
    state_shape: List[int]
    num_actions: int
    device: torch.device = torch.device('cuda:0')
    dtype: torch.dtype = torch.float32

    max_episode_len: int
    num_simulations: int

    default_reward: float = 0.0

    ucb_value_score_discount: float = 1
    value_discount: float = 1
    pb_c_init: float = 1.25
    #pb_c_init: float = 4
    pb_c_base: float = 19652
    add_exploration_noise: bool = True
    dirichlet_alpha: float = 1
    #dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25

    player_ids: List[int] = [1, 2]

    num_unroll_steps: int = 5
    td_steps: int

    num_steps_to_argmax_action_selection: int
    action_selection_temperature: float = 1

    max_training_games: int = 20

    server_port: int = 50051
    num_server_workers: int = 2

    num_training_steps: int = 64
    min_lr = 1e-5
    init_lr = 1e-4

class ConnectXHparams(GenericHparams):
    checkpoints_dir: str = 'connectx_checkpoints_1'
    log_to_stdout = True

    rows: int = 6
    columns: int = 7

    batch_size: int = 1024
    state_shape: List[int] = [6, 7]
    num_actions: int = 7

    max_episode_len: int = 42
    num_simulations: int = 800

    num_steps_to_argmax_action_selection: int = 30
    action_selection_temperature: float = 1.75

    num_unroll_steps: int = 5
    td_steps: int = 42

class TicTacToeHparams(GenericHparams):
    checkpoints_dir: str = 'tic_tac_toe_checkpoints_1'
    log_to_stdout = True

    rows: int = 3
    columns: int = 3

    batch_size: int = 1024
    state_shape: List[int] = [3, 3]
    num_actions: int = 9

    max_episode_len: int = 9
    num_simulations: int = 800

    num_steps_to_argmax_action_selection: int = 4


    num_unroll_steps: int = 5
    td_steps: int = 9
