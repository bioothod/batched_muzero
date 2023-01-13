import torch

class Hparams:
    rows: int = 6
    columns: int = 7
    inarow: int = 4
    default_reward: float = 0
    invalid_action_reward: float = -1

    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.inarow = 4
        self.default_reward = 0
        self.invalid_action_reward = -1

@torch.jit.script
def check_reward(hparams: Hparams, games: torch.Tensor, player_id: torch.Tensor):
    row_player = torch.ones([len(games), hparams.inarow], device=games.device) * player_id
    columns_end = hparams.columns - (hparams.inarow - 1)
    rows_end = hparams.rows - (hparams.inarow - 1)

    dones = torch.zeros(len(games), dtype=torch.bool, device=games.device)
    idx = torch.arange(len(games), device=games.device)

    for row in torch.arange(0, hparams.rows, dtype=torch.int64):
        for col in torch.arange(0, columns_end, dtype=torch.int64):
            window = games[idx][:, row, col:col+hparams.inarow]
            win_idx = torch.all(window == row_player[idx], -1)
            win_idx = torch.any(win_idx, 1)
            dones[idx] = torch.logical_or(dones[idx], win_idx)
            idx = idx[torch.logical_not(win_idx)]

    if len(idx) > 0:
        for col in torch.arange(0, hparams.columns, dtype=torch.int64):
            for row in torch.arange(0, rows_end, dtype=torch.int64):
                window = games[idx][:, row:row+hparams.inarow, col]
                win_idx = torch.all(window == row_player[idx], -1)
                win_idx = torch.any(win_idx, 1)
                dones[idx] = torch.logical_or(dones[idx], win_idx)
                idx = idx[torch.logical_not(win_idx)]

    if len(idx) > 0:
        for row in torch.arange(0, rows_end, dtype=torch.int64):
            row_index = torch.arange(row, row+hparams.inarow)
            for col in torch.arange(0, columns_end, dtype=torch.int64):
                col_index = torch.arange(col, col+hparams.inarow)
                window = games[idx][:, row_index, col_index]
                win_idx = torch.all(window == row_player[idx], -1)
                win_idx = torch.any(win_idx, 1)
                dones[idx] = torch.logical_or(dones[idx], win_idx)
                idx = idx[torch.logical_not(win_idx)]

    if len(idx) > 0:
        for row in torch.arange(hparams.inarow-1, hparams.rows, dtype=torch.int64):
            row_index = torch.arange(row, row-hparams.inarow, -1)
            for col in torch.arange(0, columns_end, dtype=torch.int64):
                col_index = torch.arange(col, col+hparams.inarow)
                window = games[idx][:, row_index, col_index]
                win_idx = torch.all(window == row_player[idx], -1)
                win_idx = torch.any(win_idx, 1)
                dones[idx] = torch.logical_or(dones[idx], win_idx)
                idx = idx[torch.logical_not(win_idx)]

    rewards = torch.where(dones, 1.0, hparams.default_reward)

    return rewards, dones

@torch.jit.script
def step_games(hparams: Hparams, games: torch.Tensor, player_id: torch.Tensor, actions: torch.Tensor):
    player_id = player_id.unsqueeze(1).float()

    num_games = len(games)
    non_zero = torch.count_nonzero(games[torch.arange(num_games, dtype=torch.int64), :, actions], 2).squeeze(1)

    invalid_action_index_batch = non_zero == hparams.rows
    good_action_index_batch = non_zero < hparams.rows

    good_actions_index = actions[good_action_index_batch]
    games[good_action_index_batch, hparams.rows - non_zero[good_action_index_batch] - 1, good_actions_index] = player_id[good_action_index_batch]

    rewards, dones = check_reward(hparams, games, player_id)
    rewards[invalid_action_index_batch] = torch.tensor(hparams.invalid_action_reward, dtype=torch.float32)
    dones[invalid_action_index_batch] = True

    return games, rewards, dones
