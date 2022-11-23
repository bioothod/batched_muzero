import torch

class Hparams:
    rows: int
    columns: int
    inarow: int
    default_reward: float

    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.inarow = 4
        self.default_reward = 0

@torch.jit.script
def check_reward(hparams: Hparams, games: torch.Tensor, player_id: int):
    row_player = torch.ones(hparams.inarow) * player_id
    columns_end = hparams.columns - (hparams.inarow - 1)
    rows_end = hparams.rows - (hparams.inarow - 1)

    row_player = row_player.to(games.device)
    dones = torch.zeros(len(games), dtype=torch.bool, device=games.device)
    idx = torch.arange(len(games), device=games.device)

    for row in torch.arange(0, hparams.rows, dtype=torch.int64):
        for col in torch.arange(0, columns_end, dtype=torch.int64):
            window = games[idx][:, :, row, col:col+hparams.inarow]
            win_idx = torch.all(window == row_player, -1)
            win_idx = torch.any(win_idx, 1)
            dones[idx] = torch.logical_or(dones[idx], win_idx)
            idx = idx[torch.logical_not(win_idx)]

    if len(idx) > 0:
        for col in torch.arange(0, hparams.columns, dtype=torch.int64):
            for row in torch.arange(0, rows_end, dtype=torch.int64):
                window = games[idx][:, :, row:row+hparams.inarow, col]
                win_idx = torch.all(window == row_player, -1)
                win_idx = torch.any(win_idx, 1)
                dones[idx] = torch.logical_or(dones[idx], win_idx)
                idx = idx[torch.logical_not(win_idx)]

    if len(idx) > 0:
        for row in torch.arange(0, rows_end, dtype=torch.int64):
            row_index = torch.arange(row, row+hparams.inarow)
            for col in torch.arange(0, columns_end, dtype=torch.int64):
                col_index = torch.arange(col, col+hparams.inarow)
                window = games[idx][:, :, row_index, col_index]
                win_idx = torch.all(window == row_player, -1)
                win_idx = torch.any(win_idx, 1)
                dones[idx] = torch.logical_or(dones[idx], win_idx)
                idx = idx[torch.logical_not(win_idx)]

    if len(idx) > 0:
        for row in torch.arange(hparams.inarow-1, hparams.rows, dtype=torch.int64):
            row_index = torch.arange(row, row-hparams.inarow, -1)
            for col in torch.arange(0, columns_end, dtype=torch.int64):
                col_index = torch.arange(col, col+hparams.inarow)
                window = games[idx][:, :, row_index, col_index]
                win_idx = torch.all(window == row_player, -1)
                win_idx = torch.any(win_idx, 1)
                dones[idx] = torch.logical_or(dones[idx], win_idx)
                idx = idx[torch.logical_not(win_idx)]

    rewards = torch.where(dones, 1.0, hparams.default_reward)

    return rewards, dones

@torch.jit.script
def step_games(hparams: Hparams, games: torch.Tensor, player_id: int, actions: torch.Tensor):
    player_id = torch.tensor(player_id, dtype=torch.float32)

    num_games = len(games)
    non_zero = torch.count_nonzero(games[torch.arange(num_games, dtype=torch.int64), :, :, actions], 2).squeeze(1)

    invalid_action_index_batch = non_zero == hparams.rows
    good_action_index_batch = non_zero < hparams.rows

    good_actions_index = actions[good_action_index_batch]
    games[good_action_index_batch, :, hparams.rows - non_zero[good_action_index_batch] - 1, good_actions_index] = player_id

    rewards, dones = check_reward(hparams, games, player_id)
    rewards[invalid_action_index_batch] = torch.tensor(-10., dtype=torch.float32)
    dones[invalid_action_index_batch] = True

    return games, rewards, dones
