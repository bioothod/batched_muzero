import torch

class GameHparams:
    rows: int = 3
    columns: int = 3
    inarow: int = 3
    default_reward: float = 0
    invalid_action_reward: float = -10

    def __init__(self):
        self.rows = 3
        self.columns = 3
        self.inarow = 3
        self.default_reward = 0
        self.invalid_action_reward = -10


@torch.jit.script
def invalid_actions_mask(hparams: GameHparams, games: torch.Tensor):
    batch_size = len(games)
    flat_games = games.view(batch_size, -1)
    return flat_games != 0

@torch.jit.script
def check_reward(hparams: GameHparams, games: torch.Tensor, player_id: torch.Tensor):
    player_id = player_id.unsqueeze(1).unsqueeze(1)
    columns_end = hparams.columns - (hparams.inarow - 1)
    rows_end = hparams.rows - (hparams.inarow - 1)

    dones = torch.zeros(len(games), dtype=torch.bool, device=games.device)
    idx = torch.arange(len(games), device=games.device)

    winning = torch.where(games == player_id, 1, 0)
    win_idx = winning.sum(2) == hparams.columns
    win_idx = torch.any(win_idx, 1)
    dones[idx[win_idx]] = True
    idx = idx[torch.logical_not(win_idx)]

    winning = winning[idx]
    win_idx = winning.sum(1) == hparams.rows
    win_idx = torch.any(win_idx, 1)
    dones[idx[win_idx]] = True
    idx = idx[torch.logical_not(win_idx)]

    if len(idx) > 0:
        player_id = player_id.squeeze(1)
        for row in torch.arange(0, rows_end, dtype=torch.int64):
            row_index = torch.arange(row, row+hparams.inarow)
            for col in torch.arange(0, columns_end, dtype=torch.int64):
                col_index = torch.arange(col, col+hparams.inarow)
                window = games[idx][:, row_index, col_index]
                win_idx = torch.all(window == player_id[idx], -1)
                dones[idx[win_idx]] = True
                idx = idx[torch.logical_not(win_idx)]

    if len(idx) > 0:
        for row in torch.arange(hparams.inarow-1, hparams.rows, dtype=torch.int64):
            row_index = torch.arange(row, row-hparams.inarow, -1)
            for col in torch.arange(0, columns_end, dtype=torch.int64):
                col_index = torch.arange(col, col+hparams.inarow)
                window = games[idx][:, row_index, col_index]
                win_idx = torch.all(window == player_id[idx], -1)
                dones[idx[win_idx]] = True
                idx = idx[torch.logical_not(win_idx)]

    rewards = torch.where(dones, 1.0, hparams.default_reward)

    return rewards, dones

@torch.jit.script
def step_games(hparams: GameHparams, games: torch.Tensor, player_id: torch.Tensor, actions: torch.Tensor):
    num_games = len(games)
    flat_games = games.view([num_games, -1])
    actions = actions.long()
    dst_actions = flat_games[torch.arange(num_games, dtype=torch.int64), actions]

    invalid_action_index_batch = dst_actions != 0
    good_action_index_batch = dst_actions == 0

    good_actions_index = actions[good_action_index_batch]
    flat_games[good_action_index_batch, good_actions_index] = player_id[good_action_index_batch]

    rewards, dones = check_reward(hparams, games, player_id)
    rewards[invalid_action_index_batch] = torch.tensor(hparams.invalid_action_reward, dtype=torch.float32)
    dones[invalid_action_index_batch] = True

    num_zeros = torch.count_nonzero(flat_games == 0, -1)
    finished_index = num_zeros == 0
    dones[finished_index] = True

    max_debug = 5
    # print(f'invalid_actions_index: {invalid_action_index_batch[:max_debug]}')
    # print(f'games:\n{games.detach().cpu()[:max_debug].long()}')
    return games, rewards, dones
