from typing import Callable, Dict, List, NamedTuple

import logging

from dataclasses import dataclass
from time import perf_counter

import torch
from torch.utils.tensorboard import SummaryWriter

from hparams import GenericHparams as Hparams
from inference import Inference
import module_loader
import mcts

@dataclass
class TrainElement:
    values: torch.Tensor
    last_rewards: torch.Tensor
    children_visits: torch.Tensor
    game_states: torch.Tensor
    actions: torch.Tensor
    sample_len: torch.Tensor
    player_ids: torch.Tensor

    @staticmethod
    def from_dict(sample_dict: Dict[str, torch.Tensor]) -> 'TrainElement':
        return TrainElement(
            game_states = sample_dict['game_states'],
            actions = sample_dict['actions'],
            sample_len = sample_dict['sample_len'],
            values = sample_dict['values'],
            last_rewards = sample_dict['last_rewards'],
            children_visits = sample_dict['children_visits'],
            player_ids = sample_dict['player_ids'],
        )

def roll_by_gather(mat, dim, shifts: torch.LongTensor):
    # assumes 2D array
    n_rows, n_cols = mat.shape

    if dim==0:
        arange1 = torch.arange(n_rows, device=shifts.device).view((n_rows, 1)).repeat((1, n_cols))
        arange2 = (arange1 - shifts) % n_rows
        return torch.gather(mat, 0, arange2)
    elif dim==1:
        arange1 = torch.arange(n_cols, device=shifts.device).view(( 1,n_cols)).repeat((n_rows,1))
        arange2 = (arange1 - shifts) % n_cols
        return torch.gather(mat, 1, arange2)

class GameStats:
    episode_len: torch.Tensor
    rewards: torch.Tensor
    root_values: torch.Tensor
    children_visits: torch.Tensor
    actions: torch.Tensor
    player_ids: torch.Tensor
    dones: torch.Tensor
    game_states: torch.Tensor

    def __init__(self, hparams: Hparams, logger: logging.Logger):
        self.logger = logger
        self.hparams = hparams

        self.episode_len = torch.zeros(hparams.batch_size, dtype=torch.uint8, device=hparams.device)
        self.rewards = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.float32, device=hparams.device)
        self.root_values = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.float32, device=hparams.device)
        self.children_visits = torch.zeros(hparams.batch_size, hparams.num_actions, hparams.max_episode_len, dtype=torch.float32, device=hparams.device)
        self.actions = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.uint8, device=hparams.device)
        self.player_ids = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.uint8, device=hparams.device)
        self.dones = torch.zeros(hparams.batch_size, dtype=torch.bool, device=hparams.device)
        self.game_states = torch.zeros([hparams.batch_size, hparams.max_episode_len] + hparams.state_shape, dtype=torch.uint8, device=hparams.device)

        self.stored_tensors = {
            'rewards': self.rewards,
            'root_values': self.root_values,
            'children_visits': self.children_visits,
            'actions': self.actions,
            'player_ids': self.player_ids,
            'dones': self.dones,
            'game_states': self.game_states,
        }

    def move(self, device):
        self.hparams.device = device

        self.episode_len = self.episode_len.to(device)
        self.rewards = self.rewards.to(device)
        self.root_values = self.root_values.to(device)
        self.children_visits = self.children_visits.to(device)
        self.actions = self.actions.to(device)
        self.player_ids = self.player_ids.to(device)
        self.dones = self.dones.to(device)
        self.game_states = self.game_states.to(device)
        return self

    def __len__(self):
        return self.episode_len.sum().item()

    def append(self, index: torch.Tensor, tensors_dict: Dict[str, torch.Tensor]):
        episode_len = self.episode_len[index].long()
        index = index.long()

        for key, value in tensors_dict.items():
            if not key in self.stored_tensors:
                msg = f'invalid key: {key}, tensor shape: {value.shape}, available keys: {list(self.stored_tensors.keys())}'
                self.logger.critical(msg)
                raise ValueError(msg)

            tensor = self.stored_tensors[key]
            if key == 'children_visits':
                tensor[index, :, episode_len] = value.detach().clone()
            elif key == 'dones':
                tensor[index] = value.detach().clone()
            else:
                tensor[index, episode_len] = value.detach().clone()

        self.episode_len[index] += 1

    def update_last_rewards(self, win_index: torch.Tensor):
        episode_len = self.episode_len[win_index].long()
        self.rewards[win_index, episode_len-1] = -1

    def make_target(self, start_index: torch.Tensor) -> List[TrainElement]:
        target_values = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=self.hparams.device)
        target_last_rewards = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=self.hparams.device)
        target_children_visits = torch.zeros(len(start_index), self.hparams.num_actions, self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=self.hparams.device)
        taken_actions = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=torch.int64, device=self.hparams.device)
        player_ids = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=torch.int64, device=self.hparams.device)

        sample_len = torch.zeros(len(start_index), dtype=torch.int64, device=self.hparams.device)

        batch_index = torch.arange(len(start_index), dtype=torch.int64, device=self.hparams.device)
        game_states = self.game_states[batch_index, start_index].float()

        if start_index.device != self.episode_len.device:
            msg = f'start_index: {start_index.device}, self.episode_len: {self.episode_len.device}'
            self.logger.critical(msg)
            raise ValueError(msg)


        for unroll_index in range(0, self.hparams.num_unroll_steps+1):
            current_index = start_index + unroll_index
            bootstrap_index = current_index + self.hparams.td_steps

            if bootstrap_index.device != self.episode_len.device:
                self.logger.critical(f'bootstrap_index: {bootstrap_index.device}, self.episode_len: {self.episode_len.device}')
                exit(-1)

            bootstrap_update_index = bootstrap_index < self.episode_len

            # self.logger.info(f'{unroll_index}: '
            #              f'start_index: {start_index.cpu().numpy()}, '
            #              f'current_index: {current_index.cpu().numpy()}, '
            #              f'bootstrap_index: {bootstrap_index.cpu().numpy()}, '
            #              f'bootstrap_update_index: {bootstrap_update_index.cpu().numpy()}/{bootstrap_update_index.shape}, '
            #              )
            values = torch.zeros(len(self.root_values), device=self.hparams.device).float()
            if bootstrap_update_index.sum() > 0:
                #self.logger.info(f'bootstrap_update_index: {bootstrap_update_index}')
                valid_batch_index = batch_index[bootstrap_update_index]
                last_discount = self.hparams.value_discount ** self.hparams.td_steps
                values[bootstrap_update_index] = self.root_values[valid_batch_index, bootstrap_index].float() * last_discount

            discount_mult = torch.logspace(0, self.hparams.td_steps, self.rewards.shape[1], base=self.hparams.value_discount).to(self.hparams.device)
            if discount_mult.device != start_index.device:
                msg = f'1 start_index: {start_index.device}, discount_mutl: {discount_mult.device}, hparams.device: {self.hparams.device}'
                self.logger.critical(msg)
                raise ValueError(msg)

            discount_mult = discount_mult.unsqueeze(0)
            discount_mult = discount_mult.tile([len(values), 1])
            if discount_mult.device != start_index.device:
                msg = f'2 start_index: {start_index.device}, discount_mutl: {discount_mult.device}, hparams.device: {self.hparams.device}'
                self.logger.critical(msg)
                raise ValueError(msg)

            last_index = torch.minimum(bootstrap_index, self.episode_len)
            all_rewards_index = torch.arange(self.rewards.shape[1], device=self.hparams.device).long().unsqueeze(0).tile([len(start_index), 1])

            if discount_mult.device != start_index.device:
                msg = f'start_index: {start_index.device}, discount_mutl: {discount_mult.device}'
                self.logger.critical(msg)
                raise ValueError(msg)

            discount_mult = roll_by_gather(discount_mult, 1, start_index.unsqueeze(1))
            discount_mult = torch.where(all_rewards_index < current_index.unsqueeze(1), 0, discount_mult)
            discount_mult = torch.where(all_rewards_index >= last_index.unsqueeze(1), 0, discount_mult)

            masked_rewards = torch.where(all_rewards_index < current_index.unsqueeze(1), 0, self.rewards)
            masked_rewards = torch.where(all_rewards_index >= last_index.unsqueeze(1), 0, masked_rewards)

            discounted_rewards = self.rewards.float() * discount_mult
            discounted_rewards = discounted_rewards.sum(1)
            values += discounted_rewards

            valid_index = current_index < self.episode_len

            current_valid_index = current_index[valid_index]
            target_values[valid_index, unroll_index] = values[valid_index]
            target_children_visits[valid_index, :, unroll_index] = self.children_visits[valid_index, :, current_valid_index].float()

            if unroll_index > 0:
                target_last_rewards[valid_index, unroll_index] = self.rewards[valid_index, current_valid_index-1].float()
            taken_actions[valid_index, unroll_index] = self.actions[valid_index, current_valid_index].long()
            player_ids[valid_index, unroll_index] = self.player_ids[valid_index, current_valid_index].long()
            sample_len[valid_index] += 1

        samples = []
        for i in range(len(target_values)):
            elm = TrainElement(
                values=target_values[i],
                last_rewards=target_last_rewards[i],
                children_visits=target_children_visits[i],
                game_states=game_states[i],
                actions=taken_actions[i],
                sample_len=sample_len[i],
                player_ids=player_ids[i],
            )
            samples.append(elm)

        return samples


class Train:
    def __init__(self,
                 game_ctl: module_loader.GameModule,
                 inference: Inference,
                 logger: logging.Logger,
                 summary_writer: SummaryWriter,
                 summary_prefix: str,
                 action_selection_fn: Callable):
        self.game_ctl = game_ctl
        self.hparams = game_ctl.hparams
        self.logger = logger
        self.inference = inference
        self.summary_writer = summary_writer
        self.summary_prefix = summary_prefix
        self.summary_step = 0

        self.action_selection_fn = action_selection_fn

        self.num_train_steps = 0
        self.game_stats = {player_id:GameStats(game_ctl.hparams, self.logger) for player_id in self.hparams.player_ids}

    def run_simulations(self, initial_player_id: torch.Tensor, initial_game_states: torch.Tensor):
        start_simulation_time = perf_counter()

        tree = mcts.Tree(self.hparams, initial_player_id, self.inference, self.logger)

        batch_size = len(initial_game_states)
        batch_index = torch.arange(batch_size).long()
        node_index = torch.zeros(batch_size).long().to(self.hparams.device)

        out = self.inference.initial(initial_player_id, initial_game_states)

        episode_len = torch.ones(len(node_index), dtype=torch.uint8).to(self.hparams.device)
        search_path = torch.zeros(len(node_index), 1).long().to(self.hparams.device)

        tree.store_states(search_path, episode_len, out.hidden_state)

        tree.expand(initial_player_id, batch_index, node_index, out.policy_logits)
        tree.visit_count[batch_index, node_index] = 1
        tree.value_sum[batch_index, node_index] += out.value

        if self.hparams.add_exploration_noise:
            children_index = tree.children_index(batch_index, node_index)
            tree.add_exploration_noise(batch_index, children_index, self.hparams.exploration_fraction)

        invalid_actions_mask = self.game_ctl.invalid_actions_mask(self.game_ctl.game_hparams, initial_game_states)

        for _ in range(self.hparams.num_simulations):
            search_path, episode_len = tree.run_one_simulation(initial_player_id, invalid_actions_mask)

        simulation_time = perf_counter() - start_simulation_time
        one_sim_ms = int(simulation_time / self.hparams.num_simulations * 1000)

        node_index = torch.zeros(batch_size).long().to(self.hparams.device)
        children_index = tree.children_index(batch_index, node_index)
        children_visit_counts = tree.visit_count[batch_index].gather(1, children_index).float()
        children_sum_visits = children_visit_counts.sum(1)
        children_visits = children_visit_counts / children_sum_visits.unsqueeze(1)
        root_values = tree.value(batch_index, node_index.unsqueeze(1)).squeeze(1)

        actions = self.action_selection_fn(children_visit_counts)

        actions = actions.type(torch.uint8)
        # max_debug = 10
        # self.logger.info(f'train_steps: {self.num_train_steps}, '
        #                  f'children_index:\n{children_index[:max_debug]}\n'
        #                  f'children_visit_counts:\n{children_visit_counts[:max_debug]}\n'
        #                  f'children_sum_visits:\n{children_sum_visits[:max_debug]}\n'
        #                  f'children_visits:\n{children_visits[:max_debug]}\n'
        #                  f'root_values: {root_values.shape}\n{root_values[:max_debug]}\n'
        #                  f'actions:\n{actions[:max_debug]}')
        return actions, children_visits, root_values

    def update_train_steps(self):
        self.num_train_steps += 1

def run_single_game(hparams: Hparams, train: Train, num_steps: int) -> Dict[int, GameStats]:
    game_states = torch.zeros(hparams.batch_size, *hparams.state_shape, dtype=torch.uint8).to(hparams.device)
    player_ids = torch.ones(hparams.batch_size, device=hparams.device, dtype=torch.uint8) * hparams.player_ids[0]

    active_games_index = torch.arange(hparams.batch_size).long().to(hparams.device)

    while True:
        active_game_states = game_states[active_games_index]
        active_player_ids = player_ids[active_games_index]
        actions, children_visits, root_values = train.run_simulations(active_player_ids, active_game_states)
        new_game_states, rewards, dones = train.game_ctl.step_games(train.game_ctl.game_hparams, active_game_states, active_player_ids, actions)
        game_states[active_games_index] = new_game_states.detach().clone()

        if not torch.all(active_player_ids == active_player_ids[0]):
            num_equal = (active_player_ids == active_player_ids[0]).sum().item()
            num_not_equal = (active_player_ids != active_player_ids[0]).sum().item()
            raise ValueError(f'bug: not all active_player_ids: {len(active_player_ids)} equal to active_player_ids[0]: {active_player_ids[0]}, '
                             f'equal: {num_equal}, '
                             f'not_equal: {num_not_equal}')

        player_id = active_player_ids[0].item()
        train.game_stats[player_id].append(active_games_index, {
            'children_visits': children_visits,
            'root_values': root_values,
            'game_states': active_game_states,
            'rewards': rewards,
            'actions': actions,
            'dones': dones,
            'player_ids': active_player_ids,
        })

        win_index = active_games_index[torch.logical_and((dones == True), (rewards > 0))]
        other_player_id = mcts.player_id_change(hparams, torch.tensor(player_id)).item()
        train.game_stats[other_player_id].update_last_rewards(win_index)

        train.update_train_steps()

        # max_debug = 10
        # train.logger.info(f'game:\n{game_states[0].detach().cpu().numpy().astype(int)}\n'
        #                   f'actions:\n{actions[:max_debug]}\n'
        #                   f'children_visits:\n{children_visits[:max_debug]}\n'
        #                   f'root_values:\n{root_values[:max_debug]}\n'
        #                   f'rewards:\n{rewards[:max_debug]}\n'
        #                   f'dones:\n{dones[:max_debug]}\n'
        #                   f'player_ids:\n{player_ids[:max_debug]}\n'
        #                   f'active_game_index:\n{active_games_index[:max_debug]}'
        #                   )

        if dones.sum() == len(dones):
            break

        player_ids = mcts.player_id_change(hparams, player_ids)
        active_games_index = active_games_index[dones != True]

        num_steps -= 1
        if num_steps == 0:
            break

    return train.game_stats
