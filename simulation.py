from typing import Callable, Dict, List, NamedTuple

import logging

from dataclasses import dataclass
from time import perf_counter

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from hparams import GenericHparams as Hparams
from networks import GameState, Inference
import module_loader
import mcts

@dataclass
class TrainElement:
    start_index: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    children_visits: torch.Tensor
    initial_game_state: torch.Tensor
    actions: torch.Tensor
    sample_len: torch.Tensor
    player_ids: torch.Tensor

    def __len__(self) -> int:
        return len(self.values)

    def __hash__(self) -> int:
        state_data = self.initial_game_state.detach().cpu().numpy().tostring()
        action_data = self.actions.detach().cpu().numpy().tostring()
        start_index_data = self.start_index.detach().cpu().numpy().tostring()
        sample_len_data = self.sample_len.detach().cpu().numpy().tostring()
        player_ids_data = self.player_ids.detach().cpu().numpy().tostring()
        return hash((state_data, action_data, start_index_data, sample_len_data, player_ids_data))

    def __eq__(self, other: 'TrainElement') -> bool:
        return torch.all(self.initial_game_state == other.initial_game_state) and torch.all(self.values == other.values) and torch.all(self.actions == other.actions)

    def to(self, device):
        self.start_index = self.start_index.to(device)
        self.values = self.values.to(device)
        self.rewards = self.rewards.to(device)
        self.children_visits = self.children_visits.to(device)
        self.initial_game_state = self.initial_game_state.to(device)
        self.actions = self.actions.to(device)
        self.sample_len = self.sample_len.to(device)
        self.player_ids = self.player_ids.to(device)
        return self

def roll_by_gather(mat, dim, shifts: torch.LongTensor):
    # assumes 2D array
    n_rows, n_cols = mat.shape

    if dim == 0:
        arange1 = torch.arange(n_rows, device=shifts.device).view((n_rows, 1)).repeat((1, n_cols))
        arange2 = (arange1 - shifts) % n_rows
        return torch.gather(mat, 0, arange2)
    elif dim == 1:
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
    game_states: List[torch.Tensor]
    initial_values: torch.Tensor
    initial_policy_probs: torch.Tensor

    def __init__(self, hparams: Hparams, logger: logging.Logger):
        self.logger = logger
        self.hparams = hparams

        self.episode_len = torch.zeros(hparams.batch_size, dtype=torch.int64, device=hparams.device)
        self.rewards = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.float32, device=hparams.device)
        self.root_values = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.float32, device=hparams.device)
        self.initial_values = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.float32, device=hparams.device)
        self.children_visits = torch.zeros(hparams.batch_size, hparams.num_actions, hparams.max_episode_len, dtype=torch.float32, device=hparams.device)
        self.initial_policy_probs = torch.zeros(hparams.batch_size, hparams.num_actions, hparams.max_episode_len, dtype=torch.float32, device=hparams.device)
        self.actions = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.int64, device=hparams.device)
        self.player_ids = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.int64, device=hparams.device)
        self.dones = torch.zeros(hparams.batch_size, dtype=torch.bool, device=hparams.device)
        self.game_states = []

        self.stored_tensors = {
            'episode_len': self.episode_len,
            'rewards': self.rewards,
            'root_values': self.root_values,
            'children_visits': self.children_visits,
            'actions': self.actions,
            'player_ids': self.player_ids,
            'dones': self.dones,
            'game_states': self.game_states,
            'initial_values': self.initial_values,
            'initial_policy_probs': self.initial_policy_probs,
        }

    def to(self, device):
        for key, tensor in self.stored_tensors.items():
            if key == 'game_states':
                self.game_states = [state.to(device) for state in self.game_states]
            else:
                tensor.to(device)

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

            dst = self.stored_tensors[key]
            if key == 'children_visits' or key == 'initial_policy_probs':
                dst[index, :, episode_len] = value.detach().clone()
                new_visits = dst[index, :, episode_len]

                if torch.any(value != new_visits):
                    raise ValueError(f'could not update tensor in place:\nvalue:\n{value[:10]}\nnew_visits:\n{new_visits}')
            elif key == 'dones':
                dst[index] = value.detach().clone()
            elif key == 'game_states':
                dst.append(value.detach().clone())
            elif key == 'episode_len':
                continue
            else:
                dst[index, episode_len] = value.detach().clone()

        self.episode_len[index] += 1

    def update_last_reward_and_values(self, win_index: torch.Tensor, rewards: torch.Tensor):
        episode_len = self.episode_len[win_index].long()
        self.rewards[win_index, episode_len-1] = rewards

    def index(self, batch_index: torch.Tensor) -> 'GameStats':
        dst = GameStats(self.hparams, self.logger)
        for key, value in self.stored_tensors.items():
            if key == 'game_states':
                for state in self.game_states:
                    batch = state[batch_index]
                    dst.game_states.append(batch)
                    dst.stored_tensors[key] = dst.game_states
            else:
                dst.__setattr__(key, value[batch_index])
                dst.stored_tensors[key] = dst.__getattribute__(key)

        return dst

    def make_target(self, start_index: torch.Tensor) -> List[TrainElement]:
        target_values = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=start_index.device)
        target_rewards = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=start_index.device)
        target_children_visits = torch.zeros(len(start_index), self.hparams.num_actions, self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=start_index.device)
        taken_actions = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=torch.int64, device=start_index.device)
        player_ids = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=torch.int64, device=start_index.device)

        sample_len = torch.zeros(len(start_index), dtype=torch.int64, device=start_index.device)

        if start_index.device != self.episode_len.device:
            msg = f'start_index: {start_index.device}, self.episode_len: {self.episode_len.device}'
            self.logger.critical(msg)
            raise ValueError(msg)

        td_steps = min(self.hparams.td_steps, self.hparams.max_episode_len)
        discount_mult = torch.logspace(0, 1, td_steps, base=self.hparams.value_discount).to(start_index.device)
        discount_mult = discount_mult.unsqueeze(0).tile([len(start_index), 1])
        all_rewards_index = torch.arange(0, self.rewards.shape[1]).unsqueeze(0).tile([len(start_index), 1]).to(start_index.device)

        for unroll_step in range(0, self.hparams.num_unroll_steps+1):
            start_unroll_index = start_index + unroll_step
            bootstrap_index = start_unroll_index + td_steps
            bootstrap_update_index = bootstrap_index < self.episode_len

            # self.logger.info(f'make_target: {unroll_step}:\n'
            #                  f'start_index: {start_index}\n'
            #                  f'start_unroll_index: {start_unroll_index}\n'
            #                  f'bootstrap_index: {bootstrap_index}\n'
            #                  f'bootstrap_update_index: {bootstrap_update_index}\n'
            #                  f'children_visits: {self.children_visits.shape}\n{self.children_visits[:10]}\n'
            #                  f'{self.children_visits[0]}\n'
            #              )
            values = torch.zeros(len(start_index), device=start_index.device).float()
            if bootstrap_update_index.sum() > 0:
                #self.logger.info(f'bootstrap_update_index: {bootstrap_update_index}')
                last_discount = self.hparams.value_discount ** td_steps
                values[bootstrap_update_index] = self.root_values[bootstrap_update_index, bootstrap_index].float() * last_discount

            start_unroll_valid_bool_index = start_unroll_index < self.episode_len

            rewards = torch.where(all_rewards_index < start_unroll_index.unsqueeze(1), 0, self.rewards)
            rewards = torch.where(all_rewards_index >= bootstrap_index.unsqueeze(1), 0, rewards)

            discount = roll_by_gather(discount_mult, 1, start_unroll_index.unsqueeze(1))
            discounted_rewards = rewards * discount
            discounted_rewards = discounted_rewards.sum(1)
            values += discounted_rewards

            batch_index = torch.arange(len(start_index), device=start_index.device)
            target_values[batch_index, unroll_step] = (values + self.root_values[batch_index, start_unroll_index]) / 2
            #target_values[:, unroll_step] = values

            target_children_visits[batch_index, :, unroll_step] = self.children_visits[batch_index, :, start_unroll_index].float()
            player_ids[batch_index, unroll_step] = self.player_ids[batch_index, start_unroll_index].long()

            target_rewards[batch_index, unroll_step] = self.rewards[batch_index, start_unroll_index].float()
            taken_actions[batch_index, unroll_step] = self.actions[batch_index, start_unroll_index].long()

            sample_len[start_unroll_valid_bool_index] += 1

        samples = []
        for i in range(len(target_values)):
            elm_start_index = start_index[i]

            elm = TrainElement(
                start_index=elm_start_index,
                values=target_values[i],
                rewards=target_rewards[i],
                children_visits=target_children_visits[i],
                initial_game_state=self.game_states[elm_start_index][i],
                actions=taken_actions[i],
                sample_len=sample_len[i],
                player_ids=player_ids[i],
            )
            samples.append(elm)

        return samples


class Simulation:
    @torch.no_grad()
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

    @torch.no_grad()
    def run_simulations(self, initial_player_id: torch.Tensor, initial_game_state: torch.Tensor, invalid_actions_mask: torch.Tensor):
        start_simulation_time = perf_counter()

        tree = mcts.Tree(self.hparams, initial_player_id, self.inference, self.logger)

        batch_size = len(initial_player_id)
        batch_index = torch.arange(batch_size).long().to(self.hparams.device)
        node_index = torch.zeros(batch_size, 1).long().to(self.hparams.device)

        out = self.inference.initial(initial_game_state)

        episode_len = torch.ones(batch_size, dtype=torch.int64).to(self.hparams.device)
        search_path = torch.zeros(batch_size, 1, dtype=torch.int64).to(self.hparams.device)

        tree.store_states(search_path, episode_len, out.hidden_state)

        tree.expand(node_index, out.policy_logits)
        tree.visit_count.scatter_(1, node_index, 1)
        tree.value_sum.scatter_(1, node_index, out.value)

        if self.hparams.add_exploration_noise:
            children_index = tree.children_index(batch_index, node_index)
            tree.add_exploration_noise(children_index, self.hparams.exploration_fraction)

        for _ in range(self.hparams.num_simulations):
            search_path, episode_len = tree.run_one_simulation(invalid_actions_mask.detach().clone())

        simulation_time = perf_counter() - start_simulation_time
        one_sim_ms = int(simulation_time / self.hparams.num_simulations * 1000)

        children_index = tree.children_index(batch_index, node_index)
        children_visit_counts = tree.visit_count.gather(1, children_index).float()
        root_values = tree.value(batch_index, node_index).squeeze(1)

        actions = self.action_selection_fn(children_visit_counts, episode_len)
        max_debug = 10
        self.logger.info(f'children_index:\n{children_index[:max_debug]}\n'
                         f'children_visit_counts:\n{children_visit_counts[:max_debug]}\n'
                         f'root_values: {root_values.shape}\n{root_values[:max_debug]}\n'
                         f'actions:\n{actions[:max_debug]}')
        return actions, children_visit_counts, root_values, out

    @torch.no_grad()
    def run_single_game_and_collect_stats(self, hparams: Hparams) -> Dict[int, GameStats]:
        game_stats = {player_id:GameStats(self.game_ctl.hparams, self.logger) for player_id in self.hparams.player_ids}

        game_states = torch.zeros(hparams.batch_size, *hparams.state_shape, dtype=torch.float32, device=hparams.device)
        player_ids = torch.ones(hparams.batch_size, device=hparams.device, dtype=torch.int64) * hparams.player_ids[0]

        active_games_index = torch.arange(hparams.batch_size).long().to(hparams.device)
        game_state_stacks = {player_id:GameState(hparams.batch_size, hparams, self.game_ctl.network_hparams) for player_id in hparams.player_ids}

        while True:
            active_player_ids = player_ids[active_games_index].detach().clone()

            # we do not care if it will be modified in place, we will make a copy when pushing this state into the stack of states
            active_game_states = game_states[active_games_index]
            invalid_actions_mask = self.game_ctl.invalid_actions_mask(self.game_ctl.game_hparams, active_game_states)

            player_id = active_player_ids[0].item()
            if torch.any(player_id != player_ids):
                raise ValueError(f'pushing non-consistent player_ids: player_id: {player_id}, not_equal: {(player_id != player_ids).sum()}/{len(player_ids)}')

            game_state_stacks[player_id].push_game(player_ids, game_states)
            game_state_stack_converted = game_state_stacks[player_id].create_state()

            actions, children_visits, root_values, out_initial = self.run_simulations(active_player_ids, game_state_stack_converted[active_games_index], invalid_actions_mask)
            new_game_states, rewards, dones = self.game_ctl.step_games(self.game_ctl.game_hparams, active_game_states, active_player_ids, actions)
            game_states[active_games_index] = new_game_states.detach().clone()

            game_stats[player_id].append(active_games_index, {
                'children_visits': children_visits,
                'initial_values': out_initial.value.squeeze(1),
                'initial_policy_probs': F.softmax(out_initial.policy_logits, 1),
                'root_values': root_values,
                'rewards': rewards,
                'actions': actions,
                'dones': dones,
                'player_ids': active_player_ids,

                # needs to save the whole tensor of batch_size size, because we use a list of those, not copying them into a single tensor like other fields here
                'game_states': game_state_stack_converted,
            })

            other_player_id = mcts.player_id_change(hparams, torch.tensor(player_id)).item()

            win_index = active_games_index[torch.logical_and((dones == True), (rewards > 0))]
            other_rewards = torch.ones_like(win_index).float() * -1
            game_stats[other_player_id].update_last_reward_and_values(win_index, other_rewards)

            lose_index = active_games_index[torch.logical_and((dones == True), (rewards < 0))]
            other_rewards = torch.ones_like(lose_index).float() * 1
            game_stats[other_player_id].update_last_reward_and_values(lose_index, other_rewards)

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

        return game_stats
