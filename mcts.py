from typing import Callable, Dict

import logging

import numpy as np
import torch

from copy import deepcopy

from hparams import GenericHparams as Hparams
from networks import Inference, NetworkOutput
from logger import setup_logger

class MCTSInference:
    def __init__(self, hparams: Hparams, logger: logging.Logger):
        self.num_actions = hparams.num_actions
        self.logger = logger
        self.default_reward = hparams.default_reward
        self.hparams = hparams

    def initial(self, player_id: torch.Tensor, game_states: torch.Tensor) -> NetworkOutput:
        batch_size = game_states.shape[0]
        hidden_states = game_states.view(batch_size, -1)
        policy_logits = torch.randn([batch_size, self.num_actions])
        reward = torch.zeros(batch_size).float()
        self.logger.info(f'inference: initial: game_states: {game_states.shape}: hidden_states: {hidden_states.shape}, policy_logits: {policy_logits.shape}, reward: {reward.shape}')
        return NetworkOutput(reward=reward, hidden_state=hidden_states, policy_logits=policy_logits)

    def recurrent(self, hidden_states: torch.Tensor, player_id: torch.Tensor, actions: torch.Tensor) -> NetworkOutput:
        batch_size = hidden_states.shape[0]

        new_hidden_states = hidden_states.detach().clone()
        policy_logits = torch.randn([batch_size, self.num_actions])
        reward = torch.zeros(batch_size).float()
        value = torch.rand_like(reward)
        self.logger.info(f'inference: recurrent: hidden_states: {hidden_states.shape}, policy_logits: {policy_logits.shape}, reward: {reward.shape}, value: {value.shape}')
        return NetworkOutput(reward=reward, hidden_state=new_hidden_states, policy_logits=policy_logits, value=value)

class MinMaxStats:
    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value: torch.Tensor):
        self.maximum = max(self.maximum, value.max())
        self.minimum = min(self.minimum, value.min())

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class HashKey:
    def __init__(self, full_path: np.ndarray, episode_len: int):
        self.key = full_path[:episode_len]
        self.hash = hash(self.key.tobytes())

    def __hash__(self) -> int:
        return self.hash

    def __repr__(self):
        return f'{self.key}/{self.hash}'

    def __eq__(self, other: 'HashKey') -> bool:
        return (self.hash == other.hash) and np.all(self.key == other.key, 0)

def player_id_change(hparams: Hparams, player_id: torch.Tensor):
    next_id = player_id + 1
    next_id = torch.where(next_id > hparams.player_ids[-1], hparams.player_ids[0], next_id)
    return next_id.to(hparams.device)

class Tree:
    visit_count: torch.Tensor
    reward: torch.Tensor
    value_sum: torch.Tensor
    prior: torch.Tensor
    expanded: torch.Tensor
    saved_children_index: torch.Tensor
    hidden_states: Dict[HashKey, torch.Tensor]
    hparams: Hparams
    inference: Inference
    logger: logging.Logger

    def __init__(self, hparams: Hparams, player_id: torch.Tensor, inference: Inference, logger: logging.Logger):
        self.hparams = deepcopy(hparams)
        self.hparams.batch_size = len(player_id)

        self.inference = inference
        self.logger = logger

        self.min_max_stats = MinMaxStats()
        self.simulation_index = 0


        self.start_offset = 1
        max_size = self.start_offset + (1 + self.hparams.num_simulations) * self.hparams.num_actions

        self.saved_children_index = torch.zeros([self.hparams.batch_size, max_size]).long().to(hparams.device)
        self.visit_count = torch.zeros([self.hparams.batch_size, max_size]).long().to(hparams.device)
        self.value_sum = torch.zeros([self.hparams.batch_size, max_size], dtype=torch.float32).to(hparams.device)
        self.prior = torch.zeros([self.hparams.batch_size, max_size], dtype=torch.float32).to(hparams.device)
        self.reward = torch.zeros([self.hparams.batch_size, max_size], dtype=torch.float32).to(hparams.device)
        self.expanded = torch.zeros([self.hparams.batch_size, max_size]).bool().to(hparams.device)
        self.hidden_states = {}

    def new_children_index(self, size: int) -> torch.Tensor:
        index = torch.arange(self.hparams.num_actions).unsqueeze(0).to(self.hparams.device)
        index = index.tile([size, 1])

        index += self.simulation_index * self.hparams.num_actions
        index += self.start_offset
        return index

    def children_index(self, batch_index: torch.Tensor, node_index: torch.Tensor) -> torch.Tensor:
        generation_index = self.saved_children_index[batch_index].gather(1, node_index)
        generation_index = generation_index.tile([1, self.hparams.num_actions])

        action_index = torch.arange(self.hparams.num_actions).unsqueeze(0).to(self.hparams.device)
        action_index = action_index.tile([len(batch_index), 1])

        children_index = generation_index * self.hparams.num_actions
        children_index += action_index
        children_index += self.start_offset

        # max_debug = 10
        # self.logger.info(f'children_index:'
        #              f'simulation_index: {self.simulation_index}, '
        #              f'generation_index: {generation_index[:max_debug, 0]}, '
        #              f'node_index: {node_index[:max_debug]}, '
        #              f'children_index:\n{children_index[:max_debug]}')
        return children_index

    def expand(self, parent_index: torch.Tensor, policy_logits: torch.Tensor):
        if len(parent_index) != len(self.saved_children_index):
            raise ValueError(f'invalid parent index: parent_index: {parent_index.shape} != '
                             f'saved_children_index: {self.saved_children_index.shape}, parent_index: {parent_index.shape}')

        children_index = self.new_children_index(len(parent_index))
        self.saved_children_index.scatter_(1, parent_index, self.simulation_index)
        self.simulation_index += 1

        self.expanded.scatter_(1, parent_index, True)

        probs = torch.softmax(policy_logits, 1).type(self.prior.dtype)
        self.prior.scatter_(1, children_index, probs)
        # debug_max = 10
        # self.logger.info(f'expand: new_simulation_index: {self.simulation_index}\n'
        #                  f'parent_index: {parent_index.squeeze(1)[:debug_max]}\n'
        #                  f'children_index:\n{children_index[:debug_max]}\n'
        #                  f'probs:\n{probs[:debug_max]}\n'
        #                  f'prior:\n{self.prior.gather(1, children_index).squeeze(1)[:debug_max]}')

    def value(self, batch_index: torch.Tensor, children_index: torch.Tensor) -> torch.Tensor:
        visit_count = self.visit_count[batch_index].gather(1, children_index)
        value_sum = self.value_sum[batch_index].gather(1, children_index)

        value = torch.where(visit_count == 0, 0, value_sum / visit_count)
        # debug_max = 10
        # self.logger.info(f'value func: batch_index: {batch_index[:debug_max]}\n'
        #                  f'children_index:\n{children_index[:debug_max]}\n'
        #                  f'visit_count:\n{visit_count[:debug_max]}\n'
        #                  f'value_sum:\n{value_sum[:debug_max]}\n'
        #                  f'value:\n{value[:debug_max]}')
        #return self.min_max_stats.normalize(value)
        return value

    def add_exploration_noise(self, children_index: torch.Tensor, exploration_fraction: float):
        concentration = torch.ones([len(children_index), self.hparams.num_actions]).float() * self.hparams.dirichlet_alpha
        dist = torch.distributions.Dirichlet(concentration)
        noise = dist.sample()
        noise = noise.type(self.prior.dtype).to(self.hparams.device)

        orig_priors = self.prior.gather(1, children_index)
        priors = orig_priors * (1 - exploration_fraction) + noise * exploration_fraction
        self.prior.scatter_(1, children_index, priors)

    def select_children(self, batch_index: torch.Tensor, node_index: torch.Tensor, invalid_actions_mask: torch.Tensor) -> torch.Tensor:
        max_debug = 10

        children_index = self.children_index(batch_index, node_index)

        ucb_scores = self.ucb_scores(batch_index, node_index, children_index)
        # if depth_index < 20:
        #     self.logger.info(f'depth: {depth_index}, select_children: ucb_scores:\n{ucb_scores[:max_debug]}')
        ucb_scores[invalid_actions_mask] = float('-inf')
        max_scores = ucb_scores.max(1)[0]
        not_max_indexes = ucb_scores < max_scores.unsqueeze(1)

        rnd = torch.rand(*ucb_scores.shape, requires_grad=False, device=ucb_scores.device)
        ucb_scores += rnd

        ucb_scores[not_max_indexes] = float('-inf')
        max_indexes = ucb_scores.argmax(dim=1)

        # self.logger.info(f'select_children:\n'
        #                  f'batch_index: {batch_index[:max_debug]}\n'
        #                  f'node_index: {node_index.squeeze(1)[:max_debug]}\n'
        #                  f'children_index:\n{children_index[:max_debug]}\n'
        #                  f'invalid_actions_mask:\n{invalid_actions_mask[:max_debug]}\n'
        #                  f'max_masked_with_random_ucb_scores:\n{ucb_scores[:max_debug]}\n'
        #                  f'max_scores_before_random:\n{max_scores[:max_debug]}\n'
        #                  f'max_indexes/action_indexes:\n{max_indexes[:max_debug]}')

        children_index = children_index.gather(1, max_indexes.unsqueeze(1))
        return max_indexes, children_index

    def ucb_scores(self, batch_index: torch.Tensor, parent_index: torch.Tensor, children_index: torch.Tensor) -> torch.Tensor:
        parent_visit_count = self.visit_count[batch_index].gather(1, parent_index)
        visits_score = torch.log((parent_visit_count + self.hparams.pb_c_base + 1) / self.hparams.pb_c_base) + self.hparams.pb_c_init
        visits_score = visits_score * torch.sqrt(parent_visit_count)

        children_visit_count = self.visit_count[batch_index].gather(1, children_index)
        visits_score_norm = visits_score / (children_visit_count + 1)

        children_prior = self.prior[batch_index].gather(1, children_index)
        prior_score = visits_score_norm * children_prior

        children_value = self.value(batch_index, children_index)

        value_score = self.reward[batch_index].gather(1, children_index) + self.hparams.value_discount * children_value
        score = prior_score + value_score

        # max_debug = 10
        # self.logger.info(f'ucb_scores: '
        #                  f'children_index:\n{children_index[:max_debug]}\n'
        #                  f'children_visit_counts:\n{children_visit_count[:max_debug]}\n'
        #                  f'visits_score:\n{visits_score.squeeze(1)[:max_debug]}\n'
        #                  f'visits_score_norm:\n{visits_score_norm[:max_debug]}\n'
        #                  f'children_prior:\n{children_prior[:max_debug]}\n'
        #                  f'prior_score:\n{prior_score[:max_debug]}\n'
        #                  f'children_value:\n{children_value[:max_debug]}\n'
        #                  f'reward:\n{self.reward[batch_index].gather(1, children_index)[:max_debug]}\n'
        #                  f'value_score:\n{value_score[:max_debug]}\n'
        #                  f'score:\n{score[:max_debug]}')
        return score

    def backpropagate(self, last_player_id: torch.Tensor, player_id: torch.Tensor, search_path: torch.Tensor, episode_len: torch.Tensor, value: torch.Tensor):
        for current_episode_len in torch.arange(episode_len.max(), 0, step=-1).to(self.hparams.device):
            node_index = search_path[:, current_episode_len]
            node_index = node_index.unsqueeze(1)

            valid_episode_len_index = current_episode_len <= episode_len
            current_value = torch.where(valid_episode_len_index.unsqueeze(1), value, torch.zeros_like(value))

            # node_multiplier = torch.where(player_id[:, current_episode_len-1] == last_player_id, 1, -1)
            # node_multiplier = torch.where(valid_episode_len_index, node_multiplier, 0)

            # current_value *= node_multiplier.unsqueeze(1)

            # debug_max = 10
            # self.logger.info(f'backpropagate: '
            #                  f'current_episode_len: {current_episode_len}/{episode_len.max()}/{episode_len[:debug_max]}\n'
            #                  f'last_player_id:\n{last_player_id[:debug_max]}\n'
            #                  f'player_id:\n{player_id[:debug_max, current_episode_len-1]}\n'
            #                  f'node_index: {node_index.shape}\n{node_index.squeeze(1)[:debug_max]}\n'
            #                  f'node_multiplier: {node_multiplier[:debug_max]}\n'
            #                  f'reward: {self.reward.gather(1, node_index).squeeze(1)[:debug_max]}\n'
            #                  f'value: {value.shape}\n'
            #                  f'{value.squeeze(1)[:debug_max]}\n'
            #                  f'current_value: {current_value.shape}\n{current_value.squeeze(1)[:debug_max]}\n'
            #                  f'value_sum:\n{self.value_sum.gather(1, node_index).squeeze(1)[:debug_max]}')

            self.value_sum.scatter_add_(1, node_index, current_value)
            #self.logger.info(f'backpropagate: updated value_sum:\n{self.value_sum.gather(1, node_index).squeeze(1)}')


            visit_count = torch.where(valid_episode_len_index, 1, 0).unsqueeze(1)
            self.visit_count.scatter_add_(1, node_index, visit_count)

            value = self.reward.gather(1, node_index) + self.hparams.value_discount * value
            self.min_max_stats.update(value)

        zeros_index = torch.zeros(len(episode_len), 1, dtype=torch.int64, device=self.value_sum.device)
        self.value_sum.scatter_add_(1, zeros_index, value)
        self.visit_count.scatter_add_(1, zeros_index, torch.ones_like(zeros_index))

    def _store_states(self, search_path: torch.Tensor, episode_len: torch.Tensor, hidden_states: torch.Tensor):
        pass

    def _load_states(self, search_path: torch.Tensor, episode_len: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.zeros(self.hparams.batch_size, 12, *self.hparams.state_shape).to(self.hparams.device)
        return hidden_states

    def store_states(self, search_path: torch.Tensor, episode_len: torch.Tensor, hidden_states: torch.Tensor):
        search_path = search_path.detach().cpu().numpy().astype(np.int64)
        episode_len = episode_len.detach().cpu().numpy()
        hidden_states = hidden_states.detach().clone()
        for path, elen, hidden_state in zip(search_path, episode_len, hidden_states):
            key = HashKey(path, elen)
            if key in self.hidden_states:
                continue

            #print(f'store: {key}, path: {path[:elen]}, elen: {elen}')
            self.hidden_states[key] = hidden_state

    def load_states(self, search_path: torch.Tensor, episode_len: torch.Tensor) -> torch.Tensor:
        hidden_states = []

        search_path = search_path.detach().cpu().numpy().astype(np.int64)
        episode_len = episode_len.detach().cpu().numpy()
        for path, elen in zip(search_path, episode_len):
            key = HashKey(path, elen)
            #print(f'load: {key}, path: {path[:elen]}, elen: {elen}')

            hidden_states.append(self.hidden_states[key])

        hidden_states = torch.stack(hidden_states, 0).to(self.hparams.device)
        return hidden_states

    def run_one_simulation(self, initial_player_id: torch.Tensor, invalid_actions_mask: torch.Tensor):
        search_path = torch.zeros(self.hparams.batch_size, self.hparams.max_episode_len+1).long().to(self.hparams.device)
        actions = torch.zeros(self.hparams.batch_size, self.hparams.max_episode_len).long().to(self.hparams.device)
        player_id = torch.zeros(self.hparams.batch_size, self.hparams.max_episode_len, dtype=torch.int64).to(self.hparams.device)
        episode_len = torch.zeros(self.hparams.batch_size, dtype=torch.int64, device=self.hparams.device)
        max_debug = 10

        batch_index = torch.arange(self.hparams.batch_size).to(self.hparams.device)
        node_index = torch.zeros(self.hparams.batch_size, 1).long().to(self.hparams.device)

        search_path[:, 0] = node_index.squeeze(1)
        step_player_id = initial_player_id # initial_player_id is a copy, so it can be modified in place

        for depth_index in range(0, self.hparams.max_episode_len):
            action_index, children_index = self.select_children(batch_index, node_index, invalid_actions_mask[batch_index])

            # self.logger.info(f'depth: {depth_index}\n'
            #                  f'player_id:\n{step_player_id[batch_index][:max_debug]}\n'
            #                  f'node_index: {node_index.shape}\n'
            #                  f'{node_index.squeeze(1)[:max_debug]}\n'
            #                  f'action_index: {action_index.shape}\n'
            #                  f'{action_index[:max_debug]}\n'
            #                  f'children_index: {children_index.shape}\n'
            #                  f'{children_index.squeeze(1)[:max_debug]}')

            search_path[batch_index, depth_index+1] = children_index.squeeze(1).detach().clone()
            actions[batch_index, depth_index] = action_index.detach().clone()
            episode_len[batch_index] += 1
            player_id[batch_index, depth_index] = step_player_id[batch_index].detach().clone()


            expanded_index = self.expanded[batch_index].gather(1, children_index).squeeze(1) == True
            node_index = children_index[expanded_index]
            batch_index = batch_index[expanded_index]

            #self.logger.info(f'depth: {depth_index}, node_index: {node_index.shape}\nnode_index: {node_index[:max_debug]}\nplayer_id: {player_id[batch_index][:max_debug, :episode_len.max()]}')
            if len(node_index) == 0:
                break

            step_player_id = player_id_change(self.hparams, step_player_id)

        try:
            episode_len = episode_len.long()

            last_episode = episode_len - 1
            last_episode = last_episode.unsqueeze(1)

            last_children_index = search_path.gather(1, episode_len.unsqueeze(1))

            # max_debug = 10
            # self.logger.info(f'search_path: {search_path.shape}\n{search_path[:max_debug, :episode_len.max()+1]}')
            # self.logger.info(f'actions: {actions.shape}\n{actions[:max_debug, :episode_len.max()]}')
            # self.logger.info(f'player_id: {player_id.shape}\n{player_id[:max_debug, :episode_len.max()]}')
            # self.logger.info(f'episode_len: {episode_len[:max_debug]}, episode_len_max: {episode_len.max()}')
            # self.logger.info(f'last_children_index: {last_children_index.squeeze(1)[:max_debug]}')

            hidden_states = self.load_states(search_path, episode_len)

            last_actions = actions.gather(1, last_episode).squeeze(1)
            last_player_id = player_id.gather(1, last_episode).squeeze(1)

            out = self.inference.recurrent(hidden_states, last_actions)

            self.store_states(search_path, episode_len+1, out.hidden_state)
            self.reward.scatter_(1, last_children_index, out.reward)

            self.expand(last_children_index, out.policy_logits)
            self.backpropagate(last_player_id, player_id, search_path, episode_len, out.value)
        except:
            # self.logger.error(f'search_path: {search_path.shape}\n{search_path[:max_debug, :episode_len.max()+1]}')
            # self.logger.error(f'actions: {actions.shape}\n{actions[:max_debug, :episode_len.max()]}')
            # self.logger.error(f'episode_len: {episode_len[:max_debug]}')
            # self.logger.error(f'player_id: {player_id[:max_debug]}')
            raise

        return search_path, episode_len
