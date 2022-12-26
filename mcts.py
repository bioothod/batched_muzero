from typing import Dict

import logging

import numpy as np
import torch

from copy import deepcopy

from hparams import Hparams
from inference import Inference, NetworkOutput
from logger import setup_logger

class MCTSInference(Inference):
    def __init__(self, hparams: Hparams, logger: logging.Logger):
        super().__init__()

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

    def recurrent(self, hidden_states: torch.Tensor, actions: torch.Tensor) -> NetworkOutput:
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
    def __init__(self, full_path: torch.Tensor, episode_len: torch.Tensor):
        key = full_path[:episode_len]
        #self.key = key.detach().cpu().clone().byte().numpy().tobytes()
        # this is fast
        self.key = key.tostring()
        self.hash = hash(self.key)

    def __hash__(self) -> int:
        return self.hash

    def __repr__(self):
        return f'{self.key}/{self.hash}'

    def __eq__(self, other: 'HashKey') -> bool:
        return self.hash == other.hash and self.key == other.key

def player_id_change(hparams: Hparams, player_id: torch.Tensor):
    next_id = player_id + 1
    next_id = torch.where(next_id > hparams.player_ids[-1], hparams.player_ids[0], next_id)
    return next_id.to(hparams.device)

class Tree:
    visit_count: torch.Tensor
    reward: torch.Tensor
    value_sum: torch.Tensor
    prior: torch.Tensor
    player_id: torch.Tensor
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
        self.value_sum = torch.zeros([self.hparams.batch_size, max_size]).float().to(hparams.device)
        self.prior = torch.zeros([self.hparams.batch_size, max_size]).float().to(hparams.device)
        self.reward = torch.zeros([self.hparams.batch_size, max_size]).float().to(hparams.device)
        self.expanded = torch.zeros([self.hparams.batch_size, max_size]).bool().to(hparams.device)
        self.player_id = torch.zeros([self.hparams.batch_size, max_size]).long().to(hparams.device)
        self.player_id[:, 0] = player_id.detach().clone().to(self.hparams.device)
        self.hidden_states = {}

    def new_children_index(self, batch_index: torch.Tensor) -> torch.Tensor:
        index = torch.arange(self.hparams.num_actions).unsqueeze(0).to(self.hparams.device)
        index = index.tile([len(batch_index), 1])

        index += self.simulation_index * self.hparams.num_actions
        index += self.start_offset
        return index

    def children_index(self, batch_index: torch.Tensor, node_index: torch.Tensor) -> torch.Tensor:
        generation_index = self.saved_children_index[batch_index].gather(1, node_index.unsqueeze(1))
        generation_index = generation_index.tile([1, self.hparams.num_actions])

        action_index = torch.arange(self.hparams.num_actions).unsqueeze(0).to(self.hparams.device)
        action_index = action_index.tile([len(batch_index), 1])

        children_index = generation_index * self.hparams.num_actions
        children_index += action_index
        children_index += self.start_offset

        # max_debug = 10
        # self.logger.info(f'children_index: batch_index: {batch_index[:max_debug]}, '
        #              f'simulation_index: {self.simulation_index}, '
        #              f'generation_index: {generation_index[:max_debug, 0]}, '
        #              f'node_index: {node_index[:max_debug]}, '
        #              f'children_index:\n{children_index[:max_debug]}')
        return children_index

    def expand(self, player_id: torch.Tensor, batch_index: torch.Tensor, node_index: torch.Tensor, policy_logits: torch.Tensor):
        if len(batch_index) != len(self.saved_children_index):
            # max_debug = 10
            # self.logger.critical(f'expand: batch_index: {batch_index.shape}\n{batch_index[:max_debug]}\n'
            #                   f'saved_children_index: {self.saved_children_index.shape}\n{self.saved_children_index[:max_debug]}\n'
            #                   f'node_index: {node_index.shape}\n{node_index[:max_debug]}')
            raise ValueError(f'invalid batch index')

        node_index = node_index.unsqueeze(1)

        children_index = self.new_children_index(batch_index)
        self.saved_children_index.scatter_(1, node_index, self.simulation_index)

        # max_debug = 10
        # self.logger.info(f'expand: generation: {self.simulation_index}, node_index: {node_index[:max_debug]}, children_index:\n{children_index[:max_debug]}')

        self.expanded.scatter_(1, node_index, True)
        self.player_id.scatter_(1, node_index, player_id.unsqueeze(1))

        probs = torch.softmax(policy_logits, 1)
        self.prior.scatter_(1, children_index, probs)
        #self.logger.debug(f'expand: priors:\n{self.prior[batch_index].gather(1, children_index)[:max_debug]}')

        self.simulation_index += 1

    def value(self, batch_index: torch.Tensor, children_index: torch.Tensor) -> torch.Tensor:
        visit_count = self.visit_count[batch_index].gather(1, children_index)
        value_sum = self.value_sum[batch_index].gather(1, children_index)

        return torch.where(visit_count == 0, 0, value_sum / visit_count)

    def add_exploration_noise(self, batch_index: torch.Tensor, node_index: torch.Tensor, exploration_fraction: float):
        concentration = torch.ones([len(batch_index), self.hparams.num_actions]).float() * self.hparams.dirichlet_alpha
        dist = torch.distributions.Dirichlet(concentration)
        noise = dist.sample()
        noise = noise.to(self.hparams.device)

        orig_priors = self.prior[batch_index].gather(1, node_index)
        priors = orig_priors * (1 - exploration_fraction) + noise * exploration_fraction
        #self.prior.scatter_add_(1, node_index, priors)
        self.prior.scatter_(1, node_index, priors)

        # max_debug = 10
        # self.logger.info(f'exploration:\n'
        #                  f'orig_priors:\n{orig_priors[:max_debug]}\n'
        #                  f'priors:\n{priors[:max_debug]}\n'
        #                   f'saved_priors:\n{self.prior.gather(1, node_index)[:max_debug]}\n')

    def select_children(self, batch_index: torch.Tensor, node_index: torch.Tensor) -> torch.Tensor:
        children_index = self.children_index(batch_index, node_index)
        #self.logger.debug(f'select_children: node_index: {node_index}, children_index:\n{children_index}')

        ucb_scores = self.ucb_scores(batch_index, node_index, children_index)
        # self.logger.info(f'select_children: usb_scores:\n{ucb_scores}')
        max_scores, _ = ucb_scores.max(1)
        not_max_indexes = ucb_scores < max_scores.unsqueeze(1)

        rnd = torch.rand(*ucb_scores.shape, requires_grad=False, device=ucb_scores.device)
        ucb_scores += rnd

        ucb_scores[not_max_indexes] = 0
        max_indexes = ucb_scores.argmax(dim=1)

        # max_debug = 10
        # self.logger.info(f'select_children: batch_index: {batch_index[:max_debug]}, '
        #                  f'node_index: {node_index[:max_debug]}, '
        #                  f'children_index:\n{children_index[:max_debug]}\n'
        #                  f'ucb_scores:\n{ucb_scores[:max_debug]}\n'
        #                  f'max_scores: {max_scores[:max_debug]}, '
        #                  f'max_indexes: {max_indexes[:max_debug]}')

        children_index = children_index.gather(1, max_indexes.unsqueeze(1)).squeeze(1)
        return max_indexes, children_index

    def ucb_scores(self, batch_index: torch.Tensor, parent_index: torch.Tensor, children_index: torch.Tensor) -> torch.Tensor:
        parent_visit_count = self.visit_count[batch_index, parent_index]
        children_visit_count = self.visit_count[batch_index].gather(1, children_index)
        score = torch.log((parent_visit_count + self.hparams.c2 + 1) / self.hparams.c2) + self.hparams.c1
        score *= torch.sqrt(parent_visit_count)

        score = score.unsqueeze(1) * torch.ones_like(children_visit_count)
        score /= (children_visit_count + 1)

        children_prior = self.prior[batch_index].gather(1, children_index)
        prior_score = score * children_prior

        children_value = self.value(batch_index, children_index)
        if len(self.hparams.player_ids) != 1:
            children_value *= -1

        value_score = self.reward[batch_index].gather(1, children_index) + self.hparams.discount * children_value
        score = prior_score + value_score

        # max_debug = 10
        # self.logger.info(f'ucb_scores: '
        #                  f'children_index:\n{children_index[:max_debug]}\n'
        #                  f'children_visit_counts:\n{children_visit_count[:max_debug]}\n'
        #                  f'children_prior:\n{children_prior[:max_debug]}\n'
        #                  f'prior_score:\n{prior_score[:max_debug]}\n'
        #                  f'children_value:\n{children_value[:max_debug]}\n'
        #                  f'value_score:\n{value_score[:max_debug]}\n'
        #                  f'score:\n{score[:max_debug]}')
        return score

    def backpropagate(self, player_id: torch.Tensor, search_path: torch.Tensor, episode_len: torch.Tensor, value: torch.Tensor):
        for current_episode_len in torch.arange(episode_len.max(), -1, step=-1).to(self.hparams.device):
            batch_index = torch.arange(self.hparams.batch_size)[episode_len >= current_episode_len].to(self.hparams.device)

            node_index = search_path[batch_index, current_episode_len]
            node_multiplier = torch.where(self.player_id[batch_index, node_index] == player_id[batch_index], 1., -1.)

            # debug_max = 10
            # self.logger.info(f'backpropagate: '
            #                  f'self.player_id: {self.player_id[batch_index, node_index][:debug_max]}, '
            #                  f'player_id: {player_id[batch_index][:debug_max]}, '
            #                  f'node_index: {node_index.shape}, node_index:\n{node_index[:debug_max]}')

            # self.logger.debug(f'backpropagate: batch_index: {batch_index.shape}, batch_index: {batch_index[:debug_max]}, '
            #                  f'current_episode_len: {current_episode_len}/{episode_len.max()}/{episode_len[:debug_max]}, '
            #                  f'value: {value[batch_index].shape}/{value.shape}, '
            #                  f'value: {value[batch_index][:debug_max]}/{value[:debug_max]}')

            self.value_sum[batch_index, node_index] += value[batch_index] * node_multiplier
            # self.logger.info(f'backpropagate: node_multiplier: {node_multiplier}, value_sum:\n{self.value_sum[batch_index]}')
            self.visit_count[batch_index, node_index] += 1

            value[batch_index] = self.reward[batch_index, node_index] * node_multiplier + self.hparams.discount * value[batch_index]
            self.min_max_stats.update(value)

    def _store_states(self, search_path: torch.Tensor, episode_len: torch.Tensor, hidden_states: torch.Tensor):
        pass

    def _load_states(self, search_path: torch.Tensor, episode_len: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.zeros(self.hparams.batch_size, 16, *self.hparams.state_shape[1:]).to(self.hparams.device)
        return hidden_states

    def store_states(self, search_path: torch.Tensor, episode_len: torch.Tensor, hidden_states: torch.Tensor):
        search_path = search_path.detach().cpu().numpy().astype(np.uint8)
        episode_len = episode_len.detach().cpu().numpy()
        hidden_states = hidden_states.detach().clone()
        for path, elen, hidden_state in zip(search_path, episode_len, hidden_states):
            key = HashKey(path, elen)
            if key in self.hidden_states:
                continue

            self.hidden_states[key] = hidden_state

    def load_states(self, search_path: torch.Tensor, episode_len: torch.Tensor) -> torch.Tensor:
        hidden_states = []

        search_path = search_path.detach().cpu().numpy().astype(np.uint8)
        episode_len = episode_len.detach().cpu().numpy()
        for path, elen in zip(search_path, episode_len):
            key = HashKey(path, elen)
            hidden_states.append(self.hidden_states[key])

        hidden_states = torch.stack(hidden_states, 0).to(self.hparams.device)
        return hidden_states

    def run_one_simulation(self):
        search_path = torch.zeros(self.hparams.batch_size, self.hparams.max_episode_len+1).long().to(self.hparams.device)
        actions = torch.zeros(self.hparams.batch_size, self.hparams.max_episode_len).long().to(self.hparams.device)
        episode_len = torch.zeros(self.hparams.batch_size).long().to(self.hparams.device)
        player_id = torch.ones(self.hparams.batch_size).long().to(self.hparams.device) * self.hparams.player_ids[0]
        max_debug = 10

        batch_index = torch.arange(self.hparams.batch_size).to(self.hparams.device)
        node_index = torch.zeros(self.hparams.batch_size).long().to(self.hparams.device)

        search_path[batch_index, 0] = node_index

        for depth_index in range(0, self.hparams.max_episode_len):
            action_index, children_index = self.select_children(batch_index, node_index)

            # self.logger.info(f'depth: {depth_index}, '
            #                  f'player_id: {player_id[batch_index][:max_debug]}, '
            #                  f'node_index: {node_index.shape}, '
            #                  f'node_index: {node_index[:max_debug]}, '
            #                  f'action_index: {action_index.shape}, '
            #                  f'action_index: {action_index[:max_debug]}, '
            #                  f'children_index: {children_index.shape}, '
            #                  f'children_index: {children_index[:max_debug]}')

            search_path[batch_index, depth_index+1] = children_index.detach().clone()
            actions[batch_index, depth_index] = action_index.detach().clone()
            episode_len[batch_index] += 1


            expanded_index = self.expanded[batch_index, children_index] == True
            node_index = children_index[expanded_index]
            batch_index = batch_index[expanded_index]

            # self.logger.info(f'depth: {depth_index}, node_index: {node_index.shape}, node_index: {node_index[:max_debug]}, player_id: {player_id[batch_index][:max_debug]}')
            if len(node_index) == 0:
                break

            if depth_index >= 1:
                player_id[batch_index] = player_id_change(self.hparams, player_id[batch_index])


        try:
            batch_index = torch.arange(self.hparams.batch_size).to(self.hparams.device)

            last_episode = episode_len - 1
            last_episode = last_episode.unsqueeze(1)

            hidden_states = self.load_states(search_path, episode_len)

            last_actions = actions.gather(1, last_episode).squeeze(1)
            out = self.inference.recurrent(hidden_states, last_actions)
            self.store_states(search_path, episode_len+1, out.hidden_state)


            parent_index = search_path.gather(1, episode_len.unsqueeze(1)).squeeze(1)

            # max_debug = 10
            # self.logger.info(f'search_path: {search_path.shape}\n{search_path[:max_debug, :episode_len.max()+1]}')
            # self.logger.info(f'actions: {actions.shape}\n{actions[:max_debug, :episode_len.max()]}')
            # self.logger.info(f'episode_len: {episode_len[:max_debug]}')
            # self.logger.info(f'parent_index: {parent_index[:max_debug]}')

            self.expand(player_id, batch_index, parent_index, out.policy_logits)

            self.backpropagate(player_id, search_path, episode_len, out.value)
        except:
            self.logger.error(f'search_path: {search_path.shape}\n{search_path[:max_debug, :episode_len.max()+1]}')
            self.logger.error(f'actions: {actions.shape}\n{actions[:max_debug, :episode_len.max()]}')
            self.logger.error(f'episode_len: {episode_len[:max_debug]}')
            self.logger.error(f'player_id: {player_id[:max_debug]}')
            raise

        return search_path, episode_len

def main():
    player_id = 1
    hparams = Hparams()

    game_states = torch.zeros([hparams.batch_size, 1, 6, 7]).float().cpu()

    logger = setup_logger('mcts', logfile='test.log', log_to_stdout=True)
    inference = MCTSInference(hparams, logger)
    out = inference.initial(game_states)

    tree = Tree(hparams, inference, logger)
    tree.player_id[:, 0] = player_id

    batch_index = torch.arange(hparams.batch_size).long().to(hparams.device)
    node_index = torch.zeros(hparams.batch_size).long().to(hparams.device)

    episode_len = torch.ones(len(node_index)).long().to(hparams.device)
    search_path = torch.zeros(len(node_index), 1).long().to(hparams.device)
    tree.store_states(search_path, episode_len, out.hidden_state)

    player_id = torch.ones_like(node_index) * player_id
    tree.expand(player_id, batch_index, node_index, out.policy_logits)

    for _ in range(hparams.num_simulations):
        search_path, episode_len = tree.run_one()

if __name__ == '__main__':
    main()
