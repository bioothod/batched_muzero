from typing import NamedTuple, List, Dict, Optional

import logging

import torch

from logger import setup_logger

class Hparams:
    batch_size: int = 3
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

class Inference:
    def __init__(self, hparams: Hparams, logger: logging.Logger):
        self.num_actions = hparams.num_actions
        self.logger = logger
        self.default_reward = hparams.default_reward

    def initial(self, game_states: torch.Tensor) -> NetworkOutput:
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
    def __init__(self, search_path: torch.Tensor, episode_len: torch.Tensor):
        key = search_path[:episode_len]
        self.key = key.detach().clone().byte().numpy()
        # this is fast
        self.key = self.key.tobytes()
        self.hash = hash(self.key)

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, other: 'HashKey') -> bool:
        return self.hash == other.hash and self.key == other.key

class Tree:
    visit_count: torch.Tensor
    reward: torch.Tensor
    value_sum: torch.Tensor
    prior: torch.Tensor
    player_id: torch.Tensor
    expanded: torch.Tensor
    episode_len: torch.Tensor
    hidden_states: Dict[HashKey, torch.Tensor]
    hparams: Hparams
    inference: Inference
    logger: logging.Logger

    def __init__(self, player_id: int, hparams: Hparams, inference: Inference, logger: logging.Logger):
        self.hparams = hparams
        self.inference = inference
        self.logger = logger

        self.min_max_stats = MinMaxStats()

        max_tree_depth = hparams.max_episode_len
        tree_shape = [hparams.batch_size, max_tree_depth, hparams.num_actions]

        self.visit_count = torch.zeros(tree_shape).long()
        self.value_sum = torch.zeros(tree_shape).float()
        self.prior = torch.zeros(tree_shape).float()
        self.reward = torch.zeros(tree_shape).float()
        self.expanded = torch.zeros(tree_shape).bool()
        self.player_id = torch.zeros([hparams.batch_size, max_tree_depth]).long()
        self.hidden_states = {}

        self.player_id[:, 0] = player_id

    def expand(self, batch_index: torch.Tensor, step_index: torch.Tensor, action_index: torch.Tensor, player_id: torch.Tensor, policy_logits: torch.Tensor):

        self.logger.info(f'expand: batch_index: {batch_index[:10]}, '
                         f'action_index: {action_index[:10]}, '
                         f'prior: {self.prior.shape}, '
                         f'step_index: {step_index.shape}, '
                         f'step_index: {step_index[:10]}, '
                         f'indexed_prior: {self.prior[batch_index, :].shape}, '
                         f'policy_logits: {policy_logits.shape}')

        self.expanded[batch_index, step_index, action_index] = True
        stack_index = torch.stack([batch_index, step_index, action_index], 1)
        self.logger.info(f'expand: store:\n{stack_index[:10]}')
        self.prior[batch_index, step_index, :] = torch.softmax(policy_logits, 1)

        self.player_id[batch_index, step_index] = player_id

    def value(self, batch_index: torch.Tensor, step_index: torch.Tensor) -> torch.Tensor:
        return torch.where(self.visit_count[batch_index, step_index] == 0, 0, self.value_sum[batch_index, step_index] / self.visit_count[batch_index, step_index])

    def add_exploration_noise(self, index: torch.Tensor, exploration_fraction: float):
        concentration = torch.ones([self.hparams.batch_size, self.hparams.num_actions]).float() * self.hparams.dirichlet_alpha
        dist = torch.distributions.Dirichlet(concentration)
        noise = dist.sample()
        self.prior[:, index, :] = self.prior[:, index, :] * (1 - exploration_fraction) + noise * exploration_fraction

    def select_children(self, batch_index: torch.Tensor, step_index: torch.Tensor) -> torch.Tensor:
        ucb_scores = self.ucb_scores(batch_index, step_index)
        self.logger.info(f'usb_scores:\n{ucb_scores}')
        max_scores, max_indexes = ucb_scores.max(1)
        debug_max = 10
        self.logger.info(f'select_children: batch_index: {batch_index[:debug_max]}, '
                         f'step_index: {step_index[:debug_max]}, '
                         f'max_scores: {max_scores[:debug_max]}, '
                         f'max_indexes: {max_indexes[:debug_max]}')

        return max_indexes

    def ucb_scores(self, batch_index: torch.Tensor, step_index: torch.Tensor) -> torch.Tensor:
        step_parent_index = step_index - 1

        parent_visit_count = self.visit_count[batch_index, step_parent_index]
        children_visit_count = self.visit_count[batch_index, step_index]
        score = torch.log((parent_visit_count + self.hparams.c2 + 1) / self.hparams.c2) + self.hparams.c1
        score *= torch.sqrt(parent_visit_count) / (children_visit_count + 1)

        children_prior = self.prior[batch_index, step_index]
        prior_score = score * children_prior

        children_value = self.value(batch_index, step_index)
        if len(self.hparams.player_ids) != 1:
            children_value *= -1

        value_score = self.reward[batch_index, step_index] + self.hparams.discount * children_value
        score = prior_score + value_score
        self.logger.info(f'prior_score: {prior_score.shape}, value_score: {value_score.shape}')
        return score

    def backpropagate(self, player_id: torch.Tensor, search_path: torch.Tensor, episode_len: torch.Tensor, value: torch.Tensor):
        self.logger.info(f'search_path:\n{search_path}')
        for current_episode_len in torch.arange(episode_len.max()-1, 0, step=-1):
            batch_index = torch.arange(self.hparams.batch_size)[episode_len >= current_episode_len]

            step_index = current_episode_len-1
            node_index = search_path[batch_index, step_index+1]
            node_multiplier = torch.where(self.player_id[batch_index, step_index] == player_id[batch_index], 1., -1.)

            debug_max = 10
            self.logger.info(f'backpropagate: step_index: {step_index}, '
                             f'self.player_id: {self.player_id[batch_index, step_index]}, '
                             f'player_id: {player_id[batch_index]}')

            self.logger.info(f'backpropagate: batch_index: {batch_index.shape}, '
                             f'current_episode_len: {current_episode_len}/{episode_len.max()}, '
                             f'node_index: {node_index.shape}, '
                             f'value: {value[batch_index].shape}/{value.shape}')

            self.logger.info(f'backpropagate: batch_index: {batch_index[:debug_max]}, '
                             f'current_episode_len: {current_episode_len}/{episode_len[:debug_max]}, '
                             f'node_index: {node_index[:debug_max]}, '
                             f'value: {value[batch_index][:debug_max]}/{value[:debug_max]}')

            self.value_sum[batch_index, step_index, node_index] += value[batch_index] * node_multiplier
            self.logger.info(f'backpropagate: node_multiplier: {node_multiplier}, value_sum:\n{self.value_sum[batch_index]}')
            self.visit_count[batch_index, step_index, node_index] += 1

            value[batch_index] = self.reward[batch_index, step_index, node_index] * node_multiplier + self.hparams.discount * value[batch_index]
            self.min_max_stats.update(value)

    def store_states(self, search_path: torch.Tensor, episode_len: torch.Tensor, hidden_states: torch.Tensor):
        self.logger.info(f'store_states: start: search_path: {search_path.shape}, episode_len10: {episode_len[:10]}, hidden_states: {hidden_states.shape}, saved states: {len(self.hidden_states)}')

        for path, elen, state in zip(search_path, episode_len, hidden_states):
            key = HashKey(path, elen)
            if key in self.hidden_states:
                continue

            self.logger.info(f'store_states: hash: key: {key.key}, hash: {key.hash}')
            self.hidden_states[key] = state.detach().clone()

        self.logger.info(f'store_states: finish: search_path: {search_path.shape}, episode_len10: {episode_len[:10]}, hidden_states: {hidden_states.shape}, saved states: {len(self.hidden_states)}')

    def load_states(self, search_path: torch.Tensor, episode_len: torch.Tensor) -> torch.Tensor:
        hidden_states = []

        for path, elen in zip(search_path, episode_len):
            key = HashKey(path, elen)
            self.logger.info(f'load_states: hash: key: {key.key}, hash: {key.hash}')
            hidden_states.append(self.hidden_states[key])

        hidden_states = torch.stack(hidden_states, 0)
        self.logger.info(f'load_states: search_path: {search_path.shape}, episode_len10: {episode_len[:10]}, hidden_states: {hidden_states.shape}, saved states: {len(self.hidden_states)}')
        return hidden_states

    def player_id_change(self, player_id: torch.Tensor):
        next_id = player_id + 1
        if next_id[0] > self.hparams.player_ids[-1]:
            next_id = self.hparams.player_ids[0]
        return next_id

    def run_one(self):
        search_path = torch.zeros(self.hparams.batch_size, self.hparams.max_episode_len+1).long()
        episode_len = torch.ones(self.hparams.batch_size).long()
        player_id = self.player_id[:, 0].detach().clone()
        max_debug = 10

        if self.hparams.add_exploration_noise:
            expanded_index = torch.zeros(self.hparams.batch_size).long()
            self.add_exploration_noise(expanded_index, self.hparams.exploration_fraction)

        batch_index = torch.arange(self.hparams.batch_size)

        for depth_index in range(0, self.hparams.max_episode_len):
            step_index = torch.ones_like(batch_index) * depth_index

            action_index = self.select_children(batch_index, step_index)
            search_path[batch_index, step_index+1] = action_index
            episode_len[batch_index] += 1

            self.logger.info(f'depth: {depth_index}, '
                             f'player_id: {player_id[batch_index][:max_debug]}, '
                             f'action_index: {action_index.shape}, '
                             f'action_index: {action_index[:max_debug]}')

            expanded = self.expanded[batch_index, step_index, action_index]
            index = batch_index[expanded == True]

            self.logger.info(f'depth: {depth_index}, index: {index.shape}, expanded: {expanded[:max_debug]}, player_id: {player_id[index][:max_debug]}')
            if len(index) == 0:
                break

            batch_index = index

            if depth_index >= 1:
                player_id[batch_index] = self.player_id_change(player_id[batch_index])


        self.logger.info(f'search_path:\n{search_path[:max_debug]}')
        self.logger.info(f'episode_len: {episode_len[:max_debug]}')
        self.logger.info(f'player_id: {player_id[:max_debug]}')

        hidden_states = self.load_states(search_path, episode_len-1)
        batch_index = torch.arange(self.hparams.batch_size)
        action_index = search_path[batch_index, episode_len-1]
        out = self.inference.recurrent(hidden_states, action_index)
        self.store_states(search_path, episode_len, out.hidden_state)

        step_index = episode_len-2
        self.expand(batch_index, step_index, action_index, player_id, out.policy_logits)

        self.backpropagate(player_id, search_path, episode_len, out.value)

        return search_path, episode_len

def main():
    player_id = 1
    hparams = Hparams()

    game_states = torch.zeros([hparams.batch_size, 1, 6, 7]).float().cpu()

    logger = setup_logger('mcts', logfile='test.log', log_to_stdout=True)
    inference = Inference(hparams, logger)

    tree = Tree(player_id, hparams, inference, logger)
    out = inference.initial(game_states)

    step_index = torch.zeros(hparams.batch_size).long()

    search_path = torch.zeros(len(step_index), hparams.max_episode_len).long()
    episode_len = torch.ones(len(step_index)).long()
    tree.store_states(search_path, episode_len, out.hidden_state)
    tree.prior[:, 0, :] = torch.softmax(out.policy_logits, 1)
    tree.player_id[:, 0] = player_id

    for _ in range(hparams.num_simulations):
        search_path, episode_len = tree.run_one()

###    for _ in range(42):

if __name__ == '__main__':
    main()
