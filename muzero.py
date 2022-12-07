from typing import Dict, List

import os
import logging

from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True

import connectx_impl
from evaluate_score import EvaluationDataset
from hparams import Hparams as GenericHparams
from inference import Inference as GenericInference
from inference import NetworkOutput
from logger import setup_logger
import mcts
import networks

class Hparams(GenericHparams):
    checkpoints_dir = 'checkpoints_1'
    log_to_stdout = True
    max_num_actions = 1024*16

    td_steps = 42
    num_unroll_steps = 5

    init_lr = 1e-4
    min_lr = 1e-5

    training_games_window_size = 4
    num_training_steps = 1000

def roll_by_gather(mat,dim, shifts: torch.LongTensor):
    # assumes 2D array
    n_rows, n_cols = mat.shape

    if dim==0:
        #print(mat)
        arange1 = torch.arange(n_rows, device=shifts.device).view((n_rows, 1)).repeat((1, n_cols))
        #print(arange1)
        arange2 = (arange1 - shifts) % n_rows
        #print(arange2)
        return torch.gather(mat, 0, arange2)
    elif dim==1:
        arange1 = torch.arange(n_cols, device=shifts.device).view(( 1,n_cols)).repeat((n_rows,1))
        #print(arange1)
        arange2 = (arange1 - shifts) % n_cols
        #print(arange2)
        return torch.gather(mat, 1, arange2)

class Inference(GenericInference):
    def __init__(self, hparams: Hparams, logger: logging.Logger):
        self.logger = logger
        self.hparams = hparams

        net_hparams = networks.NetworkParams(observation_shape=hparams.state_shape)
        self.logger.info(f'inference: network_params:\n{net_hparams}')

        self.representation = networks.Representation(net_hparams).to(hparams.device)
        self.prediction = networks.Prediction(net_hparams).to(hparams.device)
        self.dynamic = networks.Dynamic(net_hparams).to(hparams.device)

        self.models = [self.representation, self.prediction, self.dynamic]

    def train(self, mode: bool):
        for model in self.models:
            model.train(mode)

    def zero_grad(self):
        for model in self.models:
            model.zero_grad()

    def create_states(self, player_id: torch.Tensor, game_states: torch.Tensor):
        batch_size = len(game_states)
        input_shape = list(game_states.shape[1:]) # remove batch size

        states = torch.zeros(1 + len(self.hparams.player_ids), batch_size, *input_shape[1:]).to(self.hparams.device)

        player_id_exp = player_id.unsqueeze(1)
        player_id_exp = player_id_exp.tile([1, np.prod(input_shape)]).view([batch_size] + input_shape[1:]).to(self.hparams.device)
        states[0, ...] = player_id_exp
        for player_index, local_player_id in enumerate(self.hparams.player_ids):
            index = game_states == local_player_id
            index = index.squeeze(1)
            states[1 + player_index, index] = 1

        states = torch.transpose(states, 1, 0)
        return states

    def initial(self, player_id: torch.Tensor, game_states: torch.Tensor) -> NetworkOutput:
        batch_size = game_states.shape[0]
        states = self.create_states(player_id, game_states)
        hidden_states = self.representation(states)
        policy_logits, values = self.prediction(hidden_states)
        rewards = torch.zeros(batch_size).float().to(self.hparams.device)
        #self.logger.info(f'inference: initial: game_states: {game_states.shape}: hidden_states: {hidden_states.shape}, policy_logits: {policy_logits.shape}, rewards: {rewards.shape}')
        return NetworkOutput(reward=rewards, hidden_state=hidden_states, policy_logits=policy_logits, value=values)

    def recurrent(self, hidden_states: torch.Tensor, actions: torch.Tensor) -> NetworkOutput:
        policy_logits, values = self.prediction(hidden_states)
        actions_exp = F.one_hot(actions, self.hparams.num_actions)
        actions_exp = actions_exp.unsqueeze(1)
        fill = hidden_states.shape[2]
        actions_exp = actions_exp.tile([1, fill, 1])
        actions_exp = actions_exp.unsqueeze(1)
        inputs = torch.cat([hidden_states, actions_exp], 1)
        new_hidden_states, rewards = self.dynamic(inputs)
        #self.logger.info(f'inference: recurrent: hidden_states: {hidden_states.shape}, policy_logits: {policy_logits.shape}, rewards: {rewards.shape}, values: {values.shape}')
        return NetworkOutput(reward=rewards, hidden_state=new_hidden_states, policy_logits=policy_logits, value=values)

class Sample:
    num_steps: int
    player_id: int
    initial_game_state: torch.Tensor

    child_visits: torch.Tensor
    values: torch.Tensor
    last_rewards: torch.Tensor

    actions: torch.Tensor

class SampleDataset:
    def __init__(self, hparams: Hparams):
        self.samples = []

        self.num_unroll_steps = hparams.num_unroll_steps

    def __len__(self):
        return len(self.samples)

    def append(self, sample_dict: Dict[str, torch.Tensor]):
        game_states = sample_dict['game_states']
        actions = sample_dict['actions']
        episode_len = sample_dict['actions']
        values = sample_dict['values']
        last_rewards = sample_dict['last_rewards']
        child_visits = sample_dict['child_visits']
        player_ids = sample_dict['player_ids']

        for sample_idx in range(len(game_states)):
            s = Sample()

            s.num_steps = episode_len[sample_idx]
            s.player_id = player_ids[sample_idx]
            s.initial_game_state = game_states[sample_idx]

            s.child_visits = child_visits[sample_idx]
            s.values = values[sample_idx]
            s.last_rewards = last_rewards[sample_idx]
            s.actions = actions[sample_idx]

            self.samples.append(s)

    def __getitem__(self, index):
        s = self.samples[index]

class GameStats:
    def __init__(self, hparams: Hparams, logger: logging.Logger):
        self.logger = logger
        self.hparams = hparams

        self.episode_len = torch.zeros(hparams.batch_size, dtype=torch.int64, device=hparams.device)
        self.rewards = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=hparams.dtype, device=hparams.device)
        self.root_values = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=hparams.dtype, device=hparams.device)
        self.child_visits = torch.zeros(hparams.batch_size, hparams.num_actions, hparams.max_episode_len, dtype=hparams.dtype, device=hparams.device)
        self.actions = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.int64, device=hparams.device)
        self.player_ids = torch.zeros(hparams.batch_size, hparams.max_episode_len, dtype=torch.int64, device=hparams.device)
        self.dones = torch.zeros(hparams.batch_size, dtype=torch.bool, device=hparams.device)
        self.game_states = torch.zeros([hparams.batch_size, hparams.max_episode_len] + hparams.state_shape, dtype=hparams.dtype, device=hparams.device)

        self.stored_tensors = {
            'rewards': self.rewards,
            'root_values': self.root_values,
            'child_visits': self.child_visits,
            'actions': self.actions,
            'player_ids': self.player_ids,
            'dones': self.dones,
            'game_states': self.game_states,
        }

    def __len__(self):
        return self.episode_len.sum().item()

    def append(self, index: torch.Tensor, tensors_dict: Dict[str, torch.Tensor]):
        episode_len = self.episode_len[index]
        for key, value in tensors_dict.items():
            if not key in self.stored_tensors:
                msg = f'invalid key: {key}, tensor shape: {value.shape}, available keys: {list(self.stored_tensors.keys())}'
                self.logger.critical(msg)
                raise ValueError(msg)

            tensor = self.stored_tensors[key]
            if key == 'child_visits':
                tensor[index, :, episode_len] = value.detach().clone()
            elif key == 'dones':
                tensor[index] = value.detach().clone()
            else:
                tensor[index, episode_len] = value.detach().clone()

        self.episode_len[index] += 1

    def make_target(self, start_index: torch.Tensor):
        target_values = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=self.hparams.device)
        target_last_rewards = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=self.hparams.device)
        target_child_visits = torch.zeros(len(start_index), self.hparams.num_actions, self.hparams.num_unroll_steps+1, dtype=self.hparams.dtype, device=self.hparams.device)
        taken_actions = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=torch.int64, device=self.hparams.device)
        player_ids = torch.zeros(len(start_index), self.hparams.num_unroll_steps+1, dtype=torch.int64, device=self.hparams.device)

        sample_len = torch.zeros(len(start_index), dtype=torch.int64, device=self.hparams.device)

        batch_index = torch.arange(len(start_index), dtype=torch.int64, device=self.root_values.device)
        game_states = self.game_states[batch_index, start_index]

        for unroll_index in range(0, self.hparams.num_unroll_steps+1):
            current_index = start_index + unroll_index
            bootstrap_index = current_index + self.hparams.td_steps

            bootstrap_update_index = bootstrap_index < self.episode_len
            valid_batch_index = batch_index[bootstrap_update_index]

            if False:
                self.logger.info(f'{unroll_index}: '
                             f'start_index: {start_index.cpu().numpy()}, '
                             f'current_index: {current_index.cpu().numpy()}, '
                             f'bootstrap_index: {bootstrap_index.cpu().numpy()}, '
                             f'bootstrap_update_index: {bootstrap_update_index.cpu().numpy()}/{bootstrap_update_index.shape}, '
                             f'valid_batch_index: {valid_batch_index.cpu().numpy()}'
                             )
            values = torch.zeros(len(self.root_values), dtype=self.root_values.dtype, device=self.root_values.device)
            if bootstrap_update_index.sum() > 0:
                self.logger.info(f'bootstrap_update_index: {bootstrap_update_index}')
                values[bootstrap_update_index] = self.root_values[valid_batch_index, bootstrap_index] * self.hparams.discount ** self.hparams.td_steps

            discount_mult = torch.logspace(0, self.hparams.td_steps, self.rewards.shape[1], base=self.hparams.discount).to(self.hparams.device)
            discount_mult = discount_mult.unsqueeze(0)
            discount_mult = discount_mult.tile([len(values), 1])

            last_index = torch.minimum(bootstrap_index, self.episode_len)
            all_rewards_index = torch.arange(self.rewards.shape[1], device=self.hparams.device).long().unsqueeze(0).tile([len(start_index), 1])

            discount_mult = roll_by_gather(discount_mult, 1, start_index.unsqueeze(1))
            discount_mult = torch.where(all_rewards_index < current_index.unsqueeze(1), 0, discount_mult)
            discount_mult = torch.where(all_rewards_index >= last_index.unsqueeze(1), 0, discount_mult)

            masked_rewards = torch.where(all_rewards_index < current_index.unsqueeze(1), 0, self.rewards)
            masked_rewards = torch.where(all_rewards_index >= last_index.unsqueeze(1), 0, masked_rewards)

            discounted_rewards = self.rewards * discount_mult
            discounted_rewards = discounted_rewards.sum(1)
            values += discounted_rewards

            valid_index = current_index < self.episode_len

            current_valid_index = current_index[valid_index]
            target_values[valid_index, unroll_index] = values[valid_index]
            target_child_visits[valid_index, :, unroll_index] = self.child_visits[valid_index, :, current_valid_index]

            if unroll_index > 0:
                target_last_rewards[valid_index, unroll_index] = self.rewards[valid_index, current_valid_index-1]
            taken_actions[valid_index, unroll_index] = self.actions[valid_index, current_valid_index]
            player_ids[valid_index, unroll_index] = self.player_ids[valid_index, current_valid_index]
            sample_len[valid_index] += 1

        return {
            'values': target_values,
            'last_rewards': target_last_rewards,
            'child_visits': target_child_visits,
            'game_states': game_states,
            'actions': taken_actions,
            'sample_len': sample_len,
            'player_ids': player_ids,
        }

class Train:
    def __init__(self, hparams: Hparams, inference: Inference, logger: logging.Logger):
        self.hparams = hparams
        self.logger = logger
        self.inference = inference

        self.num_train_steps = 0
        self.game_stats = GameStats(hparams, self.logger)

    def run_simulations(self, initial_player_id: torch.Tensor, initial_game_states: torch.Tensor):
        local_hparams = deepcopy(self.hparams)
        local_hparams.batch_size = len(initial_game_states)
        tree = mcts.Tree(local_hparams, self.inference, self.logger)
        tree.player_id[:, 0] = initial_player_id

        batch_size = len(initial_game_states)
        batch_index = torch.arange(batch_size).long()
        node_index = torch.zeros(batch_size).long().to(self.hparams.device)

        out = self.inference.initial(initial_player_id, initial_game_states)

        episode_len = torch.ones(len(node_index)).long().to(self.hparams.device)
        search_path = torch.zeros(len(node_index), 1).long().to(self.hparams.device)

        tree.store_states(search_path, episode_len, out.hidden_state)

        tree.expand(initial_player_id, batch_index, node_index, out.policy_logits)

        start_simulation_time = perf_counter()

        for _ in range(self.hparams.num_simulations):
            search_path, episode_len = tree.run_one()
        simulation_time = perf_counter() - start_simulation_time
        one_sim_ms = int(simulation_time / self.hparams.num_simulations * 1000)
        if False:
            self.logger.info(f'train: {self.num_train_steps:2d}: '
                         f'batch_size: {batch_size}, '
                         f'num_simulations: {self.hparams.num_simulations}, '
                         f'time: {simulation_time:.3f} sec, '
                         f'one_sim: {one_sim_ms:3d} ms')

        actions = self.select_actions(batch_size, tree)

        children_index = tree.children_index(batch_index, node_index)
        children_visit_counts = tree.visit_count[batch_index].gather(1, children_index)
        children_sum_visits = children_visit_counts.sum(1)
        children_visits = children_visit_counts / children_sum_visits.unsqueeze(1)
        root_values = tree.value(batch_index, node_index.unsqueeze(1)).squeeze(1)

        return actions, children_visits, root_values

    def update_train_steps(self):
        self.num_train_steps += 1

    def select_actions(self, batch_size: int, tree: mcts.Tree):
        batch_index = torch.arange(batch_size, device=self.hparams.device).long()
        node_index = torch.zeros(batch_size, device=self.hparams.device).long()
        children_index = tree.children_index(batch_index, node_index)
        visit_counts = tree.visit_count[batch_index].gather(1, children_index)

        if self.num_train_steps >= 30:
            actions = torch.argmax(visit_counts, 1)
            return actions

        temperature = 1.0 # play according to softmax distribution

        dist = torch.pow(visit_counts.float(), 1 / temperature)
        actions = torch.multinomial(dist, 1)
        return actions.squeeze(1)

def run_single_game(hparams: Hparams, train: Train, num_steps: int):
    game_hparams = connectx_impl.Hparams()
    game_states = torch.zeros(hparams.batch_size, *hparams.state_shape).float().to(hparams.device)
    player_ids = torch.ones(hparams.batch_size, device=hparams.device).long() * hparams.player_ids[0]

    active_games_index = torch.arange(hparams.batch_size).long().to(hparams.device)

    while True:
        active_game_states = game_states[active_games_index]
        active_player_ids = player_ids[active_games_index]
        actions, children_visits, root_values = train.run_simulations(active_player_ids, active_game_states)
        new_game_states, rewards, dones = connectx_impl.step_games(game_hparams, active_game_states, active_player_ids, actions)
        game_states[active_games_index] = new_game_states

        train.game_stats.append(active_games_index, {
            'child_visits': children_visits,
            'root_values': root_values,
            'game_states': active_game_states,
            'rewards': rewards,
            'dones': dones,
            'player_ids': active_player_ids,
        })
        train.update_train_steps()

        if dones.sum() == len(dones):
            break

        player_ids = mcts.player_id_change(hparams, player_ids)
        active_games_index = active_games_index[dones != True]

        num_steps -= 1
        if num_steps == 0:
            break

    return train.game_stats

def scale_gradient(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return tensor * scale + tensor.detach() * (1 - scale)

class Trainer:
    def __init__(self, hparams: Hparams, logger: logging.Logger, eval_ds: EvaluationDataset):
        self.hparams = hparams
        self.logger = logger
        self.eval_ds = eval_ds

        self.max_best_score = None

        tensorboard_log_dir = os.path.join(hparams.checkpoints_dir, 'tensorboard_logs')
        first_run = True
        if os.path.exists(tensorboard_log_dir) and len(os.listdir(tensorboard_log_dir)) > 0:
            first_run = False

        self.summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.global_step = 0

        self.inference = Inference(hparams, logger)
        self.representation_opt = torch.optim.Adam(self.inference.representation.parameters(), lr=hparams.init_lr)
        self.prediction_opt = torch.optim.Adam(self.inference.prediction.parameters(), lr=hparams.init_lr)
        self.dynamic_opt = torch.optim.Adam(self.inference.dynamic.parameters(), lr=hparams.init_lr)
        self.optimizers = [self.representation_opt, self.prediction_opt, self.dynamic_opt]

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.scalar_loss = nn.MSELoss(reduction='none')

        self.all_games: List[GameStats] = []

    def collect_games_episode(self):
        start_time = perf_counter()

        self.inference.train(False)
        train = Train(self.hparams, self.inference, self.logger)
        with torch.no_grad():
            game_stats = run_single_game(self.hparams, train, num_steps=-1)

        self.all_games.append(game_stats)
        if len(self.all_games) > self.hparams.training_games_window_size:
            self.all_games.pop(0)

        collection_time = perf_counter() - start_time
        self.summary_writer.add_scalar('collect/time', collection_time, self.global_step)
        self.summary_writer.add_histogram('collect/child_visits', game_stats.child_visits[:, :, 0], self.global_step)
        self.summary_writer.add_histogram('collect/actions', game_stats.actions[:, 0], self.global_step)
        self.summary_writer.add_scalar('collect/root_values', game_stats.root_values[:, 0].mean(), self.global_step)
        self.summary_writer.add_scalar('collect/rewards', game_stats.rewards.mean(), self.global_step)
        self.summary_writer.add_scalar('collect/train_steps', train.num_train_steps, self.global_step)


    def training_forward_one_game(self, epoch: int, game_stat: GameStats):
        start_pos = []
        for i, max_episode_len in enumerate(game_stat.episode_len):
            max_start_pos = max_episode_len - self.hparams.num_unroll_steps
            if max_start_pos < 0:
                pos = 0
            else:
                pos = torch.randint(low=0, high=max_start_pos.item(), size=(1,))

            start_pos.append(pos)

        start_pos = torch.cat(start_pos).to(self.hparams.device)

        self.summary_writer.add_scalars('train/episode_len', {
            'mean': game_stat.episode_len.float().mean(),
            'min': game_stat.episode_len.min(),
            'max': game_stat.episode_len.max(),
            'start_mean': start_pos.float().mean(),
            'start_min': start_pos.min(),
            'start_max': start_pos.max(),
        }, self.global_step)

        sample_dict = game_stat.make_target(start_pos)

        game_states = sample_dict['game_states']
        actions = sample_dict['actions']
        sample_len = sample_dict['sample_len']
        values = sample_dict['values']
        last_rewards = sample_dict['last_rewards']
        child_visits = sample_dict['child_visits']
        player_ids = sample_dict['player_ids']


        train_examples = len(game_states)

        #logger.info(f'{sample_idx:2d}: game_states: {game_states.shape}, player_ids: {player_ids.shape}, actions: {actions.shape}, child_visits: {child_visits.shape}')
        out = self.inference.initial(player_ids[:, 0], game_states)
        policy_loss = self.ce_loss(out.policy_logits, child_visits[:, :, 0])
        value_loss = self.scalar_loss(out.value, values[:, 0])
        reward_loss = self.scalar_loss(out.reward, out.reward)

        iteration_loss = policy_loss + value_loss + reward_loss
        iteration_loss = scale_gradient(iteration_loss, 1/sample_len)
        total_loss = torch.mean(iteration_loss)

        self.summary_writer.add_scalars('train/initial_losses', {
                'policy': policy_loss.mean(),
                'value': value_loss.mean(),
                'total': total_loss,
        }, self.global_step)

        #policy_softmax = F.log_softmax(out.policy_logits, 1)

        #logger.info(f'child_visits:\n{child_visits[:5, :, 0]}, policy_logits:\n{out.policy_logits[:5]}\npolicy_softmax:\n{policy_softmax[:5]}\npolicy_loss:\n{policy_loss[:5]}')
        #logger.info(f'values:\n{values[:5, 0]}\nout.value:\n{out.value[:5]}\nvalue_loss: {value_loss[:5]}')

        if False:
            self.logger.info(f'{epoch:3d}: '
            f'policy_loss: {policy_loss.mean().item():.4f}, '
            f'value_loss: {value_loss.mean().item():.4f}, '
            f'reward_loss: {reward_loss.mean().item():.4f}, '
            f'total_loss: {total_loss.item():.4f}')


        for step_idx in range(1, sample_len.max()):
            batch_index = step_idx < sample_len
            sample_len = sample_len[batch_index]
            actions = actions[batch_index]
            values = values[batch_index]
            child_visits = child_visits[batch_index]
            last_rewards = last_rewards[batch_index]

            scale = torch.ones(len(last_rewards), device=self.hparams.device)*0.5
            scale = scale.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            hidden_states = scale_gradient(out.hidden_state[batch_index], scale)

            out = self.inference.recurrent(hidden_states, actions[:, step_idx-1])

            policy_loss = self.ce_loss(out.policy_logits, child_visits[:, :, step_idx])
            value_loss = self.scalar_loss(out.value, values[:, step_idx])
            reward_loss = self.scalar_loss(out.reward, last_rewards[:, step_idx])

            iteration_loss = policy_loss + value_loss + reward_loss
            iteration_loss = scale_gradient(iteration_loss, 1/sample_len)

            total_loss += torch.mean(iteration_loss)

        return total_loss, train_examples

    def training_step(self, epoch: int):
        self.inference.train(True)
        self.inference.zero_grad()
        for opt in self.optimizers:
            opt.zero_grad()

        total_loss = 0
        total_train_examples = 0
        for game_stat in self.all_games:
            loss, train_examples = self.training_forward_one_game(epoch, game_stat)
            total_loss += loss
            total_train_examples += train_examples

        total_loss.backward()

        for opt in self.optimizers:
            opt.step()

        self.summary_writer.add_scalar('train/total_loss', total_loss / len(self.all_games), self.global_step)
        self.summary_writer.add_scalar('train/samples', total_train_examples / len(self.all_games), self.global_step)
        return total_loss.item(), total_train_examples

    def run_training(self, epoch: int):
        self.summary_writer.add_scalar('train/epoch', epoch, self.global_step)

        all_losses = []
        all_train_examples = 0
        for train_idx in range(self.hparams.num_training_steps):
            total_loss, total_train_examples = self.training_step(epoch)

            all_losses.append(total_loss)
            all_train_examples += total_train_examples

            if train_idx % 10 == 0:
                best_score, good_score = self.run_evaluation(save_if_best=True)

            self.global_step += 1

        best_score, good_score = self.run_evaluation(save_if_best=True)

    def run_evaluation(self, save_if_best: bool):
        hparams = deepcopy(self.hparams)
        hparams.batch_size = len(self.eval_ds.game_states)
        train = Train(hparams, self.inference, self.logger)
        with torch.no_grad():
            game_states = torch.zeros(hparams.batch_size, *hparams.state_shape).float().to(hparams.device)
            active_games_index = torch.arange(hparams.batch_size).long().to(hparams.device)

            active_game_states = game_states[active_games_index]
            active_player_ids = self.eval_ds.game_player_ids[active_games_index]
            pred_actions, children_visits, root_values = train.run_simulations(active_player_ids, active_game_states)

        best_score, good_score = self.eval_ds.evaluate(pred_actions, debug=False)

        self.summary_writer.add_scalars('eval/ref_moves_score', {
            'good': good_score,
            'best': best_score,
        }, self.global_step)

        if save_if_best and (self.max_best_score is None or best_score > self.max_best_score):
            self.max_best_score = best_score
            checkpoint_path = os.path.join(self.hparams.checkpoints_dir, f'muzero_{best_score:.1f}.ckpt')
            self.save(checkpoint_path)
            self.logger.info(f'stored checkpoint: best_score: {best_score:.1f}, checkpoint: {checkpoint_path}')

        return best_score, good_score

    def save(self, checkpoint_path):
        torch.save({
            'representation_state_dict': self.inference.representation.state_dict(),
            'representation_optimizer_state_dict': self.representation_opt.state_dict(),
            'prediction_state_dict': self.inference.prediction.state_dict(),
            'prediction_optimizer_state_dict': self.prediction_opt.state_dict(),
            'dynamic_state_dict': self.inference.dynamic.state_dict(),
            'dynamic_optimizer_state_dict': self.dynamic_opt.state_dict(),
            'global_step': self.global_step,
            'max_best_score': self.max_best_score,
            }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        self.inference.representation.load_state_dict(checkpoint['representation_state_dict'])
        self.representation_opt.load_state_dict(checkpoint['representation_optimizer_state_dict'])
        self.inference.prediction.load_state_dict(checkpoint['prediction_state_dict'])
        self.prediction_opt.load_state_dict(checkpoint['prediction_optimizer_state_dict'])
        self.inference.dynamic.load_state_dict(checkpoint['dynamic_state_dict'])
        self.dynamic_opt.load_state_dict(checkpoint['dynamic_optimizer_state_dict'])

        self.global_step = checkpoint['global_step']
        self.max_best_score = checkpoint['max_best_score']

        self.logger.info(f'loaded checkpoint {checkpoint_path}')

def main():
    hparams = Hparams()

    epoch = 0
    hparams.batch_size = 512
    hparams.num_simulations = 64
    hparams.training_games_window_size = 4
    hparams.num_training_steps = 20
    hparams.device = torch.device('cuda:0')

    logfile = os.path.join(hparams.checkpoints_dir, 'muzero.log')
    os.makedirs(hparams.checkpoints_dir, exist_ok=True)
    logger = setup_logger('muzero', logfile, hparams.log_to_stdout)

    refmoves_fn = 'refmoves1k_kaggle'
    eval_ds = EvaluationDataset(refmoves_fn, hparams, logger)

    trainer = Trainer(hparams, logger, eval_ds)

    for epoch in range(10000):
        trainer.collect_games_episode()
        trainer.run_training(epoch)

if __name__ == '__main__':
    main()
