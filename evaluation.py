from typing import Dict, List, Optional

import argparse
import logging
import os
import random

import numpy as np
import torch

import checkpoints
import connectx_dnn_agent
from hparams import GenericHparams as Hparams
from logger import setup_logger
import module_loader
import networks
import simulation

def action_selection_fn(children_visit_counts: torch.Tensor, episode_len: torch.Tensor):
    actions_argmax = torch.argmax(children_visit_counts, 1)
    #return actions_argmax

    action_selection_temperature = 1
    dist = torch.pow(children_visit_counts.float(), 1. / action_selection_temperature)
    actions_dist = torch.multinomial(dist, 1).squeeze(1)
    return actions_dist

class Evaluation:
    @torch.no_grad()
    def __init__(self, game_ctl: module_loader.GameModule, checkpoint_path: Optional[str], logger: logging.Logger, connectx_dnn_model_dir: str, random_agent: bool):
        self.game_ctl = game_ctl
        self.hparams = game_ctl.hparams
        self.logger = logger

        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.inference = networks.Inference(self.game_ctl, logger)

        if checkpoint_path is None:
            checkpoint_path = checkpoints.find_checkpoint(self.hparams.checkpoints_dir, self.hparams.load_latest)

        checkpoint = torch.load(checkpoint_path, map_location=self.hparams.device)
        self.inference.load_state_dict(checkpoint['state_dict'])

        self.global_step = int(checkpoint['global_step'])
        self.max_best_score = float(checkpoint['max_best_score'])
        self.max_good_score = float(checkpoint['max_good_score'])

        self.logger.info(f'loaded checkpoint {checkpoint_path}, global_step: {self.global_step}, max_good_score: {self.max_good_score:.2f}, max_best_score: {self.max_best_score:.2f}')

        self.random_agent = random_agent
        if not self.random_agent:
            self.connectx_dnn = connectx_dnn_agent.CombinedModel(connectx_dnn_model_dir).to(self.hparams.device)

    def one_game(self, player_id: int):
        sim = simulation.Simulation(self.game_ctl, self.inference, action_selection_fn, self.logger, None, '', 0)

        game_states = torch.zeros(self.hparams.batch_size, *self.hparams.state_shape, dtype=torch.float32, device=self.hparams.device)
        player_ids = torch.ones(self.hparams.batch_size, device=self.hparams.device, dtype=torch.int64) * player_id
        if player_id == self.hparams.player_ids[0]:
            cx_player_id = self.hparams.player_ids[1]
        else:
            cx_player_id = self.hparams.player_ids[0]

        active_games_index = torch.arange(self.hparams.batch_size).long().to(self.hparams.device)
        game_state_stack = networks.GameState(self.hparams.batch_size, self.hparams, self.game_ctl.network_hparams)

        final_rewards = torch.zeros(self.hparams.batch_size, device=self.hparams.device, dtype=torch.float32)
        episode_len = torch.zeros(self.hparams.batch_size, device=self.hparams.device, dtype=torch.int64)

        debug = True
        while True:
            active_player_ids = player_ids[active_games_index].detach().clone()
            active_game_states = game_states[active_games_index]
            invalid_actions_mask = self.game_ctl.invalid_actions_mask(self.game_ctl.game_hparams, active_game_states)

            game_state_stack.push_game(player_ids, game_states)
            game_state_stack_converted = game_state_stack.create_state()

            actions, children_visits, root_values, out_initial = sim.run_simulations(active_player_ids, game_state_stack_converted[active_games_index], invalid_actions_mask, debug)
            debug = False

            new_game_states, rewards, dones = self.game_ctl.step_games(self.game_ctl.game_hparams, active_game_states, active_player_ids, actions)
            game_states[active_games_index] = new_game_states.detach().clone()
            final_rewards[active_games_index] = rewards.detach().clone()
            episode_len[active_games_index] += 1

            if dones.sum() == len(dones):
                break

            active_games_index = active_games_index[dones != True]
            active_game_states = game_states[active_games_index]

            if self.random_agent:
                cx_actions = np.random.randint(0, self.hparams.num_actions, size=len(active_game_states))
                cx_actions = torch.from_numpy(cx_actions).to(self.hparams.device)
            else:
                cx_probs = self.connectx_dnn.forward(cx_player_id, active_game_states.unsqueeze(1))
                cx_actions = torch.argmax(cx_probs, 1)

            cx_player_ids = torch.ones(len(cx_actions), device=self.hparams.device) * cx_player_id
            new_game_states, rewards, dones = self.game_ctl.step_games(self.game_ctl.game_hparams, active_game_states, cx_player_ids, cx_actions)
            game_states[active_games_index] = new_game_states.detach().clone()

            done_index = active_games_index[torch.logical_and((dones == True), (rewards < 0))]
            final_rewards[done_index] = torch.ones_like(done_index).float() * 1
            done_index = active_games_index[torch.logical_and((dones == True), (rewards > 0))]
            final_rewards[done_index] = torch.ones_like(done_index).float() * -1

            if dones.sum() == len(dones):
                break

            active_games_index = active_games_index[dones != True]


        wins = (final_rewards > 0).sum() / len(final_rewards)
        episode_len = episode_len.float().mean()
        return {
            'wins': wins.item(),
            'episode_len': episode_len.item(),
        }

    @torch.no_grad()
    def run_evaluation(self):
        for player_id in self.hparams.player_ids:
            stat = self.one_game(player_id)
            wins = stat['wins']
            episode_len = stat['episode_len']
            self.logger.info(f'{player_id}: wins: {wins:.4f}, episode_len: {episode_len:.1f}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--num_eval_simulations', type=int, default=400, help='Number of evaluation simulations')
    parser.add_argument('--checkpoints_dir', type=str, help='Checkpoints directory')
    parser.add_argument('--checkpoint_path', type=str, help='Load this particular checkpoint')
    parser.add_argument('--connectx_dnn_model_dir', type=str, help='ConnectX DNN model dir')
    parser.add_argument('--random_agent', action='store_true', help='Use random agent instead')
    parser.add_argument('--log_to_stdout', action='store_true', help='Whether to log messages to stdout')
    parser.add_argument('--logfile', type=str, help='Logfile')
    parser.add_argument('--load_latest', action='store_true', default=False, help='Whether to load the latest checkpoint instead of the best metrics')
    FLAGS = parser.parse_args()

    if not FLAGS.random_agent and not FLAGS.connectx_dnn_model_dir:
        print(f'Either random agent flag or connectx dnn model dir has to be provided')
        exit(-1)

    if not FLAGS.checkpoints_dir and not FLAGS.checkpoint_path:
        print(f'Either checkpoints_dir or checkpoint_path has to be provided')
        exit(-1)

    module = module_loader.GameModule('connectx', load=True)

    module.hparams.num_simulations = FLAGS.num_eval_simulations
    module.hparams.checkpoints_dir = FLAGS.checkpoints_dir
    if FLAGS.batch_size:
        module.hparams.batch_size = FLAGS.batch_size
    if torch.cuda.is_available():
        module.hparams.device = torch.device('cuda:0')
    else:
        module.hparams.device = torch.device('cpu')
    module.hparams.load_latest = FLAGS.load_latest
    module.hparams.add_exploration_noise = False

    logger = setup_logger('eval', FLAGS.logfile, FLAGS.log_to_stdout)

    evaluation = Evaluation(module, FLAGS.checkpoint_path, logger, FLAGS.connectx_dnn_model_dir, FLAGS.random_agent)
    evaluation.run_evaluation()

if __name__ == '__main__':
    main()
