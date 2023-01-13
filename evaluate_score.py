from typing import List, Optional

import argparse
import json
import logging
import os

from copy import deepcopy

import numpy as np
import torch

from logger import setup_logger
from hparams import GenericHparams as Hparams

class EvaluationDataset:
    def __init__(self, input_file: str, hparams: Hparams, logger: logging.Logger):
        self.logger = logger

        num_rows = hparams.rows
        num_columns = hparams.columns
        self.player_ids = hparams.player_ids

        self.game_states = []
        self.game_player_ids = []
        self.best_moves = []
        self.good_moves = []
        range_index = np.arange(num_columns, dtype=np.int32)
        with open(input_file, 'r') as fin:
            for line in fin:
                data = json.loads(line)

                game_state = torch.Tensor(data['board']).float().reshape(hparams.state_shape)
                score = data['score']
                action_scores = np.array(data['move score'])

                non_zero = torch.count_nonzero(game_state)
                if non_zero & 1:
                    player_id = 2
                else:
                    player_id = 1

                best_moves = range_index[action_scores == score]
                self.best_moves.append(best_moves)

                good_moves = np.array(0)
                if score < 0:
                    good_moves = range_index[action_scores < 0]
                elif score > 0:
                    good_moves = range_index[action_scores > 0]
                elif score == 0:
                    good_moves = range_index[action_scores == 0]

                self.good_moves.append(good_moves)

                self.game_states.append(game_state.detach().clone())
                self.game_player_ids.append(player_id)

        self.game_states = torch.stack(self.game_states, 0).to(hparams.device)
        self.game_player_ids = torch.tensor(self.game_player_ids).long().to(hparams.device)

        self.ref_best_moves = torch.zeros(len(self.best_moves), num_columns, dtype=torch.float32, device=hparams.device)
        self.ref_good_moves = torch.zeros(len(self.good_moves), num_columns, dtype=torch.float32, device=hparams.device)
        for idx, (best_moves, good_moves) in enumerate(zip(self.good_moves, self.best_moves)):
            self.ref_best_moves[idx, best_moves] = 1. / len(best_moves)
            self.ref_good_moves[idx, good_moves] = 1. / len(good_moves)

        player_state_rates = []
        for player_id in self.player_ids:
            rate = float(torch.count_nonzero(self.game_player_ids == player_id).cpu().numpy()) / len(self.game_player_ids)
            rate = rate * 100
            player_state_rates.append(f'{player_id}:{rate:.1f}%')

        player_state_rates = ', '.join(player_state_rates)
        self.logger.info(f'states: {len(self.game_states)}, player_state_rates: {player_state_rates}')

    def evaluate(self, pred_actions: torch.Tensor, debug: bool):
        best_count = 0
        good_count = 0
        for pred_action, best_moves, good_moves in zip(pred_actions, self.ref_best_moves, self.ref_good_moves):
            if pred_action in best_moves:
                best_count += 1
            if pred_action in good_moves:
                good_count += 1

        best_score = best_count / len(pred_actions) * 100
        good_score = good_count / len(pred_actions) * 100

        if debug:
            self.logger.info(f'perfect moves: {best_score:.1f}')
            self.logger.info(f'   good moves: {good_score:.1f}')

        return best_score, good_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_agent', type=str, required=True, help='The training agent\'s name')
    parser.add_argument('--eval_file', type=str, required=True, help='The evaluation dataset file')
    parser.add_argument('--evaluation_dir', type=str, required=True, help='Working directory')
    parser.add_argument('--log_to_stdout', action='store_true', help='Log evaluation data to stdout')
    parser.add_argument('--mcts_steps', type=int, default=0, help='Wrap training agent into mcts tree search with this many rollouts per step')
    parser.add_argument('--eval_seed', type=int, default=555, help='Random seed for generators')
    FLAGS = parser.parse_args()

    train_name = FLAGS.train_agent.split(':')[0]
    evaluation_dir = os.path.join(FLAGS.evaluation_dir, f'{train_name}_perfect_scores')

    os.makedirs(evaluation_dir, exist_ok=True)
    logfile = os.path.join(evaluation_dir, 'evalution.log')
    logger = setup_logger('e', logfile=logfile, log_to_stdout=FLAGS.log_to_stdout)

    config = edict({
        'device': 'cpu',
        'rows': 6,
        'columns': 7,
        'inarow': 4,
        'player_ids': [1, 2],
        'eval_seed': FLAGS.eval_seed,
        'logfile': logfile,
        'log_to_stdout': FLAGS.log_to_stdout,
        'default_reward': 0,

        'gamma': 0.99,
        'tau': 0.97,
        'batch_size': 128,

        'num_simulations': FLAGS.mcts_steps,
        'mcts_c1': 1.25,
        'mcts_c2': 19652,
        'mcts_discount': 0.99,
        'add_exploration_noise': False,
        'root_dirichlet_alpha': 0.3,
        'root_exploration_fraction': 0.25,
    })

    config.actions = config.columns
    eval_ds = EvaluationDataset(FLAGS.eval_file, config, logger)

    config.num_training_games = len(eval_ds.game_states)
    try:
        train_agent = create_agent(config, FLAGS.train_agent, logger, FLAGS.mcts_steps)
    except Exception as e:
        logger.critical(f'could not create an agent: {e}')
        raise


    best_score, good_score = eval_ds.evaluate(train_agent, debug=False)
    logger.info(f'{FLAGS.train_agent}: best_score: {best_score:.1f}, good_score: {good_score:.1f}')

if __name__ == '__main__':
    main()
