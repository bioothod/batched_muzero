import importlib

import hparams
import network_params

class GameModule:
    def __init__(self, game_name: str, load):
        self.game_name = game_name

        if self.game_name == 'tictactoe':
            self.hparams = hparams.TicTacToeHparams()
        elif self.game_name == 'connectx' or self.game_name == 'connect4':
            self.hparams = hparams.ConnectXHparams()
        else:
            raise NotImplementedError(f'gmae "{self.game_name}" is not implemented')

        if load:
            self.load()

    def load(self):
        if self.game_name == 'tictactoe':
            import tictactoe_impl

            self.step_games = tictactoe_impl.step_games
            self.invalid_actions_mask = tictactoe_impl.invalid_actions_mask
            self.game_hparams = tictactoe_impl.GameHparams()
            self.network_hparams = network_params.TicTacToeParams(observation_shape=self.hparams.state_shape, num_actions=self.hparams.num_actions)
        elif self.game_name == 'connectx':
            import connectx_impl

            self.step_games = connectx_impl.step_games
            self.invalid_actions_mask = connectx_impl.invalid_actions_mask
            self.game_hparams = connectx_impl.GameHparams()
            self.network_hparams = network_params.ConnectXParams(observation_shape=self.hparams.state_shape, num_actions=self.hparams.num_actions)
