# -*- coding: utf-8 -*-

"""
Created on 2020/4/4

@author: Siqi Miao
"""

import pynode
import numpy as np
from utils import idx_2_loc


class Board(object):
    """
    A class to represent the state and to encode the rule of game.
    """

    def __init__(self, board_size):
        self.board_size = board_size
        self.winning_flag = False
        self.init_state = np.zeros([self.board_size] * 2, dtype=np.int)
        self.state = self.init_state.copy()

    def reset_board(self):
        self.winning_flag = False
        self.state = self.init_state.copy()

    def get_current_state(self):
        return self.state

    @staticmethod
    def get_new_state(state, action, player_id):
        """Get the new state of the state given the action.

        :param state: the current state of the game
        :param action: a two-dimensional tuple representing the position for the new stone, e.g. (1, 2)
        :param player_id: a integer to indicate which player_id is playing
        :return:
            state: the new state state
            winning_flag: true for wining; false for losing
        """
        if not isinstance(action, tuple):
            action = idx_2_loc(action, state.shape[0])

        assert state[action] == 0
        new_state = state.copy()
        new_state[action] = player_id
        return new_state

    @staticmethod
    def has_won(action, state, board_size):
        """Check if the action would result in a win.

        :param action: a two-dimensional tuple representing the position for the new stone, e.g. (1, 2)
        :param state: the current state of the game
        :param board_size: size of the board
        :return:
            winning_flag: true for wining; false for losing
        """
        if action is None:
            return False

        if not isinstance(action, tuple):
            action = idx_2_loc(action, board_size)

        player_id = state[action]
        x, y = action
        return pynode.is_five_in_a_row(x, y, state.tolist(), board_size, player_id)
