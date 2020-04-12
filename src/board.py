# -*- coding: utf-8 -*-

"""
Created on 2020/4/4

@author: Siqi Miao
"""

import numpy as np


class Board(object):
    """
    A class to represent the state and to encode the rule of game.
    """

    def __init__(self, board_size):
        self.board_size = board_size
        self.winning_flag = False
        self.init_state = np.zeros([self.board_size] * 2, dtype=np.uint8)
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

        player_id = state[action]

        x, y = action
        # horizontal
        if any([(y-i >= 0 and y-i+5 <= board_size and np.all(state[x, y - i:y - i + 5] == player_id))
                for i in range(5)]):
            return True
        # vertical
        elif any([(x-i >= 0 and x-i+5 <= board_size and np.all(state[x - i:x - i + 5, y] == player_id))
                  for i in range(5)]):
            return True

        # diagonal \
        elif any([set([state[x - i + j][y - i + j]
                       if 0 <= x - i + j <= board_size - 1 and 0 <= y - i + j <= board_size - 1
                       else False for j in range(5)]) == {player_id} for i in range(5)]):
            return True

        # diagonal /
        elif any([set([state[x - i + j][y + i - j]
                       if 0 <= x - i + j <= board_size - 1 and 0 <= y + i - j <= board_size - 1
                       else False for j in range(5)]) == {player_id} for i in range(5)]):
            return True
        else:
            return False
