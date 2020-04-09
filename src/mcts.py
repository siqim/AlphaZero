# -*- coding: utf-8 -*-

"""
Created on 2020/4/5

@author: Siqi Miao
"""


import numpy as np
from board import Board
from utils import idx_2_loc


class Node(object):

    def __init__(self, parent_node, P, player_id):
        # one node is associated with one state, but storing each state is too costly, so we don't store the state here.
        self.parent = parent_node
        self.children = {}

        self.P = P
        self.N = 0
        self.Q = 0
        self.player_id = player_id

    def expand(self, probs):
        """To expand a leaf node.

        :param probs:
        :return:
        """
        if self.player_id == 1:
            next_player_id = 2
        elif self.player_id == 2:
            next_player_id = 1
        else:
            raise ValueError('Unknown value!!!')

        for action, P in probs:
            self.children[action] = Node(self, P, next_player_id)

    @staticmethod
    def get_ucb(node):
        Q = node.Q
        P = node.P
        N = node.N
        return np.random.uniform(-1, 1, 1).item()

    @staticmethod
    def is_leaf_node(node):
        if node.children == {}:
            return True
        else:
            return False

    @staticmethod
    def is_root_node(node):
        if node.parent is None:
            return True
        else:
            return False


class MCTS(object):

    def __init__(self, board, c_puct):
        self.board = board
        self.c_puct = c_puct
        self.use_nn = False

    def search(self, action, start_node, start_state):
        if Board.has_won(action, start_state, self.board.board_size):
            v = 1
            return -v

        if Node.is_leaf_node(start_node):
            probs, v = self.get_probs_and_v(start_state)
            start_node.expand(probs)
            return -v

        best_U = -float('inf')
        best_action = None
        best_child_node = None
        for action, child_node in start_node.children.items():
            U = Node.get_ucb(child_node)
            if U > best_U:
                best_action = action
                best_child_node = child_node
                best_U = U
        best_state = Board.get_new_state(start_state, best_action, start_node.player_id)

        v = self.search(best_action, best_child_node, best_state)

        start_node.Q = ((start_node.Q * start_node.N) + v) / (start_node.N + 1)
        start_node.N += 1

        return -v

    def get_one_move(self, node, state, num_simulations):
        for _ in range(num_simulations):
            self.search(action=None, start_node=node, start_state=state)

        action = np.random.choice(a=[action for action in node.children.keys()],
                                  p=[child_node.N for child_node in node.children.values()])

        return action

    def get_probs_and_v(self, state):
        indices = np.arange(self.board.board_size ** 2)
        locations = [idx_2_loc(idx, self.board.board_size) for idx in indices]

        if self.use_nn:
            probs = {loc: np.random.uniform(0, 1, 1).item() for loc in locations}
            v = np.random.uniform(-1, 1, 1).item()
        else:
            probs = {loc: np.random.uniform(0, 1, 1).item() for loc in locations}
            v = np.random.uniform(-1, 1, 1).item()
        return probs, v


if __name__ == '__main__':

    c_puct = 1
    num_simulations = 1600
    board_size = 11
    init_player_id = 1

    board = Board(board_size)
    node = Node(parent_node=None, P=1, player_id=init_player_id)
    state = board.state

    mcts = MCTS(board, c_puct)
    mcts.get_one_move(node, state, num_simulations)
