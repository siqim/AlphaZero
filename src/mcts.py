# -*- coding: utf-8 -*-

"""
Created on 2020/4/5

@author: Siqi Miao
"""


import time
import numpy as np
from board import Board


class Node(object):

    def __init__(self, parent_node, P, player_id):
        """Class for representing nodes in the monte carlo tree.

        :param parent_node: an instance of the Node class, which is the parent of the current node
        :param P: proir probs for the current node
        :param player_id: the player who is gonna play based on the current state
        """
        # one node is associated with one state, but storing each state is too costly, so we don't store the state here.
        self.parent = parent_node
        self.children = {}

        self.P = P
        self.N = 0
        self.Q = 0
        self.player_id = player_id

    def expand(self, probs):
        """To expand a leaf node.

        :param probs: a dict with prior prob for each action
        :return:
        """
        if self.player_id == 1:
            next_player_id = 2
        elif self.player_id == 2:
            next_player_id = 1
        else:
            raise ValueError('Unknown value!!!')

        for action, P in probs.items():
            self.children[action] = Node(self, P, next_player_id)

    @staticmethod
    def calc_ucb(node):
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

        best_action, best_child_node, best_state = MCTS.choose_max_ucb_move(start_node, start_state)

        v = self.search(best_action, best_child_node, best_state)

        start_node.Q = ((start_node.Q * start_node.N) + v) / (start_node.N + 1)
        start_node.N += 1

        return -v

    def get_one_move_by_simulations(self, node, state, num_simulations):
        for _ in range(num_simulations):
            self.search(action=None, start_node=node, start_state=state)

        actions = list(node.children.keys())
        Ns = [child_node.N for child_node in node.children.values()]

        idx = np.random.choice(a=range(len(actions)),
                               p=[N / sum(Ns) for N in Ns])
        action = actions[idx]
        return action

    def get_probs_and_v(self, state):
        """Given the current state, return prior prob for each valid action and v for the current state.

        :param state: the current states
        :return:
        """
        valid_actions = np.argwhere(state == 0)

        if self.use_nn:
            probs = {tuple(action): np.random.uniform(0, 1, 1).item() for action in valid_actions}
            v = np.random.uniform(-1, 1, 1).item()
        else:
            probs = {tuple(action): np.random.uniform(0, 1, 1).item() for action in valid_actions}
            v = np.random.uniform(-1, 1, 1).item()
        return probs, v

    @staticmethod
    def choose_max_ucb_move(start_node, start_state):

        best_U = -float('inf')
        best_action = None
        best_child_node = None

        for action, child_node in start_node.children.items():
            U = Node.calc_ucb(child_node)
            if U > best_U:
                best_action = action
                best_child_node = child_node
                best_U = U

        best_state = Board.get_new_state(start_state, best_action, start_node.player_id)
        return best_action, best_child_node, best_state


if __name__ == '__main__':

    c_puct = 1
    num_simulations = 1600
    board_size = 11
    init_player_id = 1

    board = Board(board_size)
    node = Node(parent_node=None, P=1, player_id=init_player_id)
    state = board.state

    mcts = MCTS(board, c_puct)

    while 1:
        tik = time.time()
        action = mcts.get_one_move_by_simulations(node, state, num_simulations)
        tok = time.time()
        print(tok - tik)
