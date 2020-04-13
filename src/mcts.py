# -*- coding: utf-8 -*-

"""
Created on 2020/4/5

@author: Siqi Miao
"""


import time
from utils import switch_player
import numpy as np
from board import Board


class Node(object):

    def __init__(self, parent_node, P, action, player_id):
        """Class for representing nodes in the monte carlo tree.

        :param parent_node: an instance of the Node class, which is the parent of the current node
        :param P: prior probs for the current node
        :param player_id: the player who is gonna play based on the current state
        """
        # one node is associated with one state, but storing each state is too costly, so we don't store the state here.
        self.parent = parent_node
        self.children = {}

        self.P = P
        self.N = 0
        self.Q = 0
        self.action = action  # the action that leads to this node
        self.player_id = player_id

    def expand(self, probs):
        """To expand a leaf node.

        :param probs: a dict with prior prob for each action
        :return:
        """

        next_player_id = switch_player(self.player_id)

        for action, P in probs.items():
            self.children[action] = Node(self, P, action, next_player_id)

    @staticmethod
    def calc_ucb(node, c_puct):
        U = c_puct * node.P * np.sqrt(node.parent.N) / (1 + node.N)
        return node.Q + U

    @staticmethod
    def is_leaf_node(node):
        if node.children == {}:
            return True
        else:
            return False


class MCTS(object):

    def __init__(self, board_size, c_puct, strategy='stochastically', tau=1, use_nn=False):
        self.board_size = board_size
        self.c_puct = c_puct
        self.strategy = strategy
        self.tau = tau
        self.use_nn = use_nn

    def search(self, action, start_node, start_state):
        if Board.has_won(action, start_state, self.board_size):
            v = 1
            return -v

        if Node.is_leaf_node(start_node):
            probs, v = self.get_probs_and_v(start_state)
            start_node.expand(probs)
            return -v

        best_action, best_child_node, best_state = self.choose_max_ucb_move(start_node, start_state)

        v = self.search(best_action, best_child_node, best_state)

        start_node.Q = ((start_node.Q * start_node.N) + v) / (start_node.N + 1)
        best_child_node.N += 1

        return -v

    def get_one_move_by_simulations(self, node, state, num_simulations):
        for _ in range(num_simulations):
            self.search(action=None, start_node=node, start_state=state)

        action, next_node = self.sample_actions(node)
        return action, next_node  # the action will lead to the next_node, that is, the next state

    def sample_actions(self, node):
        actions, child_nodes = zip(*node.children.items())
        Ns = [child_node.N ** (1/self.tau) for child_node in child_nodes]

        if self.strategy == 'deterministically':
            idx = np.argmax(Ns)
        elif self.strategy == 'stochastically':
            idx = np.random.choice(a=range(len(actions)), p=[N / sum(Ns) for N in Ns])
        else:
            raise ValueError('Unknown value!!!')

        action = actions[idx]
        next_node = child_nodes[idx]

        return action, next_node

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

    def choose_max_ucb_move(self, start_node, start_state):

        best_U = -float('inf')
        best_action = None
        best_child_node = None

        for action, child_node in start_node.children.items():
            U = Node.calc_ucb(child_node, self.c_puct)
            if U > best_U:
                best_action = action
                best_child_node = child_node
                best_U = U

        best_state = Board.get_new_state(start_state, best_action, start_node.player_id)
        return best_action, best_child_node, best_state


if __name__ == '__main__':

    c_puct = 1
    num_simulations = 400
    board_size = 11
    player_id = 1
    max_moves = 11**2
    max_games = 25000

    board = Board(board_size)
    mcts = MCTS(board_size, c_puct)

    num_games = 0
    while num_games < max_games:

        tik = time.time()

        node = Node(parent_node=None, P=1, action=None, player_id=player_id)
        state = board.init_state

        num_moves = 0
        while 1:

            action, node = mcts.get_one_move_by_simulations(node, state, num_simulations)
            state = Board.get_new_state(state, action, player_id)
            num_moves += 1

            if Board.has_won(action, state, board_size):
                print('Play {player_id} has won the game!'.format(player_id=player_id))
                num_games += 1
                break
            elif num_moves == max_moves:
                break

            player_id = switch_player(player_id)

        tok = time.time()
        print(num_games, tok-tik, num_moves)
