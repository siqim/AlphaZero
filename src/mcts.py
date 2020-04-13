# -*- coding: utf-8 -*-

"""
Created on 2020/4/5

@author: Siqi Miao
"""


import time
from collections import deque
from utils import switch_player, idx_2_loc, change_sampling_strategy
import numpy as np
from board import Board


class Node(object):

    def __init__(self, parent_node, P, action, player_id):
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

    def __init__(self, board_size, strategy='stochastically', c_puct=5,
                 use_nn=True, add_noise=True, alpha=0.03, eps=0.25):
        self.board_size = board_size

        self.strategy = strategy
        self.c_puct = c_puct

        self.use_nn = use_nn
        self.add_noise = add_noise
        self.alpha = alpha
        self.eps = eps

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
        Ns = [child_node.N for child_node in child_nodes]

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

        # tell if we are gonna use neural net to get probs and v
        if self.use_nn:
            probs = np.random.dirichlet([self.board_size**2]*self.board_size**2)
            v = np.random.uniform(-1, 1, 1).item()
        else:
            probs = np.random.dirichlet([self.board_size**2]*self.board_size**2)
            v = np.random.uniform(-1, 1, 1).item()

        # tell if we are gonna add dirichlet noise every time we expand a leaf node in order to enhance exploration
        if self.add_noise:
            noise = np.random.dirichlet([self.alpha]*self.board_size**2)
            probs_dict = {idx_2_loc(idx, self.board_size):
                          (1 - self.eps) * prob + self.eps * noise[idx]
                          for idx, prob in enumerate(probs)}
        else:
            probs_dict = {idx_2_loc(idx, self.board_size): prob for idx, prob in enumerate(probs)}

        probs_dict = {tuple(valid_action): probs_dict[tuple(valid_action)] for valid_action in valid_actions}

        return probs_dict, v

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


def collect_self_play_data(player_id, state, history_buffer_black, history_buffer_white):

    # 1 for black 2 for white
    if player_id == 1:
        player_indicator = np.ones_like(state, dtype=np.float32)

        black_plane = np.zeros_like(state, dtype=np.float32)
        black_plane[state == 1] = 1
        history_buffer_black.append(black_plane)
    else:
        player_indicator = np.zeros_like(state, dtype=np.float32)

        white_plane = np.zeros_like(state, dtype=np.float32)
        white_plane[state == 2] = 1
        history_buffer_white.append(white_plane)

    x = np.stack((*history_buffer_black, *history_buffer_white, player_indicator))
    p = None
    v = None
    return x, (p, v)




if __name__ == '__main__':

    strategy_change_point = 10
    history_buffer_len_per_player = 7

    num_simulations = 400
    board_size = 11
    player_id = 1  # 1 for black 2 for white
    max_moves = 11**2
    max_games = 25000

    board = Board(board_size)
    mcts = MCTS(board_size)

    # generate self-play games until we have max number of games.
    num_games = 0
    while num_games < max_games:

        tik = time.time()

        node = Node(parent_node=None, P=None, action=None, player_id=player_id)
        state = board.init_state

        history_buffer_black = deque(maxlen=history_buffer_len_per_player)
        history_buffer_white = deque(maxlen=history_buffer_len_per_player)
        for _ in range(history_buffer_len_per_player):
            history_buffer_black.append(np.zeros_like(state, dtype=np.float32))
            history_buffer_white.append(np.zeros_like(state, dtype=np.float32))

        # self-play until we have a winner or the number of moves exceeds the max_moves
        num_moves = 0
        while 1:

            action, node = mcts.get_one_move_by_simulations(node, state, num_simulations)

            x, y = collect_self_play_data(player_id, state, history_buffer_black, history_buffer_white)

            state = Board.get_new_state(state, action, player_id)
            num_moves += 1

            change_sampling_strategy(mcts, strategy_change_point, num_moves)

            if Board.has_won(action, state, board_size):
                print('Play {player_id} has won the game!'.format(player_id=player_id))
                num_games += 1
                break
            elif num_moves == max_moves:
                break

            player_id = switch_player(player_id)

        tok = time.time()
        print(num_games, tok-tik, num_moves)
