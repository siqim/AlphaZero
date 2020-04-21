# -*- coding: utf-8 -*-

"""
Created on 2020/4/5

@author: Siqi Miao
"""


import torch
import time
from collections import deque
from utils import switch_player
from model import Model
import threading
from multiprocessing import Queue


import os
import numpy as np
from board import Board
import pynode


class Node(object):
    __slots__ = ('parent', 'p', 'player_id', 'N', 'Q', 'actions', 'child_nodes')

    def __init__(self, parent, p, player_id):
        """Class for representing nodes in the monte carlo tree.

        :param parent: an instance of the Node class, which is the parent of the current node
        :param p: prior probs for the current node
        :param player_id: the player who is gonna play based on the current state
        """
        # one node is associated with one state, but storing each state is too costly, so we don't store the state here.
        self.parent = parent

        self.p = p
        self.N = 0
        self.Q = 0
        self.player_id = player_id

    def expand(self, actions, probs):
        next_player_id = switch_player(self.player_id)
        self.actions = actions
        self.child_nodes = [Node(self, prob, next_player_id) for prob in probs]

    @staticmethod
    def calc_ucb(node, c_puct):
        return pynode.calc_ucb(node.Q, c_puct, node.p, node.parent.N, node.N)

    @staticmethod
    def is_leaf_node(node):
        if hasattr(node, 'child_nodes'):
            return False
        else:
            return True


class MCTS(object):

    def __init__(self, board_size, strategy='stochastically', c_puct=5,
                 use_nn=True, add_noise=True, alpha=0.03, eps=0.25):
        self.board_size = board_size
        self.num_actions = board_size ** 2

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
            with torch.no_grad():
                actions, probs, v = self.get_probs_and_v(start_state)
            start_node.expand(actions, probs)
            return -v

        best_action, best_child_node, best_state = self.choose_max_ucb_move(start_node, start_state)

        v = self.search(best_action, best_child_node, best_state)

        start_node.Q = ((start_node.Q * start_node.N) + v) / (start_node.N + 1)
        best_child_node.N += 1

        return -v

    def get_one_move_by_simulations(self, node, state, num_simulations):
        for _ in range(num_simulations):
            self.search(action=None, start_node=node, start_state=state)
        action, next_node, pi = self.sample_actions(node)

        # discard all other branches except the branch of the new node,
        # otherwise we would always keep the whole tree in the memory, which is too costly.
        next_node.parent = None
        return action, next_node, pi  # the action will lead to the next_node, that is, the next state

    def sample_actions(self, node):

        Ns = [child_node.N for child_node in node.child_nodes]
        sum_N = sum(Ns)
        pi = [N/sum_N for N in Ns]

        if self.strategy == 'deterministically':
            idx = np.argmax(pi)
        elif self.strategy == 'stochastically':
            idx = np.random.choice(range(len(pi)), p=pi)
        else:
            raise ValueError('Unknown value!!!')

        action = node.actions[idx]
        next_node = node.child_nodes[idx]
        return action, next_node, pi

    def get_probs_and_v(self, state):
        """Given the current state, return prior prob for each valid action and v for the current state.

        :param state: the current states
        :return:
        """
        valid_actions = np.argwhere(state.reshape(-1) == 0).squeeze()

        # tell if we are gonna use neural net to get probs and v
        if self.use_nn:
            dummy_input = torch.from_numpy(
                np.random.random((batch_size, in_channels, board_size, board_size)).astype(np.float32)
            )
            probs, v = model(dummy_input)
            probs = probs.numpy().squeeze()
            v = v.item()
        else:
            # probs = np.random.dirichlet([self.num_actions]*self.num_actions)
            probs = np.random.random(self.num_actions)
            v = np.random.uniform(-1, 1, 1).item()

        # tell if we are gonna add dirichlet noise every time we expand a leaf node in order to enhance exploration
        if self.add_noise:
            noise = np.random.dirichlet([self.alpha]*self.num_actions)
            probs = (1 - self.eps) * probs + self.eps * noise

        return valid_actions, probs[valid_actions], v

    def choose_max_ucb_move(self, start_node, start_state):

        child_stats = [[child_node.Q, child_node.p, child_node.N] for child_node in start_node.child_nodes]
        best_idx = pynode.get_max_ucb_child(self.c_puct, start_node.N, child_stats)

        best_action = start_node.actions[best_idx]
        best_child_node = start_node.child_nodes[best_idx]
        best_state = Board.get_new_state(start_state, best_action, start_node.player_id)
        return best_action, best_child_node, best_state


def change_sampling_strategy(mcts, strategy_change_point, num_moves):
    if num_moves <= strategy_change_point:
        mcts.strategy = 'stochastically'
    else:
        mcts.strategy = 'deterministically'


def collect_self_play_data(player_id, state, history_buffer_black, history_buffer_white):

    # 1 for black 2 for white
    if player_id == 1:
        player_indicator = np.ones_like(state, dtype=np.float32)

        # append last player's move
        white_plane = np.zeros_like(state, dtype=np.float32)
        white_plane[state == 2] = 1
        history_buffer_white.append(white_plane)

    else:
        player_indicator = np.zeros_like(state, dtype=np.float32)

        # append last player's move
        black_plane = np.zeros_like(state, dtype=np.float32)
        black_plane[state == 1] = 1
        history_buffer_black.append(black_plane)

    x = np.stack((player_indicator, *history_buffer_white, *history_buffer_black))
    return x


def init_history_buffer(history_buffer_len_per_player, state):
    history_buffer_black = deque(maxlen=history_buffer_len_per_player)
    history_buffer_white = deque(maxlen=history_buffer_len_per_player)
    for _ in range(history_buffer_len_per_player):
        history_buffer_black.append(np.zeros_like(state, dtype=np.float32))
        history_buffer_white.append(np.zeros_like(state, dtype=np.float32))
    return history_buffer_black, history_buffer_white


def self_play(max_games):
    num_games = 0
    while num_games < max_games:
        player_id = 1
        node = Node(parent=None, p=None, player_id=player_id)
        state = board.init_state
        history_buffer_black, history_buffer_white = init_history_buffer(history_buffer_len_per_player, state)

        # self-play until we have a winner or the number of moves exceeds the max_moves
        num_moves = 0
        data_points = []
        while 1:
            action, node, pi = mcts.get_one_move_by_simulations(node, state, num_simulations)
            x = collect_self_play_data(player_id, state, history_buffer_black, history_buffer_white)
            data_points.append([player_id, x, pi])

            state = Board.get_new_state(state, action, player_id)
            num_moves += 1
            print(os.getpid(), threading.current_thread().name, num_moves)

            change_sampling_strategy(mcts, strategy_change_point, num_moves)

            if Board.has_won(action, state, board_size):
                num_games += 1
                print('{pid}, {thread_id}: {num} of games! Play {player_id} has won the game!'
                      .format(pid=os.getpid(), thread_id=threading.current_thread().name,
                              num=num_games, player_id=player_id))

                for player_id_for_x, x, pi in data_points:
                    if player_id_for_x == player_id:
                        self_play_buffer.put((x, pi, 1))
                    else:
                        self_play_buffer.put((x, pi, -1))

                break
            elif num_moves == max_moves:
                break

            player_id = switch_player(player_id)


def self_play_multi_threads(max_game_per_thread):

    t_list = [threading.Thread(target=self_play, name='thread_{idx}'.format(idx=idx), daemon=True,
                               args=(max_game_per_thread,))
              for idx in range(num_threads)]

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()


if __name__ == '__main__':
    batch_size = 1

    player_id = 1  # 1 for black 2 for white
    board_size = 11
    max_moves = board_size**2
    num_simulations = 400
    strategy_change_point = 10

    history_buffer_len_per_player = 2
    in_channels = history_buffer_len_per_player * 2 + 1
    self_play_buffer_len = 5000
    self_play_buffer = Queue(maxsize=self_play_buffer_len)

    max_game_per_thread = 1
    num_threads = 1

    model = Model(in_channels, num_filters=128, num_blocks=5, board_size=board_size)
    # model.cuda()
    model.eval()

    board = Board(board_size)
    mcts = MCTS(board_size, use_nn=True)

    tik = time.time()

    if num_threads != 1:
        self_play_multi_threads(max_game_per_thread)
    else:
        self_play(max_game_per_thread)

    tok = time.time()
    print((tok-tik) / (max_game_per_thread * num_threads))
