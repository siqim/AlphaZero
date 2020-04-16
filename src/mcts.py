# -*- coding: utf-8 -*-

"""
Created on 2020/4/5

@author: Siqi Miao
"""


import sys
import torch
import time
from collections import deque
from utils import switch_player
from model import Model
import threading
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, Queue

import os
import numpy as np
from board import Board
from math import sqrt


class Node(object):

    def __init__(self, parent_node, p, player_id):
        """Class for representing nodes in the monte carlo tree.

        :param parent_node: an instance of the Node class, which is the parent of the current node
        :param p: prior probs for the current node
        :param player_id: the player who is gonna play based on the current state
        """
        # one node is associated with one state, but storing each state is too costly, so we don't store the state here.
        self.parent = parent_node
        self.children = {}

        self.p = p
        self.N = 0
        self.Q = 0
        self.player_id = player_id

    def expand(self, probs):
        """To expand a leaf node.

        :param probs: a dict with prior prob for each action
        :return:
        """

        next_player_id = switch_player(self.player_id)
        for action, p in probs:
            self.children[action] = Node(self, p, next_player_id)

    @staticmethod
    def calc_ucb(node, c_puct):
        U = c_puct * node.p * sqrt(node.parent.N) / (1 + node.N)
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
        action, next_node, pi = self.sample_actions(node)

        # discard all other branches except the branch of the new node,
        # otherwise we would always keep the whole tree in the memory, which is too costly.
        next_node.parent = None
        return action, next_node, pi  # the action will lead to the next_node, that is, the next state

    def sample_actions(self, node):

        Ns = {action: child_node.N for action, child_node in node.children.items()}
        sum_N = sum(Ns.values())
        pi = np.array([Ns.get(action_idx, 0) / sum_N for action_idx in range(self.num_actions)])

        if self.strategy == 'deterministically':
            action = np.argmax(pi)
        elif self.strategy == 'stochastically':
            action = np.random.choice(range(self.num_actions), p=pi)
        else:
            raise ValueError('Unknown value!!!')
        next_node = node.children[action]
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
            probs = np.random.dirichlet([self.num_actions]*self.num_actions)
            v = np.random.uniform(-1, 1, 1).item()

        # tell if we are gonna add dirichlet noise every time we expand a leaf node in order to enhance exploration
        if self.add_noise:
            noise = np.random.dirichlet([self.alpha]*self.num_actions)
            probs = (1 - self.eps) * probs + self.eps * noise

        probs_dict = [(action, probs[action]) for action in valid_actions]
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


def change_sampling_strategy(mcts, strategy_change_point, num_moves):
    if num_moves <= strategy_change_point:
        mcts.strategy = 'stochastically'
    else:
        mcts.strategy = 'deterministically'


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
    return x


def init_history_buffer(history_buffer_len_per_player, state):
    history_buffer_black = deque(maxlen=history_buffer_len_per_player)
    history_buffer_white = deque(maxlen=history_buffer_len_per_player)
    for _ in range(history_buffer_len_per_player):
        history_buffer_black.append(np.zeros_like(state, dtype=np.float32))
        history_buffer_white.append(np.zeros_like(state, dtype=np.float32))
    return history_buffer_black, history_buffer_white


def self_play(num_games):
    while num_games < max_games:
        player_id = 1
        node = Node(parent_node=None, p=None, player_id=player_id)
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


def self_play_multi_threads(num_games):

    t_list = [threading.Thread(target=self_play, name='thread_{idx}'.format(idx=idx), daemon=True, args=(num_games,))
              for idx in range(num_threads)]

    tik = time.time()
    for t in t_list:
        t.start()
    for t in t_list:
        t.join()
    print('asdasd')
    tok = time.time()
    print('test', (tok-tik)/num_games)


if __name__ == '__main__':

    in_channels = 5
    board_size = 11
    batch_size = 1
    num_filters = 128
    num_blocks = 5
    model = Model(in_channels, num_filters=num_filters, num_blocks=num_blocks, board_size=board_size)
    model.eval()

    strategy_change_point = 10
    history_buffer_len_per_player = 2
    num_simulations = 100
    player_id = 1  # 1 for black 2 for white
    max_moves = board_size**2
    num_games = 0
    max_games = 1
    num_threads = 2
    num_processes = 2
    self_play_buffer_len = 5000
    self_play_buffer = Queue(maxsize=self_play_buffer_len)

    board = Board(board_size)
    mcts = MCTS(board_size, use_nn=False)

    pool = Pool(num_processes)
    for i in range(num_processes):
        pool.apply_async(func=self_play_multi_threads, args=(num_games,))

    pool.close()
    pool.join()
