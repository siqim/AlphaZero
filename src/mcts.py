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
from queue import Queue

import pickle
import datetime
import os
from copy import deepcopy
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

    def search(self, action, start_node, start_state, history_buffer_black, history_buffer_white):
        update_history_buffer(start_node.player_id, start_state, history_buffer_black, history_buffer_white)
        if Board.has_won(action, start_state, self.board_size):
            v = 1
            return -v

        if Node.is_leaf_node(start_node):
            actions, probs, v = self.get_probs_and_v(start_state, start_node.player_id,
                                                     history_buffer_black, history_buffer_white)
            start_node.expand(actions, probs)
            return -v

        best_action, best_child_node, best_state = self.choose_max_ucb_move(start_node, start_state)

        v = self.search(best_action, best_child_node, best_state, history_buffer_black, history_buffer_white)

        start_node.Q = ((start_node.Q * start_node.N) + v) / (start_node.N + 1)
        best_child_node.N += 1

        return -v

    def get_one_move_by_simulations(self, node, state, num_simulations, history_buffer_black, history_buffer_white):
        for _ in range(num_simulations):
            self.search(action=None, start_node=node, start_state=state,
                        history_buffer_black=deepcopy(history_buffer_black),
                        history_buffer_white=deepcopy(history_buffer_white))
        action, next_node, pi = self.sample_actions(node)

        # discard all other branches except the branch of the new node,
        # otherwise we would always keep the whole tree in the memory, which is too costly.
        next_node.parent = None
        return action, next_node, pi  # the action will lead to the next_node, that is, the next state

    def sample_actions(self, node):

        Ns = [child_node.N for child_node in node.child_nodes]
        sum_N = sum(Ns)
        pi = {node.actions[idx]: N/sum_N for idx, N in enumerate(Ns)}

        if self.strategy == 'deterministically':
            idx = np.argmax(list(pi.values()))
        elif self.strategy == 'stochastically':
            idx = np.random.choice(range(len(pi)), p=list(pi.values()))
        else:
            raise ValueError('Unknown value!!!')

        action = node.actions[idx]
        next_node = node.child_nodes[idx]
        pi = [pi.get(i, 0) for i in range(self.num_actions)]
        return action, next_node, pi

    def get_probs_and_v(self, state, player_id, history_buffer_black, history_buffer_white):
        """Given the current state, return prior prob for each valid action and v for the current state.

        :param state: the current state
        :param player_id: the player who is gonna put the stone on the current state
        :param history_buffer_black: history states for black stones, not including the current state
        :param history_buffer_white: history states for black stones, not including the current state
        :return:
        """
        global batch_res
        valid_actions = np.argwhere(state.reshape(-1) == 0).reshape(-1)

        # tell if we are gonna use neural net to get probs and v
        if self.use_nn:
            x = collect_self_play_data(player_id, state, history_buffer_black,
                                       history_buffer_white, update_buffer=False)
            idx = threading.current_thread().name
            batch_buffer.put([idx, x], block=True)

            while idx not in batch_res:
                time.sleep(0.0001)

            probs, v = batch_res[idx]
            del batch_res[idx]

            probs = probs.cpu().numpy()
            v = v.item()

        else:
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


def update_history_buffer(player_id, state, history_buffer_black, history_buffer_white):
    # 1 for black 2 for white
    if player_id == 1:
        # append last player's move
        white_plane = np.zeros_like(state, dtype=np.float32)
        white_plane[state == 2] = 1
        history_buffer_white.append(white_plane)
    else:
        # append last player's move
        black_plane = np.zeros_like(state, dtype=np.float32)
        black_plane[state == 1] = 1
        history_buffer_black.append(black_plane)


def collect_self_play_data(player_id, state, history_buffer_black, history_buffer_white, update_buffer=True):
    if update_buffer:
        update_history_buffer(player_id, state, history_buffer_black, history_buffer_white)

    # 1 for black 2 for white
    if player_id == 1:
        player_indicator = np.ones_like(state, dtype=np.float32)
    else:
        player_indicator = np.zeros_like(state, dtype=np.float32)

    x = np.stack((player_indicator, *history_buffer_white, *history_buffer_black))
    return x


def init_history_buffer(history_buffer_len_per_player, state):
    history_buffer_black = deque(maxlen=history_buffer_len_per_player)
    history_buffer_white = deque(maxlen=history_buffer_len_per_player)
    for _ in range(history_buffer_len_per_player):
        history_buffer_black.append(np.zeros_like(state, dtype=np.float32))
        history_buffer_white.append(np.zeros_like(state, dtype=np.float32))
    return history_buffer_black, history_buffer_white


def self_play(max_games, lock):
    global num_games
    while num_games < max_games:
        player_id = 1  # 1 for black 2 for white
        node = Node(parent=None, p=None, player_id=player_id)
        state = board.init_state
        history_buffer_black, history_buffer_white = init_history_buffer(history_buffer_len_per_player, state)

        # self-play until we have a winner or the number of moves exceeds the max_moves
        num_moves = 0
        data_points = []
        tik = time.time()
        while 1:
            action, node, pi = mcts.get_one_move_by_simulations(node, state, num_simulations,
                                                                deepcopy(history_buffer_black),
                                                                deepcopy(history_buffer_white))
            x = collect_self_play_data(player_id, state, history_buffer_black, history_buffer_white)
            data_points.append([player_id, x, pi])
            state = Board.get_new_state(state, action, player_id)
            num_moves += 1
            lock.acquire()
            print(os.getpid(), threading.current_thread().name, num_moves, time.time() - tik)
            tik = time.time()
            lock.release()

            change_sampling_strategy(mcts, strategy_change_point, num_moves)

            if Board.has_won(action, state, board_size):
                lock.acquire()
                num_games += 1
                print('{pid}, {thread_id}: {num} of games! Play {player_id} has won the game!'
                      .format(pid=os.getpid(), thread_id=threading.current_thread().name,
                              num=num_games, player_id=player_id))
                lock.release()

                for player_id_for_x, x, pi in data_points:
                    if player_id_for_x == player_id:
                        self_play_buffer.put((x, pi, 1), block=True)
                    else:
                        self_play_buffer.put((x, pi, -1), block=True)

                break
            elif num_moves == max_moves:
                break

            player_id = switch_player(player_id)


def self_play_multi_threads(max_games, lock):

    t_list = [threading.Thread(target=self_play, name='thread_{idx}'.format(idx=idx), daemon=True,
                               args=(max_games, lock))
              for idx in range(num_threads)]

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()


def batch_inference():
    global num_games, batch_res

    with torch.no_grad():
        while 1:
            if num_games >= max_games:
                break

            if batch_buffer.full():
                data = [batch_buffer.get(block=False) for _ in range(batch_size)]
                idx, x = list(zip(*data))
                x = torch.from_numpy(np.stack(x)).cuda()
                load_model_event.wait()
                probs, v = model(x)
                for i, id_ in enumerate(idx):
                    batch_res[id_] = [probs[i], v[i]]

            if self_play_buffer.full():
                data = [self_play_buffer.get(block=False) for _ in range(self_play_buffer_len)]
                file_name = '_'.join(['data', str(os.getpid()), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')])
                with open(file_name, 'wb') as f:
                    pickle.dump(data, f)


def get_model():
    # TODO: get new model according to the number of self-play games, rather than scan the folder
    global current_model_name, model

    while 1:
        models = sorted([each for each in os.listdir('../models') if 'model_' in each])
        if models:
            newest_model_name = models[-1]
            if current_model_name == newest_model_name:
                time.sleep(60*10)
            else:
                load_model_event.clear()
                state_dict = torch.load('../models/' + newest_model_name)
                model.load_state_dict(state_dict)
                current_model_name = newest_model_name
                load_model_event.set()
        else:
            time.sleep(60*10)


if __name__ == '__main__':

    board_size = 11
    max_moves = board_size**2
    num_simulations = 400
    strategy_change_point = 10

    history_buffer_len_per_player = 2
    in_channels = history_buffer_len_per_player * 2 + 1
    self_play_buffer_len = 5000
    self_play_buffer = Queue(maxsize=self_play_buffer_len)

    num_games = 0
    max_games = 63
    num_threads = 64
    lock = threading.Lock()

    batch_res = {}
    batch_size = 32
    batch_buffer = Queue(maxsize=batch_size)

    current_model_name = None
    model = Model(in_channels, num_filters=128, num_blocks=5, board_size=board_size)
    model.cuda()
    model.eval()
    load_model_event = threading.Event()
    load_model_event.set()

    board = Board(board_size)
    mcts = MCTS(board_size, use_nn=True)

    model_scan_thread = threading.Thread(target=get_model, daemon=True)
    batch_inference_thread = threading.Thread(target=batch_inference, daemon=True)

    tik = time.time()
    batch_inference_thread.start()
    if num_threads != 1:
        self_play_multi_threads(max_games, lock)
    else:
        self_play(max_games, lock)
    batch_inference_thread.join()

    tok = time.time()
    print((tok-tik) / max_games)
