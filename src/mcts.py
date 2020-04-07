# -*- coding: utf-8 -*-

"""
Created on 2020/4/5

@author: Siqi Miao
"""


import numpy as np


def get_ps_and_v(state):
    ps = np.random.uniform(0, 1, 121)
    v = np.random.uniform(-1, 1, 1).item()
    return ps, v


class Node(object):

    def __init__(self, parent_node, p):
        self.parent = parent_node
        self.children = {}

        self.p = p
        self.N = 0
        self.Q = 0

    def expand(self, ps):
        for action, p in ps:
            self.children[action] = Node(self, p)


class MCTS(object):

    def __init__(self, board):
        self.board = board

    def one_simulation(self, action, start_node, start_state):
        if self.board.winning_check(start_state, action):
            start_node.v = 1
            return None

        ps, v = get_ps_and_v(start_state)
        start_node.v = v
        if not start_node.children:
            start_node.expand(ps)

        best_U = -float('inf')
        best_action = None
        best_child = None
        for action, child in start_node.children:
            U = self.get_ucb()
            if U > best_U:
                best_action = action
                best_child = child
                best_U = U
        best_state = self.board.get_new_state(start_state, best_action)

        self.one_simulation(best_action, best_child, best_state)

        start_node.Q = ((start_node.Q * start_node.N) + best_child.v) / (start_node.N + 1)
        start_node.N += 1

        return None


    def get_ucb(self):
        return np.random.uniform(-1, 1, 1).item()

def search(s, game, nnet):
    if game.gameEnded(s): return -game.gameReward(s)

    if s not in visited:
        visited.add(s)
        P[s], v = nnet.predict(s)
        return -v

    max_u, best_a = -float("inf"), -1
    for a in game.getValidActions(s):
        u = Q[s][a] + c_puct*P[s][a]*sqrt(sum(N[s]))/(1+N[s][a])
        if u>max_u:
            max_u = u
            best_a = a
    a = best_a

    sp = game.nextState(s, a)
    v = search(sp, game, nnet)

    Q[s][a] = (N[s][a]*Q[s][a] + v)/(N[s][a]+1)
    N[s][a] += 1
    return -v