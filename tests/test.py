# -*- coding: utf-8 -*-

"""
Created on 2020/4/20

@author: Siqi Miao
"""

import pynode
import pytest
from copy import deepcopy
import numpy as np


class TestClass(object):

    def test_init(self):
        parent = None
        p = 1.0
        player_id = 1
        num_actions = 11*11
        root = pynode.Node(parent, p, player_id, num_actions)
        assert root.parent is None
        assert root.p == p
        assert root.Q == 0
        assert root.N == 0
        assert root.num_actions == num_actions
        assert root.player_id == 1
        assert len(root.actions) == 0
        assert len(root.child_nodes) == 0

    def test_expand(self):
        parent = None
        p = 1.0
        player_id = 1
        num_actions = 11*11
        root = pynode.Node(parent, p, player_id, num_actions)

        actions = [4, 7, 9, 10]
        probs = [0.1, 0.2, 0.3, 0.4]
        root.expand(actions, probs)

        assert root.actions == actions
        assert len(root.child_nodes) == len(actions)

        for i in range(len(actions)):
            assert abs(root.child_nodes[i].p - probs[i]) < 1e-5
            assert root.child_nodes[i].player_id == 2
            assert pynode.is_leaf_node(root.child_nodes[i]) is True

        assert pynode.is_leaf_node(root) is False

    def test_ucb(self):
        parent = None
        p = 1.0
        player_id = 1
        num_actions = 11*11
        root = pynode.Node(parent, p, player_id, num_actions)

        actions = [4, 7, 9, 10]
        probs = [0.1, 0.2, 0.3, 0.4]
        root.expand(actions, probs)

        Qs = [0.3, 0.2, 0.5, 0.6]
        c_puct = 0.5
        Ns = [30, 15, 40, 50]
        root.N = sum(Ns)

        ucbs = np.zeros(len(actions))
        for i in range(len(actions)):
            assert root.child_nodes[i].parent.N == sum(Ns)
            root.child_nodes[i].Q = Qs[i]
            root.child_nodes[i].N = Ns[i]

            ucbs[i] = Qs[i] + c_puct * probs[i] * np.sqrt(sum(Ns)) / (1 + Ns[i])
            assert abs(pynode.calc_ucb(root.child_nodes[i], c_puct) - ucbs[i]) < 1e-5

        assert pynode.get_max_ucb_child(root, c_puct) == np.argmax(ucbs).item()

    def test_update_QN(self):
        parent = None
        p = 1.0
        player_id = 1
        num_actions = 11*11
        root = pynode.Node(parent, p, player_id, num_actions)

        actions = [0, 1, 2, 3]
        probs = [0.1, 0.2, 0.3, 0.4]
        root.expand(actions, probs)

        root.Q = 0.1
        root_Q = deepcopy(root.Q)
        v = 0.5
        pynode.update_Q(root, v)
        assert abs(root.Q - (root_Q * root.N + v) / (1.0 + root.N)) <= 1e-5

        root.child_nodes[1].N = 30
        pynode.update_N(root.child_nodes[1])
        assert root.child_nodes[1].N == 30 + 1

    def test_multi_expansion(self):
        parent = None
        p = 1.0
        player_id = 1
        num_actions = 11*11
        root = pynode.Node(parent, p, player_id, num_actions)

        actions = [4, 7, 9, 10]
        probs = [0.1, 0.2, 0.3, 0.4]
        root.expand(actions, probs)

        Qs = [0.3, 0.2, 0.5, 0.6]
        c_puct = 0.5
        Ns = [30, 15, 40, 50]
        root.N = sum(Ns)

        for i in range(len(actions)):
            root.child_nodes[i].Q = Qs[i]
            root.child_nodes[i].N = Ns[i]

        idx = pynode.get_max_ucb_child(root, c_puct)

        root = root.child_nodes[idx]
        actions_2 = [10, 15, 17, 20, 15]
        probs_2 = [0.4, 0.3, 0.1, 0.1, 0.2]
        root.expand(actions_2, probs_2)

        assert root.actions == actions_2
        assert len(root.child_nodes) == len(actions_2)

        for i in range(len(actions_2)):
            assert abs(root.child_nodes[i].p - probs_2[i]) < 1e-5
            assert root.child_nodes[i].player_id == 1
            assert pynode.is_leaf_node(root.child_nodes[i]) is True

        assert pynode.is_leaf_node(root) is False
        assert root.parent.p == p
        assert root.parent.actions[idx] == actions[idx]

        root = root.child_nodes[3]
        assert pynode.is_leaf_node(root) is True
        assert abs(root.parent.p - probs[idx]) < 1e-5
        assert root.parent.actions[2] == actions_2[2]
        assert root.parent.parent.p == p
        assert root.parent.parent.actions[idx] == actions[idx]

        actions_3 = [7, 8, 9, 6, 15]
        probs_3 = [0.5, 0.1, 0.9, 0.1, 0.3]
        root.expand(actions_3, probs_3)

        # pynode.clear_parent_info(root)
        assert abs(root.p - probs_2[3]) < 1e-5
        assert root.actions[4] == actions_3[4]
