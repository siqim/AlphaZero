# -*- coding: utf-8 -*-

"""
Created on 2020/4/5

@author: Siqi Miao
"""


def idx_2_loc(idx, board_size):
    """
    transfer a move value to a location value
    :param idx: an int type move value such as 34
    :param board_size: size of the state
    :return: an 1*2 dimension location value such as (2, 3)
    """
    return idx // board_size, idx % board_size


def loc_2_idx(loc, board_size):
    """
    transfer a location value to a move value
    :param loc: an 1*2 dimension location value such as (2, 3)
    :param board_size: size of the state
    :return: an int type move value such as 34
    """
    return loc[0] * board_size + loc[1]


def switch_player(player_id):
    return 2 if player_id == 1 else 1
