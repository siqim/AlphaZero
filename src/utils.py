# -*- coding: utf-8 -*-

"""
Created on 2020/4/5

@author: Siqi Miao
"""


def move_2_loc(move, board_size):
    """
    transfer a move value to a location value
    :param move: an int type move value such as 34
    :param board_size: size of the state
    :return: an 1*2 dimension location value such as (2, 3)
    """
    return move // board_size, move % board_size


def loc_2_move(loc, board_size):
    """
    transfer a location value to a move value
    :param loc: an 1*2 dimension location value such as (2, 3)
    :param board_size: size of the state
    :return: an int type move value such as 34
    """
    return loc[0] * board_size + loc[1]
