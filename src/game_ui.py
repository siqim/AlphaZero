# -*- coding: utf-8 -*-

"""
Created on 2020-04-02

@author: Siqi Miao
"""

# adapted from https://github.com/initial-h/AlphaZero_Gomoku_MPI/blob/master/GUI_v1_4.py


from board import Board
from itertools import product
import numpy as np
import pygame
from pygame.locals import *
import time


class GUI(object):

    def __init__(self, board_size=11):
        pygame.init()
        icon = pygame.image.load('../img/icon.jpg')
        pygame.display.set_icon(icon)

        self.board_size = board_size
        self.board = Board(self.board_size)

        self.score = [0, 0]
        self.unit_size = 50      # the basic size of all elements, try a different value!
        self.text_size = int(self.unit_size * 0.625)
        self.board_length = self.unit_size * self.board_size
        self.state = {}         # a dictionary for pieces on state. filled with move-player_id pairs, such as 34:1
        self.areas = {}         # a dictionary for button areas. filled with name-Rect pairs
        self.screen_size = None  # save the screen size for some calculation
        self.screen = None
        self.last_action_player = None
        self.round_counter = 0
        self.messages = ''
        self._background_color = (212, 212, 212)
        self._board_color = (245, 185, 127)

        self.reset()

        # restart_game() must be called before reset_score() because restart_game() will add value to self.round_counter
        self.restart_game()
        self.reset_score()

    def reset(self):

        self.screen_size = (self.board_size * self.unit_size + 2 * self.unit_size,
                            self.board_size * self.unit_size + 3 * self.unit_size)
        self.screen = pygame.display.set_mode(self.screen_size, 0, 32)
        pygame.display.set_caption('AlphaZero_Gomoku')

        # button areas
        self.areas['ResetScore'] = Rect(0, self.screen_size[1] - self.unit_size,
                                        self.unit_size * 2.5, self.unit_size)

        self.areas['Restart'] = Rect(self.unit_size * 2.6, self.screen_size[1] - self.unit_size,
                                     self.unit_size * 2.5, self.unit_size)

        self.areas['Man vs Man'] = Rect(self.unit_size * 5.2, self.screen_size[1] - self.unit_size,
                                        self.unit_size * 2.6, self.unit_size)

        self.areas['Man vs AI'] = Rect(self.unit_size * 7.9, self.screen_size[1] - self.unit_size,
                                       self.unit_size * 2.5, self.unit_size)

        self.areas['AI vs AI'] = Rect(self.unit_size * 10.5, self.screen_size[1] - self.unit_size,
                                      self.unit_size * 2.5, self.unit_size)

        self.areas['state'] = Rect(self.unit_size, self.unit_size, self.board_length, self.board_length)

    def restart_game(self, highlight_button=None):
        """
        restart for a new round
        """
        self.board.reset_board()
        self.round_counter += 1
        self._draw_static()
        if highlight_button is not None:
            self._draw_button(highlight_button, 2, True)

        self.state = {}
        self.last_action_player = None
        pygame.display.update()

    def reset_score(self):
        """
        reset score and round
        """
        self.score = [0, 0]
        self.round_counter = 1
        self.show_messages()

    def add_score(self, player_id):
        """
        add score for winner
        :param player_id: the name of the winner
        """
        if player_id == 1:
            self.score[0] += 1
        elif player_id == 2:
            self.score[1] += 1
        else:
            raise ValueError('player_id number error')
        self.show_messages()

    def render_step(self, action, player_id):
        """
        render a step of the game
        :param action: 1*2 dimension location value such as (2, 3) or an int type move value such as 34
        :param player_id: the name of the player
        """

        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

        if self.last_action_player:     # draw a cross on the last piece to mark the last move
            self._draw_pieces(self.last_action_player[0], self.last_action_player[1], False)

        self.board.get_new_state(action, player_id)
        self._draw_pieces(action, player_id, True)
        self.state[action] = player_id
        self.last_action_player = action, player_id

    def get_input(self):
        """
        get inputs from clicks
        :return: variable-length array.[0] is the name. Additional information behind (maybe not exist).
        """
        while True:
            event = pygame.event.wait()
            if event.type == QUIT:
                return 'quit',

            if event.type == MOUSEBUTTONDOWN:   # check mouse click event
                if event.button == 1:
                    mouse_pos = event.pos

                    for name, rec in self.areas.items():
                        if self._in_area(mouse_pos, rec):
                            if name != 'state':
                                return name,

                            else:
                                x = (mouse_pos[1] - self.unit_size) // self.unit_size
                                y = (mouse_pos[0] - self.unit_size) // self.unit_size
                                action = (x, y)
                                if not self.board.winning_flag and (x, y) not in self.state:
                                    return 'move', action

    def show_messages(self, messages=None):
        """
        show extra messages on screen
        :param messages:
        :return:
        """
        if messages:
            self.messages = messages
        pygame.draw.rect(self.screen, self._background_color, (0, self.screen_size[1] - self.unit_size * 2,
                                                               self.screen_size[0], self.unit_size))
        self._draw_round(False)
        self._draw_text(self.messages, (self.screen_size[0] / 2, self.screen_size[1] - self.unit_size * 1.5),
                        text_height=self.text_size)
        self._draw_score()

    def _draw_score(self, update=True):
        score = 'Score: ' + str(self.score[0]) + ' : ' + str(self.score[1])
        self._draw_text(score, (self.screen_size[0] * 0.11, self.screen_size[1] - self.unit_size * 1.5),
                        background_color=self._background_color, text_height=self.text_size)
        if update:
            pygame.display.update()

    def _draw_round(self, update=True):
        self._draw_text('Round: ' + str(self.round_counter),
                        (self.screen_size[0] * 0.88, self.screen_size[1] - self.unit_size * 1.5),
                        background_color=self._background_color, text_height=self.text_size)
        if update:
            pygame.display.update()

    def _draw_pieces(self, loc, player, last_step=False):
        """
        draw pieces
        :param loc:  1*2 dimension location value such as (2, 3) or an int type move value such as 34
        :param player: the name of the player_id
        :param last_step: whether it is the last step
        """

        x, y = loc
        pos = int(self.unit_size * 1.5 + y * self.unit_size), \
            int(self.unit_size * 1.5 + x * self.unit_size)
        if player == 1:
            c = (0, 0, 0)
        elif player == 2:
            c = (255, 255, 255)
        else:
            raise ValueError('num input ValueError')
        pygame.draw.circle(self.screen, c, pos, int(self.unit_size * 0.45))
        if last_step:
            if player == 1:
                c = (255, 255, 255)
            elif player == 2:
                c = (0, 0, 0)

            start_p1 = pos[0] - self.unit_size * 0.3, pos[1]
            end_p1 = pos[0] + self.unit_size * 0.3, pos[1]
            pygame.draw.line(self.screen, c, start_p1, end_p1)

            start_p2 = pos[0], pos[1] - self.unit_size * 0.3
            end_p2 = pos[0], pos[1] + self.unit_size * 0.3
            pygame.draw.line(self.screen, c, start_p2, end_p2)

    def _draw_static(self):
        """
        Draw static elements that will not change in a round.
        """
        # draw background
        self.screen.fill(self._background_color)
        # draw state
        pygame.draw.rect(self.screen, self._board_color, self.areas['state'])
        for i in range(self.board_size):
            # draw grid lines
            start = self.unit_size * (i + 0.5)
            pygame.draw.line(self.screen, (0, 0, 0), (start + self.unit_size, self.unit_size * 1.5),
                             (start + self.unit_size, self.board_length + self.unit_size * 0.5))
            pygame.draw.line(self.screen, (0, 0, 0), (self.unit_size * 1.5, start + self.unit_size),
                             (self.board_length + self.unit_size * 0.5, start + self.unit_size))
            pygame.draw.rect(self.screen, (0, 0, 0), (self.unit_size, self.unit_size,
                                                      self.board_length, self.board_length), 1)
            # coordinate values
            self._draw_text(i, (self.unit_size / 2, start + self.unit_size),
                            text_height=self.text_size)  # vertical
            self._draw_text(i, (start + self.unit_size, self.unit_size / 2), text_height=self.text_size)  # horizontal

        # draw buttons
        for name in self.areas.keys():
            if name != 'state':
                self._draw_button(name)

        self.show_messages()

    def _draw_text(self, text, position, text_height=25, font_color=(0, 0, 0), background_color=None, pos='center',
                   angle=0):
        """
        draw text
        :param text: a string type text
        :param position: the location point
        :param text_height: text height
        :param font_color: font color
        :param background_color: background color
        :param pos: the location point is where in the text rectangle.
        'center','top','bottom','left','right'and their combination such as 'topleft' can be selected
        :param angle: the rotation angle of the text
        """
        posx, posy = position
        font_obj = pygame.font.Font(None, int(text_height))
        text_surface_obj = font_obj.render(str(text), True, font_color, background_color)
        text_surface_obj = pygame.transform.rotate(text_surface_obj, angle)
        text_rect_obj = text_surface_obj.get_rect()
        if pos == 'center':
            text_rect_obj.center = (posx, posy)
        self.screen.blit(text_surface_obj, text_rect_obj)

    def _draw_button(self, name, high_light=0, update=False):
        rec = self.areas[name]
        if not high_light:
            button_color = (225, 225, 225)
        elif high_light == 1:
            button_color = (245, 245, 245)
        elif high_light == 2:
            button_color = (255, 255, 255)
        else:
            raise ValueError('high_light value error')
        pygame.draw.rect(self.screen, button_color, rec)
        pygame.draw.rect(self.screen, (0, 0, 0), rec, 1)
        self._draw_text(name, rec.center, text_height=self.text_size)
        if update:
            pygame.display.update()

    @staticmethod
    def _in_area(loc, area):
        """
        check whether the location is in area
        :param loc: a 1*2 dimension location value such as (123, 45)
        :param area: a Rect type value in pygame
        """
        return True if area[0] < loc[0] < area[0] + area[2] and area[1] < loc[1] < area[1] + area[3] else False


states = list(product(range(11), range(11)))


def get_ai_action():
    time.sleep(0.1)
    idx = np.random.choice(range(len(states)))
    action = states[idx]
    states.pop(idx)
    return action


def process_one_move(mode, click, player_id, ui, ai_first):
    if mode == 'Man vs Man':
        action = click[1]
        ui.render_step(action, player_id)

    elif mode == 'AI vs AI':
        action = get_ai_action()
        ui.render_step(action, player_id)

    elif mode == 'Man vs AI':
        if ai_first:
            if player_id == 1:
                action = get_ai_action()
                ui.render_step(action, player_id)
            else:
                action = click[1]
                ui.render_step(action, player_id)

        else:
            if player_id == 1:
                action = click[1]
                ui.render_step(action, player_id)
            else:
                action = get_ai_action()
                ui.render_step(action, player_id)

    else:
        raise ValueError("Unknown mode!!!")

    return action


def main():
    board_size = 11
    ui = GUI(board_size)
    player_id = 1
    ai_first = False
    mode = 'Man vs Man'
    while True:
        first_player, second_player = mode.split(' vs ')
        if not ui.board.winning_flag and player_id == 1:
            ui.show_messages("First Player [{player}] is thinking...".format(player=first_player))
        elif not ui.board.winning_flag and player_id == 2:
            ui.show_messages("Second Player [{player}] is thinking...".format(player=second_player))
        elif ui.board.winning_flag and player_id == 1:
            ui.show_messages("First Player [{player}] has won!!!".format(player=first_player))
        elif ui.board.winning_flag and player_id == 2:
            ui.show_messages("Second Player [{player}] has won!!!".format(player=second_player))

        if not ui.board.winning_flag and mode == 'AI vs AI' \
                or ((mode == 'Man vs AI') and ((ai_first and player_id == 1) or (not ai_first and player_id == 2))):
            click = 'move',
        else:
            click = ui.get_input()

        if click[0] == 'quit':
            exit()
        elif click[0] == 'Restart':
            ui.restart_game()
        elif click[0] == 'ResetScore':
            ui.reset_score()
        elif click[0] in ('Man vs Man', 'Man vs AI', 'AI vs AI'):
            player_id = 1
            ui.restart_game(click[0])
            ui.reset_score()
            mode = click[0]

        elif click[0] == 'move':
            action = process_one_move(mode, click, player_id, ui, ai_first)
            print(action)
            if ui.board.winning_flag:
                ui.add_score(player_id)
            else:
                player_id %= 2
                player_id += 1
        else:
            raise ValueError('Unknown inputs!!!')


if __name__ == '__main__':
    main()
