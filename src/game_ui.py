# -*- coding: utf-8 -*-

"""
Created on 2020-04-02

@author: Siqi Miao
"""

# adapted from https://github.com/initial-h/AlphaZero_Gomoku_MPI/blob/master/GUI_v1_4.py


from itertools import product
import numpy as np
import pygame
from pygame.locals import *


class GUI(object):

    def __init__(self, board_size=11):
        pygame.init()

        self.score = [0, 0]
        self.BoardSize = board_size
        self.UnitSize = 50      # the basic size of all elements, try a different value!
        self.TextSize = int(self.UnitSize * 0.625)
        self.state = {}         # a dictionary for pieces on board. filled with move-player pairs, such as 34:1
        self.areas = {}         # a dictionary for button areas. filled with name-Rect pairs
        self.ScreenSize = None  # save the screen size for some calculation
        self.screen = None
        self.last_action_player = None
        self.round_counter = 0
        self.messages = ''
        self._background_color = (212, 212, 212)
        self._board_color = (245, 185, 127)

        self.reset(board_size)

        # restart_game() must be called before reset_score() because restart_game() will add value to self.round_counter
        self.restart_game()
        self.reset_score()

    def reset(self, bs):
        """
        reset screen
        :param bs: board size
        """

        # # you can add limits for board size
        # bs = int(bs)
        # if bs < 5:
        #     raise ValueError('board size too small')

        self.BoardSize = bs
        self.ScreenSize = (self.BoardSize * self.UnitSize + 2 * self.UnitSize,
                           self.BoardSize * self.UnitSize + 3 * self.UnitSize)
        self.screen = pygame.display.set_mode(self.ScreenSize, 0, 32)
        pygame.display.set_caption('AlphaZero_Gomoku')

        # button areas
        self.areas['ResetScore'] = Rect(0, self.ScreenSize[1] - self.UnitSize,
                                        self.UnitSize*2.5, self.UnitSize)

        self.areas['Restart'] = Rect(self.UnitSize*2.6, self.ScreenSize[1] - self.UnitSize,
                                     self.UnitSize*2.5, self.UnitSize)

        self.areas['Man vs Man'] = Rect(self.UnitSize*5.2, self.ScreenSize[1] - self.UnitSize,
                                        self.UnitSize*2.6, self.UnitSize)

        self.areas['Man vs AI'] = Rect(self.UnitSize*7.9, self.ScreenSize[1] - self.UnitSize,
                                       self.UnitSize*2.5, self.UnitSize)

        self.areas['AI vs AI'] = Rect(self.UnitSize*10.5, self.ScreenSize[1] - self.UnitSize,
                                      self.UnitSize*2.5, self.UnitSize)

        board_lenth = self.UnitSize * self.BoardSize
        self.areas['board'] = Rect(self.UnitSize, self.UnitSize, board_lenth, board_lenth)

    def restart_game(self, highlight_button=None):
        """
        restart for a new round
        """
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

    def add_score(self, winner):
        """
        add score for winner
        :param winner: the name of the winner
        """
        if winner == 1:
            self.score[0] += 1
        elif winner == 2:
            self.score[1] += 1
        else:
            raise ValueError('player number error')
        self.show_messages()

    def render_step(self, action, player):
        """
        render a step of the game
        :param action: 1*2 dimension location value such as (2, 3) or an int type move value such as 34
        :param player: the name of the player
        """
        try:
            action = int(action)
        except Exception:
            pass
        if type(action) != int:
            move = self.loc_2_move(action)
        else:
            move = action

        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

        if self.last_action_player:     # draw a cross on the last piece to mark the last move
            self._draw_pieces(self.last_action_player[0], self.last_action_player[1], False)

        self._draw_pieces(action, player, True)
        self.state[move] = player
        self.last_action_player = move, player

    def move_2_loc(self, move):
        """
        transfer a move value to a location value
        :param move: an int type move value such as 34
        :return: an 1*2 dimension location value such as (2, 3)
        """
        return move % self.BoardSize, move // self.BoardSize

    def loc_2_move(self, loc):
        """
        transfer a location value to a move value
        :param loc: an 1*2 dimension location value such as (2, 3)
        :return: an int type move value such as 34
        """
        return loc[0] + loc[1] * self.BoardSize

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
                            if name != 'board':
                                return name,

                            else:
                                x = (mouse_pos[0] - self.UnitSize)//self.UnitSize
                                y = (mouse_pos[1] - self.UnitSize)//self.UnitSize
                                move = self.loc_2_move((x, y))
                                if move not in self.state:
                                    return 'move', move

    def show_messages(self, messages=None):
        """
        show extra messages on screen
        :param messages:
        :return:
        """
        if messages:
            self.messages = messages
        pygame.draw.rect(self.screen, self._background_color, (0, self.ScreenSize[1] - self.UnitSize*2,
                                                               self.ScreenSize[0], self.UnitSize))
        self._draw_round(False)
        self._draw_text(self.messages, (self.ScreenSize[0]/2, self.ScreenSize[1]-self.UnitSize*1.5),
                        text_height=self.TextSize)
        self._draw_score()

    def _draw_score(self, update=True):
        score = 'Score: ' + str(self.score[0]) + ' : ' + str(self.score[1])
        self._draw_text(score, (self.ScreenSize[0] * 0.11, self.ScreenSize[1] - self.UnitSize*1.5),
                        backgroud_color=self._background_color, text_height=self.TextSize)
        if update:
            pygame.display.update()

    def _draw_round(self, update=True):
        self._draw_text('Round: ' + str(self.round_counter),
                        (self.ScreenSize[0]*0.88, self.ScreenSize[1] - self.UnitSize*1.5),
                        backgroud_color=self._background_color, text_height=self.TextSize)
        if update:
            pygame.display.update()

    def _draw_pieces(self, loc, player, last_step=False):
        """
        draw pieces
        :param loc:  1*2 dimension location value such as (2, 3) or an int type move value such as 34
        :param player: the name of the player
        :param last_step: whether it is the last step
        """
        try:
            loc = int(loc)
        except Exception:
            pass

        if type(loc) is int:
            x, y = self.move_2_loc(loc)
        else:
            x, y = loc
        pos = int(self.UnitSize * 1.5 + x * self.UnitSize), \
            int(self.UnitSize * 1.5 + y * self.UnitSize)
        if player == 1:
            c = (0, 0, 0)
        elif player == 2:
            c = (255, 255, 255)
        else:
            raise ValueError('num input ValueError')
        pygame.draw.circle(self.screen, c, pos, int(self.UnitSize * 0.45))
        if last_step:
            if player == 1:
                c = (255, 255, 255)
            elif player == 2:
                c = (0, 0, 0)

            start_p1 = pos[0] - self.UnitSize * 0.3, pos[1]
            end_p1 = pos[0] + self.UnitSize * 0.3, pos[1]
            pygame.draw.line(self.screen, c, start_p1, end_p1)

            start_p2 = pos[0], pos[1] - self.UnitSize * 0.3
            end_p2 = pos[0], pos[1] + self.UnitSize * 0.3
            pygame.draw.line(self.screen, c, start_p2, end_p2)

    def _draw_static(self):
        """
        Draw static elements that will not change in a round.
        """
        # draw background
        self.screen.fill(self._background_color)
        # draw board
        board_lenth = self.UnitSize * self.BoardSize
        pygame.draw.rect(self.screen, self._board_color, self.areas['board'])
        for i in range(self.BoardSize):
            # draw grid lines
            start = self.UnitSize * (i + 0.5)
            pygame.draw.line(self.screen, (0, 0, 0), (start + self.UnitSize, self.UnitSize*1.5),
                             (start + self.UnitSize, board_lenth + self.UnitSize*0.5))
            pygame.draw.line(self.screen, (0, 0, 0), (self.UnitSize*1.5, start + self.UnitSize),
                             (board_lenth + self.UnitSize*0.5, start + self.UnitSize))
            pygame.draw.rect(self.screen, (0, 0, 0), (self.UnitSize, self.UnitSize, board_lenth, board_lenth), 1)
            # coordinate values
            self._draw_text(i, (self.UnitSize / 2, start + self.UnitSize),
                            text_height=self.TextSize)  # vertical
            self._draw_text(i, (start + self.UnitSize, self.UnitSize / 2), text_height=self.TextSize)  # horizontal

        # draw buttons
        for name in self.areas.keys():
            if name != 'board':
                self._draw_button(name)

        self.show_messages()

    def _draw_text(self, text, position, text_height=25, font_color=(0, 0, 0), backgroud_color=None, pos='center',
                   angle=0):
        """
        draw text
        :param text: a string type text
        :param position: the location point
        :param text_height: text height
        :param font_color: font color
        :param backgroud_color: background color
        :param pos: the location point is where in the text rectangle.
        'center','top','bottom','left','right'and their combination such as 'topleft' can be selected
        :param angle: the rotation angle of the text
        """
        posx, posy = position
        font_obj = pygame.font.Font(None, int(text_height))
        text_surface_obj = font_obj.render(str(text), True, font_color, backgroud_color)
        text_surface_obj = pygame.transform.rotate(text_surface_obj, angle)
        text_rect_obj = text_surface_obj.get_rect()
        exec('text_rect_obj.' + pos + ' = (posx, posy)')
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
        self._draw_text(name, rec.center, text_height=self.TextSize)
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

    print(action)


def main():

    ui = GUI()
    player_id = 1
    ai_first = False
    mode = 'Man vs Man'
    while True:
        if player_id == 1:
            ui.show_messages("First player's turn")
        else:
            ui.show_messages("Second player's turn")

        if mode == 'AI vs AI' or ((mode == 'Man vs AI')
                                  and ((ai_first and player_id == 1)
                                       or (not ai_first and player_id == 2))):
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
            process_one_move(mode, click, player_id, ui, ai_first)
            player_id %= 2
            player_id += 1

        else:
            raise ValueError('Unknown inputs!!!')


if __name__ == '__main__':
    main()
