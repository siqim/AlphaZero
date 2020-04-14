# -*- coding: utf-8 -*-

"""
Created on 2020/4/5

@author: Siqi Miao
"""


import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class BasicBlock(nn.Module):

    def __init__(self, in_channels, num_filters=256):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Model(nn.Module):

    def __init__(self, in_channels, num_filters=256, board_size=15, block=BasicBlock, num_blocks=19):
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.num_filters = num_filters
        self.board_size = board_size
        self.num_actions = self.board_size**2

        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=False)

        self.residual_layers = self.make_residual_layers(block, num_blocks)
        self.policy_head = self.make_policy_head(final_num_filters=2)
        self.value_head = self.make_value_head(final_num_filters=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.residual_layers(out)

        probs = self.policy_head(out)
        v = self.value_head(out)
        return probs, v

    def make_residual_layers(self, block, num_blocks):
        layers = [block(self.num_filters, self.num_filters) for _ in range(num_blocks)]
        return nn.Sequential(*layers)

    def make_policy_head(self, final_num_filters=2):
        policy_head = nn.Sequential(
            nn.Conv2d(self.num_filters, final_num_filters, kernel_size=1, stride=1, bias=False),
            nn.Flatten(1),
            nn.BatchNorm1d(final_num_filters*self.num_actions),
            nn.ReLU(inplace=False),
            nn.Linear(final_num_filters*self.num_actions, self.num_actions)
        )
        return policy_head

    def make_value_head(self, final_num_filters=1):
        value_head = nn.Sequential(
            nn.Conv2d(self.num_filters, final_num_filters, kernel_size=1, stride=1, bias=False),
            nn.Flatten(1),
            nn.BatchNorm1d(final_num_filters*self.num_actions),
            nn.ReLU(inplace=False),
            nn.Linear(final_num_filters*self.num_actions, self.num_filters),
            nn.ReLU(inplace=False),
            nn.Linear(self.num_filters, 1),
            nn.Tanh()
        )
        return value_head


if __name__ == '__main__':
    writer = SummaryWriter('../logs/model_test_0')

    in_channels = 3
    board_size = 15
    batch_size = 512
    num_filters = 128
    num_blocks = 5

    dummy_input = np.random.random((batch_size, in_channels, board_size, board_size)).astype(np.float32)
    dummy_input = torch.from_numpy(dummy_input).cuda()

    model = Model(in_channels, num_filters=num_filters, num_blocks=num_blocks)
    model.eval()
    model.cuda()
    cnt = 0
    while cnt <= 100:
        with torch.no_grad():
            tik = time.time()
            probs, v = model(dummy_input)
            tok = time.time()
            print(tok-tik)
            cnt += 1

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    torch.save(model.state_dict(), '../models/model.pth')

    writer.add_graph(model, dummy_input)
    writer.close()
