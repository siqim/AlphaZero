# 1. Introduction
This repo is aimed at implementing the AlphaZero algorithm on the Gomoku game. We hope the model we train would learn the way to beat human.

# 2. Requirements
The code is based on `Python 3.7.7` and you can create a virtual environment by the following:
```
conda create -n AlphaZero python=3.7.7
conda activate AlphaZero
pip install -r ./requirements.txt
pip install ./src/node
```

In `./src/node`, we implemented some computational intensive parts in c++ and used pybind11 to access them in Python. In this way, the MCTS process gets much faster.

# 3. TODO
There are too many engineering details in implementing AlphaZero. We would come around to this project later... Hope within half a year...
