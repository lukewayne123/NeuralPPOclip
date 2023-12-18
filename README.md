***Install process:***

- System: Using docker image pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

`docker(or podman) run -it pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime`

- Update system: 

`apt-get update`

`apt-get install gcc g++ cmake git vim zlib1g zlib1g-dev swig`

- python packages:

`pip install --upgrade pip`

`pip install -r requirements.txt`


***Training command:***

```
python train.py --algo {ALOGS} --classifier {CLASSIFIER} --env {GAMES} --seed {SEEDS} --tensorboard-log $PWD/tensorboards
```

- E.g.: For NeuralPPO (Clip-log) in game MiniAtar breakout with random seed 123

`python train.py --algo neuralppo --classifier Clip-log --env mini-breakout-v4 --seed 123 --tensorboard-log $PWD/tensorboards`

- Training command arguments:

`GAMES = {CartPole-v1, mini-space_invaders-v4, mini-breakout-v4, LunarLander-v2}`

`SEEDS = {123, 666, 987, 285, 517}`

`ALGO = {neuralppo, ppo, a2c}`

***Training log:***
Record for experiments.


