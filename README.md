# rl-kingkong

![King Kong on the Atari 2600](https://ale.farama.org/_images/king_kong.gif)

A DQN agent that learns to play the Atari 2600 game King Kong. I wrote it from scratch in PyTorch to learn reinforcement learning, with [Gymnasium](https://gymnasium.farama.org/) running the game.

## The game

You climb a building to rescue the girl King Kong has carried to the top, while he throws bombs down at you.

| Event | Points |
| --- | --- |
| Jump over a bomb | 25 |
| Jump over a magic bomb (also lifts you one floor) | 125 |
| Reach the top | whatever is left of the bonus |

The bonus starts at 990 and drops 10 points per second. If it reaches zero you lose a life, and you have three.

The environment is [`ALE/KingKong-v5`](https://ale.farama.org/environments/king_kong/), which has 6 actions.

## What's in the repo

| Path | What it is |
| --- | --- |
| `src/main.py` | Plays one game with random actions. Good for checking that the install works. |
| `src/dqn.py` | The agent and the training loop. |
| `src/watch.py` | Loads a trained model and plays a game in a window. |
| `src/plot.py` | Plots the average score from one or more training logs. |
| `src/find_height.py` | Lists the bytes of game memory that could hold the player's height. |
| `logs/` | Training logs, one file per run. |
| `models/dqn_kingkong_clipped.pt` | The best model so far. |
| `models/dqn_kingkong.pt` | The model from the latest run. |

Run everything from the repo root, since the paths to `models/` and `logs/` are relative to it.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt
```

Train, and keep a log of the run:

```bash
python -u src/dqn.py | tee logs/log.txt
```

Three million steps take about two hours on a MacBook Air. The model is saved to `models/dqn_kingkong.pt` every 50,000 steps, so you can watch it or plot the log while training is still going:

```bash
python src/watch.py
python src/plot.py logs/log.txt
```

## The agent

The agent is a plain DQN, close to the one in the original DeepMind papers.

The screen is turned into 84×84 grayscale, each action is held for 4 frames, and the last 4 frames are stacked so the network can see movement. The network has three convolution layers and two linear layers, and outputs one Q-value per action.

| Setting | Value |
| --- | --- |
| Training steps | 3,000,000 |
| Learning rate | 0.0001 (Adam) |
| Gamma | 0.99 |
| Batch size | 32 |
| Replay buffer | 50,000 transitions |
| Epsilon | 1.0 down to 0.05 over the first 500,000 steps |
| Target network update | every 1,000 steps |
| Training frequency | one update per 4 steps, after 10,000 warm-up steps |

Three choices are specific to this game. Losing a life is stored as the end of an episode, although the game keeps going. Rewards are divided by 25, so a bomb is worth 1 and a magic bomb 5. The agent also gets a bonus of 2 the first time it reaches each floor in a life. The player's height is read from byte 33 of the game's memory.

## Results so far

Random play scores about 100 points per game.

| Run | Average over the last 100 games | What the agent does |
| --- | --- | --- |
| 500,000 steps, rewards clipped to +1 | about 100 | No better than random. |
| 3,000,000 steps, rewards clipped to +1, life loss as episode end | about 410 | Climbs to the first floor and jumps bombs. It never gets to the second floor. |
| 1,100,000 steps (stopped), rewards divided by 100 | about 65 | Worse than random. |
| 850,000 steps (stopped), floor bonus paid on every step by mistake | about 15 | Climbs to the first floor and stops jumping. |
| 3,000,000 steps, rewards divided by 25, floor bonus once per floor per life | about 55 | Stands still on the ground floor until a bomb hits it. |

The clipped run is still the best one. Every run with unclipped rewards dropped below random once epsilon reached 0.05 and never recovered, though with one run per setting some of that may be luck.

The ladders turned out to be the hard part: pressing up only works when the player is lined up with a ladder almost to the pixel, so random play rarely climbs.

Next I want to keep training the clipped model with clipped rewards plus the floor bonus, to see whether it gets past the first floor.

## Sources

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (Mnih et al., 2015)
- [Gymnasium documentation](https://gymnasium.farama.org/)
- [Arcade Learning Environment: King Kong](https://ale.farama.org/environments/king_kong/), which is also where the animation comes from
