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

| File | What it does |
| --- | --- |
| `main.py` | Plays one game with random actions. Good for checking that the install works. |
| `dqn.py` | The agent and the training loop. |
| `watch.py` | Loads a trained model and plays a game in a window. |
| `plot.py` | Plots the average score from one or more training logs. |

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt
```

Train, and keep a log of the run:

```bash
python -u dqn.py | tee log.txt
```

Three million steps take about two hours on a MacBook Air. The model is saved to `dqn_kingkong.pt` every 50,000 steps, so you can watch it or plot the log while training is still going:

```bash
python watch.py
python plot.py log.txt
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

Two choices are specific to this game. Losing a life is stored as the end of an episode, although the game keeps going. Rewards are divided by 100, so a magic bomb and the top bonus stay worth more than an ordinary bomb.

## Results so far

Random play scores about 100 points per game.

| Run | Average over 100 games | What the agent does |
| --- | --- | --- |
| 500,000 steps, rewards clipped to +1 | about 100 | No better than random. |
| 3,000,000 steps, rewards clipped to +1, life loss as episode end | about 410 | Stands on the ground floor and jumps bombs. It never climbs. |
| 3,000,000 steps, rewards divided by 100 | in progress | |

With clipped rewards the top bonus looked the same to the agent as a single bomb, which gave it no reason to climb. The current run tests whether scaled rewards change that. Raising gamma is next on the list, since at 0.99 the agent only looks about 7 seconds ahead.

## Sources

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (Mnih et al., 2015)
- [Gymnasium documentation](https://gymnasium.farama.org/)
- [Arcade Learning Environment: King Kong](https://ale.farama.org/environments/king_kong/), which is also where the animation comes from
