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
| `models/dqn_kingkong_clipped.pt` | The highest-scoring model. It stays low in the building. |
| `models/dqn_kingkong_demo3.pt` | The best all-rounder: scores almost as much and climbs higher. |
| `models/dqn_kingkong_ladder1.pt`, `models/dqn_kingkong_demo2.pt` | Models from the earlier demonstration runs. |
| `models/dqn_kingkong.pt`, `models/dqn_kingkong_best.pt` | The last and the best checkpoint of the latest run. Each new run overwrites them. |

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

Two million steps take a bit over two hours on a MacBook Air. The model is saved to `models/dqn_kingkong.pt` every 50,000 steps, so you can watch it or plot the log while training is still going. The checkpoint with the best 100-game average goes to `models/dqn_kingkong_best.pt`.

```bash
python src/watch.py
python src/plot.py logs/log.txt
```

## The agent

The agent is a plain DQN, close to the one in the original DeepMind papers.

The screen is turned into 84×84 grayscale, each action is held for 4 frames, and the last 4 frames are stacked so the network can see movement. The network has three convolution layers and two linear layers, and outputs one Q-value per action.

| Setting | Value |
| --- | --- |
| Training steps | 2,000,000, starting from an earlier model |
| Learning rate | 0.0001 (Adam) |
| Gamma | 0.99 |
| Batch size | 32 |
| Replay buffer | 100,000 transitions |
| Epsilon | 0.1 when fine-tuning (1.0 down to 0.05 over 500,000 steps when training from scratch) |
| Target network update | every 1,000 steps |
| Training frequency | one update per 4 steps, after 10,000 warm-up steps |

Some choices are specific to this game:

- Losing a life is stored as the end of an episode, although the game keeps going.
- Rewards are clipped to +1, so a bomb, a magic bomb and reaching the top all count the same.
- The agent gets a bonus of 2 the first time it reaches each floor in a life. The player's height is read from byte 33 of the game's memory.
- In half of the lives a small script takes over when the agent arrives on the first floor, walks to the middle ladder and climbs it. The agent learns from those steps like any others, since DQN can learn from actions it did not choose itself. The script finds the player from the grey pixels of his shoes.

The log prints two numbers per game: `score` is the game's own points, and `learn reward` is the sum of what the agent was trained on.

## Results so far

Random play scores about 100 points per game. The first five runs are judged by the average score over the last 100 training games.

| Run | Average score | What the agent does |
| --- | --- | --- |
| 500,000 steps, rewards clipped to +1 | about 100 | No better than random. |
| 3,000,000 steps, rewards clipped to +1, life loss as episode end | about 410 | Jumps bombs on the ground floor. Magic bombs sometimes lift it to the first floor. |
| 1,100,000 steps (stopped), rewards divided by 100 | about 65 | Worse than random. |
| 850,000 steps (stopped), floor bonus paid on every step by mistake | about 15 | Reaches the first floor and stops jumping. |
| 3,000,000 steps, rewards divided by 25, floor bonus once per floor per life | about 55 | Stands still on the ground floor until a bomb hits it. |

The later runs all continue from an earlier model with clipped rewards and the floor bonus. The training log is misleading for these, because it also counts the lives where a script helped. So I played 15 games with each model, with no script, and counted how often it reached each floor in its 45 lives.

| Model | Score | Floor 1 | Floor 2 | Floor 3 | Floor 4 |
| --- | --- | --- | --- | --- | --- |
| Clipped rewards only (`dqn_kingkong_clipped.pt`) | 427 | 24 | 4 | 0 | 0 |
| A script climbs the middle ladder in half the lives (`dqn_kingkong_demo3.pt`) | 378 | 43 | 29 | 3 | 2 |

What I learned along the way:

- The floor bonus alone changed nothing in 6,000,000 steps. Pressing up only works when the player is lined up with a ladder almost to the pixel, so random actions never climb one, and the agent had no climb to learn from.
- Showing it the climb worked for the first ladder. After a script climbed the ground-floor ladder in 30% of the lives, the agent took that ladder by itself in almost every life.
- It did not work for the middle ladder on the first floor, in three runs. The walk there is about 100 steps with no reward on the way. The agent gets above the first floor by jumping magic bombs.
- Every run peaked and then slid back, so the best checkpoint is better than the last one.

Next I want to try Double DQN and a lower learning rate, to see if that stops the slide.

## Sources

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (Mnih et al., 2015)
- [Gymnasium documentation](https://gymnasium.farama.org/)
- [Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732) (Hester et al., 2017)
- [Arcade Learning Environment: King Kong](https://ale.farama.org/environments/king_kong/), which is also where the animation comes from
