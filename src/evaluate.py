"""Play a number of games with one or more saved models and report how they do.

No training and no script: the model plays by itself, on the same games every time,
so the numbers can be compared between models.

    python3 src/evaluate.py                      # the two default models
    python3 src/evaluate.py models/some_model.pt # any model(s) you name
    python3 src/evaluate.py --games 10           # fewer games, quicker but noisier
"""
import argparse
import random
from collections import deque

import numpy as np
import torch

from dqn import make_env, QNetwork, select_action, get_floor

DEFAULT_MODELS = ["models/dqn_kingkong_clipped.pt", "models/dqn_kingkong_demo3.pt"]
EPSILON = 0.05      # a few random actions, so the model cannot get stuck in a loop
TOP_FLOOR = 8       # floors above this are glitches in the height byte during animations


def evaluate(path, games, device):
    random.seed(0)
    env = make_env()
    n_actions = env.action_space.n
    q_net = QNetwork(n_actions).to(device)
    q_net.load_state_dict(torch.load(path, map_location=device))

    scores = []
    lives_played = 0
    by_ladder = [0] * (TOP_FLOOR + 1)   # lives that first reached floor N by a ladder
    by_bomb = [0] * (TOP_FLOOR + 1)     # lives that first reached floor N by a magic bomb

    for game in range(games):
        obs, info = env.reset(seed=1000 + game)   # game number N is the same for every model
        lives = info["lives"]
        score = 0
        best_floor = 0                    # highest floor reached in this life
        recent_points = deque(maxlen=12)  # points from the last 12 steps
        done = False

        while not done:
            action = select_action(q_net, obs, EPSILON, n_actions, device)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            recent_points.append(reward)

            floor = get_floor(env)
            if best_floor < floor <= TOP_FLOOR:
                # a magic bomb pays 125 points just before it lifts the player
                if sum(recent_points) >= 125:
                    by_bomb[floor] += 1
                else:
                    by_ladder[floor] += 1
                best_floor = floor

            if info["lives"] < lives:
                lives_played += 1
                best_floor = 0
            lives = info["lives"]

        scores.append(score)
        print(f"  game {game + 1:>2}/{games}: score {score:>5.0f}", flush=True)

    env.close()

    scores = np.array(scores)
    # the standard error says how far the mean could be off by luck:
    # two models are only clearly different if their means are several errors apart
    error = scores.std(ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0

    print(f"\n{path}")
    print(f"  score: mean {scores.mean():.0f} (give or take {error:.0f}), "
          f"lowest {scores.min():.0f}, highest {scores.max():.0f}")
    print(f"  lives played: {lives_played}")
    print("  lives that reached each floor:")
    for floor in range(1, TOP_FLOOR + 1):
        total = by_ladder[floor] + by_bomb[floor]
        if total == 0:
            continue
        share = 100 * total / lives_played
        print(f"    floor {floor}: {total:>3} ({share:>3.0f}%)   "
              f"by ladder {by_ladder[floor]}, by magic bomb {by_bomb[floor]}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--games", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for path in args.models:
        print(f"Evaluating {path} over {args.games} games")
        evaluate(path, args.games, device)
