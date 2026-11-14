import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import random
from collections import Counter

gym.register_envs(ale_py)

def make_env(render_mode=None):
    env = gym.make("ALE/KingKong-v5", frameskip=1, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=4, screen_size=84, grayscale_obs=True
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env

class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            # 4 input channels = our 4 stacked frames
            nn.Conv2d(4, 32, kernel_size=8, stride=4),   # 84x84 -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 20x20 -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 9x9 -> 7x7
            nn.ReLU(),
            nn.Flatten(),                                # 64 * 7 * 7 = 3136 numbers
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),                   # one Q-value per action
        )

    def forward(self, x):
        # pixels are 0-255; networks learn better with inputs between 0 and 1
        return self.net(x / 255.0)

def get_epsilon(step, start=1.0, end=0.05, decay_steps=100_000):
    # slides linearly from start to end over decay_steps, then stays at end
    fraction = min(step / decay_steps, 1.0)
    return start + fraction * (end - start)

def select_action(q_net, obs, epsilon, n_actions, device):
    if random.random() < epsilon:
        return random.randrange(n_actions)  # explore

    with torch.no_grad(): # just asking the network, not training it
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(obs_tensor)
        return q_values.argmax(dim=1).item() # exploit

env = make_env()
obs, info = env.reset(seed=0)

#print("Observation shape:", obs.shape)
#print("Observation dtype:", obs.dtype)
#print("Number of actions:", env.action_space.n)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device: ", device)

q_net = QNetwork(env.action_space.n).to(device)

for step in [0, 50_000, 100_000, 200_000]:
    print(f"Epsilon at step {step}: {get_epsilon(step):.2f}")

# Play one episode at two different epsilons and count which actions get picked
for epsilon in [1.0, 0.05]:
    obs, info = env.reset()
    counts = Counter()
    done = False
    while not done:
        action = select_action(q_net, obs, epsilon, env.action_space.n, device)
        counts[action] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    print(f"epsilon={epsilon}: {sorted(counts.items())}")