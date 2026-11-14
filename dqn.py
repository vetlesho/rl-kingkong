import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import random
from collections import Counter
import numpy as np

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

class ReplayBuffer: 
    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        # uint8 keeps pixels as 0-255 whole numbers: 4x less memory than floats
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0  # where the next transition gets written
        self.size = 0      # how many transitions are stored so far

    def add(self, obs, action, reward, next_obs, done):
        i = self.position
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        # when full, wrap around and overwrite the oldest transition
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx], dtype=torch.float32, device=device),
            torch.tensor(self.actions[idx], device=device),
            torch.tensor(self.rewards[idx], device=device),
            torch.tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            torch.tensor(self.dones[idx], device=device),
        )

def train_step(q_net, target_net, optimizer, buffer, batch_size, gamma, device):
    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size, device)

    # 1. Prediction: Q-values for all 6 actions, then keep only the one for the action that was actually taken
    q_values = q_net(obs)                                         # shape (32, 6)
    q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # shape (32,)

    # 2. Target: reward + discounted best Q-value of the next screen
    with torch.no_grad():
        next_q = target_net(next_obs).max(dim=1).values
        target = rewards + gamma * next_q * (1 - dones)  # (1 - dones) removes the future if the game ended

    # 3. Loss: how far the prediction is from the target
    loss = nn.functional.smooth_l1_loss(q_pred, target)

    # 4. Nudge the weights to reduce the loss
    optimizer.zero_grad()   # clear the nudges from last time
    loss.backward()         # work out which direction to nudge each weight
    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)  # cap the nudge size for stability
    optimizer.step()        # apply the nudges

    return loss.item()

    
env = make_env()
obs, info = env.reset(seed=0)

#print("Observation shape:", obs.shape)
#print("Observation dtype:", obs.dtype)
#print("Number of actions:", env.action_space.n)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device: ", device)

q_net = QNetwork(env.action_space.n).to(device)

buffer = ReplayBuffer(capacity=50_000, obs_shape=env.observation_space.shape)

# Fill the buffer with 1000 random steps
obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    buffer.add(obs, action, np.sign(reward), next_obs, terminated)
    obs = next_obs
    if terminated or truncated:
        obs, info = env.reset()

print("Buffer size:", buffer.size)

target_net = QNetwork(env.action_space.n).to(device)
target_net.load_state_dict(q_net.state_dict())  # start as an exact copy

optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)

for i in range(200):
    loss = train_step(q_net, target_net, optimizer, buffer, 32, 0.99, device)
    if i % 20 == 0:
        print(f"train step {i}: loss = {loss:.5f}")