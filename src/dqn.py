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
    #env = gym.make("ALE/Pong-v5", frameskip=1, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=4, screen_size=84, grayscale_obs=True
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env

def get_floor(env):
    # byte 33 of the game's memory is the player's height:
    # 94 on the ground, 12 less for every floor climbed
    y = int(env.unwrapped.ale.getRAM()[33])
    return (94 - y) // 12

NOOP, FIRE, UP, RIGHT, LEFT, DOWN = range(6)

def new_demo():
    # the script's memory: which way it walks, and a step counter
    #return {"direction": random.choice([LEFT, RIGHT]), "t": 0}
    return {"direction": random.choice([LEFT, RIGHT]), "t": 0, "floor": 0}

def demo_action(env, demo):
    # a simple script that finds a ladder and climbs it
    y = int(env.unwrapped.ale.getRAM()[33])
    if y > 94:
        return NOOP                  # intro or death animation
    if (94 - y) % 12 != 0:
        return UP                    # between two floors = on a ladder
    floor = (94 - y) // 12
    if floor != demo["floor"]:       # just arrived on a new floor: turn around
        demo["floor"] = floor
        demo["t"] = 0
        demo["direction"] = LEFT if demo["direction"] == RIGHT else RIGHT
    demo["t"] += 1
    if demo["t"] % 200 == 0:         # no ladder this way, turn around
        demo["direction"] = LEFT if demo["direction"] == RIGHT else RIGHT
    if demo["t"] % 3 == 0:
        return UP                    # try the ladder
    return demo["direction"]

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

def get_epsilon(step, start=1.0, end=0.05, decay_steps=500_000):
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



if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    TOTAL_STEPS = 6_000_000
    LEARNING_STARTS = 10_000        # collect this many steps before training begins
    TRAIN_EVERY = 4                 # one training step per 4 game steps
    TARGET_UPDATE_EVERY = 1_000     # how often the target network is refreshed
    BATCH_SIZE = 32
    GAMMA = 0.99
    FLOOR_BONUS = 2.0               # extra reward for reaching a new highest floor in this life
    DEMO_PROB = 0.3                 # share of lives where the script climbs the first ladder

    env = make_env()
    n_actions = env.action_space.n
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    q_net = QNetwork(n_actions).to(device)
    q_net.load_state_dict(torch.load("models/dqn_kingkong_ladder1.pt", map_location=device))
    target_net = QNetwork(n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)
    buffer = ReplayBuffer(capacity=100_000, obs_shape=env.observation_space.shape)

    obs, info = env.reset(seed=0)
    lives = info["lives"]        # how many lives we have right now
    print("Starting lives:", lives) 
    best_floor = 0               # highest floor reached in this life
    demo = new_demo() if random.random() < DEMO_PROB else None
    episode_reward = 0           # the game score in this game
    episode_learn_reward = 0     # the reward the agent learns from, summed over this game
    episode_rewards = []
    episode_top_floor = 0        # highest floor reached in this game
    best_average = 0             # best avg(100) so far, for saving the best model

    for step in range(TOTAL_STEPS):
        # 1-3: act, step the game, remember what happened
        #epsilon = get_epsilon(step)    # when training new models
        epsilon = 0.1                   # retrain from clipped model, giving some chance to randomness
        #action = select_action(q_net, obs, epsilon, n_actions, device)
        if demo is not None:
            action = demo_action(env, demo)
        else:
            action = select_action(q_net, obs, epsilon, n_actions, device)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # did this step cost a life?
        life_lost = info["lives"] < lives
        lives = info["lives"]
        
        #shaped_reward = reward / 25    # old shaped reward
        shaped_reward = np.sign(reward)
        floor = get_floor(env)
        episode_top_floor = max(episode_top_floor, floor)
        if floor > best_floor:
            shaped_reward += FLOOR_BONUS
            best_floor = floor
            if best_floor >= 2:
                demo = None    # the agent takes over
        if life_lost:
            best_floor = 0  # the player restarts on the ground
            demo = new_demo() if random.random() < DEMO_PROB else None

        buffer.add(obs, action, shaped_reward, next_obs, terminated or life_lost)
        obs = next_obs
        episode_reward += reward  # the real score, not the clipped one
        episode_learn_reward += shaped_reward

        if terminated or truncated:
            episode_rewards.append(episode_reward)
            average = np.mean(episode_rewards[-20:])
            average_100 = np.mean(episode_rewards[-100:])
            print(
                f"step {step:>7} | episode {len(episode_rewards):>4} | "
                f"score {episode_reward:>6.0f} | learn reward {episode_learn_reward:>5.1f} | "
                f"avg(20) {average:>7.1f} | "
                f"avg(100) {average_100:>7.1f} | "
                f"top floor {episode_top_floor} | "
                f"epsilon {epsilon:.2f}"
                
            )
            if len(episode_rewards) >= 100 and average_100 > best_average:
                best_average = average_100
                torch.save(q_net.state_dict(), "models/dqn_kingkong_best.pt")

            episode_reward = 0
            episode_learn_reward = 0
            episode_top_floor = 0
            obs, info = env.reset()
            lives = info["lives"]
            best_floor = 0
            demo = new_demo() if random.random() < DEMO_PROB else None

        # 4: learn
        if step >= LEARNING_STARTS and step % TRAIN_EVERY == 0:
            train_step(q_net, target_net, optimizer, buffer, BATCH_SIZE, GAMMA, device)

        # 5: refresh the frozen copy
        if step % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(q_net.state_dict())

        # save progress so a crash or Ctrl+C doesn't lose everything
        if step % 50_000 == 0 and step > 0:
            torch.save(q_net.state_dict(), "models/dqn_kingkong.pt")

    torch.save(q_net.state_dict(), "models/dqn_kingkong.pt")
    env.close()