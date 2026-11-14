import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/KingKong-v5", render_mode="human")
obs, info = env.reset(seed=0)

print("Observation shape: ", env.observation_space.shape)
print("Actions: ", env.unwrapped.get_action_meanings())

done = False
total_reward = 0

while not done:
    action = env.action_space.sample()      # random action
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    done = terminated or truncated

print("Episode reward: ", total_reward)
env.close()