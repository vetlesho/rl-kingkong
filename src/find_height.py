import numpy as np
from dqn import make_env

env = make_env()
obs, info = env.reset(seed=0)

snapshots = []
for _ in range(20_000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    snapshots.append(env.unwrapped.ale.getRAM().copy())  # the 128 bytes of game memory
    if terminated or truncated:
        obs, info = env.reset()

snapshots = np.array(snapshots)
for index in range(128):
    values = np.unique(snapshots[:, index])
    if 2 <= len(values) <= 12:
        print(f"byte {index:>3}: {values}")