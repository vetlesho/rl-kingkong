import torch
from dqn import make_env, QNetwork, select_action

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
env = make_env(render_mode="human")

q_net = QNetwork(env.action_space.n).to(device)
q_net.load_state_dict(torch.load("dqn_kingkong_clipped.pt", map_location=device))

obs, info = env.reset()
done = False
total_reward = 0
while not done:
    action = select_action(q_net, obs, 0.05, env.action_space.n, device)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print("Score:", total_reward)
env.close()