import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
from torch.utils.data import DataLoader, TensorDataset

# Example policies
def random_policy(state, env):
    return env.action_space.sample()

def angle_based_policy(states):
    """
    Vectorized heuristic policy for CartPole.
    states: array-like or tensor of shape [batch, 4]
    Returns: array/tensor of shape [batch] with actions {0,1}
    """
    if isinstance(states, np.ndarray):
        actions = (states[:, 2] > 0).astype(np.int64)  # pole_angle > 0 → action 1
    elif isinstance(states, torch.Tensor):
        actions = (states[:, 2] > 0).long()
    else:
        raise TypeError("Input must be a numpy array or torch tensor")
    return actions

# Q-network
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

# Collect dataset with a random policy
def collect_data(env, n_episodes=50):
    data = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = angle_based_policy(s[None,:])[0]
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            data.append((s, a, r, s2, done))
            s = s2
    return data

# Fitted Q Iteration
def fitted_q_iteration(data, state_dim, action_dim, policy, n_iters=50, gamma=0.99, lr=1e-4):
    qnet = QNet(state_dim, action_dim)
    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Convert dataset to tensors once
    states = torch.FloatTensor([t[0] for t in data])
    actions = torch.LongTensor([t[1] for t in data]).unsqueeze(1)
    rewards = torch.FloatTensor([t[2] for t in data]).unsqueeze(1)
    next_states = torch.FloatTensor([t[3] for t in data])
    dones = torch.FloatTensor([t[4] for t in data]).unsqueeze(1)

    dataset = TensorDataset(states, actions, rewards, next_states, dones)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for it in range(n_iters):
        for batch in loader:
            states_b, actions_b, rewards_b, next_states_b, dones_b = batch

            with torch.no_grad():
                q_next = qnet(next_states_b)                      # [B, action_dim]
                a_next = policy(next_states_b)                    # [B] or [B,1]
                if a_next.ndim == 1:
                    a_next = a_next.unsqueeze(1)                  # ensure [B,1]
                targets = rewards_b + gamma * (1 - dones_b) * q_next.gather(1, a_next)

            q_values = qnet(states_b).gather(1, actions_b)         # [B,1]
            loss = loss_fn(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if it % 10 == 0:
            print(f"Iter {it}, Loss = {loss.item():.4f}")

    return qnet

# Monte Carlo evaluation with discount gamma
def mc_evaluate(env, policy, episodes=50, gamma=0.99):
    rewards = []
    for _ in range(episodes):
        s, _ = env.reset()
        done, ep_r, t = False, 0, 0
        while not done:
            a = policy(s[None,:])[0]
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_r += (gamma ** t) * r
            t += 1
        rewards.append(ep_r)
    return np.mean(rewards), np.std(rewards)


# Q-based evaluation (discount is implicit in Q)
def q_evaluate(env, qnet, policy, n_samples=50):
    values = []
    for _ in range(n_samples):
        s, _ = env.reset()
        with torch.no_grad():
            a = policy(s[None,:])[0]
            qvals = qnet(torch.FloatTensor(s).unsqueeze(0))
            v = qvals[0, a].item()  # already discounted estimate
        values.append(v)
    return np.mean(values), np.std(values)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n

    # Step 1: collect dataset with random policy
    dataset = collect_data(env, n_episodes=1000)

    # Step 2: run Fitted Q Iteration
    qnet = fitted_q_iteration(dataset, state_dim, action_dim, angle_based_policy, n_iters=100)

    # Step 3: evaluate learned greedy policy
    mean_r, std_r = q_evaluate(env, qnet, policy=angle_based_policy, n_samples=50)
    print(f"Evaluation: mean reward = {mean_r:.1f} ± {std_r:.1f}")

    true_r, _ = mc_evaluate(env, policy=angle_based_policy, episodes=50)
    print(f"True Evaluation: mean reward = {true_r:.1f}")
