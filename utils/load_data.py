import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict, Mapping


class RandomActionWrapper(gym.Wrapper):
    """With probability `eps`, take a random action instead of the policy's action."""
    def __init__(self, env, eps=0.1):
        super().__init__(env)
        self.eps = eps

    def step(self, action):
        if np.random.rand() < self.eps:
            action = self.env.action_space.sample()
        return self.env.step(action)

def make_environment(env_name: str, noise_level: float = 0.0):
    """Create a Gymnasium environment and return it."""
    env = gym.make(env_name)

    # Add random action noise if requested
    if noise_level > 0.0:
        env = RandomActionWrapper(env, eps=noise_level)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    environment_spec = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "obs_space": env.observation_space,
        "act_space": env.action_space,
    }
    return env, environment_spec


def collect_offline_dataset(env, num_episodes, policy):
    """Collect offline dataset from the environment using a behavior policy."""
    dataset = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            dataset.append((obs, action, reward, done, next_obs))
            obs = next_obs

    return dataset


def load_data_and_env(
    task_name: str,
    noise_level: float = 0.0,
    near_policy_dataset: bool = False,
    policy: Mapping = None,
    batch_size: int = 128,
    max_dev_size: int = 1000,
) -> Tuple[DataLoader, DataLoader, gym.Env, Dict]:
    """
    Returns: dataset_loader, dev_dataset_loader, environment, environment_spec
    """
    # 1. Make environment
    env, env_spec = make_environment(task_name, noise_level)

    # 2. Collect dataset
    dataset = collect_offline_dataset(env, num_episodes=2000, policy=policy)

    # 3. Convert to tensors
    states = torch.tensor([d[0] for d in dataset], dtype=torch.float32)
    actions = torch.tensor([d[1] for d in dataset], dtype=torch.long)
    rewards = torch.tensor([d[2] for d in dataset], dtype=torch.float32)
    dones = torch.tensor([d[3] for d in dataset], dtype=torch.float32)
    next_states = torch.tensor([d[4] for d in dataset], dtype=torch.float32)
    
    full_dataset = TensorDataset(states, actions, rewards, dones, next_states)

    # 4. Split into train/dev
    dev_size = min(max_dev_size, len(full_dataset) // 5)
    train_size = len(full_dataset) - dev_size
    train_dataset, dev_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, dev_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, dev_loader, env, env_spec


