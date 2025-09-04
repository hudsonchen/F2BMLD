import numpy as np
import torch


def cal_mse(value_func, policy, environment, mse_samples, discount, device="cpu"):
    """Compute Bellman residual MSE under the policy in PyTorch."""
    sample_count = 0
    obs, _ = environment.reset()
    mse = 0.0

    while sample_count < mse_samples:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = policy(obs_tensor)             # (batch, action_dim) or discrete action
        if isinstance(action, torch.Tensor):
            action_np = action.squeeze(0).detach().cpu().numpy()
        else:
            action_np = action

        next_obs, reward, terminated, truncated, _ = environment.step(action_np)
        done = terminated or truncated
        reward = float(reward)

        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        action_tensor = action if isinstance(action, torch.Tensor) else torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)

        if done:
            target = torch.tensor([reward], dtype=torch.float32, device=device)
            q_val = value_func(obs_tensor, action_tensor)
            mse_one = (target - q_val) ** 2
            # restart episode
            obs, _ = environment.reset()
        else:
            next_action = policy(next_obs_tensor)
            target = reward + discount * value_func(next_obs_tensor, next_action).detach()
            q_val = value_func(obs_tensor, action_tensor)
            mse_one = (target - q_val) ** 2
            obs = next_obs

        mse += mse_one.item()
        sample_count += 1

    return mse / mse_samples


def ope_evaluation(value_func, policy, environment, num_init_samples,
                   mse_samples=0, discount=0.99, counter=None, logger=None, device="cpu"):
    """Run OPE evaluation in PyTorch."""
    mse = -1
    if mse_samples > 0:
        mse = cal_mse(value_func, policy, environment, mse_samples, discount, device=device)

    # Estimate initial Q-values from the starting state distribution
    q0s = []
    for _ in range(num_init_samples):
        obs, _ = environment.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = policy(obs_tensor)
        q0 = value_func(obs_tensor, action).item()
        q0s.append(q0)

    q0s = np.array(q0s)
    
    results = {
        "Bellman_Residual_MSE": mse,
        "Q0_mean": float(np.mean(q0s)),
        "Q0_std_err": float(np.std(q0s, ddof=0) / np.sqrt(len(q0s))),
    }

    if counter is not None:
        counts = counter.increment(steps=1)
        results.update(counts)

    if logger is not None:
        logger.write(results)

    return results


def estimate_true_value(policy, environment, discount=0.99, 
                        num_episodes=100, device="cpu"):
    """
    Estimate the true discounted return of a policy by Monte Carlo rollouts.

    Args:
        policy: Callable/nn.Module mapping obs -> action (PyTorch).
        environment: Gym-like env with reset() and step().
        discount: float, discount factor.
        num_episodes: int, number of episodes to average over.
        device: str, "cpu" or "cuda".

    Returns:
        mean_return: average discounted return across episodes.
        stderr_return: standard error of returns.
    """
    returns = []
    for _ in range(num_episodes):
        obs, _ = environment.reset()
        done, ep_return, t = False, 0.0, 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = policy(obs_tensor)

            # handle discrete vs continuous
            if isinstance(action, torch.Tensor):
                if action.numel() == 1:  # discrete scalar
                    action = int(action.item())
                else:  # e.g. policy outputs logits/probs
                    action = int(torch.argmax(action, dim=-1).item())

            next_obs, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
            ep_return += (discount ** t) * reward
            obs = next_obs
            t += 1

        returns.append(ep_return)

    mean_return = float(np.mean(returns))
    return mean_return