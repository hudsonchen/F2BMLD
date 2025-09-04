from collections import Counter
import pathlib
import sys
ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))
import torch
import utils
from utils.logger import StandardLogger
from methods import dfiv
import numpy as np
from functools import partial
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class Config:
    dataset_path: str
    value_layer_sizes: str = "50,50"
    instrumental_layer_sizes: str = "50,50"
    batch_size: int = 1024
    value_learning_rate: float = 1e-4
    instrumental_learning_rate: float = 1e-3
    stage1_reg: float = 1e-5
    stage2_reg: float = 1e-5
    instrumental_reg: float = 1e-5
    value_reg: float = 1e-5
    instrumental_iter: int = 1
    value_iter: int = 1
    max_dev_size: int = 10 * 1024
    evaluate_every: int = 100
    evaluate_init_samples: int = 1000
    max_steps: int = 100_000
    noise_level: float = 0.1
    policy_noise_level: float = 0.0
    
config = Config(dataset_path=str(ROOT_PATH / "offline_dataset" / "stochastic"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def target_policy(obs_batch, act_dim: int = 2):
#     """
#     Random policy for CartPole (discrete actions 0 or 1).
#     obs_batch: torch.Tensor or np.ndarray, shape [batch, 4] or [4]
#     Returns: actions of shape [batch] (int64)
#     """
#     if isinstance(obs_batch, np.ndarray):
#         if obs_batch.ndim == 1:
#             return np.random.randint(act_dim)
#         else:
#             return np.random.randint(act_dim, size=obs_batch.shape[0])
#     elif isinstance(obs_batch, torch.Tensor):
#         if obs_batch.ndim == 1:
#             return torch.randint(0, act_dim, (1,)).item()
#         else:
#             return torch.randint(0, act_dim, (obs_batch.shape[0],))
#     else:
#         raise TypeError("Input must be a numpy array or torch tensor")
    
# def target_policy(obs_batch: torch.Tensor) -> torch.Tensor:
#     """
#     Heuristic policy for CartPole.
#     obs = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
#     """
#     if isinstance(obs_batch, np.ndarray):
#         if obs_batch.ndim == 1:
#             angle = obs_batch[2]  # pole angle
#         else:
#             angle = obs_batch[:, 2] 
#         return (angle > 0).astype(np.int64)
#     elif isinstance(obs_batch, torch.Tensor):
#         if obs_batch.ndim == 1:
#             angle = obs_batch[2]  # pole angle
#         else:
#             angle = obs_batch[:, 2]  # pole angle
#         action = (angle > 0).long()  # 0 = left, 1 = right
#         return action

def target_policy(obs_batch: torch.Tensor, policy_dqn: torch.nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return policy_dqn(obs_batch).argmax(dim=-1)


def behavior_policy(obs_batch: torch.Tensor, policy_dqn: torch.nn.Module, epsilon=0.2) -> torch.Tensor:
    """
    Policy that follows the trained DQN policy but with probability epsilon takes a random action.
    """
    with torch.no_grad():
        greedy_actions = policy_dqn(obs_batch).argmax(dim=-1)  # [batch]
    
    random_mask = torch.rand(len(obs_batch)) < epsilon
    random_actions = torch.randint(0, 2, size=(len(obs_batch),), dtype=torch.long).to(obs_batch.device)
    final_actions = greedy_actions.clone()
    final_actions[random_mask] = random_actions[random_mask]
    return final_actions


def main():
    # Load pretrained DQN network
    policy_dqn = utils.load_pretrained_dqn("policy_net.pth", device=device)

    # Load the offline dataset and environment.
    dataset_loader, dev_loader, env, env_spec = utils.load_data_and_env(
        task_name="CartPole-v1",
        noise_level=config.noise_level,
        policy=partial(behavior_policy, policy_dqn=policy_dqn, epsilon=config.policy_noise_level),
        batch_size=config.batch_size,
        max_dev_size=config.max_dev_size,
        device=device
    )

    value_func, instrumental_feature = dfiv.make_ope_networks(
        "bsuite_cartpole",
        env_spec,
        value_layer_sizes=config.value_layer_sizes,
        instrumental_layer_sizes=config.instrumental_layer_sizes,
        device=device
    )

    counter = Counter()
    log_dir=f'./results/dfiv_env_noise_{config.noise_level}__policy_noise_{config.policy_noise_level}'
    logger = StandardLogger(name='train', log_dir=log_dir)
    eval_logger = StandardLogger(name='val', log_dir=log_dir)

    learner = dfiv.DFIVLearner(
        value_func=value_func,
        instrumental_feature=instrumental_feature,
        policy=partial(target_policy, policy_dqn=policy_dqn),
        discount=0.99,
        value_learning_rate=config.value_learning_rate,
        instrumental_learning_rate=config.instrumental_learning_rate,
        stage1_reg=config.stage1_reg,
        stage2_reg=config.stage2_reg,
        value_reg=config.value_reg,
        instrumental_reg=config.instrumental_reg,
        instrumental_iter=config.instrumental_iter,
        value_iter=config.value_iter,
        dataset=dataset_loader,
        device=device,
        counter=counter,
        logger=logger)
    
    truth_value = utils.estimate_true_value(
        partial(target_policy, policy_dqn=policy_dqn), env, discount=0.99, num_episodes=1000, device=device
    )
    print(f"Ground-truth policy value: {truth_value}")

    # Keep history outside loop
    train_logs = []
    eval_logs = []

    while True:
        # --- Training step ---
        train_results = learner.step()
        train_results["num_steps"] = learner._num_steps  # make sure steps are logged
        train_logs.append(train_results)

        steps = learner._num_steps

        # --- Evaluation ---
        if steps % config.evaluate_every == 0:
            eval_results = {}
            if dev_loader is not None:
                eval_results["dev_mse"] = learner.cal_validation_err(dev_loader)
            eval_results.update(utils.ope_evaluation(
                value_func=value_func,
                policy=partial(target_policy, policy_dqn=policy_dqn),
                environment=env,
                num_init_samples=config.evaluate_init_samples,
                discount=0.99,
                mse_samples=100,
                device=device,
            ))
            eval_results["num_steps"] = steps  # log step for x-axis
            eval_logs.append(eval_results)

            # --- Convert to DataFrame ---
            train_df = pd.DataFrame(train_logs)
            eval_df = pd.DataFrame(eval_logs)

            # Compute moving averages for losses
            train_df["stage1_loss_ma"] = train_df["stage1_loss"].rolling(window=100).mean()
            train_df["stage2_loss_ma"] = train_df["stage2_loss"].rolling(window=100).mean()

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()  # easy indexing

            # --- Stage 1 Loss (smoothed) ---
            axes[0].plot(train_df["num_steps"], train_df["stage1_loss_ma"], label="Stage 1 Loss (MA100)")
            axes[0].set_xlabel("Steps")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Stage 1 Loss")
            axes[0].legend()

            # --- Stage 2 Loss (smoothed) ---
            axes[1].plot(train_df["num_steps"], train_df["stage2_loss_ma"], label="Stage 2 Loss (MA100)", color="orange")
            axes[1].set_xlabel("Steps")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Stage 2 Loss")
            axes[1].legend()

            # --- Bellman Residual MSE ---
            if "Bellman_Residual_MSE" in eval_df:
                axes[2].plot(eval_df["num_steps"], eval_df["Bellman_Residual_MSE"], "r--", label="Bellman Residual MSE")
            axes[2].set_xlabel("Steps")
            axes[2].set_ylabel("MSE")
            axes[2].set_title("Bellman Residual MSE")
            axes[2].legend()

            # --- Q0 Mean ± SE ---
            if "Q0_mean" in eval_df and "Q0_std_err" in eval_df:
                axes[3].errorbar(eval_df["num_steps"], eval_df["Q0_mean"], 
                                yerr=eval_df["Q0_std_err"], fmt="bo-", label="Q0 Mean ± SE")
            axes[3].set_xlabel("Steps")
            axes[3].set_ylabel("Q0")
            axes[3].set_title("Q0 Mean ± StdErr")
            axes[3].axhline(y=truth_value, color='g', linestyle='--', label='True Value')
            axes[3].legend()

            plt.tight_layout()
            plt.savefig(f"{log_dir}/training_progress_{steps}.png")
            plt.close(fig)
        # --- Exit condition ---
        if steps >= config.max_steps:
            torch.save(value_func.state_dict(), f"{log_dir}/value_func.pth")
            torch.save(instrumental_feature.state_dict(), f"{log_dir}/instrumental_feature.pth")
            break

if __name__ == "__main__":
    main()