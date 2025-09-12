import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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
import argparse
import random
import pwd
import pickle

if pwd.getpwuid(os.getuid())[0] == 'zongchen':
    os.chdir('/home/zongchen/F2BMLD/')
    sys.path.append('/home/zongchen/F2BMLD/')
elif pwd.getpwuid(os.getuid())[0] == 'ucabzc9':
    os.chdir('/home/ucabzc9/Scratch/F2BMLD/')
    sys.path.append('/home/ucabzc9/Scratch/F2BMLD/')
else:
    pass


def target_policy(obs_batch: torch.Tensor, policy_dqn: torch.nn.Module, epsilon) -> torch.Tensor:
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


def behaviour_policy(obs_batch: torch.Tensor, policy_dqn: torch.nn.Module, epsilon) -> torch.Tensor:
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


def main(config):
    # Load pretrained DQN network
    policy_dqn = utils.load_pretrained_dqn(f"pretrained_dqns/policy_net_{config.noise_level}.pth", device=device)

    # Load the offline dataset and environment.
    dataset_loader, dev_loader, env, env_spec = utils.load_data_and_env(
        task_name="CartPole-v1",
        noise_level=config.noise_level,
        policy=partial(behaviour_policy, policy_dqn=policy_dqn, epsilon=0.1),
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
    log_dir += f'_seed_{config.seed}'
    logger = StandardLogger(name='train', log_dir=log_dir)
    eval_logger = StandardLogger(name='val', log_dir=log_dir)

    learner = dfiv.DFIVLearner(
        value_func=value_func,
        instrumental_feature=instrumental_feature,
        policy=partial(target_policy, policy_dqn=policy_dqn, epsilon=config.policy_noise_level),
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
    
    truth_value, stderr_value = utils.estimate_true_value(
        partial(target_policy, policy_dqn=policy_dqn, epsilon=config.policy_noise_level),
        env, discount=0.99, num_episodes=100, device=device
    )
    print(f"Ground-truth policy treatment: {truth_value} ± {stderr_value}")

    # Keep history outside loop
    train_logs = []
    eval_logs = []

    plt.ion()  # interactive mode
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()
    # initialize line handles (empty for now)
    line_stage1, = axes[0].plot([], [], label="Stage 1 Loss (MA100)")
    axes[0].set_title("Stage 1 Loss"); axes[0].set_xlabel("Steps"); axes[0].set_ylabel("Loss"); axes[0].legend()

    line_stage2, = axes[1].plot([], [], color="orange", label="Stage 2 Loss (MA100)")
    axes[1].set_title("Stage 2 Loss"); axes[1].set_xlabel("Steps"); axes[1].set_ylabel("Loss"); axes[1].legend()

    line_bellman, = axes[2].plot([], [], "r--", label="Bellman Residual MSE")
    axes[2].set_title("Bellman Residual MSE"); axes[2].set_xlabel("Steps"); axes[2].set_ylabel("MSE"); axes[2].legend()

    (line_q0,) = axes[3].plot([], [], "bo-", label="Q0 Mean ± SE")
    axes[3].axhline(y=truth_value, color='g', linestyle='--', label='True value')
    axes[3].set_title("Q0 Mean ± StdErr"); axes[3].set_xlabel("Steps"); axes[3].set_ylabel("Q0"); axes[3].legend()

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
                policy=partial(target_policy, policy_dqn=policy_dqn, epsilon=config.policy_noise_level),
                environment=env,
                num_init_samples=config.evaluate_init_samples,
                discount=0.99,
                mse_samples=100,
                device=device,
            ))
            eval_logger.write(eval_results)
            eval_results["num_steps"] = steps  # log step for x-axis
            eval_logs.append(eval_results)

            logger.write(train_results)
            # --- Convert to DataFrame ---
            train_df = pd.DataFrame(train_logs)
            eval_df = pd.DataFrame(eval_logs)

            # Compute moving averages for losses

            line_stage1.set_data(train_df["num_steps"], train_df["stage1_loss"])
            line_stage2.set_data(train_df["num_steps"], train_df["stage2_loss"])
            if "Bellman_Residual_MSE" in eval_df:
                line_bellman.set_data(eval_df["num_steps"], eval_df["Bellman_Residual_MSE"])
            if "Q0_mean" in eval_df:
                line_q0.set_data(eval_df["num_steps"], eval_df["Q0_mean"])

            # rescale axes so new data fits
            for ax in axes:
                ax.relim()
                ax.autoscale_view()

            plt.pause(0.01)
            plt.draw()   # <--- force update of the figure
            fig.tight_layout()
            fig.savefig(f"{log_dir}/training_progress.png")

        # --- Exit condition ---
        if steps >= config.max_steps:
            torch.save(value_func.state_dict(), f"{log_dir}/value_func.pth")
            torch.save(instrumental_feature.state_dict(), f"{log_dir}/instrumental_feature.pth")
            with open(f"{log_dir}/eval_df.pkl", "wb") as f:
                pickle.dump(eval_df, f)
            break

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--value_layer_sizes", type=str, default="50, 50")
    parser.add_argument("--instrumental_layer_sizes", type=str, default="50, 50")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--value_learning_rate", type=float, default=1e-4)
    parser.add_argument("--instrumental_learning_rate", type=float, default=1e-3)
    parser.add_argument("--stage1_reg", type=float, default=1e-5)
    parser.add_argument("--stage2_reg", type=float, default=1e-5)
    parser.add_argument("--instrumental_reg", type=float, default=1e-5)
    parser.add_argument("--value_reg", type=float, default=1e-5)
    parser.add_argument("--instrumental_iter", type=int, default=10)
    parser.add_argument("--value_iter", type=int, default=1)
    parser.add_argument("--max_dev_size", type=int, default=10 * 1024)
    parser.add_argument("--evaluate_every", type=int, default=100)
    parser.add_argument("--evaluate_init_samples", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--policy_noise_level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    config = parser.parse_args()
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    main(config)
    print("Done!")