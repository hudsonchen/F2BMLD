import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from collections import Counter
import pathlib
import sys
ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))
import torch
import random
import utils
from utils.logger import StandardLogger
from methods import f2bmld
import numpy as np
from functools import partial
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import pickle
import pwd

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
    policy_dqn = utils.load_pretrained_dqn(f"pretrained_dqns/policy_net_{config.noise_level}.pth", device=device)

    dataset_loader, dev_loader, env, env_spec = utils.load_data_and_env(
        task_name="CartPole-v1",
        noise_level=config.noise_level,
        policy=partial(behaviour_policy, policy_dqn=policy_dqn, epsilon=0.1),
        batch_size=config.batch_size,
        max_dev_size=config.max_dev_size,
        device=device
    )

    treatment_net, instrument_net, instrument_tilde_net = f2bmld.make_ope_networks(
        env_spec,
        treatment_layer_sizes=config.treatment_layer_sizes,
        instrument_layer_sizes=config.instrument_layer_sizes,
        device=device
    )

    counter = Counter()
    log_dir=f'./results/f2bmld_env_noise_{config.noise_level}__policy_noise_{config.policy_noise_level}'
    log_dir += f'__lagrange_{config.lagrange_reg}__stage1_reg_{config.stage1_reg}__stage2_reg_{config.stage2_reg}'
    log_dir += f'__treat_lr_{config.treatment_learning_rate}__instr_lr_{config.instrument_learning_rate}'
    log_dir += f'__instr_iter_{config.instrument_iter}__instr_tilde_iter_{config.instrument_tilde_iter}'
    log_dir += f'__bs_{config.batch_size}__seed_{config.seed}'
    logger = StandardLogger(name='train', log_dir=log_dir)
    eval_logger = StandardLogger(name='val', log_dir=log_dir)
    logger.write(vars(config))


    learner = f2bmld.F2BMLDLearner(
        treatment_net=treatment_net,
        instrument_net=instrument_net,
        instrument_tilde_net=instrument_tilde_net,
        policy=partial(target_policy, policy_dqn=policy_dqn, epsilon=config.policy_noise_level),
        discount=0.99,
        treatment_learning_rate=config.treatment_learning_rate,
        instrument_learning_rate=config.instrument_learning_rate,
        stage1_reg=config.stage1_reg,
        stage2_reg=config.stage2_reg,
        lagrange_reg=config.lagrange_reg,
        instrument_iter=config.instrument_iter,
        instrument_tilde_iter=config.instrument_tilde_iter,
        treatment_iter=config.treatment_iter,
        stage1_ent=config.stage1_ent,
        stage2_ent=config.stage2_ent,
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


    ### Plotting code ###

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

    fig.tight_layout()

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
                treatment_net,
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
            torch.save(treatment_net.state_dict(), f"{log_dir}/treatment_net.pth")
            torch.save(instrument_net.state_dict(), f"{log_dir}/instrument_net.pth")
            torch.save(instrument_tilde_net.state_dict(), f"{log_dir}/instrument_tilde_net.pth")
            with open(f"{log_dir}/eval_df.pkl", "wb") as f:
                pickle.dump(eval_df, f)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--treatment_layer_sizes", type=str, default="50,1")
    parser.add_argument("--instrument_layer_sizes", type=str, default="50,1")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--treatment_learning_rate", type=float, default=1e-3)
    parser.add_argument("--instrument_learning_rate", type=float, default=1e-3)
    parser.add_argument("--stage1_reg", type=float, default=1e-5)
    parser.add_argument("--stage2_reg", type=float, default=1e-5)
    parser.add_argument("--stage1_ent", type=float, default=1e-5)
    parser.add_argument("--stage2_ent", type=float, default=1e-5)
    parser.add_argument("--lagrange_reg", type=float, default=0.3)
    parser.add_argument("--instrument_iter", type=int, default=10)
    parser.add_argument("--instrument_tilde_iter", type=int, default=10)
    parser.add_argument("--treatment_iter", type=int, default=1)
    parser.add_argument("--max_dev_size", type=int, default=10 * 1024)
    parser.add_argument("--evaluate_every", type=int, default=100)
    parser.add_argument("--evaluate_init_samples", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--policy_noise_level", type=float, default=0.1)
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

    print("Configuration:")
    print(config)
    main(config)
    print("Done!")