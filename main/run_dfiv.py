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

from dataclasses import dataclass

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
    
config = Config(dataset_path=str(ROOT_PATH / "offline_dataset" / "stochastic"))


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
    
def target_policy(obs_batch: torch.Tensor) -> torch.Tensor:
    """
    Heuristic policy for CartPole.
    obs = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    """
    if isinstance(obs_batch, np.ndarray):
        if obs_batch.ndim == 1:
            angle = obs_batch[2]  # pole angle
        else:
            angle = obs_batch[:, 2] 
        return (angle > 0).astype(np.int64)
    elif isinstance(obs_batch, torch.Tensor):
        if obs_batch.ndim == 1:
            angle = obs_batch[2]  # pole angle
        else:
            angle = obs_batch[:, 2]  # pole angle
        action = (angle > 0).long()  # 0 = left, 1 = right
        return action

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the offline dataset and environment.
    dataset_loader, dev_loader, env, env_spec = utils.load_data_and_env(
        task_name="CartPole-v1",
        noise_level=config.noise_level,
        near_policy_dataset=True,
        policy=target_policy,
        batch_size=config.batch_size,
        max_dev_size=config.max_dev_size,
    )

    value_func, instrumental_feature = dfiv.make_ope_networks(
        "bsuite_cartpole",
        env_spec,
        value_layer_sizes=config.value_layer_sizes,
        instrumental_layer_sizes=config.instrumental_layer_sizes,
        device=device
    )

    counter = Counter()
    logger = StandardLogger(name='train', log_dir='./results/dfiv')

    learner = dfiv.DFIVLearner(
        value_func=value_func,
        instrumental_feature=instrumental_feature,
        policy_net=target_policy,
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
    
    eval_logger = StandardLogger(name='val', log_dir='./results/dfiv')

    while True:
        learner.step()
        steps = learner._num_steps

        if steps % config.evaluate_every == 0:
            eval_results = {}
            if dev_loader is not None:
                eval_results = {'dev_mse': learner.cal_validation_err(dev_loader)}
            eval_results.update(utils.ope_evaluation(
                value_func=value_func,
                policy_net=target_policy,
                environment=env,
                num_init_samples=config.evaluate_init_samples,
                discount=0.99,
                mse_samples=100
                )
            )
            eval_logger.write(eval_results)

        if steps >= config.max_steps:
            break


if __name__ == "__main__":
    main()