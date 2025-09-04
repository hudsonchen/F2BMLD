import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple


def add_const_col(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    return torch.cat([x, ones], dim=-1)


def linear_reg_pred(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Predict y = x @ W"""
    return x @ W


def linear_reg_loss(y: torch.Tensor, x: torch.Tensor, reg: float) -> torch.Tensor:
    """Ridge regression loss: ||y - xW||^2 + λ||W||^2"""
    # Closed form ridge not used here, this is for gradient-based training
    W = torch.linalg.lstsq(x, y).solution
    pred = x @ W
    loss = ((y - pred) ** 2).mean() + reg * (W**2).sum()
    return loss


def fit_linear(y: torch.Tensor, x: torch.Tensor, reg: float) -> torch.Tensor:
    """Fit linear regression weights W with ridge penalty."""
    # x: [N, d], y: [N, k]
    N, d = x.shape
    xtx = x.T @ x + reg * torch.eye(d, device=x.device)
    xty = x.T @ y
    W = torch.linalg.solve(xtx, xty)
    return W


class DFIVLearner:
    def __init__(
        self,
        value_func: nn.Module,
        instrumental_feature: nn.Module,
        policy: nn.Module,
        discount: float,
        value_learning_rate: float,
        instrumental_learning_rate: float,
        value_reg: float,
        instrumental_reg: float,
        stage1_reg: float,
        stage2_reg: float,
        instrumental_iter: int,
        value_iter: int,
        dataset: torch.utils.data.DataLoader,
        device: str = "cpu",
        counter=None,
        logger=None,
    ):
        self.stage1_reg = stage1_reg
        self.stage2_reg = stage2_reg
        self.instrumental_iter = instrumental_iter
        self.value_iter = value_iter
        self.discount = discount
        self.value_reg = value_reg
        self.instrumental_reg = instrumental_reg

        self.dataset = dataset
        self._iterator = iter(dataset)
        self.value_func = value_func
        self.value_feature = value_func._feature
        self.instrumental_feature = instrumental_feature
        self.policy = policy

        self._value_optimizer = optim.Adam(self.value_feature.parameters(),
                                           lr=value_learning_rate, betas=(0.5, 0.9))
        self._instrumental_optimizer = optim.Adam(self.instrumental_feature.parameters(),
                                                  lr=instrumental_learning_rate, betas=(0.5, 0.9))

        self.stage1_weight = torch.zeros(
            instrumental_feature.feature_dim(), value_func.feature_dim(),
            requires_grad=False
        ).to(device)
        self.device = device
        self._num_steps = 0

        self._counter = counter
        self._logger = logger

    def update_batch(self):
        try:
            stage1_input = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataset)
            stage1_input = next(self._iterator)

        try:
            stage2_input = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataset)
            stage2_input = next(self._iterator)

        stage1_input_ = [x.to(self.device) for x in stage1_input]
        stage2_input_ = [x.to(self.device) for x in stage2_input]
        return stage1_input_, stage2_input_

    def update_instrumental(self, current_obs, action, reward, dones, next_obs):
        next_action = self.policy(next_obs)
        target = add_const_col(self.value_feature(current_obs, action))
        target = target - self.discount * (1 - dones[:, None]) * add_const_col(self.value_feature(next_obs, next_action))

        feature = self.instrumental_feature(current_obs, action)
        # Ridge regression loss approximation
        pred = linear_reg_pred(feature, self.stage1_weight)
        loss = ((target - pred) ** 2).mean()
        loss += self.instrumental_reg * sum(p.pow(2).sum() for p in self.instrumental_feature.parameters())

        self._instrumental_optimizer.zero_grad()
        loss.backward()
        self._instrumental_optimizer.step()
        return loss.item()

    def update_value(self, stage1_input, stage2_input):
        current_obs_1st, action_1st, _, dones_1st, next_obs_1st = stage1_input[:5]
        current_obs_2nd, action_2nd, reward_2nd, dones_2nd, _ = stage2_input[:5]
        next_action_1st = self.policy(next_obs_1st)

        instrumental_feature_1st = self.instrumental_feature(current_obs_1st, action_1st)
        instrumental_feature_2nd = self.instrumental_feature(current_obs_2nd, action_2nd)

        target_1st = add_const_col(self.value_feature(current_obs_1st, action_1st))
        target_1st = target_1st - self.discount * (1 - dones_1st[:, None]) * add_const_col(
            self.value_feature(next_obs_1st, next_action_1st) * (1 - dones_1st[:, None])
        )
    
        stage1_weight = fit_linear(target_1st, instrumental_feature_1st, self.stage1_reg)

        predicted_feature = linear_reg_pred(instrumental_feature_2nd, stage1_weight)
        stage2_weight = self.value_func._weight.data
        predict = linear_reg_pred(predicted_feature, stage2_weight)
        loss = ((reward_2nd.unsqueeze(-1) - predict) ** 2).mean()
        loss += self.value_reg * sum(p.pow(2).sum() for p in self.value_feature.parameters())

        self._value_optimizer.zero_grad()
        loss.backward()
        self._value_optimizer.step()
        return loss.item()

    def update_final_weight(self, stage1_input, stage2_input):
        current_obs_1st, action_1st, _, dones_1st, next_obs_1st = stage1_input[:5]
        current_obs_2nd, action_2nd, reward_2nd, dones_2nd, _ = stage2_input[:5]
        next_action_1st = self.policy(next_obs_1st)

        instrumental_feature_1st = self.instrumental_feature(current_obs_1st, action_1st)
        instrumental_feature_2nd = self.instrumental_feature(current_obs_2nd, action_2nd)

        target_1st = add_const_col(self.value_feature(current_obs_1st, action_1st)) 
        target_1st = target_1st - self.discount * (1 - dones_1st[:, None]) * add_const_col(self.value_feature(next_obs_1st, next_action_1st))
        stage1_weight = fit_linear(target_1st, instrumental_feature_1st, self.stage1_reg)
        self.stage1_weight = stage1_weight.detach()

        predicted_feature = linear_reg_pred(instrumental_feature_2nd, stage1_weight)
        stage2_weight = fit_linear(reward_2nd.unsqueeze(-1), predicted_feature, self.stage2_reg)
        self.value_func._weight.data = stage2_weight
        return stage1_weight, stage2_weight

    def step(self):
        stage1_input, stage2_input = self.update_batch()
        stage1_loss = None
        stage2_loss = None
        for _ in range(self.value_iter):
            for _ in range(max(1, self.instrumental_iter // self.value_iter)):
                stage1_loss = self.update_instrumental(*stage1_input[:5])
            stage2_loss = self.update_value(stage1_input, stage2_input)

        self.update_final_weight(stage1_input, stage2_input)
        self._num_steps += 1

        result = {"stage1_loss": stage1_loss, "stage2_loss": stage2_loss, "num_steps": self._num_steps}
        if self._logger is not None and self._num_steps % 100 == 0:
            self._logger.write({key: value for key, value in result.items()})
        return result
    

    def cal_validation_err(self, valid_input):
        """Return prediction MSE and std error on the validation dataset (PyTorch)."""
        stage1_weight = self.stage1_weight        # (d_instru, d_value)
        stage2_weight = self.value_func.weight    # (d_value, 1) or similar

        se_sum = 0.0
        se2_sum = 0.0
        count = 0.0

        for sample in valid_input:
            sample = [s.to(self.device) for s in sample]
            current_obs, action, reward, done, _ = sample[:5]

            # Forward pass through instrumental net
            instrumental_feature = self.instrumental_feature(
                obs=current_obs, act=action
            )

            # Stage 1 linear regression prediction
            predicted_feature = linear_reg_pred(instrumental_feature, stage1_weight)

            # Stage 2 linear regression prediction
            predict = linear_reg_pred(predicted_feature, stage2_weight)

            # Compute squared error
            reward = reward.to(predict.device).unsqueeze(-1)  # ensure shape match
            sq_err = (reward - predict) ** 2

            se_sum += sq_err.sum().item()
            se2_sum += (sq_err ** 2).sum().item()
            count += sq_err.shape[0]

        mse = se_sum / count
        mse_err_std = ((se2_sum / count - mse ** 2) / count) ** 0.5
        return mse, mse_err_std