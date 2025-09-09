import torch
import torch.nn as nn
import torch.optim as optim


def add_langevin_noise(params, noise_level):
    """
    Add Langevin dynamics noise to network parameters.
    """
    with torch.no_grad():
        for p in params:
            if p.requires_grad:
                p.add_(torch.randn_like(p) * noise_level)

class F2BMLDLearner:
    def __init__(
        self,
        treatment_net: nn.Module,
        instrument_net: nn.Module,
        instrument_tilde_net: nn.Module,
        policy: nn.Module,
        discount: float,
        treatment_learning_rate: float,
        instrument_learning_rate: float,
        stage1_reg: float,
        stage2_reg: float,
        lagrange_reg: float,
        stage1_ent: float,
        stage2_ent: float,
        instrument_iter: int,
        instrument_tilde_iter: int,
        treatment_iter: int,
        dataset: torch.utils.data.DataLoader,
        device: str = "cpu",
        counter=None,
        logger=None,
    ):
        self.stage1_reg = stage1_reg
        self.stage2_reg = stage2_reg
        self.lagrange_reg = lagrange_reg
        self.instrument_iter = instrument_iter
        self.instrument_tilde_iter = instrument_tilde_iter
        self.treatment_iter = treatment_iter
        self.stage1_ent = stage1_ent
        self.stage2_ent = stage2_ent
        self.discount = discount
        self.treatment_net = treatment_net
        self.instrument_net = instrument_net
        self.instrument_tilde_net = instrument_tilde_net

        self.dataset = dataset
        self._iterator = iter(dataset)
        self.policy = policy

        self._treatment_optimizer = optim.Adam(self.treatment_net.parameters(),
                                           lr=treatment_learning_rate, betas=(0.5, 0.9))
        self._instrument_optimizer = optim.Adam(self.instrument_net.parameters(),
                                                  lr=instrument_learning_rate, betas=(0.5, 0.9))
        self._instrument_tilde_optimizer = optim.Adam(self.instrument_tilde_net.parameters(),
                                                  lr=instrument_learning_rate, betas=(0.5, 0.9))

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

    def update_instrument(self, stage1_input):
        current_obs, action, reward, dones, next_obs = stage1_input[:5]
        next_action = self.policy(next_obs)
        target = self.treatment_net(current_obs, action).detach()
        target = target - self.discount * (1 - dones[:, None]) * self.treatment_net(next_obs, next_action).detach()

        loss = ((target - self.instrument_net(current_obs, action)) ** 2).mean()
        loss += self.stage1_reg * sum(p.pow(2).sum() for p in self.instrument_net.parameters())

        self._instrument_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.instrument_net.parameters(), 10)
        self._instrument_optimizer.step()
        add_langevin_noise(self.instrument_net.parameters(), self.stage1_ent)
        return loss.item()

    def update_instrument_tilde(self, stage1_input, stage2_input):
        current_obs_1st, action_1st, reward_1st, dones_1st, next_obs_1st = stage1_input[:5]
        current_obs_2nd, action_2nd, reward_2nd, dones_2nd, next_obs_2nd = stage2_input[:5]
        next_action_1st = self.policy(next_obs_1st)
        target = self.treatment_net(current_obs_1st, action_1st).detach()
        target = target - self.discount * (1 - dones_1st[:, None]) * self.treatment_net(next_obs_1st, next_action_1st).detach()

        loss_1 = ((target - self.instrument_tilde_net(current_obs_1st, action_1st)) ** 2).mean()
        loss_2 = ((reward_2nd[:, None] - self.instrument_tilde_net(current_obs_2nd, action_2nd)) ** 2).mean()
        loss = self.lagrange_reg * loss_1 + loss_2
        loss += self.lagrange_reg * self.stage1_reg * sum(p.pow(2).sum() for p in self.instrument_tilde_net.parameters())

        self._instrument_tilde_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.instrument_tilde_net.parameters(), 10)
        self._instrument_tilde_optimizer.step()
        add_langevin_noise(self.instrument_tilde_net.parameters(), self.stage1_ent * self.lagrange_reg)
        return loss.item()
    

    def update_treatment(self, stage1_input, stage2_input):
        current_obs_1st, action_1st, reward_1st, dones_1st, next_obs_1st = stage1_input[:5]
        current_obs_2nd, action_2nd, reward_2nd, dones_2nd, next_obs_2nd = stage2_input[:5]
        next_action_1st = self.policy(next_obs_1st)
        next_action_2nd = self.policy(next_obs_2nd)

        target_1 = self.treatment_net(current_obs_1st, action_1st)
        target_1 = target_1 - self.discount * (1 - dones_1st[:, None]) * self.treatment_net(next_obs_1st, next_action_1st)
        loss_1 = ((target_1 - self.instrument_tilde_net(current_obs_1st, action_1st)) ** 2).mean()
        loss_2 = ((target_1 - self.instrument_net(current_obs_1st, action_1st)) ** 2).mean()
        loss = self.lagrange_reg * (loss_1 - loss_2)
        loss += self.stage2_reg * sum(p.pow(2).sum() for p in self.treatment_net.parameters())

        self._treatment_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.treatment_net.parameters(), 10)
        self._treatment_optimizer.step()
        add_langevin_noise(self.treatment_net.parameters(), self.stage2_ent)
        return loss.item()

    def step(self):
        stage1_input, stage2_input = self.update_batch()
        stage1_loss = None
        stage2_loss = None
        for _ in range(self.treatment_iter):
            for _ in range(self.instrument_iter):
                stage1_loss = self.update_instrument(stage1_input)
                # print(f"Stage 1 Instrument Loss: {stage1_loss:.4f}")
            for _ in range(self.instrument_tilde_iter):
                stage1_tilde_loss = self.update_instrument_tilde(stage1_input, stage2_input)
                # print(f"Stage 1 Instrument Tilde Loss: {stage1_tilde_loss:.4f}")
            stage1_input, stage2_input = self.update_batch()
            stage2_loss = self.update_treatment(stage1_input, stage2_input)

        self._num_steps += 1

        result = {"stage1_loss": stage1_loss, "stage1_tilde_loss": stage1_tilde_loss, "stage2_loss": stage2_loss, "num_steps": self._num_steps}
        return result
    

    def cal_validation_err(self, valid_input):
        """Return prediction MSE and std error on the validation dataset (PyTorch)."""
        se_sum = 0.0
        se2_sum = 0.0
        count = 0.0

        for sample in valid_input:
            sample = [s.to(self.device) for s in sample]
            current_obs, action, reward, done, _ = sample[:5]
            predict = self.instrument_net(obs=current_obs, act=action)

            # Compute squared error
            reward = reward.to(predict.device).unsqueeze(-1)  # ensure shape match
            sq_err = (reward - predict) ** 2

            se_sum += sq_err.sum().item()
            se2_sum += (sq_err ** 2).sum().item()
            count += sq_err.shape[0]

        mse = se_sum / count
        mse_err_std = ((se2_sum / count - mse ** 2) / count) ** 0.5
        return mse, mse_err_std