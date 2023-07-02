import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from offlinerlkit.modules.dynamics_module import DecoupledDynamicsModel

class DecoupledDynamics(BaseDynamics):
    def __init__(
        self,
        model: DecoupledDynamicsModel,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self.best_model = self.model.state_dict()


    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        device = self.model.device
        normalized_obs = self.scaler.transform(obs)
        normalized_obs = torch.as_tensor(normalized_obs, dtype=torch.float32).to(device)
        action = torch.as_tensor(action, dtype=torch.float32).to(device)
        mean, logvar, reward = self.model.predict(normalized_obs, action)
        reward = reward.cpu().numpy()
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean += obs
        std = np.sqrt(np.exp(logvar))
        next_obs = mean + np.random.normal(size=mean.shape) * std
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward
        return next_obs, reward, terminal, info

    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:
        data['delta'] = data["next_observations"] - data["observations"]
        del data['next_observations']
        data_size = data['observations'].shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_data, holdout_data = {}, {}
        for k,v in data.items():
            train_data[k] = v[train_splits.indices]
            holdout_data[k] = v[holdout_splits.indices]

        self.scaler.fit(train_data['observations'])
        train_data["observations"] = self.scaler.transform(train_data['observations'])
        holdout_data['observations'] = self.scaler.transform(holdout_data['observations'])
        holdout_loss = 1e10
        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            train_loss = self.learn(train_data, batch_size, logvar_loss_coef)
            new_holdout_loss = self.validate(holdout_data)
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", new_holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])
            improvement = (holdout_loss - new_holdout_loss) / holdout_loss
            if improvement > 0.01:
                cnt = 0
                holdout_loss = new_holdout_loss
                self.best_model = self.model.state_dict()
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break
        self.model.load_state_dict(self.best_model)
        self.save(logger.model_dir)
        self.model.eval()
    
    def learn(
        self,
        train_data: dict,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        train_size = train_data['observations'].shape[0]
        total_loss = 0
        for batch_num in range(int(np.ceil(train_size / batch_size))):
            tmp = {}
            for k, v in train_data.items():
                item = v[batch_num * batch_size:(batch_num + 1) * batch_size]
                tmp[k] = torch.as_tensor(item).to(self.model.device)
            mean, logvar, reward = self.model.predict(tmp['observations'], tmp['actions'])
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - tmp['delta'], 2) * inv_var).mean()
            var_loss = logvar.mean()
            reward_loss = torch.pow(reward-tmp['rewards'], 2).mean() 
            loss = mse_loss_inv + var_loss + reward_loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            total_loss+=loss.item()
        return total_loss/batch_num
    
    @ torch.no_grad()
    def validate(self, holdout_data: dict,) -> List[float]:
        self.model.eval()
        for k,v in holdout_data.items():
            holdout_data[k] = torch.as_tensor(v).to(self.model.device)
        mean, _, reward = self.model.predict(holdout_data['observations'], holdout_data['actions'])
        loss = ((mean - holdout_data['delta']) ** 2).mean()
        return loss.item()
    
    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)
