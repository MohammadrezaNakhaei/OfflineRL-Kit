import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.dynamics import SequenceDynamics, BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from offlinerlkit.modules.dynamics_module import KoopmanDynamicModel

class KoopmanDynamics(SequenceDynamics):
    def __init__(
        self,
        model: KoopmanDynamicModel,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        super().__init__(model, optim, scaler, terminal_fn)

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        device = self.model.device
        normalized_obs = obs
        normalized_obs = torch.as_tensor(normalized_obs, dtype=torch.float32).to(device)
        action = torch.as_tensor(action, dtype=torch.float32).to(device)
        next_obs, reward = self.model.predict(normalized_obs, action)
        reward = reward.cpu().numpy()
        next_obs = next_obs.cpu().numpy()
        next_obs = next_obs
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward
        return next_obs, reward, terminal, info

    def learn(
        self,
        train_loader: DataLoader,
    ) -> float:
        self.model.train()
        losses = []
        for obss, actions, deltas, rewards, masks in train_loader:
            next_obss = obss + deltas
            obss = obss.to(self.device)
            actions = actions.to(self.device)
            next_obss = next_obss.to(self.device)
            rewards = rewards.to(self.device)
            masks = masks.to(self.device)
            batch_size, steps, obs_dim = obss.shape
            loss = torch.zeros(1).to(self.device)
            aug_loss = torch.zeros(1).to(self.device)
            x_current = self.model.encode(obss[:,0,:])
            beta, gamma, beta_sum = 1, 0.99, 0
            for i in range(steps-1):
                x_current = self.model(x_current, actions[:, i, :])
                loss += beta*F.mse_loss(x_current[:,:obs_dim], next_obss[:,i,:])
                x_current_encoded = self.model.encode(x_current[:, :obs_dim])
                aug_loss += beta*F.mse_loss(x_current_encoded, x_current)
                beta*=gamma
                beta_sum+=beta
            loss = loss/beta_sum+0.5*aug_loss/beta_sum
            A = self.model.lA.weight
            c = torch.linalg.eigvals(A).abs()-torch.ones(1).to(self.device)
            mask = c>0
            loss += c[mask].sum()
            pred_reward = self.model.reward(obss, actions)
            loss += F.mse_loss(pred_reward, rewards)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses = loss.item()
        return np.mean(losses)
    
    @ torch.no_grad()
    def validate(self, holdout_loader) -> float:
        self.model.eval()
        losses = []
        for obss, actions, deltas, rewards, masks in holdout_loader:
            next_obss = obss + deltas
            obss = obss.to(self.device)
            actions = actions.to(self.device)
            next_obss = next_obss.to(self.device)
            rewards = rewards.to(self.device)
            masks = masks.to(self.device)
            batch_size, steps, obs_dim = obss.shape
            loss = torch.zeros(1).to(self.device)
            aug_loss = torch.zeros(1).to(self.device)
            x_current = self.model.encode(obss[:,0,:])
            beta, gamma, beta_sum = 1, 0.99, 0
            for i in range(steps-1):
                x_current = self.model(x_current, actions[:, i, :])
                loss += beta*F.mse_loss(x_current[:,:obs_dim], next_obss[:,i,:])
                x_current_encoded = self.model.encode(x_current[:, :obs_dim])
                aug_loss += beta*F.mse_loss(x_current_encoded, x_current)
                beta*=gamma
                beta_sum+=beta
            loss = loss/beta_sum+0.5*aug_loss/beta_sum
            pred_reward = self.model.reward(obss, actions)
            loss += F.mse_loss(pred_reward, rewards)
            losses.append(loss.item())
        return np.mean(losses)
    


class KoopmanDynamicsOneStep(BaseDynamics):
    def __init__(
        self,
        model: KoopmanDynamicModel,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        super().__init__(model, optim,)
        self.terminal_fn = terminal_fn
        self.scaler = scaler

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        device = self.model.device
        normalized_obs = obs
        normalized_obs = torch.as_tensor(normalized_obs, dtype=torch.float32).to(device)
        action = torch.as_tensor(action, dtype=torch.float32).to(device)
        next_obs, reward = self.model.predict(normalized_obs, action)
        reward = reward.cpu().numpy()
        next_obs = next_obs.cpu().numpy()
        next_obs = next_obs
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
        data_size = data['observations'].shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_data, holdout_data = {}, {}
        for k,v in data.items():
            train_data[k] = v[train_splits.indices]
            holdout_data[k] = v[holdout_splits.indices]

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
            
            encoded = self.model.encode(tmp['observations'])
            next_encoded = self.model(encoded, tmp['actions'])
            obs_dim = tmp['next_observations'].shape[1]
            loss_pred = F.mse_loss(next_encoded[:, :obs_dim], tmp['next_observations'])
            x_hat = self.model.encode(next_encoded[:, :obs_dim])
            aug_loss = F.mse_loss(x_hat, next_encoded)
            loss = loss_pred
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
        next_obs, reward = self.model.predict(holdout_data['observations'], holdout_data['actions'])
        loss = F.mse_loss(next_obs, holdout_data['next_observations'])
        return loss.item()
    
    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)
