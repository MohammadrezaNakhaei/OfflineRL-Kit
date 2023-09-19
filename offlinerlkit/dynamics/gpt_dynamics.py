import random
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from offlinerlkit.utils.load_dataset import SequenceDynamicDataset, transform_to_episodic

class SequenceDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],

    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self.device = self.model.device

    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        max_len:int = 10,
        num_workers: int = 4,

    ) -> None:
        #self.scaler.fit(data['observations'])
        #self.scaler.transform(data['observations'])
        #self.scaler.transform(data['next_observations'])
        episodes = transform_to_episodic(data)
        data_size = len(episodes)
        random.shuffle(episodes)
        holdout_size = min(int(data_size * holdout_ratio), 25)
        train_size = data_size - holdout_size        
        train_episodes, holdout_episodes = episodes[0:train_size], episodes[train_size:]
        train_dataset = SequenceDynamicDataset(train_episodes, max_len=max_len)
        holdout_dataset = SequenceDynamicDataset(holdout_episodes, max_len=max_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        holdout_loader = DataLoader(holdout_dataset, batch_size=batch_size)
        old_holdout_loss = 1e10

        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            train_loss = self.learn(train_loader)
            holdout_loss = self.validate(holdout_loader)
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])
            improvement = (old_holdout_loss - holdout_loss) / old_holdout_loss
            if improvement > 0.01:
                cnt = 0
                old_holdout_loss = holdout_loss
            else:
                cnt += 1
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break
    
    def learn(
        self,
        train_loader: DataLoader,
    ) -> float:
        
        self.model.train()
        losses = []
        for obss, actions, deltas, rewards, masks in train_loader:
            obss = obss.to(self.device)
            actions = actions.to(self.device)
            deltas = deltas.to(self.device)
            rewards = rewards.to(self.device)
            masks = masks.to(self.device)
            mean, logvar, pred_rewards = self.model(obss, actions)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - deltas, 2) * inv_var).mean()
            var_loss = logvar.mean()
            loss_reward = torch.pow(pred_rewards-rewards,2).mean()
            loss = mse_loss_inv + var_loss + loss_reward
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(loss.item())
        return np.mean(losses)
    
    @ torch.no_grad()
    def validate(self, holdout_loader) -> float:
        self.model.eval()
        losses = []
        for obss, actions, deltas, rewards, masks in holdout_loader:
            obss = obss.to(self.device)
            actions = actions.to(self.device)
            deltas = deltas.to(self.device)
            rewards = rewards.to(self.device)
            masks = masks.to(self.device)
            mean, logvar, pred_rewards = self.model(obss, actions)
            loss = torch.pow(mean - deltas, 2).mean() + torch.pow(pred_rewards-rewards,2).mean()
            losses.append(loss.item())        
        return np.mean(losses)

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)
