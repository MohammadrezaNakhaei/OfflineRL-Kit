from typing import Any
import torch
import torch.nn.functional as F
from torch.optim import Adam

from typing import Tuple, List, Mapping

from offlinerlkit.nets import MLP
from offlinerlkit.buffer import ReplayBuffer


class RewardModel():
    def __init__(self, state_dim:int, hidden_dims:Tuple[int], lr:float=1e-4, device:str='cpu'):
        self.reward = MLP(state_dim, hidden_dims, 1).to(device)
        self.optim = Adam(self.reward.parameters(), lr)
        self.device = device

    @torch.no_grad()
    def predict(self, state:torch.Tensor):
        return self.reward(state)
    
    def __call__(self, state:torch.Tensor): 
        return self.predict(state)
    
    def train(self, buffer:ReplayBuffer, num_iter:int=10000, batch_size=256):
        for t in range(num_iter):
            data = buffer.sample(batch_size)
            next_obs = data['next_observations']
            reward = data['rewards']
            self.optim.zero_grad()
            pred = self.reward(next_obs)
            loss = F.mse_loss(pred, reward)
            loss.backward()
            self.optim.step()

    def save(self, path):
        torch.save(self.reward.state_dict(), path)


        
