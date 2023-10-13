import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional
from offlinerlkit.nets import EnsembleLinear
from offlinerlkit.nets import MLP


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x


def soft_clamp(
    x : torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation: nn.Module = Swish,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        with_reward: bool = True,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        self.device = torch.device(device)

        self.activation = activation()

        assert len(weight_decays) == (len(hidden_dims) + 1)

        module_list = []
        hidden_dims = [obs_dim+action_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        for in_dim, out_dim, weight_decay in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
        self.backbones = nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(
            hidden_dims[-1],
            2 * (obs_dim + self._with_reward),
            num_ensemble,
            weight_decays[-1]
        )

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * -10, requires_grad=True)
        )

        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )

        self.to(self.device)

    def forward(self, obs_action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self.device)
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs
    

class DecoupledDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        activation: nn.Module = Swish,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.activation = activation()     

        module_list = []
        hidden_dims = [obs_dim] + list(hidden_dims)
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            module_list.append(nn.Linear(in_dim, out_dim))
            module_list.append(self.activation)
        self.backbones = nn.Sequential(*module_list)

        self.output_layer = nn.Linear(
            hidden_dims[-1],
            obs_dim*obs_dim+obs_dim*action_dim+obs_dim, # A: n*n, B: n*m, n for std
        )

        self.reward_model = nn.Sequential(
            nn.Linear(obs_dim+action_dim, 64),
            self.activation,
            nn.Linear(64,64),
            self.activation,
            nn.Linear(64,1)
        )

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(obs_dim) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(obs_dim) * -10, requires_grad=True)
        )
        self.to(self.device)

    def forward(self, obs: torch.Tensor, action: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        output = self.output_layer(self.backbones(obs))
        n = self.obs_dim
        m = self.action_dim
        A = output[:, :n*n]
        B = output[:, n*n:n*n+n*m]
        logvar = output[:, -n:]
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        A = A.reshape(-1, n, n)
        B = B.reshape(-1, n, m)        
        return A, B, logvar
    
    def predict(self, obs: torch.Tensor, action: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        A, B, logvar = self.forward(obs, action)
        n = self.obs_dim
        m = self.action_dim        
        delta = A@obs.reshape(-1,n,1) + B@action.reshape(-1,m, 1)
        delta = delta.squeeze(-1)
        reward = self.reward_model(torch.cat([obs, action], dim=-1))
        return delta, logvar, reward
    

class KoopmanDynamicModel(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            latent_dim: int,
            hidden_dims: Union[List[int], Tuple[int]],
            activation: nn.Module = nn.ReLU,
            device: str = "cpu",
            ) -> None:
        super().__init__()
        n_koopman = obs_dim + latent_dim
        self.encoder = MLP(obs_dim, hidden_dims, latent_dim, activation)
        self.lA = nn.Linear(n_koopman, n_koopman, bias=False)
        nn.init.normal_(self.lA.weight.data, 0, 1/latent_dim)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(action_dim, n_koopman, bias=False)  
        self.reward_model = MLP(obs_dim+action_dim, [64,64], 1)
        self.to(device)
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim  
        self.n_koopman = n_koopman   

    def reward(self, state:torch.Tensor, u:torch.Tensor,):
        return self.reward_model(torch.cat([state, u], dim=-1))

    def encode_only(self, state:torch.Tensor):
        return self.encoder(state)

    # encoded: [state, latent]
    def encode(self, state:torch.Tensor):
        return torch.cat([state, self.encoder(state)],axis=-1)
    
    # take latent state as input!
    def forward(self, x:torch.Tensor, u:torch.Tensor):
        return self.lA(x)+self.lB(u) 
    
    def predict(self, state:torch.Tensor, u:torch.Tensor):
        x = self.encode(state)
        reward = self.reward(state, u)
        next_x= self.forward(x, u)
        return next_x[:, :state.shape[1]], reward
        
    @torch.no_grad()
    def A(self,):
        return self.lA.weight.data
    
    @torch.no_grad()
    def B(self,):
        return self.lB.weight.data
    