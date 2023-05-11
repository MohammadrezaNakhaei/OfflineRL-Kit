import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import COMBOPolicy
from offlinerlkit.dynamics import BaseDynamics


class SECOMBOPolicy(COMBOPolicy):
    """
    Conservative Offline Model-Based Policy Optimization <Ref: https://arxiv.org/abs/2102.08363>
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        cql_weight: float = 1.0,
        temperature: float = 1.0,
        max_q_backup: bool = False,
        deterministic_backup: bool = True,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        cql_alpha_lr: float = 1e-4,
        num_repeart_actions:int = 10,
        uniform_rollout: bool = False,
        rho_s: str = "mix"
    ) -> None:
        super().__init__(
            dynamics,
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            action_space,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            cql_weight=cql_weight,
            temperature=temperature,
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            lagrange_threshold=lagrange_threshold,
            cql_alpha_lr=cql_alpha_lr,
            num_repeart_actions=num_repeart_actions,
            uniform_rollout=uniform_rollout,
            rho_s=rho_s
        )

    @torch.no_grad()
    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        batch_size = observations.shape[0]
        act_dim = self.action_space.shape[0]
        obs_dim = observations.shape[-1]
        device = self.dynamics.model.device
        seq_obss = torch.as_tensor(observations.reshape(batch_size,1,obs_dim), dtype=torch.float32, device=device)        
        seq_acts = torch.zeros(size=(batch_size,1,act_dim), dtype=torch.float32, device=device)
        nonterm_mask = np.ones(batch_size, dtype=bool)
        for i in range(rollout_length):
            if self._uniform_rollout:
                actions = np.random.uniform(
                    self.action_space.low[0],
                    self.action_space.high[0],
                    size=(len(observations), self.action_space.shape[0])
                )
            else:
                actions = self.select_action(observations)
            seq_acts[:,-1,:] = torch.as_tensor(actions, dtype=torch.float32, device=device)
            seq_mu, seq_logvar, seq_reward = self.dynamics.model(seq_obss, seq_acts)
            seq_delta = seq_mu+seq_logvar.exp()*torch.randn_like(seq_mu, device=device)
            seq_next_obs = seq_delta+seq_obss
            seq_obss = torch.cat([seq_obss,torch.zeros((batch_size,1, obs_dim), device=device)], dim=1)
            seq_acts = torch.cat([seq_acts, torch.zeros((batch_size,1,act_dim), device=device)], dim=1)
            seq_obss[:,-1] = seq_next_obs[:,-1,:]
            next_observations = seq_next_obs[:,-1,:].cpu().numpy()
            rewards = seq_reward[:,-1].cpu().numpy()
            terminals = self.dynamics.terminal_fn(observations, actions, next_observations)
            #next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
            rollout_transitions["obss"].append(observations[nonterm_mask])
            rollout_transitions["next_obss"].append(next_observations[nonterm_mask])
            rollout_transitions["actions"].append(actions[nonterm_mask])
            rollout_transitions["rewards"].append(rewards[nonterm_mask])
            rollout_transitions["terminals"].append(terminals[nonterm_mask])

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = np.minimum((~terminals).flatten(), nonterm_mask)
           
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}