import gym
import numpy as np
import torch
from tqdm import tqdm
import pickle

from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy, SACPolicy, TD3Policy
from offlinerlkit.buffer import ReplayBuffer
from typing import Tuple

class Normalizer():
    warmp_up = 200
    def __init__(self, shape:Tuple[int], eps:float=1e-2):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32)
        self.counts = eps
        self.total_sum = np.zeros(shape, dtype=np.float32)
        self.total_sumsq = np.zeros(shape, dtype=np.float32)

    def update(self, batch: np.ndarray):
        assert batch.ndim == 2
        self.total_sum += np.sum(batch, axis=0)
        self.total_sumsq += np.sum(batch**2, axis=0)
        self.counts+=batch.shape[0]
        if self.counts>self.warmp_up:
            self.std = np.sqrt((self.total_sumsq / self.counts) - np.square(self.total_sum / self.counts))
            self.mean = self.total_sum/self.counts

    def normalize(self, state: np.ndarray):
        return (state-self.mean)/self.std
    
    def normalize_torch(self, state: torch.Tensor, device='cpu'):
        mean = torch.from_numpy(self.mean).to(device)
        std = torch.from_numpy(self.std).to(device)
        return (state-mean)/std



class ResidualAgentTrainer:
    EVAL_EVERY = 5000
    SAVE_EVERY = 50000
    def __init__(
            self, 
            env: gym.Env,
            eval_env: gym.Env,
            policy: BasePolicy, 
            dynamics: BaseDynamics, 
            residual_agent: (SACPolicy, TD3Policy), 
            real_buffer: ReplayBuffer, 
            buffer: ReplayBuffer,
            logger: Logger,
            res_action_coef:float = 1,
    ):
        self.env = env
        self.eval_env = eval_env
        self.policy = policy
        self.dynamics = dynamics
        self.agent = residual_agent
        self.buffer = buffer
        self.real_buffer = real_buffer
        self.logger = logger
        self.res_action_coef = res_action_coef
        self.normalizer = Normalizer(self.buffer.obs_shape)
        self.device = self.real_buffer.device

    def pre_train(self, num_epoch:int, step_per_epoch: int, batch_size:int=256):
        self.agent.train()
        num_timesteps = 0
        self.logger.log('Pre training the agent')
        for epoch in range(1,num_epoch+1):
            pbar = tqdm(range(step_per_epoch), desc=f"Epoch #{epoch}/{num_epoch}")
            for it in pbar:
                batch = self.real_buffer.sample(batch_size)
                # augment state
                obs = batch['observations']
                next_obs = batch['next_observations']
                with torch.no_grad():
                    act, _ = self.policy.actforward(obs, True)
                    next_act, _ = self.policy.actforward(next_obs, True)
                batch['observations'] = torch.cat(
                    [obs, obs.clone(), act], 
                    dim=-1)
                batch['next_observations'] = torch.cat(
                    [next_obs, next_obs.clone(), next_act], 
                    dim=-1)
                batch['actions'] = torch.zeros_like(batch['actions'])
                loss = self.agent.learn(batch)
                pbar.set_postfix(**loss)
                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                num_timesteps+=1
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()
        return num_timesteps
    
    def _prepare_state(self, obs, x_hat, policy_act, k=None):
        return np.concatenate([obs, obs-x_hat, policy_act], -1)
    
    def _prepare_action(self, policy_action, res_action, residual_agent=True):
        if residual_agent:
            return np.clip(policy_action + self.res_action_coef*res_action, -1, 1)
        else:
            return policy_action
                
    
    @torch.no_grad()
    def evaluate(self, k:float=0.1, res_agent:bool=True, deterministic:bool=True, num_eval:int=20):
        device = self.agent.actor.device
        rewards = np.zeros(num_eval)
        for n in range(num_eval):
            obs, done = self.eval_env.reset(), False
            x_hat = obs # estimated state
            ep_reward = 0
            while not done:
                policy_act = self.policy.select_action(torch.as_tensor(obs).to(device), True) # policy action
                state = self._prepare_state(obs, x_hat, policy_act, k)
                normalized_state = self.normalizer.normalize(state)
                res_action = self.agent.select_action(torch.as_tensor(normalized_state).to(device), deterministic)
                action = self._prepare_action(policy_act, res_action, res_agent) 
                next_obs, reward, done, info = self.eval_env.step(action)
                pred_next_obs, *_ = self.dynamics.step(obs.reshape(1,-1), action.reshape(1,-1))
                pred_next_obs = pred_next_obs.reshape(-1,)
                x_hat_dot = pred_next_obs-obs+k*(obs-x_hat)
                next_x_hat = x_hat + x_hat_dot
                ep_reward+=reward
                obs = next_obs
                x_hat = next_x_hat
            rewards[n] = ep_reward
        ep_reward = self.eval_env.get_normalized_score(np.mean(rewards))*100
        std = self.eval_env.get_normalized_score(np.std(rewards))*100
        return ep_reward, std       
    
    def save(self, tag:str='best'):
        path = f'{self.logger.model_dir}/residual_agent_{tag}.pth'
        path_normalizer = f'{self.logger.model_dir}/normalizer_{tag}.pkl'
        torch.save(self.agent.state_dict(), path)
        with open(path_normalizer, 'wb') as f:
            pickle.dump(self.normalizer, f, pickle.HIGHEST_PROTOCOL)

    
    def train_epoch(self, num_step:int, batch_size:int=256):
        self.agent.train()
        pbar = tqdm(range(num_step),)
        for i in pbar:
            batch = self.buffer.sample(batch_size)
            batch['observations'] = self.normalizer.normalize_torch(batch['observations'], self.device)
            batch['next_observations'] = self.normalizer.normalize_torch(batch['next_observations'], self.device)
            loss = self.agent.learn(batch)
            for k, v in loss.items():
                self.logger.logkv_mean(k, v)
        self.logger.dumpkvs()
    
    def train_sample(self, num_step:int, batch_size:int=256):
        self.agent.train()
        for i in range(num_step):
            batch = self.buffer.sample(batch_size)
            batch['observations'] = self.normalizer.normalize_torch(batch['observations'], self.device)
            batch['next_observations'] = self.normalizer.normalize_torch(batch['next_observations'], self.device)
            loss = self.agent.learn(batch)
        return loss
        

    def train_episodic(self, max_steps:int, update_ratio: int=2, batch_size:int=256, k:float=0.1):
        device = self.real_buffer.device
        policy_reward, std = self.evaluate(k, res_agent=False)
        self.logger.logkv('eval_reward', policy_reward)
        self.logger.logkv('eval_reward_std', std)
        self.logger.dumpkvs()
        self.logger.log('Training the agent')
        self.logger.log(f'Initial policy reward: {policy_reward:.2f}')
        total_t = 0
        t = 0
        best_reward = 0
        while True:
            obs, done = self.env.reset(), False
            x_hat = obs # estimated state
            ep_reward = 0
            while not done and total_t<=max_steps:
                total_t+=1
                policy_act = self.policy.select_action(torch.as_tensor(obs).to(device), True) # policy action
                state = self._prepare_state(obs, x_hat, policy_act, k)
                self.normalizer.update(state.reshape(1,-1)) # add batch dimention
                normalized_state = self.normalizer.normalize(state)
                res_action = self.agent.select_action(torch.as_tensor(normalized_state).to(device), False)
                action = self._prepare_action(policy_act, res_action, True) 
                next_obs, reward, done, info = self.env.step(action)
                pred_next_obs, *_ = self.dynamics.step(obs.reshape(1,-1), action.reshape(1,-1))
                pred_next_obs = pred_next_obs.reshape(-1,)
                x_hat_dot = pred_next_obs-obs+k*(obs-x_hat)
                next_x_hat = x_hat + x_hat_dot
                next_policy_action = self.policy.select_action(torch.as_tensor(next_obs).to(device), True)
                next_state = self._prepare_state(next_obs, next_x_hat, next_policy_action, k)
                self.buffer.add(state, next_state, res_action, reward, done)
                ep_reward+=reward
                obs = next_obs
                x_hat = next_x_hat 
                # update the policy
                n_trainint_step = update_ratio*(total_t-t)
                self.train_epoch(n_trainint_step, batch_size)
                if total_t%self.EVAL_EVERY==0:
                    eval_reward, std = self.evaluate(k, res_agent=True, deterministic=True, num_eval=20)
                    self.logger.set_timestep(total_t)
                    self.logger.logkv('eval_reward', eval_reward)
                    self.logger.logkv('eval_reward_std', std)  
                    self.logger.dumpkvs() 
                    if eval_reward>best_reward:
                        self.save()
                        best_reward = eval_reward
                if total_t%self.SAVE_EVERY==0:
                    self.save(str(total_t))
            if total_t>max_steps:
                break
            ep_reward = self.env.get_normalized_score(ep_reward)*100
            self.logger.logkv('reward', ep_reward)      
            self.logger.set_timestep(total_t) 
            self.logger.dumpkvs()  


    def train_continuous(self, max_steps:int, update_ratio: int=2, batch_size:int=256, k:float=0.1):
        device = self.real_buffer.device
        policy_reward, std = self.evaluate(k, res_agent=False)
        self.logger.logkv('eval_reward', policy_reward)
        self.logger.logkv('eval_reward_std', std)  
        self.logger.dumpkvs()
        self.logger.log('Training the agent')
        self.logger.log(f'Initial policy reward: {policy_reward:.2f}')
        total_t = 0
        best_reward = 0
        while True:
            obs, done = self.env.reset(), False
            x_hat = obs # estimated state
            ep_reward = 0
            while not done and total_t<=max_steps:
                total_t+=1
                policy_act = self.policy.select_action(torch.as_tensor(obs).to(device), True) # policy action
                state = self._prepare_state(obs, x_hat, policy_act, k)
                self.normalizer.update(state.reshape(1,-1)) # add batch dimention
                normalized_state = self.normalizer.normalize(state)
                res_action = self.agent.select_action(torch.as_tensor(normalized_state).to(device), False)
                action = self._prepare_action(policy_act, res_action, True) 
                next_obs, reward, done, info = self.env.step(action)
                pred_next_obs, *_ = self.dynamics.step(obs.reshape(1,-1), action.reshape(1,-1))
                pred_next_obs = pred_next_obs.reshape(-1,)
                x_hat_dot = pred_next_obs-obs+k*(obs-x_hat)
                next_x_hat = x_hat + x_hat_dot
                next_policy_action = self.policy.select_action(torch.as_tensor(next_obs).to(device), True)
                next_state = self._prepare_state(next_obs, next_x_hat, next_policy_action, k)
                self.buffer.add(state, next_state, res_action, reward, done)
                ep_reward+=reward
                obs = next_obs
                x_hat = next_x_hat 
                # update the policy
                loss = self.train_sample(update_ratio, batch_size)
                for key,val in loss.items():
                    self.logger.logkv_mean(key, val)
                if total_t%self.EVAL_EVERY==0:
                    eval_reward, std = self.evaluate(k, res_agent=True, deterministic=True, num_eval=20)
                    self.logger.set_timestep(total_t)
                    self.logger.logkv('eval_reward', eval_reward)
                    self.logger.logkv('eval_reward_std', std)  
                    self.logger.dumpkvs()
                    if eval_reward>best_reward:
                        self.save()
                        best_reward = eval_reward
                if total_t%self.SAVE_EVERY==0:
                    self.save(str(total_t))
            if total_t>max_steps:
                break
            ep_reward = self.env.get_normalized_score(ep_reward)*100
            self.logger.logkv('reward', ep_reward)      
            self.logger.set_timestep(total_t) 
            self.logger.dumpkvs()                   


class AdaptiveAgentTrainer(ResidualAgentTrainer):
    def __init__(
            self, 
            env: gym.Env,
            eval_env:gym.Env,
            policy: BasePolicy, 
            dynamics: BaseDynamics, 
            residual_agent: (SACPolicy, TD3Policy), 
            real_buffer: ReplayBuffer, 
            buffer: ReplayBuffer,
            logger: Logger,
            res_action_coef:float = 1,
    ):
        super().__init__(
            env, eval_env, policy, dynamics, residual_agent, real_buffer, buffer, logger, res_action_coef
        )
    
    def train_episodic(self, max_steps:int, update_ratio: int=2, batch_size:int=256, k:float=0.1):
        device = self.real_buffer.device
        policy_reward, std = self.evaluate(k, res_agent=False)
        self.logger.logkv('eval_reward', eval_reward)
        self.logger.logkv('eval_reward_std', std)  
        self.logger.dumpkvs()
        self.logger.log('Training the agent')
        self.logger.log(f'Initial policy reward: {policy_reward:.2f}')
        total_t = 0
        t = 0
        best_reward = 0
        while True:
            obs, done = self.env.reset(), False
            x_hat = obs # estimated state
            ep_reward = 0
            ep_residual_reward = 0
            while not done and total_t<=max_steps:
                total_t+=1
                policy_act = self.policy.select_action(torch.as_tensor(obs).to(device), True) # policy action
                state = self._prepare_state(obs, x_hat, policy_act, k)
                res_action = self.agent.select_action(torch.as_tensor(state).to(device), False)
                action = self._prepare_action(policy_act, res_action, True) 
                next_obs, reward, done, info = self.env.step(action)
                pred_next_obs, *_ = self.dynamics.step(obs.reshape(1,-1), action.reshape(1,-1))
                pred_next_obs = pred_next_obs.reshape(-1,)
                x_hat_dot = pred_next_obs-obs+k*(obs-x_hat)
                next_x_hat = x_hat + x_hat_dot
                next_policy_action = self.policy.select_action(torch.as_tensor(next_obs).to(device), True)
                next_state = self._prepare_state(next_obs, next_x_hat, next_policy_action, k)
                res_reward = -np.log(np.mean((next_obs-pred_next_obs)**2))
                ep_residual_reward+=res_reward
                self.buffer.add(state, next_state, res_action, res_reward, done)
                ep_reward+=reward
                obs = next_obs
                x_hat = next_x_hat 
                # update the policy
                n_trainint_step = update_ratio*(total_t-t)
                self.train_epoch(n_trainint_step, batch_size)
                if total_t%self.EVAL_EVERY==0:
                    eval_reward, std = self.evaluate(k, res_agent=True, deterministic=True, ts=ts, num_eval=20)
                    self.logger.set_timestep(total_t)
                    self.logger.logkv('eval_reward', eval_reward)
                    self.logger.logkv('eval_reward_std', std)  
                    self.logger.dumpkvs()
                    if eval_reward>best_reward:
                        self.save()
                        best_reward = eval_reward
                if total_t%self.SAVE_EVERY==0:
                    self.save(str(total_t))
            if total_t>max_steps:
                break
            ep_reward = self.env.get_normalized_score(ep_reward)*100
            self.logger.logkv('reward', ep_reward)     
            self.logger.logkv('residual reward', ep_residual_reward) 
            self.logger.set_timestep(total_t) 
            self.logger.dumpkvs()

    def train_continuous(self, max_steps:int, update_ratio: int=2, batch_size:int=256, k:float=0.1, ts:float=0.05):
        device = self.real_buffer.device
        policy_reward, std= self.evaluate(k, res_agent=False)
        self.logger.logkv('eval_reward', policy_reward)
        self.logger.logkv('eval_reward_std', std)  
        self.logger.dumpkvs()
        self.logger.log('Training the agent')
        self.logger.log(f'Initial policy reward: {policy_reward:.2f}')
        total_t = 0
        best_reward = 0
        while True:
            obs, done = self.env.reset(), False
            x_hat = obs # estimated state
            ep_reward = 0
            ep_residual_reward = 0
            while not done and total_t<=max_steps:
                total_t+=1
                policy_act = self.policy.select_action(torch.as_tensor(obs).to(device), True) # policy action
                state = self._prepare_state(obs, x_hat, policy_act, k)
                res_action = self.agent.select_action(torch.as_tensor(state).to(device), False)
                action = self._prepare_action(policy_act, res_action, True) 
                next_obs, reward, done, info = self.env.step(action)
                pred_next_obs, *_ = self.dynamics.step(obs.reshape(1,-1), action.reshape(1,-1))
                pred_next_obs = pred_next_obs.reshape(-1,)
                x_hat_dot = pred_next_obs-obs+k*(obs-x_hat)
                next_x_hat = x_hat + x_hat_dot
                next_policy_action = self.policy.select_action(torch.as_tensor(next_obs).to(device), True)
                next_state = self._prepare_state(next_obs, next_x_hat, next_policy_action, k)
                res_reward = -np.log(np.mean((next_obs-pred_next_obs)**2))
                ep_residual_reward+=res_reward
                self.buffer.add(state, next_state, res_action, res_reward, done)
                ep_reward+=reward
                obs = next_obs
                x_hat = next_x_hat 
                # update the policy
                loss = self.train_sample(update_ratio, batch_size)
                for key,val in loss.items():
                    self.logger.logkv_mean(key, val)
                    
                if total_t%self.EVAL_EVERY==0:
                    self.logger.set_timestep(total_t)
                    eval_reward, std = self.evaluate(k, res_agent=True, deterministic=True, ts=ts, num_eval=20)
                    self.logger.set_timestep(total_t)
                    self.logger.logkv('eval_reward', eval_reward)
                    self.logger.logkv('eval_reward_std', std)  
                    self.logger.dumpkvs()
                    if eval_reward>best_reward:
                        self.save()
                        best_reward = eval_reward
                if total_t%self.SAVE_EVERY==0:
                    self.save(str(total_t))
            if total_t>max_steps:
                break
            ep_reward = self.env.get_normalized_score(ep_reward)*100
            self.logger.logkv('reward', ep_reward)     
            self.logger.logkv('residual_reward', ep_residual_reward) 
            self.logger.set_timestep(total_t) 
            self.logger.dumpkvs() 