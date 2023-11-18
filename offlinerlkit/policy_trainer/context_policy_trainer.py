import gym
import numpy as np
import torch
from torch.utils.data import DataLoader

from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import SACPolicy, TD3Policy
from offlinerlkit.buffer import NSequenceBuffer
from offlinerlkit.nets import EncoderModule

class ContextAgentTrainer:
    EVAL_EVERY = 5000
    SAVE_EVERY = 50000
    def __init__(
            self, 
            env: gym.Env,
            eval_env: gym.Env,
            policy: SACPolicy, 
            residual_agent: (SACPolicy, TD3Policy), 
            encoder: EncoderModule,
            buffer: NSequenceBuffer,
            logger: Logger,
            num_worker: int=4,
            batch_size: int=64,
            seq_len: int=10,
            device: str='cpu', 
            coeff:float = 0.9,
    ):
        self.env = env
        self.eval_env = eval_env
        self.policy = policy
        self.agent = residual_agent
        self.buffer = buffer
        self.logger = logger
        self.encoder = encoder
        self.data_loader = DataLoader(self.buffer, batch_size=batch_size, num_workers=num_worker)
        
        self.seq_len = seq_len
        self.device = device
        self.agent.to(device)
        self.policy.to(device)
        self.total_t = 0
        self.coeff = coeff

    # collect data only using offline policy
    def warm_up(self, n_episodes:int=20):
        for i in range(n_episodes):
            obs, done = self.env.reset(), False
            ep_states = [obs,]
            ep_actions = []
            ep_rewards = []
            while not done:
                action = self.policy.select_action(torch.as_tensor(obs).to(self.device), False) # offline policy action
                obs, reward, done, info = self.env.step(action)
                ep_states.append(obs)
                ep_actions.append(action)
                ep_rewards.append(reward)
                self.total_t+=1
            ep_states, ep_actions, ep_rewards = [np.array(lst, dtype=np.float32) for lst in (ep_states, ep_actions, ep_rewards)]
            self.buffer.add_traj(dict(
                states=ep_states, actions=ep_actions, rewards=ep_rewards
            ))
        self.train_iter = iter(self.data_loader) # call next each time for training

    @torch.no_grad()
    def evaluate(self, res_agent:bool=True, deterministic:bool=True, num_eval:int=20):
        rewards = np.zeros(num_eval)
        for n in range(num_eval):
            obs, done = self.eval_env.reset(), False
            ep_reward = 0
            # used for saving episode to the buffer
            ep_states = [obs,]
            ep_actions = []
            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
                # action of offline RL agent
                with torch.no_grad():
                    offline_act, _ = self.policy.actforward(obs_tensor, True) # offline policy action
                # first step, no context encoder to be used 
                if len(ep_actions)<self.seq_len or not res_agent:
                    action = offline_act.cpu().numpy()
                else:
                    start_seq = -self.seq_len if len(ep_states)<-self.seq_len else -self.seq_len-1 # same length for state sequence 
                    context_states = np.array(ep_states[start_seq:-1], dtype=np.float32)
                    context_actions = np.array(ep_actions[-self.seq_len:], dtype=np.float32)
                    context_state = torch.from_numpy(context_states).unsqueeze(0).to(self.device) # remove last state, add batch dimension
                    context_actions = torch.from_numpy(context_actions).unsqueeze(0).to(self.device)
                    time_step = torch.arange(context_actions.shape[1]).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        encoded = self.encoder(context_state, context_actions, time_step).squeeze(0) # context encoder
                        res_state = self._res_state(obs_tensor, offline_act, encoded) # augmented state
                        res_action = self.agent.select_action(res_state, deterministic=deterministic) 
                        action = self._total_action(offline_act.cpu().numpy(), res_action)                        
                obs, reward, done, info = self.eval_env.step(action)
                ep_actions.append(action)
                ep_states.append(obs)
                ep_reward += reward
            rewards[n] = ep_reward
        ep_reward = self.eval_env.get_normalized_score(np.mean(rewards))*100
        std = self.eval_env.get_normalized_score(np.std(rewards))*100
        return ep_reward, std   

    def _res_state(self, obs:torch.Tensor, policy_action:torch.Tensor, encoded:torch.Tensor):
        return torch.cat([obs, policy_action, encoded], dim=-1)
    
    def _total_action(self, offline_action, context_action):
        return self.coeff*offline_action + (1-self.coeff)*context_action
    
    def _agent_action(self, total_action, offline_action):
        return (total_action-self.coeff*offline_action)/(1-self.coeff)
    
    def _prepare_training_samples(self, device:str='cpu'):
        batch = next(self.train_iter)
        seq_states, seq_actions, seq_masks, state, action, reward, next_state, done = [tensor.to(device) for tensor in batch]
        batch_encoder = dict(
            seq_states = seq_states, 
            seq_actions = seq_actions, 
            seq_masks = seq_masks, 
            state = state, 
            action = action, 
            next_state = next_state
            )
        with torch.no_grad():
            offline_act, _ = self.policy.actforward(state, True) # offline policy action
            next_offline_act, _ = self.policy.actforward(next_state, True)
            latents = self.encoder.encode_multiple(seq_states, seq_actions, seq_masks)
            latents = latents.mean(1) # batch dim, N, latent dim
        observations = self._res_state(state, offline_act, latents)
        next_observations = self._res_state(next_state, next_offline_act, latents)
        batch_agent = dict(
            observations = observations,
            actions = self._agent_action(action, offline_act),
            next_observations=next_observations,
            rewards = reward,
            terminals = done.to(torch.float32),
        )
        return batch_encoder, batch_agent
    
    def save(self, tag:str='best'):
        path_agent = f'{self.logger.model_dir}/residual_agent_{tag}.pth'
        path_encoder = f'{self.logger.model_dir}/encoder_{tag}.pth'
        path_predictor = f'{self.logger.model_dir}/predictor_{tag}.pth'
        torch.save(self.agent.state_dict(), path_agent)
        torch.save(self.encoder.encoder.state_dict(), path_encoder)
        torch.save(self.encoder.predictor.state_dict(), path_predictor)
    
    def train_sample(self, num_step:int,):
        self.agent.train()
        self.encoder.train()
        for _ in range(num_step):
            batch_encoder, batch_agent = self._prepare_training_samples(self.device)
            loss_encoder = self.encoder.learn_batch(batch_encoder)
            loss_agent = self.agent.learn(batch_agent)
        loss_agent.update(loss_encoder)
        return loss_agent
        
    def run(self, max_step:int=int(5e5), warm_up:int=20, update_ratio:int=1):
        policy_reward, std = self.evaluate(res_agent=False)
        self.logger.logkv('eval_reward', policy_reward)
        self.logger.logkv('eval_reward_std', std)  
        self.logger.dumpkvs()
        self.logger.log('Training the agent')
        self.logger.log(f'Initial policy reward: {policy_reward:.2f}')
        self.warm_up(warm_up)
        self.train_sample(1000)
        best_reward = 0
        while True:
            # start of the episode
            obs, done = self.env.reset(), False
            ep_reward = 0
            # used for saving episode to the buffer
            ep_states = [obs,]
            ep_actions = []
            ep_rewards = []
            while not done and self.total_t<=max_step:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
                # action of offline RL agent
                with torch.no_grad():
                    offline_act, _ = self.policy.actforward(obs_tensor, True) # offline policy action
                # first step, no context encoder to be used 
                if len(ep_actions)<self.seq_len:
                    action = offline_act.cpu().numpy()
                else:
                    start_seq = -self.seq_len if len(ep_states)<-self.seq_len else -self.seq_len-1 # same length for state sequence 
                    context_states = np.array(ep_states[start_seq:-1], dtype=np.float32)
                    context_actions = np.array(ep_actions[-self.seq_len:], dtype=np.float32)
                    context_state = torch.from_numpy(context_states).unsqueeze(0).to(self.device) # remove last state, add batch dimension
                    context_actions = torch.from_numpy(context_actions).unsqueeze(0).to(self.device)
                    time_step = torch.arange(context_actions.shape[1]).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        encoded = self.encoder(context_state, context_actions, time_step).squeeze(0) # context encoder, remove batch dim
                        res_state = self._res_state(obs_tensor, offline_act, encoded) # augmented state
                        res_action = self.agent.select_action(res_state)
                        action = self._total_action(offline_act.cpu().numpy(), res_action)
                obs, reward, done, info = self.env.step(action)
                ep_states.append(obs)
                ep_actions.append(action)
                ep_rewards.append(reward)
                ep_reward += reward
                self.total_t+=1

                loss = self.train_sample(update_ratio)
                for key,val in loss.items():
                    self.logger.logkv_mean(key, val)
                if self.total_t%self.EVAL_EVERY==0:
                    eval_reward, std = self.evaluate(res_agent=True, deterministic=True, num_eval=20)
                    self.logger.set_timestep(self.total_t)
                    self.logger.logkv('eval_reward', eval_reward)
                    self.logger.logkv('eval_reward_std', std)  
                    self.logger.dumpkvs()
                    if eval_reward>best_reward:
                        self.save()
                        best_reward = eval_reward
                if self.total_t%self.SAVE_EVERY==0:
                    self.save(str(self.total_t))
            if self.total_t>max_step:
                break
            ep_states, ep_actions, ep_rewards = [np.array(lst, dtype=np.float32) for lst in (ep_states, ep_actions, ep_rewards)]
            self.buffer.add_traj(dict(
                states=ep_states, actions=ep_actions, rewards=ep_rewards
            ))        
            self.train_iter = iter(self.data_loader)   
            ep_reward = self.env.get_normalized_score(ep_reward)*100
            self.logger.logkv('reward', ep_reward)      
            self.logger.set_timestep(self.total_t) 
            self.logger.dumpkvs()  


                    







        
    

