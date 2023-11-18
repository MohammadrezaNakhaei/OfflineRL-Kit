import gym
import numpy as np
import torch
from torch.utils.data import DataLoader

from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import SACPolicy, TD3Policy
from offlinerlkit.buffer import NSequenceBuffer
from offlinerlkit.nets import EncoderModule


from dataclasses import dataclass
@dataclass
class MPPI:
    horizon:int = 10
    iteration:int = 6
    num_samples:int = 512
    num_elites:int = 64
    temperature:float = 0.5
    min_std:float = 0.05
    max_std:float = 2
    num_pi_trajs:int = 48
    discount: float = 0.99



class ContextAgentPlanner:
    EVAL_EVERY = 5000
    SAVE_EVERY = 50000
    def __init__(
            self, 
            env: gym.Env,
            eval_env: gym.Env,
            policy: SACPolicy, 
            reward_model, 
            termination_fn,
            encoder: EncoderModule,
            buffer: NSequenceBuffer,
            logger: Logger,
            num_worker: int=4,
            batch_size: int=64,
            seq_len: int=10,
            device: str='cpu', 
    ):
        self.env = env
        self.eval_env = eval_env
        self.policy = policy
        self.reward_model = reward_model
        self.termination_fn = termination_fn
        self.buffer = buffer
        self.logger = logger
        self.encoder = encoder
        self.data_loader = DataLoader(self.buffer, batch_size=batch_size, num_workers=num_worker)
        
        self.seq_len = seq_len
        self.device = device
        self.policy.to(device)
        self.total_t = 0
        self.action_dim = self.env.action_space.shape[0]
        self.mppi = MPPI()

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
                        t0 = len(ep_actions)==1
                        action = self.plan(obs_tensor, encoded, t0, deterministic)
                obs, reward, done, info = self.eval_env.step(action)
                ep_actions.append(action)
                ep_states.append(obs)
                ep_reward += reward
            rewards[n] = ep_reward
        ep_reward = self.eval_env.get_normalized_score(np.mean(rewards))*100
        std = self.eval_env.get_normalized_score(np.std(rewards))*100
        return ep_reward, std   

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
        return batch_encoder
    
    def save(self, tag:str='best'):
        path_encoder = f'{self.logger.model_dir}/encoder_{tag}.pth'
        path_predictor = f'{self.logger.model_dir}/predictor_{tag}.pth'
        torch.save(self.encoder.encoder.state_dict(), path_encoder)
        torch.save(self.encoder.predictor.state_dict(), path_predictor)
    
    def train_sample(self, num_step:int,):
        self.encoder.train()
        for _ in range(num_step):
            batch_encoder = self._prepare_training_samples(self.device)
            loss_encoder = self.encoder.learn_batch(batch_encoder)
        return loss_encoder
        
    def run(self, max_step:int=int(5e5), warm_up:int=20, update_ratio:int=1):
        policy_reward, std = self.evaluate(res_agent=False)
        self.logger.logkv('eval_reward', policy_reward)
        self.logger.logkv('eval_reward_std', std)  
        self.logger.dumpkvs()
        self.logger.log('Training the agent')
        self.logger.log(f'Initial policy reward: {policy_reward:.2f}')
        self.warm_up(warm_up)
        self.train_sample(5000)
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
                        encoded = self.encoder(context_state, context_actions, time_step).squeeze(0) # context encoder
                        t0 = len(ep_actions)==self.seq_len
                        action = self.plan(obs_tensor, encoded, t0)
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

    @torch.no_grad()
    def estimate_value(self, state:torch.Tensor, z:torch.Tensor, actions:torch.Tensor):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.mppi.horizon):
            next_state = state+self.encoder.predictor(state, actions[t], z)
            reward = self.reward_model(next_state)
            state = next_state
            G += discount * reward
            discount *= self.mppi.discount
        G += discount * self.value(state) # critic bootstrapping
        return G

    @torch.no_grad()
    def plan(self, state:torch.Tensor, latent:torch.Tensor, t0:bool=False, deterministic:bool=False):
        """Plan a sequence of actions using the learned world model."""		
        obs = state.unsqueeze(0) # add batch dim
        action_dim = self.action_dim #TODO
        # Sample policy trajectories
        if self.mppi.num_pi_trajs > 0:
            pi_actions = torch.empty(self.mppi.horizon, self.mppi.num_pi_trajs, action_dim, device=self.device)
            _obs = obs.repeat(self.mppi.num_pi_trajs, 1)
            _latent = latent.repeat(self.mppi.num_pi_trajs, 1)
            for t in range(self.mppi.horizon-1):
                pi_actions[t], _ = self.policy.actforward(_obs, False)
                _obs = self.encoder.predictor(_obs, pi_actions[t], _latent)
            pi_actions[-1], _ = self.policy.actforward(_obs, False) # TODO: decide std
            self._pi_mean = pi_actions.mean(1)
            self._pi_std = pi_actions.std(1)

        # Initialize state and parameters
        obs = obs.repeat(self.mppi.num_samples, 1)
        latent = latent.repeat(self.mppi.num_samples, 1)
        mean = torch.zeros(self.mppi.horizon, action_dim, device=self.device)
        std = self.mppi.max_std*torch.ones(self.mppi.horizon, action_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.mppi.horizon, self.mppi.num_samples, action_dim, device=self.device)
        if self.mppi.num_pi_trajs > 0:
            actions[:, :self.mppi.num_pi_trajs] = pi_actions
    
        # Iterate MPPI
        for i in range(self.mppi.iteration):
            # Sample actions
            actions[:, self.mppi.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(self.mppi.horizon, self.mppi.num_samples-self.mppi.num_pi_trajs, action_dim, device=std.device)) \
                .clamp(-1, 1)
            # Compute elite actions
            value = self.estimate_value(obs, latent, actions).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.mppi.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.mppi.temperature*(elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
                .clamp_(self.mppi.min_std, self.mppi.max_std)
        
        # for warm start
        self._prev_mean = mean
        # Select action
        score = score.squeeze(1).cpu().numpy()
        elite_actions = elite_actions.cpu().numpy()
        if deterministic:
            actions = elite_actions[:, 0]
        else:
            actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        return actions[0].clip(-1, 1)
    
    @torch.no_grad()
    def value(self, state):
        """Compute the value function according to the offline policy, used to bootstrap in planning"""
        action, _ = self.policy.actforward(state)
        q1 = self.policy.critic1(state, action)
        q2 = self.policy.critic2(state, action)
        return torch.min(q1, q2)
    
    
    

        
    

