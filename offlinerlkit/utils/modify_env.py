import gym
import d4rl
import numpy as np

class ModifiedENV(gym.Env):
    def __init__(self, env, max_delta=0.2):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._max_delta = max_delta
        self.original_body_mass = env.env.wrapped_env.model.body_mass.copy()
    
    def reset(self,):
        model = self._env.env.wrapped_env.model
        n_link = model.body_mass.shape[0]
        ind = np.random.randint(n_link)
        delta = np.random.uniform(-self._max_delta, self._max_delta)
        for i in range(n_link):
            model.body_mass[i] = self.original_body_mass[i]
        model.body_mass[ind]*=(delta+1)
        return self._env.reset()
    
    def step(self, action):
        return self._env.step(action)
    
    def get_normalized_score(self, score):
        return self._env.get_normalized_score(score)
    

class SimpleModifiedENV(gym.Env):
    def __init__(self, env, max_delta=0.2):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        model = self._env.env.wrapped_env.model
        model.body_mass[1]*=1+max_delta

    def reset(self,):
        return self._env.reset()
    
    def step(self, action):
        return self._env.step(action)
    
    def get_normalized_score(self, score):
        return self._env.get_normalized_score(score)
    

class SemiSimpleModifiedENV(gym.Env):
    def __init__(self, env, max_delta=0.2):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._max_delta = max_delta
        self.original_body_mass = env.env.wrapped_env.model.body_mass.copy()
    
    def reset(self,):
        model = self._env.env.wrapped_env.model
        n_link = model.body_mass.shape[0]
        ind = 1
        delta = np.random.uniform(-self._max_delta, self._max_delta)
        for i in range(n_link):
            model.body_mass[i] = self.original_body_mass[i]
        model.body_mass[ind]*=(delta+1)
        return self._env.reset()
    
    def step(self, action):
        return self._env.step(action)
    
    def get_normalized_score(self, score):
        return self._env.get_normalized_score(score)
    
class MassDampingENV(gym.Env):
    def __init__(self, env, max_delta=0.2):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._max_delta = max_delta
        self.mass_ratios = (0.75, 0.85, 1, 1.15, 1.25)
        self.damping_ratios = (0.75, 0.85, 1, 1.15, 1.25)
        self.original_body_mass = env.env.wrapped_env.model.body_mass.copy()
        self.original_damping = env.env.wrapped_env.model.dof_damping.copy()
    
    def reset(self,):
        model = self._env.env.wrapped_env.model
        n_link = model.body_mass.shape[0]
        ind_mass = np.random.randint(len(self.mass_ratios))
        ind_damp = np.random.randint(len(self.damping_ratios))
        for i in range(n_link):
            model.body_mass[i] = self.original_body_mass[i]*self.mass_ratios[ind_mass]
            model.dof_damping[i] = self.original_damping[i]*self.damping_ratios[ind_damp]
        return self._env.reset()
    
    def step(self, action):
        return self._env.step(action)
    
    def get_normalized_score(self, score):
        return self._env.get_normalized_score(score)