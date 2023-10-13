import argparse
import os
import random
import gym
import pickle
import numpy as np
import torch
import pandas as pd
import d4rl 
from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, KoopmanDynamicModel
from offlinerlkit.dynamics import KoopmanDynamics, KoopmanDynamicsOneStep, KoopmanDynamicsCosine
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.logger import make_log_dirs
from offlinerlkit.policy import COMBOPolicy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="combo")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    parser.add_argument("--uniform-rollout", type=bool, default=True)
    parser.add_argument("--rho-s", type=str, default="model", choices=["model", "mix"])

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--load-dynamics-path", type=str, default=None)

    parser.add_argument("--embedding-dim", type=int, default=7)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--dynamic-epoch", type=int, default=None)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--adaptive_gain", type=float, default=-100)
    parser.add_argument("--k", type=float, default=0.01,) 
    return parser.parse_args()


def train(args=get_args(), delta_m=1):
    # create env and dataset
    env = gym.make(args.task)
    model = env.env.wrapped_env.model
    model.body_mass[1] *= delta_m    
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    dynamics_model = KoopmanDynamicModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        latent_dim=args.embedding_dim,
        hidden_dims=args.dynamics_hidden_dims,
        device=args.device, 
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr,
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)
    dynamics = KoopmanDynamicsCosine(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn
    )

    # create policy
    policy = COMBOPolicy(
        dynamics,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions,
        uniform_rollout=args.uniform_rollout,
        rho_s=args.rho_s
    )

    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    dynamic_base_dirs = make_log_dirs(args.task, 'koopman-onestep-cosine', args.seed, vars(args))
    # key: output file name, value: output handler type
    load_policy_path = os.path.join(log_dirs, 'checkpoint/policy.pth')
    load_dynamic_path = os.path.join(dynamic_base_dirs, 'model')
    dynamics.load(load_dynamic_path)
    policy.load_state_dict(torch.load(load_policy_path))
    l1controller = L1Controller(dynamics_model, args.adaptive_gain, args.k, 0.002)
    print(f'Evaluation: Delta: {delta_m}, adaptive gain:{args.adaptive_gain}, k:{args.k}')
    results = evaluate(env, policy, n_eval=10, L1=l1controller)
    os.makedirs(f'result/{args.task}_seed{args.seed}_cos2', exist_ok=True)
    with open(f'result/{args.task}_seed{args.seed}_cos2/adaptive-gain_{abs(args.adaptive_gain)}_k_{args.k}_delta_{delta_m}.pkl', 'wb') as f:
        pickle.dump(results, f)


class L1Controller():
    """
    L1 addaptive controller based on nearal network approximator for dynamic equation 
    Arguements:
        addaptive_gain: control gain used to estimate uncertainties
        k: control gain used in determining adaptive action
        ts: time sampling rate (could be different from the real time sampling, better to be the same)        
    """
    def __init__(self, dynamic_model, adaptive_gain, k, ts=0.02):
        # system nominal parameter
        self.ts = ts
        self.net = dynamic_model.to('cpu')
        self.f = dynamic_model.A().cpu().numpy()
        self.g = dynamic_model.B().cpu().numpy()
        self.n = dynamic_model.n_koopman # state size
        self.m = dynamic_model.action_dim # action size
        self.adaptive_gain = adaptive_gain*np.diag([1.0]*self.n)
        self.k = k
        self.x_hat = np.zeros(self.n) # state estimation
        self.u_adaptive = np.zeros(self.m)

    def reset(self, x_hat = None):
        if x_hat is None:
            self.x_hat = np.zeros(self.n)
        else:
            self.x_hat = np.array(x_hat)
        self.u_adaptive = np.zeros(self.m)

    # estimate uncertainties
    def adaptive_law(self, x,):
        x_tilda = self.x_hat-x
        b_sigma = np.matmul(self.adaptive_gain, x_tilda.reshape(self.n,1))
        sigma_m = np.linalg.pinv(self.g)@b_sigma
        return b_sigma.reshape(self.n), sigma_m.reshape(self.m)

    def state_predictor(self, u, b_sigma, x):
        x_hat_dot = self.f@x.reshape(self.n,1) + self.g@u.reshape(self.m,1)
        x_hat_dot = x_hat_dot.reshape(self.n)+b_sigma
        return x_hat_dot
 
    # correct the action
    def u_next(self, u_policy, sigma_m):
        u = -self.u_adaptive + u_policy
        eta_hat = self.u_adaptive + sigma_m
        u_dot = -self.k*eta_hat
        next_u = u + u_dot*self.ts
        self.u_adaptive = next_u-u_policy
        return next_u

    # main function for each timestep
    def step(self, x, u_policy):
        # approximate f and b vector (dynamic equation) from learned NN
        with torch.no_grad():
            x = self.net.encode(torch.as_tensor(x, dtype=torch.float32))
        x = x.cpu().numpy()
        b_sigma, sigma_m = self.adaptive_law(x)
        x_hat_dot = self.state_predictor(u_policy+self.u_adaptive, b_sigma, x)
        u_next = self.u_next(u_policy, sigma_m)
        self.x_hat = self.x_hat + x_hat_dot*self.ts
        return u_next, sigma_m, x

def evaluate(env, policy, L1=None, n_eval=10,):
    results = {}
    rewards = np.zeros(n_eval)
    for i in range(n_eval):
        obs = env.reset()
        if L1:
            L1.reset()
        ep_reward = 0
        states, x_hats, sigma_hats, adaptive_actions, policy_actions = [],[],[],[],[]
        while True:
            action = policy.select_action(obs)
            if L1:

                action, sigma_hat, x = L1.step(obs, action)
                states.append(x)
                x_hats.append(L1.x_hat)
                sigma_hats.append(sigma_hat)
                adaptive_actions.append(L1.u_adaptive)
                policy_actions.append(action-L1.u_adaptive)
            obs, reward, done, _ = env.step(action)
            ep_reward+=reward
            if done:
                rewards[i] = ep_reward
                ep_result = {
                    'states': states,
                    'x_hats': x_hats,
                    'sigma_hats': sigma_hats,
                    'adaptive_actions': adaptive_actions,
                    'policy_actions': policy_actions,
                    'total_reward': ep_reward
                }
                results[i] = ep_result
                break
    rewards = env.get_normalized_score(rewards)
    print(f'Mean reward {100*np.mean(rewards):.2f}, std reward {100*np.std(rewards):.2f}')
    return results

if __name__ == "__main__":
    for delta_m in (1, 1.2, 1.5, 2, 0.8, 0.6, 0.5):
        for k in (0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5):
            for adaptive_gain in (0, -1, -10, -50, -75, -100, -200, -500):
                args = get_args()
                args.k = k
                args.adaptive_gain = adaptive_gain
                train(args=args, delta_m=delta_m)