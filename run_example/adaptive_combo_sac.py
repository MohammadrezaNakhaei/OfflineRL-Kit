import argparse
import os
import random
import gym
import numpy as np
import torch
import d4rl 

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.logger import make_log_dirs, load_args
from offlinerlkit.policy import COMBOPolicy, SACPolicy
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.utils.modify_env import ModifiedENV, SemiSimpleModifiedENV, SimpleModifiedENV
from offlinerlkit.policy_trainer import ResidualAgentTrainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="combo")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--max-delta", type=float, default=0.2, help='maximum perturbation in mass')
    parser.add_argument("--env-mode", type=str, default="complex", 
                        help='simple, semi-simple or complex environment, different types of perturbation')
    parser.add_argument("--coeff-residual", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline-seed", type=int, default=40, help='seed of loaded offline agent')
    parser.add_argument("--max-steps", type=int, default=int(5e5))
    parser.add_argument('--n-update', type=int, default=1, help='number of updates in each samples')
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)    
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--tag", type=str, default='')
    return parser.parse_args()


def train(mainargs=get_args()):
    load_path = f'log/{mainargs.task}/{mainargs.algo_name}/seed_{mainargs.offline_seed}'
    args = load_args(os.path.join(load_path, 'record/hyper_param.json'))
    args.action_dim = int(args.action_dim) # when loaded from json file, default is float
    # create env and dataset
    env = gym.make(args.task)
    eval_env = gym.make(args.task)
    dataset = qlearning_dataset(env)
    EnvMode = {
        'simple': SimpleModifiedENV,
        'semi-simple': SemiSimpleModifiedENV,
        'complex': ModifiedENV,
    }
    env = EnvMode[mainargs.env_mode](env, mainargs.max_delta)
    eval_env = EnvMode[mainargs.env_mode](eval_env, mainargs.max_delta)
    # seed
    random.seed(mainargs.seed)
    np.random.seed(mainargs.seed)
    torch.manual_seed(mainargs.seed)
    torch.cuda.manual_seed_all(mainargs.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(mainargs.seed)
    eval_env.seed(mainargs.seed)
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
    actor_optim = None
    critic1_optim = None
    critic2_optim = None
    alpha = None

    # create dynamics
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = None
    scaler = StandardScaler()
    scaler.load_scaler(f'{load_path}/model')
    termination_fn = get_termination_fn(task=args.task)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn
    )
    dynamics.load(f'{load_path}/model')

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
    
    policy.load_state_dict(torch.load(f'{load_path}/checkpoint/policy.pth'))

    res_actor_backbone = MLP(input_dim=2*np.prod(args.obs_shape)+args.action_dim, hidden_dims=mainargs.hidden_dims)
    res_critic1_backbone = MLP(input_dim=2*np.prod(args.obs_shape)+2*args.action_dim, hidden_dims=mainargs.hidden_dims)
    res_critic2_backbone = MLP(input_dim=2*np.prod(args.obs_shape)+2*args.action_dim, hidden_dims=mainargs.hidden_dims)
    res_dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    res_actor = ActorProb(res_actor_backbone, res_dist, args.device)
    res_critic1 = Critic(res_critic1_backbone, args.device)
    res_critic2 = Critic(res_critic2_backbone, args.device)
    res_actor_optim = torch.optim.Adam(res_actor.parameters(), lr=mainargs.actor_lr)
    res_critic1_optim = torch.optim.Adam(res_critic1.parameters(), lr=mainargs.critic_lr)
    res_critic2_optim = torch.optim.Adam(res_critic2.parameters(), lr=mainargs.critic_lr)   

    if mainargs.auto_alpha:
        target_entropy = mainargs.target_entropy if mainargs.target_entropy \
            else -np.prod(env.action_space.shape)
        mainargs.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=mainargs.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = mainargs.alpha

    res_policy = SACPolicy(
        res_actor,
        res_critic1,
        res_critic2,
        res_actor_optim,
        res_critic1_optim,
        res_critic2_optim,
        tau=mainargs.tau,
        gamma=mainargs.gamma,
        alpha=alpha,
    )

    
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=tuple(args.obs_shape),
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)
    res_obs_shape = (args.obs_shape[0]*2+args.action_dim, )
    aug_buffer = ReplayBuffer(
        buffer_size=mainargs.buffer_size,
        obs_shape=res_obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    # log
    log_dirs = make_log_dirs(f'{args.task}-adaptive', f'{args.algo_name}_sac', mainargs.seed, vars(mainargs), record_params=['tag'])
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    res_agent = ResidualAgentTrainer(
        env, eval_env, policy, dynamics, res_policy, 
        real_buffer, aug_buffer, logger, 
        res_action_coef=mainargs.coeff_residual,
        )
    # res_agent.pre_train(10, 100)
    res_agent.train_continuous(mainargs.max_steps, mainargs.n_update, args.batch_size,)



if __name__ == '__main__':
    mainargs=get_args()
    train(mainargs)
