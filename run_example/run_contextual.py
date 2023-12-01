import argparse
import os
import random
import gym
import numpy as np
import torch
from torch.utils.data import DataLoader
import d4rl 

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.logger import make_log_dirs, load_args
from offlinerlkit.policy import COMBOPolicy, SACPolicy
from offlinerlkit.buffer import NSequenceBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.utils.modify_env import ModifiedENV, SemiSimpleModifiedENV, SimpleModifiedENV, MassDampingENV
from offlinerlkit.policy_trainer import ContextAgentTrainer
from offlinerlkit.nets import ContextEncoder, ContextPredictor, EncoderModule2, EncoderConv, EncoderModule
from offlinerlkit.utils.load_dataset import qlearning_dataset, transform_to_episodic

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="combo")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--max-delta", type=float, default=0.2, help='maximum perturbation in mass')
    parser.add_argument("--env-mode", type=str, default="def", 
                        help='def, simple, semi-simple or complex environment, different types of perturbation')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline-seed", type=int, default=40, help='seed of loaded offline agent')
    parser.add_argument("--max-steps", type=int, default=int(1e6))
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
    parser.add_argument("--buffer-size", type=int, default=1000, help='number of trajectories in the buffer') 
    parser.add_argument("--latent-dim", type=int, default=8, help='encoder output dim')
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4, help='number of cpu cores used to prepare data')
    parser.add_argument('--encoder-type', type=str, default='cnn') # or transformer
    parser.add_argument("--encoder-layers", type=int, default=1, help='number of layers in encoder (transformer)')
    parser.add_argument("--encoder-heads", type=int, default=4, help='number of heads in multi-head attention')
    parser.add_argument("--encoder-lr", type=float, default=1e-4, help='learning rate for encoder')
    parser.add_argument("--pre-train-steps", type=int, default=20000, help='number of gradient steps to pretrain encoder from offline data')
    parser.add_argument("--encoder-points", type=int, default=4, 
                        help='number of subtrajectories used to train the encoder, mean value is used for training')
    parser.add_argument("--coeff", type=float, default=0.6, help='coefficient for context/offline actions')
    parser.add_argument("--hidden-dims-predictor", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--loss-sim", type=float, default=0.2, help='Coefficient for similarity loss in N points from one trajectory')
    parser.add_argument("--k-step", type=int, default=5, help='Number of future steps for prediction loss in training the encoder')
    parser.add_argument("--tag", type=str, default='', help='used for logging')
    return parser.parse_args()


def train(mainargs=get_args()):
    load_path = f'log/{mainargs.task}/{mainargs.algo_name}/seed_{mainargs.offline_seed}'
    args = load_args(os.path.join(load_path, 'record/hyper_param.json'))
    args.action_dim = int(args.action_dim) # when loaded from json file, default is float
    mainargs.device = args.device
    # create env and dataset
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)
    eval_env = gym.make(args.task)
    EnvMode = {
        'simple': SimpleModifiedENV,
        'semi-simple': SemiSimpleModifiedENV,
        'complex': ModifiedENV,
        'def': MassDampingENV
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

    if mainargs.encoder_type == 'transformer':
        encoder = ContextEncoder(
            state_dim = np.prod(args.obs_shape), 
            action_dim = args.action_dim, 
            seq_len = mainargs.seq_len, 
            output_dim = mainargs.latent_dim, 
            num_layers=mainargs.encoder_layers,
            num_heads=mainargs.encoder_heads,
            )
    else:
        encoder = EncoderConv(
            state_dim = np.prod(args.obs_shape),
            action_dim = args.action_dim,
            seq_len = mainargs.seq_len, 
            output_dim = mainargs.latent_dim,
        )
    predictor = ContextPredictor(
        state_dim = np.prod(args.obs_shape), 
        action_dim = args.action_dim, 
        latent_dim = mainargs.latent_dim, 
        hidden_dim = mainargs.hidden_dims_predictor       
    )
    encoder.to(args.device)
    predictor.to(args.device)
    mu_state = torch.tensor(scaler.mu, device=args.device, dtype=torch.float32).squeeze(0)[:-args.action_dim]
    std_state = torch.tensor(scaler.std, device=args.device, dtype=torch.float32).squeeze(0)[:-args.action_dim]
    # print(mu_state.shape, std_state.shape)
    encoder_module = EncoderModule2(
        encoder, predictor, lr=mainargs.encoder_lr, 
        alpha_sim=mainargs.loss_sim, k_steps=mainargs.k_step, 
        mu=mu_state, std=std_state, 
        )
    
    # pre_train_encoder(encoder_module, dataset, mainargs)

    buffer = NSequenceBuffer(mainargs.buffer_size, mainargs.seq_len+mainargs.k_step, mainargs.encoder_points)
    obs_shape_res = np.prod(args.obs_shape) + args.action_dim + mainargs.latent_dim

    res_actor_backbone = MLP(input_dim=obs_shape_res, hidden_dims=mainargs.hidden_dims)
    res_critic1_backbone = MLP(input_dim=obs_shape_res+args.action_dim, hidden_dims=mainargs.hidden_dims)
    res_critic2_backbone = MLP(input_dim=obs_shape_res+args.action_dim, hidden_dims=mainargs.hidden_dims)
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

    

    # log
    log_dirs = make_log_dirs(f'{args.task}-context', f'normalized_encoder', mainargs.seed, vars(mainargs), record_params=['tag'])
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(mainargs))

    res_agent = ContextAgentTrainer(
        env = env, eval_env = eval_env, policy = policy,
        residual_agent = res_policy, encoder = encoder_module,
        buffer=buffer, logger=logger, 
        batch_size = mainargs.batch_size, 
        num_worker = mainargs.num_workers,
        seq_len = mainargs.seq_len,
        device=args.device,
        coeff=mainargs.coeff,
        )
    # res_agent.pre_train(10, 100)
    res_agent.run(max_step=mainargs.max_steps)



def pre_train_encoder( encoder:EncoderModule2, data, args):
    episodes = transform_to_episodic(data)
    n_episodes = len(episodes)
    print(episodes)
    buffer = NSequenceBuffer(n_episodes, args.seq_len, args.encoder_points)
    for episode in episodes:
        buffer.add_traj({
            'states': episode['observations'],
            'actions': episode['actions'],
            'rewards': episode['rewards'],
        })
    data_loader = DataLoader(buffer, batch_size=args.batch_size, num_workers=args.num_workers)
    itr = iter(data_loader)
    for i in range(args.pre_train_steps):
        batch = next(itr)
        seq_states, seq_actions, seq_masks, state, action, reward, next_state, done = [tensor.to(args.device) for tensor in batch]
        batch_encoder = dict(
            seq_states = seq_states, 
            seq_actions = seq_actions, 
            seq_masks = seq_masks, 
            state = state, 
            action = action, 
            next_state = next_state
            )
        losses = encoder.learn_batch(batch_encoder)
        if (i+1)%1000 ==0:
            print(f'Pre training step {i+1}')
            print('    '.join(f'{k}: {v:.5f}' for k,v in losses.items()))

if __name__ == '__main__':
    mainargs=get_args()
    train(mainargs)
