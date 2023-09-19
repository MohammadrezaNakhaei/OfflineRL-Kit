import argparse
import os
import sys
import random

import gym
import d4rl

import numpy as np
import torch


from offlinerlkit.modules import KoopmanDynamicModel
from offlinerlkit.dynamics import KoopmanDynamics, KoopmanDynamicsOneStep
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MBPolicyTrainer
from offlinerlkit.policy import COMBOPolicy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="koopman-onestep")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--model-retain-epochs", type=int, default=5)

    parser.add_argument("--embedding-dim", type=int, default=7)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--dynamic-epoch", type=int, default=None)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)
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
    dynamics = KoopmanDynamicsOneStep(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn
    )

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))
    dynamics.train(
        dataset, logger, 
        max_epochs_since_update=5, 
        max_epochs=args.dynamic_epoch)

if __name__ == "__main__":
    train()
    