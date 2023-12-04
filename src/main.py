import os
import time
import torch
import wandb
import random
import argparse
import numpy as np

from envs.env import CellEnv
from algorithms.iql.agent import IQL

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="CellEnv Training or Test module")

# env
parser.add_argument("-is", "--islet_num", type=int, default=20, help="Environment Islet number, default=20")
parser.add_argument("-mt", "--max_time", type=int, default=200, help="Environment Max time (min) per episode, default=20")
parser.add_argument("-g", "--glu_fix", action='store_true', help="Init glucose level fix on, action='store_true'")
parser.add_argument("-gl", "--glu_level", type=float, default=None, help="Init glucose level value, use when glu_fix is true, default=None")
parser.add_argument("-eg", "--external_glu", type=float, default=0, help="External glucose every timestep, default=0")
parser.add_argument("-rm", "--reward_mode", type=str, help="Choose reward mode, individual hormone secretion or total hormone secretion, ['local', 'global']")
parser.add_argument("-sd", "--seed", type=int, default=0, help="Seed")
# agent
parser.add_argument("-tr", "--train", action='store_true', help="Training mode on. action='store_true'")
parser.add_argument("-cu", "--cuda", action='store_true', help="Use cuda, action='store_true'")
parser.add_argument("-cn", "--cuda_num", type=int, default=0, help="Using cuda number, default=0")

# train
parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Training batch size, default=64")
parser.add_argument("-uf", "--target_update_freq", type=int, default=500, help="Target network update frequency, default=500")
parser.add_argument("-gm", "--gamma", type=float, default=0.995, help="Gamma(Reward decay) value, default=0.995")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Optimizer learning rate, default=1e-4")
parser.add_argument("-mc", "--memory_cap", type=int, default=1e5, help="Replay memory size, default=100_000")
parser.add_argument("-me", "--max_epi", type=int, default=10000, help="Maximum training episode, default=10000")
parser.add_argument("-el", "--eps_linear", action='store_true', help="Using epsilon decay as linear. If not, epsilon will decay exponentially, action='store_true'")
parser.add_argument("-ed", "--eps_decay", type=float, default=0.995, help="Epsilon decay on exponential decay epsilon")

# train/test
parser.add_argument("-pl", "--param_loc", type=str, default="../parameters/iql", help="Torch parameter saving/loading location, default='../parameters/iql'")
parser.add_argument("-ps", "--param_suffix", type=str, default=None, help="Parameter suffix, if empty, save/load name [alpha_cell.pth, beta_cell.pth, delta_cell.pth] in param_location")

# test
parser.add_argument("-p", "--plot", action='store_true', help="Used in test mode, plotting result, action='store_true'")
parser.add_argument("-pc", "--plot_dir", type=str, default="../image", help="Main directory plot saved, default='../image'")
parser.add_argument("-pn", "--plot_subdir", type=str, default=None, help="Sub directory plot saved under main directory, default=None")
parser.add_argument("-av", "--action_view", action='store_true', help="View action on terminal, action='store_true'")
parser.add_argument("-as", "--action_share", action='store_true', help="Using action share in one timestep, action='store_true'")


if __name__=="__main__":
    args = parser.parse_args()    
    seed_everything(args.seed)
    env = CellEnv(
        islet_num=args.islet_num,
        max_time=args.max_time,
        reward_mode=args.reward_mode)
    
    agent = IQL(
        env,
        is_train=args.train,
        cuda=args.cuda,
        cuda_num=args.cuda_num)

    if args.train:
        wandb.init(project="Tripartite Cell IQL - Server network 64")
        wandb.config.update(args)

        agent.train(
            batch_size=args.batch_size,
            target_update_freq=args.target_update_freq,
            gamma=args.gamma,
            memory_capacity=args.memory_cap,
            max_epi=args.max_epi,
            eps_decay=args.eps_decay,
            eps_linear=args.eps_linear,
            external_glu=args.external_glu,
            param_location=args.param_loc,
            param_suffix=args.param_suffix)

    else:
        agent.test(
            param_location=args.param_loc,
            param_suffix=args.param_suffix,
            external_glu=args.external_glu,
            glucose_fix=args.glu_fix,
            glucose_level=args.glu_level,
            action_share=args.action_share,
            action_view=args.action_view,
            plot=args.plot,
            plot_dir=args.plot_dir,
            plot_subdir=args.plot_subdir)
