import os
import wandb
import argparse
import numpy as np

from envs.env import CellEnv
from algorithms.iql.agent import IQL

parser = argparse.ArgumentParser(description="CellEnv Training or Test module")

# env
parser.add_argument("-is", "--islet_num", type=int, default=20, help="Environment Islet number, default=20")
parser.add_argument("-mt", "--max_time", type=int, default=200, help="Environment Max time (min) per episode, default=200")
parser.add_argument("-g", "--glu_fix", action='store_true', help="Init glucose level fix on, action='store_true'")
parser.add_argument("-gl", "--glu_level", type=float, default=None, help="Init glucose level value, use when glu_fix is true, default=None")
parser.add_argument("-eg", "--external_glu", type=float, default=0, help="External glucose every timestep, default=0")
parser.add_argument("-rm", "--reward_mode", type=str, help="Choose reward mode, individual hormone secretion or total hormone secretion, ['local', 'global']")

# agent
parser.add_argument("-tr", "--train", action='store_true', help="Training mode on. action='store_true'")
parser.add_argument("-cu", "--cuda", action='store_true', help="Use cuda, action='store_true'")
parser.add_argument("-cn", "--cuda_num", type=int, default=0, help="Using cuda number, default=0")

# train
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Training batch size, default=256")
parser.add_argument("-uf", "--target_update_freq", type=int, default=2000, help="Target network update frequency, default=2000")
parser.add_argument("-gm", "--gamma", type=float, default=0.95, help="Gamma(Reward decay) value, default=0.95")
parser.add_argument("-mc", "--memory_cap", type=int, default=2e5, help="Replay memory size, default=200_000")
parser.add_argument("-me", "--max_epi", type=int, default=1000, help="Maximum training episode, default=1000")
parser.add_argument("-el", "--eps_linear", action='store_true', help="Using epsilon decay as linear. If not, epsilon will decay exponentially, action='store_true'")

# train/test
parser.add_argument("-pl", "--param_loc", type=str, default="../parameters/iql", help="Torch parameter saving/loading location, default='../parameters/iql'")
parser.add_argument("-ps", "--param_suffix", type=str, default=None, help="Parameter suffix, if empty, save/load name [alpha_cell.pth, beta_cell.pth, delta_cell.pth] in param_location")

if __name__=="__main__":
	args = parser.parse_args()
	env = CellEnv(
		islet_num=args.islet_num,
		max_time=args.max_time,
		glucose_fix=args.glu_fix,
		reward_mode=args.reward_mode)
	
	agent = IQL(
		env,
		is_train=args.train,
		cuda=args.cuda,
		cuda_num=args.cuda_num)

	if args.train:
		agent.train(
			batch_size=args.batch_size,
			target_update_freq=args.target_update_freq,
			gamma=args.gamma,
			memory_capacity=args.memory_cap,
			max_epi=args.max_epi,
			eps_linear=args.eps_linear,
			external_glu=args.external_glu,
			param_location=args.param_loc,
			param_suffix=args.param_suffix)

	else:
	    agent.test(
	    	param_location=args.param_loc,
	    	param_suffix=args.param_suffix,
	    	external_glu=args.external_glu)
    
