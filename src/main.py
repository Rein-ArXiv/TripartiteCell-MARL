import os
import time
import torch
import wandb
import argparse
import numpy as np

from envs.env import CellEnv
from algorithms.iql.agent import IQL
from algorithms.mab.agent import Bandit

parser = argparse.ArgumentParser(description="CellEnv Training or Test module")

# env
parser.add_argument("-is", "--islet_num", type=int, default=20, help="Number of environment islets. Default=20")
parser.add_argument("-mt", "--max_time", type=int, default=200, help="Maximum time (minutes) per episode in the environment. Default=200")
parser.add_argument("-rm", "--reward_mode", type=str, help="Choose reward mode: individual hormone secretion or total hormone secretion. Options: ['local', 'global']")

# model
parser.add_argument("-m", "--model", type=str, help="Choose one of model [MAB, DDQN]")

args, unknown = parse.parse_known_args()

if (args.model == "DDQN"):
	# agent
	parser.add_argument("-tr", "--train", action='store_true', help="Turn on training mode. Use to enable.")
	parser.add_argument("-cu", "--cuda", action='store_true', help="Enable CUDA. Use to enable.")
	parser.add_argument("-cn", "--cuda_num", type=int, default=0, help="Specify CUDA device number. Default=0")

	# train IQL
	parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size for training. Default=256")
	parser.add_argument("-uf", "--target_update_freq", type=int, default=2000, help="Frequency of target network updates. Default=2000")
	parser.add_argument("-gm", "--gamma", type=fleoat, default=0.95, help="Gamma (reward decay) value. Default=0.95")
	parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Optimizer's learning rate. Default=1e-4")
	parser.add_argument("-mc", "--memory_cap", type=int, default=2e5, help="Replay memory capacity. Default=200,000")
	parser.add_argument("-me", "--max_epi", type=int, default=1000, help="Maximum number of training episodes. Default=1000")
	parser.add_argument("-el", "--eps_linear", action='store_true', help="Enable linear epsilon decay. If not enabled, epsilon will decay exponentially. Use 'action=store_true' to enable.")
	parser.add_argument("-ed", "--eps_decay", type=float, help="Epsilon decay rate for exponential decay.")

	# train/test IQL
	parser.add_argument("-eg", "--external_glu", type=float, default=0, help="External glucose provided every timestep. Default=0")
	parser.add_argument("-pl", "--param_loc", type=str, default="../parameters/iql", help="Location for saving/loading torch parameters. Default='../parameters/iql'")
	parser.add_argument("-ps", "--param_suffix", type=str, default=None, help="Parameter file suffix. If empty, default save/load names are [alpha_cell.pth, beta_cell.pth, delta_cell.pth] in the specified parameter location.")

	# test IQL
	parser.add_argument("-g", "--glu_fix", action='store_true', help="Turn on fixed initial glucose level. Use to enable.")
	parser.add_argument("-gl", "--glu_level", type=float, default=None, help="Initial glucose level value. Use when glu_fix is true. Default=None")
	parser.add_argument("-p", "--plot", action='store_true', help="Enable plotting results in test mode. Use 'action=store_true' to enable.")
	parser.add_argument("-pc", "--plot_dir", type=str, default="../image", help="Main directory where plots are saved. Default='../image'")
	parser.add_argument("-pn", "--plot_subdir", type=str, default=None, help="Subdirectory under the main directory where plots are saved. Default=None")
	parser.add_argument("-av", "--action_view", action='store_true', help="Enable viewing actions on the terminal. Use to enable.")
	parser.add_argument("-as", "--action_share", action='store_true', help="Enable shared actions within a single timestep. Use to enable.")

elif (args.model == "MAB"):


if __name__=="__main__":
	args = parser.parse_args()
	env = CellEnv(
		islet_num=args.islet_num,
		max_time=args.max_time,
		reward_mode=args.reward_mode)
	
	agent = IQL(
		env,
		is_train=args.train,
		cuda=args.cuda,
		cuda_num=args.cuda_num)

	if args.model == "DDQN":
		if args.train:
			agent.train(
				batch_size=args.batch_size,
				target_update_freq=args.target_update_freq,
				gamma=args.gamma,
				memory_capacity=args.memory_cap,
				max_epi=args.max_epi,
				eps_linear=args.eps_linear,
				eps_decay=args.eps_decay,
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

	elif args.model == "MAB":

	else:
		print("You put wrong model name")
