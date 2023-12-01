import os, sys
sys.path.append(os.pardir)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from experiments import Experiment
from envs.env import CellEnv
from algorithms.iql.model import DDQN

parser = argparse.ArgumentParser(description="Experiment for cell interaction analysis")

parser.add_argument("-is", "--islet_num", type=int, default=20, help="Environment Islet number, default=20")
parser.add_argument("-mt", "--max_time", type=int, default=100, help="Environment Max time (min) per episode, default=100")
parser.add_argument("-rm", "--reward_mode", type=str, help="Choose reward mode, individual hormone secretion or total hormone secretion, ['local', 'global']")
parser.add_argument("-cu", "--cuda", action='store_true', help="Use cuda, action='store_true'")
parser.add_argument("-cn", "--cuda_num", type=int, default=0, help="Using cuda number, default=0")
parser.add_argument("-pl", "--param_loc", type=str, default="../../parameters/iql", help="Torch parameter saving/loading location, default='../../parameters/iql'")
parser.add_argument("-ps", "--param_suffix", type=str, default=None, help="Parameter suffix, if empty, save/load name [alpha_cell.pth, beta_cell.pth, delta_cell.pth] in param_location")

#parser.add_argument("-pc", "--plot_dir", type=str, default="../image", help="Main directory plot saved, default='../image'")
#parser.add_argument("-pn", "--plot_subdir", type=str, default=None, help="Sub directory plot saved under main directory, default=None")

if __name__=="__main__":
    args = parser.parse_args()
    env = CellEnv(
        islet_num=args.islet_num,
        max_time=args.max_time,
        reward_mode=args.reward_mode)
    
    experiment = Experiment(env=env, 
                            model=DDQN,
                            cuda=args.cuda,
                            cuda_num = args.cuda_num,
                            param_location=args.param_loc,
                            param_suffix=args.param_suffix)
    
    while(True):
        analysis_type = input("Input which you want to experiment [hormone, glucose, sync, exit]: ")
        if analysis_type == "hormone":
            experiment.hormone_secretion()
        elif analysis_type == "glucose":
            experiment.glucose_fluctuation()
        elif analysis_type == "sync":
            experiment.cell_synchronization()
        elif analysis_type == "exit":
            break


