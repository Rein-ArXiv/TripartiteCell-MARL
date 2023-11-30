import os, sys
sys.path.append(os.pardir)

import torch
import numpy as np
import matplotlib.pyplot as plt

from envs.env import CellEnv
from algorithms.iql.agent import IQL

class Experiment:
	def __init__(self, env, model, param_suffix):
		self.islet_num = env.islet_num
		self.max_time = env.max_time
		pass


if __name__=="__main__":
	red_normal = []
	red_high = []
	blue_normal = []
	blue_high = []
	green_normal = []
	green_high = []
	sky_normal = []
	sky_high = []
	grey_normal = []
	grey_high = []

	count = 0
	env = CellEnv(max_time=200, reward_mode="local")
    
	for glu in [0.0, env.glucose_0]:
		for alpha_action in range(9):
			for beta_action in range(9):
				for delta_action in range(9):
					action = np.array([alpha_action, beta_action, delta_action])
					actions = np.tile(action, (env.islet_num, 1))
					env.reset(glucose_fix=True, glucose_level=(env.glucose_0 + glu))
					terminated = False
					truncated = False
					score = 0
					total_hormone = 0
					while not (terminated or truncated):
					    state, reward, terminated, truncated, info = env.step(actions, external_glucose=(glu*2))
					     #env.glucose = glu
					    for i in range(env.islet_num):	
					        total_hormone += state[i][0] + state[i][1]
					    score += reward

					print(f"{actions[0]} : {np.average(score)}")
                    