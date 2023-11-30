import os, sys
sys.path.append(os.pardir)

import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch

from envs.env import CellEnv
from algorithms.iql.agent import IQL


def one_tiemstep_action_share(action_list):
	alpha_cell_action = action_list[:, 0]
	beta_cell_action = action_list[:, 1]
	delta_cell_action = action_list[:, 2]

	alpha_unique_actions, alpha_action_counts = np.unique(alpha_cell_action, return_counts=True)
	beta_unique_actions, beta_action_counts = np.unique(beta_cell_action, return_counts=True)
	delta_unique_actions, delta_action_counts = np.unique(delta_cell_action, return_counts=True)

	alpha_most_action = alpha_unique_actions[np.argmax(alpha_action_counts)]
	beta_most_action = beta_unique_actions[np.argmax(beta_action_counts)]
	delta_most_action = delta_unique_actions[np.argmax(delta_action_counts)]

	action_share_array = np.array([[alpha_most_action, beta_most_action, delta_most_action]] * action_list.shape[0])

	return action_share_array

def one_timestep_action_frequency(action_list):
	alpha_cell_action = action_list[:, 0]
	beta_cell_action = action_list[:, 1]
	delta_cell_action = action_list[:, 2]

	alpha_unique_actions, alpha_action_counts = np.unique(alpha_cell_action, return_counts=True)
	beta_unique_actions, beta_action_counts = np.unique(beta_cell_action, return_counts=True)
	delta_unique_actions, delta_action_counts = np.unique(delta_cell_action, return_counts=True)

	alpha_action_frequency = dict()
	beta_action_frequency = dict()
	delta_action_frequency = dict()
	
	for alpha_action, alpha_count in zip(alpha_unique_actions, alpha_action_counts):
		alpha_action_frequency[alpha_action] = alpha_count
	for beta_action, beta_count in zip(beta_unique_actions, beta_action_counts):
		beta_action_frequency[beta_action] = beta_count
	for delta_action, delta_count in zip(delta_unique_actions, delta_action_counts):
		delta_action_frequency[delta_action] = delta_count
	
	return alpha_action_frequency, beta_action_frequency, delta_action_frequency

def all_timestep_cell_action_frequency(all_action_list, action_share=None):
	assert action_share != None, "Put action share True/False"

	all_timestep = all_action_list.shape[0]

	if action_share:
		all_alpha_cell_action = all_action_list[:, 0, 0].flatten() # length = all_timestep
		all_beta_cell_action = all_action_list[:, 0, 1].flatten()
		all_delta_cell_action = all_action_list[:, 0, 2].flatten()

	else:
		all_alpha_cell_action = all_action_list[:, :, 0].flatten() # length = all_timestep * islet num
		all_beta_cell_action = all_action_list[:, :, 1].flatten()
		all_delta_cell_action = all_action_list[:, :, 2].flatten()

	all_alpha_unique_actions, all_alpha_action_counts = np.unique(all_alpha_cell_action, return_counts=True)
	all_beta_unique_actions, all_beta_action_counts = np.unique(all_beta_cell_action, return_counts=True)
	all_delta_unique_actions, all_delta_action_counts = np.unique(all_delta_cell_action, return_counts=True)
	
	all_alpha_action_frequency = dict()
	all_beta_action_frequency = dict()
	all_delta_action_frequency = dict()
	
	for all_alpha_action, all_alpha_count in zip(all_alpha_unique_actions, all_alpha_action_counts):
		all_alpha_action_frequency[all_alpha_action] = all_alpha_count
	for all_beta_action, all_beta_count in zip(all_beta_unique_actions, all_beta_action_counts):
		all_beta_action_frequency[all_beta_action] = all_beta_count
	for all_delta_action, all_delta_count in zip(all_delta_unique_actions, all_delta_action_counts):
		all_delta_action_frequency[all_delta_action] = all_delta_count

	return all_alpha_action_frequency, all_beta_action_frequency, all_delta_action_frequency

def all_timestep_cell_action_set_frequency(all_action_list, action_share=None):
	assert action_share != None, "Put action share True/False"

	all_timestep = all_action_list.shape[0]

	if action_share:
		all_action_set = all_action_list[:, 0].reshape((-1, 3))
	else:
		all_action_set = all_action_list[:, :].reshape((-1, 3))

	all_alpha_unique_actions, all_alpha_action_counts = np.unique(all_alpha_cell_action, return_counts=True)
	all_beta_unique_actions, all_beta_action_counts = np.unique(all_beta_cell_action, return_counts=True)
	all_delta_unique_actions, all_delta_action_counts = np.unique(all_delta_cell_action, return_counts=True)
	
	all_alpha_action_frequency = dict()
	all_beta_action_frequency = dict()
	all_delta_action_frequency = dict()
	
	for all_alpha_action, all_alpha_count in zip(all_alpha_unique_actions, all_alpha_action_counts):
		all_alpha_action_frequency[all_alpha_action] = all_alpha_count
	for all_beta_action, all_beta_count in zip(all_beta_unique_actions, all_beta_action_counts):
		all_beta_action_frequency[all_beta_action] = all_beta_count
	for all_delta_action, all_delta_count in zip(all_delta_unique_actions, all_delta_action_counts):
		all_delta_action_frequency[all_delta_action] = all_delta_count

def action_to_cell

if __name__=="__main__":
	action_list = np.random.randint(low=0, high=9, size=(20,3), dtype=int)
	print("action list")
	print(action_list)
	print("action_share_array")
	print(one_tiemstep_action_share(action_list))
	print("action frequency")
	print(one_timestep_action_frequency(action_list))

	all_action_list = np.random.randint(low=0, high=9, size=(100, 20,3), dtype=int)
	print("all_action")
	print(all_timestep_cell_action_frequency(all_action_list, action_share=True))