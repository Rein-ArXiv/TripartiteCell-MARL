import os, sys
sys.path.append(os.pardir)

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from util import param_load

class Experiment:
    def __init__(self, env, model, state_dim=4, action_dim=9, cuda=False, cuda_num=0, param_location=None, param_suffix=None):
        self.env = env
        self.islet_num = env.islet_num
        self.device = torch.device(f"cuda:{cuda_num}" if (cuda and torch.cuda.is_available()) else "cpu")
        
        self.alpha_cell_network = model(state_dim, action_dim).to(self.device)
        self.beta_cell_network = model(state_dim, action_dim).to(self.device)
        self.delta_cell_network = model(state_dim, action_dim).to(self.device)

        param_load(self.alpha_cell_network, self.beta_cell_network, self.delta_cell_network,
                   param_location=param_location, param_suffix=param_suffix)
    
    def score(self):
        self._init_log()
        for glu_index, add_glucose in enumerate([-self.env.glucose_0, 0, self.env.glucose_0]):
            for alpha_action in range(9):
                for beta_action in range(9):
                    for delta_action in range(9):
                        # action = [(atb, atd), (btd, bta), (dta, dtb)]
                        
                        cell_action = np.array([alpha_action, beta_action, delta_action])
                        actions = np.tile(cell_action, (self.islet_num, 1))
                        self.env.reset(glucose_fix=True, glucose_level=(self.env.glucose_0 + add_glucose))
                        
                        terminated = False
                        truncated = False
                        score = 0

                        while not (terminated or truncated):
                            state, reward, terminated, truncated, info = self.env.step(actions, add_glucose)
                            score += np.average(reward)
                        print(f"Actions: {cell_action}\t Score: {score}")
                        self._log(cell_action=cell_action, glu_index=glu_index, log_item=score)



        self._log_save(log_dir="../../logs/iql", log_name="score")
        self._plot(plot_location="../../image/experiment/iql", plot_name="score")

    def hormone_secretion(self):
        self._init_log()
        for glu_index, add_glucose in enumerate([-self.env.glucose_0, 0, self.env.glucose_0]):
            for alpha_action in range(9):
                for beta_action in range(9):
                    for delta_action in range(9):
                        # action = [(atb, atd), (btd, bta), (dta, dtb)]
                        
                        cell_action = np.array([alpha_action, beta_action, delta_action])
                        actions = np.tile(cell_action, (self.islet_num, 1))
                        self.env.reset(glucose_fix=True, glucose_level=(self.env.glucose_0 + add_glucose))
                        
                        terminated = False
                        truncated = False
                        total_hormone = 0

                        while not (terminated or truncated):
                            state, reward, terminated, truncated, info = self.env.step(actions, add_glucose)
                            hormones = np.sum(state, axis=0)
                            total_hormone += np.sum(hormones[0] + hormones[1]) / self.islet_num  # alpha_hormone + beta_hormone
                        print(f"Actions: {cell_action}\t Hormone: {total_hormone}")

                        #self._log(cell_action=cell_action, glu_index=glu_index, log_item=total_hormone)

        #self._log_save(log_dir="../../logs/iql", log_name="score")
        #self._plot(plot_location="", plot_name="")
        
    def glucose_fluctuation(self):
        self._init_log()

    def cell_synchronization(self):
        self._init_log()
        pass

    
    def _plot(self, plot_location=None, plot_name=None):
        pass


    def _log(self, cell_action, glu_index, log_item):
        alpha_action, beta_action, delta_action = cell_action
        
        alpha_to_beta_action = alpha_action // 3
        alpha_to_delta_action = alpha_action % 3
        beta_to_delta_action = beta_action // 3
        beta_to_alpha_action = beta_action % 3
        delta_to_alpha_action = delta_action // 3
        delta_to_beta_action = delta_action % 3

        if (alpha_to_beta_action==2 and beta_to_alpha_action==2):
            if (glu_index == 0):
                self.red_low[cell_action] = log_item
            elif (glu_index == 1):
                self.red_normal[cell_action] = log_item
            elif (glu_index == 2):
                self.red_high[cell_action] = log_item

        elif (alpha_to_beta_action==0 and beta_to_alpha_action==0):
            if (glu_index == 0):
                self.blue_low[cell_action] = log_item
            elif (glu_index == 1):
                self.blue_normal[cell_action] = log_item
            elif (glu_index == 2):
                self.blue_high[cell_action] = log_item

        elif (alpha_to_beta_action==2 and beta_to_alpha_action==0):
            if (glu_index == 0):
                self.green_low[cell_action] = log_item
            elif (glu_index == 1):
                self.green_normal[cell_action] = log_item
            elif (glu_index == 2):
                self.green_high[cell_action] = log_item

        elif (alpha_to_beta_action==0 and beta_to_alpha_action==2):
            if (glu_index == 0):
                self.sky_low[cell_action] = log_item
            elif (glu_index == 1):
                self.sky_normal[cell_action] = log_item
            elif (glu_index == 2):
                self.sky_high[cell_action] = log_item

        else:
            if (glu_index == 0):
                self.grey_low[cell_action] = log_item
            elif (glu_index == 1):
                self.grey_normal[cell_action] = log_item
            elif (glu_index == 2):
                self.grey_high[cell_action] = log_item


    def _log_save(self, log_dir=None, log_name=None):
        pass

    def network_select_action(state):
        action_list = []
        alpha_cell_action_list = []
        beta_cell_action_list = []
        delta_cell_action_list= []


        for islet_i in range(self.islet_num):
            alpha_cell_action = network_get_action(alpha_cell_online_dqn, state[islet_i])
            beta_cell_action = network_get_action(beta_cell_online_dqn, state[islet_i])
            delta_cell_action = network_get_action(delta_cell_online_dqn, state[islet_i])

            action_list.append([alpha_cell_action, beta_cell_action, delta_cell_action])

        return np.array(action_list)

    def network_get_action(ddqn, state):
        action = ddqn((torch.from_numpy(state)).to(self.device, dtype=torch.float)).argmax()
        action = action.detach().cpu().numpy()
        return action

    def _init_log(self):
        # red -> atb, bta = (pos, pos)
        self.red_low = dict()
        self.red_normal = dict()
        self.red_high = dict()
        
        # blue -> atb, bta = (neg, neg)
        self.blue_low = dict()
        self.blue_normal = dict()
        self.blue_high = dict()

        # green -> atb, bta = (pos, neg)
        self.green_low = dict()
        self.green_normal = dict()
        self.green_high = dict()
        
        # sky -> atb, bta = (neg, pos)
        self.sky_low = dict()
        self.sky_normal = dict()
        self.sky_high = dict()
        
        self.grey_low = dict()
        self.grey_normal = dict()
        self.grey_high = dict()
