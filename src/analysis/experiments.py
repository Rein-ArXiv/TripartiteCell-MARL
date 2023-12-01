import os, sys
sys.path.append(os.pardir)

import torch
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

        param_load(self.alpha_cell_network, self.beta_cell_network, self.delta_cell_network, self.device,
                   param_location=param_location, param_suffix=param_suffix)
    
    def hormone_secretion(self):
        self._init_log()
        for glu in [-self.env.glucose_0, 0.0, self.env.glucose_0]:
            for alpha_action in range(9):
                for beta_action in range(9):
                    for delta_action in range(9):
                        # action = [(atb, atd), (btd, bta), (dta, dtb)]
                        
                        cell_action = np.array([alpha_action, beta_action, delta_action])
                        actions = np.tile(cell_action, (self.islet_num, 1))
                        self.env.reset(glucose_fix=True, glucose_level=(self.env.glucose_0 + glu))
                        terminated = False
                        truncated = False
                        total_hormone = 0
                        while not (terminated or truncated):
                            state, reward, terminated, truncated, info = self.env.step(actions, external_glucose=(glu*2))
                            for i in range(self.islet_num):	
                                total_hormone += state[i][0] + state[i][1] # alpha_hormone + beta_hormone
                            
                        print(f"{actions[0]} hormone: {total_hormone}")

    def glucose_fluctuation(self):
        self._init_log()

    def cell_synchronization(self):
        self._init_log()
        pass

    

    
    def _init_log(self):
        # red -> atb, bta = (pos, pos)
        self.red_low = []
        self.red_normal = []
        self.red_high = []
        
        # blue -> atb, bta = (neg, neg)
        self.blue_low = []
        self.blue_normal = []
        self.blue_high = []

        # green -> atb, bta = (pos, neg)
        self.green_low = []
        self.green_normal = []
        self.green_high = []
        
        # sky -> atb, bta = (neg, pos)
        self.sky_low = []
        self.sky_normal = []
        self.sky_high = []
        
        self.grey_low = []
        self.grey_normal = []
        self.grey_high = []

    def plot(self):
        pass

if __name__=="__main__":

    
