import os, sys
sys.path.append(os.pardir)

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from util import param_load, one_tiemstep_action_share

class Experiment:
    def __init__(self, env, model, state_dim=4, action_dim=9, cuda=False, cuda_num=0, param_location=None, param_suffix=None, plot=False):
        self.env = env
        self.islet_num = env.islet_num
        self.device = torch.device(f"cuda:{cuda_num}" if (cuda and torch.cuda.is_available()) else "cpu")
        
        self.alpha_cell_network = model(state_dim, action_dim).to(self.device)
        self.beta_cell_network = model(state_dim, action_dim).to(self.device)
        self.delta_cell_network = model(state_dim, action_dim).to(self.device)
        self.use_plot = plot

        param_load(self.alpha_cell_network, self.beta_cell_network, self.delta_cell_network,
                   param_location=param_location, param_suffix=param_suffix)
    
    def rl_action_view(self):
        for glu_index, add_glucose in enumerate([-self.env.glucose_0, 0, self.env.glucose_0]):
            # RL action no share
            state, info = self.env.reset(glucose_fix=True, glucose_level=(self.env.glucose_0 + add_glucose))
            
            terminated = False
            truncated = False
            score = 0
            
            state_action_record = dict()
            q_value_record = []

            while not (terminated or truncated):
                actions, q_value = self.network_select_action(state)
                state_action_record

                state, reward, terminated, truncated, info = self.env.step(actions, add_glucose)
            
            if glu_index == 0:
                self.rl_no_share_low = score
                
            elif glu_index == 1:
                self.rl_no_share_normal = score
            elif glu_index == 2:
                self.rl_no_share_high = score

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
            

            # RL action no share
            state, info = self.env.reset(glucose_fix=True, glucose_level=(self.env.glucose_0 + add_glucose))
            
            terminated = False
            truncated = False
            score = 0

            while not (terminated or truncated):
                actions = self.network_select_action(state)
                state, reward, terminated, truncated, info = self.env.step(actions, add_glucose)
                score += np.average(reward)
            print(f"Actions: RL(DDQN) action no share \t Score: {score}")
            
            if glu_index == 0:
                self.rl_no_share_low = score
            elif glu_index == 1:
                self.rl_no_share_normal = score
            elif glu_index == 2:
                self.rl_no_share_high = score

            # RL action share
            state, info = self.env.reset(glucose_fix=True, glucose_level=(self.env.glucose_0 + add_glucose))
            
            terminated = False
            truncated = False
            score = 0

            while not (terminated or truncated):
                actions = one_tiemstep_action_share(self.network_select_action(state))
                state, reward, terminated, truncated, info = self.env.step(actions, add_glucose)
                score += np.average(reward)
            print(f"Actions: RL(DDQN) action share \t Score: {score}")
            
            if glu_index == 0:
                self.rl_share_low = score
            elif glu_index == 1:
                self.rl_share_normal = score
            elif glu_index == 2:
                self.rl_share_high = score

        if self.use_plot:
            self._plot(plot_location="../../image/experiment/iql", plot_name="Score", experiment_type="score")

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
                        total_hormone /= self.env.max_time
                        
                        print(f"Actions: {cell_action}\t Hormone: {total_hormone}")

                        self._log(cell_action=cell_action, glu_index=glu_index, log_item=total_hormone)
            
            # RL action no share
            state, info = self.env.reset(glucose_fix=True, glucose_level=(self.env.glucose_0 + add_glucose))
            
            terminated = False
            truncated = False
            total_hormone = 0

            while not (terminated or truncated):
                actions = self.network_select_action(state)
                state, reward, terminated, truncated, info = self.env.step(actions, add_glucose)
                hormones = np.sum(state, axis=0)
                total_hormone += np.sum(hormones[0] + hormones[1]) / self.islet_num  # alpha_hormone + beta_hormone
            total_hormone /= self.env.max_time
            print(f"Actions: RL(DDQN) action no share \t Hormone: {total_hormone}")
            
            if glu_index == 0:
                self.rl_no_share_low = total_hormone
            elif glu_index == 1:
                self.rl_no_share_normal = total_hormone
            elif glu_index == 2:
                self.rl_no_share_high = total_hormone


            # RL action share
            state, info = self.env.reset(glucose_fix=True, glucose_level=(self.env.glucose_0 + add_glucose))
            
            terminated = False
            truncated = False
            total_hormone = 0

            while not (terminated or truncated):
                actions = one_tiemstep_action_share(self.network_select_action(state))
                state, reward, terminated, truncated, info = self.env.step(actions, add_glucose)
                hormones = np.sum(state, axis=0)
                total_hormone += np.sum(hormones[0] + hormones[1]) / self.islet_num  # alpha_hormone + beta_hormone
            total_hormone /= self.env.max_time
            print(f"Actions: RL(DDQN) action share \t Hormone: {total_hormone}")
            
            if glu_index == 0:
                self.rl_share_low = total_hormone
            elif glu_index == 1:
                self.rl_share_normal = total_hormone
            elif glu_index == 2:
                self.rl_share_high = total_hormone

        if self.use_plot:
            self._plot(plot_location="../../image/experiment/iql", plot_name="Hormone", experiment_type="hormone")
        
    def glucose_fluctuation(self):
        self._init_log()
        glucose_epsilon = 1e-5
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
                        
                        total_glucose = 0
                        glucose_list = []

                        while not (terminated or truncated):
                            state, reward, terminated, truncated, info = self.env.step(actions, add_glucose)
                            total_glucose += info["glucose"] + glucose_epsilon
                            glucose_list.append(info["glucose"] + glucose_epsilon)
                        
                        total_glucose /= self.env.max_time
                        glucose_delta = np.sqrt(np.sum((np.array(glucose_list) - total_glucose)**2) / self.env.max_time)
                        glucose_delta_bar = glucose_delta / total_glucose
                        print(f"Actions: {cell_action}\t Glucose fluctuation: {glucose_delta_bar}")

                        self._log(cell_action=cell_action, glu_index=glu_index, log_item=glucose_delta_bar)

            state, info = self.env.reset(glucose_fix=True, glucose_level=(self.env.glucose_0 + add_glucose))
            
            terminated = False
            truncated = False
            
            total_glucose = 0
            glucose_list = []

            while not (terminated or truncated):
                actions = self.network_select_action(state)
                state, reward, terminated, truncated, info = self.env.step(actions, add_glucose)
                total_glucose += info["glucose"] + glucose_epsilon
                glucose_list.append(info["glucose"] + glucose_epsilon)
            
            total_glucose /= self.env.max_time
            glucose_delta = np.sqrt(np.sum((np.array(glucose_list) - total_glucose)**2) / self.env.max_time)
            glucose_delta_bar = glucose_delta / total_glucose
            print(f"Actions: RL(DDQN) action no share \t Glucose fluctuation: {glucose_delta_bar}")
            
            if glu_index == 0:
                self.rl_no_share_low = glucose_delta_bar
            elif glu_index == 1:
                self.rl_no_share_normal = glucose_delta_bar
            elif glu_index == 2:
                self.rl_no_share_high = glucose_delta_bar

            
            state, info = self.env.reset(glucose_fix=True, glucose_level=(self.env.glucose_0 + add_glucose))
            
            terminated = False
            truncated = False
            
            total_glucose = 0
            glucose_list = []

            while not (terminated or truncated):
                actions = one_tiemstep_action_share(self.network_select_action(state))
                state, reward, terminated, truncated, info = self.env.step(actions, add_glucose)
                total_glucose += info["glucose"] + glucose_epsilon
                glucose_list.append(info["glucose"] + glucose_epsilon)
            
            total_glucose /= self.env.max_time
            glucose_delta = np.sqrt(np.sum((np.array(glucose_list) - total_glucose)**2) / self.env.max_time)
            glucose_delta_bar = glucose_delta / total_glucose
            print(f"Actions: RL(DDQN) action share \t Glucose fluctuation: {glucose_delta_bar}")
            
            if glu_index == 0:
                self.rl_share_low = glucose_delta_bar
            elif glu_index == 1:
                self.rl_share_normal = glucose_delta_bar
            elif glu_index == 2:
                self.rl_share_high = glucose_delta_bar

        if self.use_plot:
            self._plot(plot_location="../../image/experiment/iql", plot_name="Glucose fluctuation", experiment_type="glu_fluc")

    def cell_synchronization(self):
        self._init_log()
        for glu_index, add_glucose in enumerate([-self.env.glucose_0, 0, self.env.glucose_0]):
            for alpha_action in range(9):
                for beta_action in range(9):
                    for delta_action in range(9):
                        cell_action = np.array([alpha_action, beta_action, delta_action])
                        actions = np.tile(cell_action, (self.islet_num, 1))
                        self.env.reset(glucose_fix=True, glucose_level=(self.env.glucose_0 + add_glucose))
                        
                        terminated = False
                        truncated = False

                        total_synchronization_rho = 0

                        while not (terminated or truncated):
                            state, reward, terminated, truncated, info = self.env.step(actions, add_glucose)
                            if self.env.curr_time > (self.env.max_time/2):
                                alpha_sync = []
                                beta_sync = []
                                delta_sync = []
                                
                                for i in range(self.islet_num):
                                    alpha_sync.append(self.env.phases[i][0])
                                    beta_sync.append(self.env.phases[i][1])
                                    delta_sync.append(self.env.phases[i][2])

                                alpha_sync_arr = np.array(alpha_sync)
                                beta_sync_arr = np.array(beta_sync)
                                delta_sync_arr = np.array(delta_sync)

                                alpha_mean = np.mean(alpha_sync)
                                beta_mean = np.mean(beta_sync)
                                delta_mean = np.mean(delta_sync)

                                alpha_rho = np.sum(np.exp(1j * (alpha_sync_arr - alpha_mean)))/self.islet_num 
                                beta_rho = np.sum(np.exp(1j * (beta_sync_arr - beta_mean)))/self.islet_num
                                delta_rho = np.sum(np.exp(1j * (delta_sync_arr - delta_mean)))/self.islet_num

                                total_rho = alpha_rho + beta_rho + delta_rho
                                total_synchronization_rho += total_rho
                        
                        total_synchronization_rho = abs(total_synchronization_rho) / (self.env.max_time*3/2)

                        print(f"Actions: {cell_action}\t Synchronization: {total_synchronization_rho}")
                        self._log(cell_action=cell_action, glu_index=glu_index, log_item=total_synchronization_rho)
            
            state, info = self.env.reset(glucose_fix=True, glucose_level=(self.env.glucose_0 + add_glucose))
            terminated = False
            truncated = False
            total_synchronization_rho = 0
            
            while not (terminated or truncated):
                actions = self.network_select_action(state)
                state, reward, terminated, truncated, info = self.env.step(actions, add_glucose)
                if self.env.curr_time > (self.env.max_time/2):
                    alpha_sync = []
                    beta_sync = []
                    delta_sync = []
            
                    for i in range(self.islet_num):
                        alpha_sync.append(self.env.phases[i][0])
                        beta_sync.append(self.env.phases[i][1])
                        delta_sync.append(self.env.phases[i][2])

                    alpha_sync_arr = np.array(alpha_sync)
                    beta_sync_arr = np.array(beta_sync)
                    delta_sync_arr = np.array(delta_sync)

                    alpha_mean = np.mean(alpha_sync)
                    beta_mean = np.mean(beta_sync)
                    delta_mean = np.mean(delta_sync)

                    alpha_rho = np.sum(np.exp(1j * (alpha_sync_arr - alpha_mean)))/self.islet_num 
                    beta_rho = np.sum(np.exp(1j * (beta_sync_arr - beta_mean)))/self.islet_num
                    delta_rho = np.sum(np.exp(1j * (delta_sync_arr - delta_mean)))/self.islet_num

                    total_rho = alpha_rho + beta_rho + delta_rho
                    total_synchronization_rho += total_rho
            total_synchronization_rho = abs(total_synchronization_rho) / (self.env.max_time*3/2)
            
            print(f"Actions: RL(DDQN) action no share \t Synchronization: {total_synchronization_rho}")
            
            if glu_index == 0:
                self.rl_no_share_low = total_synchronization_rho
            elif glu_index == 1:
                self.rl_no_share_normal = total_synchronization_rho
            elif glu_index == 2:
                self.rl_no_share_high = total_synchronization_rho



            state, info = self.env.reset(glucose_fix=True, glucose_level=(self.env.glucose_0 + add_glucose))
            terminated = False
            truncated = False
            total_synchronization_rho = 0
            
            while not (terminated or truncated):
                actions = one_tiemstep_action_share(self.network_select_action(state))
                state, reward, terminated, truncated, info = self.env.step(actions, add_glucose)
                if self.env.curr_time > (self.env.max_time/2):
                    alpha_sync = []
                    beta_sync = []
                    delta_sync = []
            
                    for i in range(self.islet_num):
                        alpha_sync.append(self.env.phases[i][0])
                        beta_sync.append(self.env.phases[i][1])
                        delta_sync.append(self.env.phases[i][2])

                    alpha_sync_arr = np.array(alpha_sync)
                    beta_sync_arr = np.array(beta_sync)
                    delta_sync_arr = np.array(delta_sync)

                    alpha_mean = np.mean(alpha_sync)
                    beta_mean = np.mean(beta_sync)
                    delta_mean = np.mean(delta_sync)

                    alpha_rho = np.sum(np.exp(1j * (alpha_sync_arr - alpha_mean)))/self.islet_num 
                    beta_rho = np.sum(np.exp(1j * (beta_sync_arr - beta_mean)))/self.islet_num
                    delta_rho = np.sum(np.exp(1j * (delta_sync_arr - delta_mean)))/self.islet_num

                    total_rho = alpha_rho + beta_rho + delta_rho
                    total_synchronization_rho += total_rho
            total_synchronization_rho = abs(total_synchronization_rho) / (self.env.max_time*3/2)
            
            print(f"Actions: RL(DDQN) action share \t Synchronization: {total_synchronization_rho}")
            
            if glu_index == 0:
                self.rl_share_low = total_synchronization_rho
            elif glu_index == 1:
                self.rl_share_normal = total_synchronization_rho
            elif glu_index == 2:
                self.rl_share_high = total_synchronization_rho

        #self._log_save(log_dir="../../logs/iql", log_name="cell_sync")
        if self.use_plot:
            self._plot(plot_location="../../image/experiment/iql", plot_name="Cell synchronization", experiment_type="sync")
    
    def _plot(self, plot_location=None, plot_name=None, experiment_type=None):
        if not os.path.exists(plot_location):
            os.makedirs(plot_location)
        
        plt.figure(figsize=(10, 10))
        plt.title(f"{plot_name} Normal/High")
        plt.scatter(self.red_normal.values(), self.red_high.values(), c='red', s=10)
        plt.scatter(self.blue_normal.values(), self.blue_high.values(), c='blue', s=10)
        plt.scatter(self.sky_normal.values(), self.sky_high.values(), c='skyblue', s=10)
        plt.scatter(self.green_normal.values(), self.green_high.values(), c='green', s=10)
        plt.scatter(self.human_normal.values(), self.human_high.values(), c='magenta', marker='^', s=100)
        plt.scatter(self.no_interaction_normal.values(), self.no_interaction_high.values(), c='purple', marker='v', s=100)
        plt.scatter(self.grey_normal.values(), self.grey_high.values(), c='grey', alpha=0.2, s=10)
        plt.scatter(self.rl_no_share_normal, self.rl_no_share_low, c='gold', marker='*', s=100)
        plt.scatter(self.rl_share_normal, self.rl_share_low, c='cyan', marker='*', s=100)

        if experiment_type == "hormone":
            plt.grid(linestyle=':', color="grey")
            #plt.xlim(-0.1, 1.0)
            #plt.ylim(-0.1, 1.0)

        elif experiment_type == "glu_fluc":
            #plt.xlim(0.001, 1)
            #plt.ylim(0.001, 1)
            plt.xscale('log', base=10)
            plt.yscale('log', base=10)

        elif experiment_type == "sync":
            #plt.xlim(0.001, 1)
            #plt.ylim(0.001, 1)
            plt.xscale('log', base=10)
            plt.yscale('log', base=10)

        plt.savefig(f"{plot_location}/{experiment_type}_normal_high.png")
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.title(f"{plot_name} Normal/Low")
        plt.scatter(self.red_normal.values(), self.red_low.values(), c='red', s=10)
        plt.scatter(self.blue_normal.values(), self.blue_low.values(), c='blue', s=10)
        plt.scatter(self.sky_normal.values(), self.sky_low.values(), c='skyblue', s=10)
        plt.scatter(self.green_normal.values(), self.green_low.values(), c='green', s=10)
        plt.scatter(self.human_normal.values(), self.human_low.values(), c='magenta', marker='^', s=100)
        plt.scatter(self.no_interaction_normal.values(), self.no_interaction_low.values(), c='purple', marker='v', s=100)
        plt.scatter(self.grey_normal.values(), self.grey_low.values(), c='grey', alpha=0.2, s=10)
        plt.scatter(self.rl_no_share_normal, self.rl_no_share_high, c='gold', marker='*', s=100)
        plt.scatter(self.rl_share_normal, self.rl_share_high, c='cyan', marker='*', s=100)

        if experiment_type == "hormone":
            plt.grid(linestyle=':', color="grey")
            #plt.xlim(0.4, 1.0)
            #plt.ylim(0.4, 1.0)

        elif experiment_type == "glu_fluc":
            #plt.xlim(0.01, 1)
            #plt.ylim(0.01, 1)
            plt.xscale('log', base=10)
            plt.yscale('log', base=10)

        elif experiment_type == "sync":
            #plt.xlim(0.01, 1)
            #plt.ylim(0.01, 1)
            plt.xscale('log', base=10)
            plt.yscale('log', base=10)

        plt.savefig(f"{plot_location}/{experiment_type}_normal_low.png")
        plt.close()

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
                self.red_low[f"{cell_action}"] = log_item
            elif (glu_index == 1):
                self.red_normal[f"{cell_action}"] = log_item
            elif (glu_index == 2):
                self.red_high[f"{cell_action}"] = log_item

        elif (alpha_to_beta_action==0 and beta_to_alpha_action==0):
            if (glu_index == 0):
                self.blue_low[f"{cell_action}"] = log_item
            elif (glu_index == 1):
                self.blue_normal[f"{cell_action}"] = log_item
            elif (glu_index == 2):
                self.blue_high[f"{cell_action}"] = log_item

        elif (alpha_to_beta_action==2 and beta_to_alpha_action==0):
            # Human action log
            if (alpha_to_delta_action==2 and delta_to_alpha_action==0):
                if (beta_to_delta_action==2 and delta_to_beta_action==0):
                    if (glu_index == 0):
                        self.human_low[f"{cell_action}"] = log_item
                    elif (glu_index == 1):
                        self.human_normal[f"{cell_action}"] = log_item
                    elif (glu_index == 2):
                        self.human_high[f"{cell_action}"] = log_item

            if (glu_index == 0):
                self.green_low[f"{cell_action}"] = log_item
            elif (glu_index == 1):
                self.green_normal[f"{cell_action}"] = log_item
            elif (glu_index == 2):
                self.green_high[f"{cell_action}"] = log_item

        elif (alpha_to_beta_action==0 and beta_to_alpha_action==2):
            if (glu_index == 0):
                self.sky_low[f"{cell_action}"] = log_item
            elif (glu_index == 1):
                self.sky_normal[f"{cell_action}"] = log_item
            elif (glu_index == 2):
                self.sky_high[f"{cell_action}"] = log_item

        else:
            if (alpha_to_beta_action==1 and beta_to_alpha_action==1):
                if (alpha_to_delta_action==1 and delta_to_alpha_action==1):
                    if (beta_to_delta_action==1 and delta_to_beta_action==1):
                        if (glu_index == 0):
                            self.no_interaction_low[f"{cell_action}"] = log_item
                        elif (glu_index == 1):
                            self.no_interaction_normal[f"{cell_action}"] = log_item
                        elif (glu_index == 2):
                            self.no_interaction_high[f"{cell_action}"] = log_item
            if (glu_index == 0):
                self.grey_low[f"{cell_action}"] = log_item
            elif (glu_index == 1):
                self.grey_normal[f"{cell_action}"] = log_item
            elif (glu_index == 2):
                self.grey_high[f"{cell_action}"] = log_item


    def _log_save(self, log_dir=None, log_name=None):
        assert log_dir, "log directory has no input. Please input log directory"
        print(f"log path: {os.path.abspath(log_dir)}")
        
        with open(f"{log_name}_red_low.pkl", "wb") as file:
            pickle.dump(self.red_low, file)

    def _log_load(self, log_dir=None, log_name=None):
        pass

    def network_select_action(self, state):
        action_list = []
        q_value_list = []
        for islet_i in range(self.islet_num):
            alpha_cell_action, alpha_cell_q_value = self.network_get_action(self.alpha_cell_network, state[islet_i])
            beta_cell_action, beta_cell_q_value = self.network_get_action(self.beta_cell_network, state[islet_i])
            delta_cell_action, delta_cell_q_value = self.network_get_action(self.delta_cell_network, state[islet_i])

            action_list.append([alpha_cell_action, beta_cell_action, delta_cell_action])
            q_value_list.append([alpha_cell_q_value, beta_cell_q_value, delta_cell_q_value])
        return np.array(action_list), np.array(q_value_list)

    def network_get_action(self, ddqn, state):
        q_value = ddqn((torch.from_numpy(state)).to(self.device, dtype=torch.float))
        action = q_value.argmax()
        q_value = q_value.detach().cpu().numpy()
        action = action.detach().cpu().numpy()
        return action, q_value

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

        # human -> atb, atd, btd, bta, dta, dtb = (pos, pos, pos, neg, neg, neg)
        self.human_low = dict()
        self.human_normal = dict()
        self.human_high = dict()
        
        # no interaction
        self.no_interaction_low = dict()
        self.no_interaction_normal = dict()
        self.no_interaction_high = dict()
