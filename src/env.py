import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Cellenv(gym.Env):
    def __init__(self, time_interval=1, islet_num=1, max_time=200, glucose_fix=False):
        self.cell_num = 3
        self.cell_interaction_type_num = 9
        self.islet_num = islet_num
        self.observation_space = spaces.Box(low=0.0, high=40.0, shape=(self.islet_num, int(self.cell_num + 1)))
        self.action_space = spaces.MultiDiscrete(np.full((self.islet_num), self.cell_interaction_type_num))
        self.state = None
        self.glucose_0 = 4.
        self.time_interval = time_interval
        self.max_time = max_time
        self.glucose_fix = glucose_fix
        
        # 단위 mM, 시간 간격 1 sec
        self.kapa = 0.4
        self.eta = 5
        
        # Env parameters
        self.terminated = None
        self.truncated = None

    def reset(self, glucose_level=None, seed=None):
        self.curr_time = 0
        self.terminated = False
        self.truncated = False
        
        if self.glucose_fix:
            glucose_init = glucose_level
            assert glucose_init is not None, 'glucose fix is True but glucose level is not assigned.'
        else:
            glucose_init = random.uniform(0.0, 8.0)
            
        self.phases = []
        self.amps = []
        self.hormones = []
        self.glucose = glucose_init
        
        for i in range(self.islet_num):
            cell_phase = np.random.uniform(0.0, 2*np.pi, 3).tolist()
            cell_amp = np.random.uniform(0.1, 0.5, 3).tolist()
            
            self.phases.append(cell_phase)
            self.amps.append(cell_amp)
            self.hormones.append(self.hormone_cal(self.phases[i], self.amps[i]))
        
        glucose_column = np.full((self.isliet_num, 1), glucose_init)
        self.state = np.hstack((np.array(self.hormones), glucose_column))
    
        self.phases = np.array(self.phases)
        self.amps = np.array(self.amps)
        self.hormones = np.array(self.hormones)

        info = {"phase":self.phases, "amplitude":self.amps, "hormones":self.hormones, "glucose":self.glucose}

        return self.state
        
    def step(self, actions, external_glucose=0):
        assert self.state is not None, "Call reset before using step method"

        info = dict()
        total_reward = 0
        for i_time in range(self.time_interval):
            self.total_glucose_delta = 0
            
            hormone_before = self.hormones.copy()
            for i in range(self.islet_num):
                atb, atd, bta, btd, dta, dtb = (np.array(actions[i]) - 1)
                alpha_phase_delta = self.phase_oscilation(0, self.phases[i], self.amps[i], self.glucose, bta, dta)
                beta_phase_delta = self.phase_oscilation(1, self.phases[i], self.amps[i], self.glucose, dtb, atb)
                delta_phase_delta = self.phase_oscilation(2, self.phases[i], self.amps[i], self.glucose, atd, btd)
                
                alpha_amp_delta = self.amp_oscilation(0, self.phases[i], self.amps[i], self.glucose, bta, dta)
                beta_amp_delta = self.amp_oscilation(1, self.phases[i], self.amps[i], self.glucose, dtb, atb)
                delta_amp_delta = self.amp_oscilation(2, self.phases[i], self.amps[i], self.glucose, atd, btd)
                
                
                self.phases[i][0] += alpha_phase_delta
                self.phases[i][1] += beta_phase_delta
                self.phases[i][2] += delta_phase_delta
                
                self.amps[i][0] += alpha_amp_delta
                self.amps[i][1] += beta_amp_delta
                self.amps[i][2] += delta_amp_delta
                
                self.phases[self.phases >= 2 * np.pi] = self.phases[self.phases >= 2 * np.pi] % (2 * np.pi)
                self.phases[self.phases < 0] = self.phases[self.phases < 0] + (2 * np.pi)
                self.amps[self.amps < 0] = 1e-5
                
                self.hormones[i] = self.hormone_cal(self.phases[i], self.amps[i])
                
                self.total_glucose_delta += self.glucose_delta(self.hormones[i], self.glucose)
            #print(beta_amp_delta)
            hormone_curr = self.hormones.copy()
            #print("hormone_curr [1]: ", hormone_curr)
            self.glucose_fluctuation = self.total_glucose_delta / self.islet_num
            self.glucose += self.glucose_fluctuation
            
            if i_time == 0:
                self.glucose += external_glucose
            if self.glucose < 0:
                self.glucose = 0.
            
            reward = abs(self.glucose_fluctuation) + (np.sum(hormone_curr) * 2) / self.islet_num
            total_reward += reward
            self.curr_time += 1
            
            self.state = np.array(self.hormones)
            glucose_column = np.full((self.state.shape[0], 1), self.glucose)
            self.state = np.hstack((self.state, glucose_column))
            
            if self.curr_time > self.max_time:
                self.truncated = True
            
            #if self.glucose < 0 or self.glucose > 40:
                #self.terminated = True
        total_reward = np.clip(-(total_reward/self.time_interval) / 1000, -0.1, 0)
        
        return self.state, total_reward, self.terminated, self.truncated, info
                
    def phase_oscilation(self, cell_num, phase_list, amp_list, glucose, first_action, second_action):
        w_sigma = 2 * np.pi / (np.random.normal(5, 0.1))
        g_sigma = self._phase_modulation(cell_num, glucose)
        phase_delta = w_sigma + g_sigma * np.cos(phase_list[cell_num]) + self._phase_couple(cell_num, phase_list, amp_list, first_action, second_action)
        return phase_delta
    
    def amp_oscilation(self, cell_num, phase_list, amp_list, glucose, first_action, second_action):
        amp_delta = (self._amp_modulation(cell_num, glucose) - amp_list[cell_num] ** 2) * amp_list[cell_num] + self._amp_couple(cell_num, phase_list, amp_list, first_action, second_action)
        return amp_delta

    def hormone_cal(self, phase_array, amp_array):
        hormone_array = (amp_array * (1 + np.cos(phase_array)) / 2)
        return hormone_array
    
    def glucose_delta(self, hormone_array, glucose):
        lamb = 1
        glucose_delta = lamb * (self.glucose_0 * hormone_array[0] - glucose * hormone_array[1])
        return glucose_delta
        
    def _phase_modulation(self, cell_num, glucose):
        mu = 0.1
        #mu = 0.5
        if cell_num != 0:
            mu = mu * (-1)
            
        factor = mu * (glucose - self.glucose_0)
        return factor
    
    def _phase_couple(self, cell_num, phase_list, amp_list, first_action, second_action):
        # alpha -> bta, dta
        # beta -> dtb, atb
        # delta -> atd, btd
        interaction_1 = self.kapa * first_action * amp_list[(cell_num + 1) % 3] / amp_list[cell_num] * np.sin(phase_list[(cell_num + 1) % 3] - phase_list[cell_num])
        interaction_2 = self.kapa * second_action * amp_list[(cell_num + 2) % 3] / amp_list[cell_num] * np.sin(phase_list[(cell_num + 2) % 3] - phase_list[cell_num])
        interaction = interaction_1 + interaction_2
        return interaction
    
    def _amp_modulation(self, cell_num, glucose):
        if cell_num == 0:
            alpha_const = -1
        else:
            alpha_const = 1
        
        if cell_num == 2:
            abs_concentration = 0.5
            somato_response = 2
        else:
            abs_concentration = 1
            somato_response = 0
            
        factor = 1/2 * abs_concentration * (1 + alpha_const*np.tanh((glucose + somato_response - self.glucose_0)/self.eta)) # factor > r^2
        return factor
    
    def _amp_couple(self, cell_num, phase_list, amp_list, first_action, second_action):
        interaction_1 = self.kapa * first_action * amp_list[(cell_num + 1) % 3] * np.cos(phase_list[(cell_num + 1) % 3] - phase_list[cell_num])
        interaction_2 = self.kapa * second_action * amp_list[(cell_num + 2) % 3] * np.cos(phase_list[(cell_num + 2) % 3] - phase_list[cell_num])
        interaction = interaction_1 + interaction_2
        return interaction