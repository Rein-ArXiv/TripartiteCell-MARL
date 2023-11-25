import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CellEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 1}
    
    def __init__(self, islet_num=20, max_time=200, glucose_fix=False):
        super(CellEnv, self).__init__()

        self.cell_num = 3
        self.cell_interaction_type_num = 9
        self.islet_num = islet_num
        self.observation_space = spaces.Box(low=0.0, high=40.0, shape=(self.islet_num, int(self.cell_num + 1)))
        self.action_space = spaces.MultiDiscrete(np.full((self.islet_num, self.cell_num), self.cell_interaction_type_num))
        self.state = None
        self.glucose_0 = 4.     # default glucose level
        self.max_time = max_time
        self.glucose_fix = glucose_fix
        
        self.kapa = 0.4 # changable
        self.eta = 5
        
        # Env parameters
        self.reward_scale = 0.01
        self.terminated = None
        self.truncated = None

    def reset(self, glucose_level=None, seed=None):
        self.curr_time = 0
        self.terminated = False
        self.truncated = False
        self.info = {}
        
        if self.glucose_fix:
            glucose_init = glucose_level
            assert glucose_init is not None, 'glucose fix is True but glucose level is not assigned.'
        else:
            glucose_init = random.uniform(0.0, 8.0)
            
        self.phases = []
        self.amps = []
        self.hormones = []
        self.glucose = glucose_init
        self.glucose_fluctuations = np.zeros(self.islet_num)

        for i in range(self.islet_num):
            cell_phase = np.random.uniform(0.0, 2*np.pi, 3)
            cell_amp = np.random.uniform(0.0, 0.5, 3)
            
            self.phases.append(cell_phase)
            self.amps.append(cell_amp)
            self.hormones.append(self.hormone_cal(self.phases[i], self.amps[i]))
        
        self.phases = np.array(self.phases)
        self.amps = np.array(self.amps)
        self.hormones = np.array(self.hormones)
        self.state = np.hstack((self.hormones, np.full((self.islet_num, 1), glucose_init)))
        
        self.info["phase"] = self.phases
        self.info["amplitudes"] = self.amps
        self.info["hormones"] = self.hormones
        self.info["glucose"] = self.glucose
        
        return self.state, self.info
    
    def render(self):
        pass

    def step(self, actions, external_glucose=0):
        assert self.state is not None, "Call reset before using step method"

        self.info = {}
        total_reward = 0
        self.total_glucose_delta = 0
        
        hormone_before = self.hormones.copy()
        for i in range(self.islet_num):
            alpha_action, beta_action, delta_action = np.array(actions[i])
            alpha_phase_delta = self.phase_oscilation(0, self.phases[i], self.amps[i], self.glucose, alpha_action)
            beta_phase_delta = self.phase_oscilation(1, self.phases[i], self.amps[i], self.glucose, beta_action)
            delta_phase_delta = self.phase_oscilation(2, self.phases[i], self.amps[i], self.glucose, delta_action)
            
            alpha_amp_delta = self.amp_oscilation(0, self.phases[i], self.amps[i], self.glucose, alpha_action)
            beta_amp_delta = self.amp_oscilation(1, self.phases[i], self.amps[i], self.glucose, beta_action)
            delta_amp_delta = self.amp_oscilation(2, self.phases[i], self.amps[i], self.glucose, delta_action)
            
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
            self.glucose_fluctuations[i] = self.glucose_delta(self.hormones[i], self.glucose)
            
        self.total_glucose_delta = np.sum(self.glucose_fluctuations) / self.islet_num
        self.glucose += self.total_glucose_delta + external_glucose
        
        if self.glucose < 0:
            self.glucose = 0.
        
        reward = np.clip(-abs(self.total_glucose_delta) - (np.sum(self.hormones) / self.islet_num), -3.0, 0.0)
        self.curr_time += 1
        
        self.state = np.hstack((self.hormones, np.full((self.islet_num, 1), self.glucose)))
        if self.curr_time > self.max_time:
            self.truncated = True
        
        self.info["phase"] = self.phases
        self.info["amplitudes"] = self.amps
        self.info["hormones"] = self.hormones
        self.info["glucose"] = self.glucose

        return self.state, reward, self.terminated, self.truncated, self.info
                
    def phase_oscilation(self, cell_num, phase_list, amp_list, glucose, cell_action):
        w_sigma = 2 * np.pi / (np.random.normal(5, 0.1))
        g_sigma = self._phase_modulation(cell_num, glucose)
        phase_delta = w_sigma + g_sigma * np.cos(phase_list[cell_num]) + self._phase_couple(cell_num, phase_list, amp_list, cell_action)
        return phase_delta
    
    def amp_oscilation(self, cell_num, phase_list, amp_list, glucose, cell_action):
        amp_delta = (self._amp_modulation(cell_num, glucose) - amp_list[cell_num] ** 2) * amp_list[cell_num] + self._amp_couple(cell_num, phase_list, amp_list, cell_action)
        return amp_delta
        
    def hormone_cal(self, phase_list, amp_list):
        hormone_array = (amp_list * (1 + np.cos(phase_list)) / 2)
        return hormone_array
    
    def glucose_delta(self, hormone_array, glucose):
        lamb = 1
        glucose_delta = lamb * (self.glucose_0 * hormone_array[0] - glucose * hormone_array[1])
        return glucose_delta
        
    def _phase_modulation(self, cell_num, glucose):
        mu = 0.1
        #mu = 0.5
        if cell_num != 0:
            mu *= -1
            
        factor = mu * (glucose - self.glucose_0)
        return factor
    
    def _phase_couple(self, cell_num, phase_list, amp_list, cell_action):
        # alpha -> bta, dta
        # beta -> dtb, atb
        # delta -> atd, btd
        first_action, second_action = self._interaction(cell_action)
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
    
    def _amp_couple(self, cell_num, phase_list, amp_list, cell_action):
        first_action, second_action = self._interaction(cell_action)
        interaction_1 = self.kapa * first_action * amp_list[(cell_num + 1) % 3] * np.cos(phase_list[(cell_num + 1) % 3] - phase_list[cell_num])
        interaction_2 = self.kapa * second_action * amp_list[(cell_num + 2) % 3] * np.cos(phase_list[(cell_num + 2) % 3] - phase_list[cell_num])
        interaction = interaction_1 + interaction_2
        return interaction

    def _interaction(self, cell_action):
        action_type = np.argmax(cell_action)
        first_action = action_type // 3 - 1
        second_action = action_type % 3 - 1
        return first_action, second_action

if __name__ == "__main__":
    print("CellEnv test start.")
    env = CellEnv()
    print("CellEnv made. Resetting env")
    state, info = env.reset()
    print(f"CellEnv resetted. state: {state}, \ninfo: {info}")  
    print("Testing Env...")
    while(not env.terminated and not env.truncated):
        state, reward, _, _, _ = env.step(env.action_space.sample())

        print(f"Reward: {reward}, \tGlucose Reward: {env.total_glucose_delta}, \tHormone Reward: {np.sum(env.hormones)/env.islet_num}")

    print("CellEnv test end.")