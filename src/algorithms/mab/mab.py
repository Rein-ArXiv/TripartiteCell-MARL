import os
import pickle
import numpy as np

from .agent import Agent, GradientAgent, BetaAgent
from .bandit import GaussianBandit, BinomialBandit, BernoulliBandit
from .policy import StaticEpsilonGreedy, UCB, Softmax



class MAB(object):
    def __init__(self, env, agent, bandit, policy, gamma=None, is_train=None):
        self.env = env
        self.islet_num = env.islet_num
        self.reward_mode = env.reward_mode
        self.gamma = gamma
        self.alpha_agent = agent(self.islet_num, bandit, policy, gamma=gamma)
        self.beta_agent = agent(self.islet_num, bandit, policy, gamma=gamma)
        self.delta_agent = agent(self.islet_num, bandit, policy, gamma=gamma)

        assert is_train != None, "Please select an execution mode: [is_train=True, is_train=False]"
        self.is_train = is_train

    def train(self,
        max_epi=1000,
        #eps_linear=False,
        #eps_decay=0.995,
        external_glu=0,
        #param_location="../../../parameters/mab",
        #param_suffix=None
        ):

        assert self.is_train, "Initialized in test mode."

        #self.max_eps = 1.0
        #self.min_eps = 0.01
        #self.eps = self.max_eps
        #self.eps_decay = eps_decay
        
        self.update_count = 0
        self.episode = 0

        self.alpha_agent.reset()
        self.beta_agent.reset()
        self.delta_agent.reset()

        max_frame = int((max_epi + 1) * self.env.max_time)
        terminated = True
        truncated = True

        print("--Train Start--")
        for frame_idx in range(max_frame):
            if terminated or truncated:
                state, info = self.env.reset()
                terminated = False
                truncated = False
                
                if (self.episode != 0):
                    #wandb.log({"Episode Reward": episode_reward})
                    print(f"Episode: {self.episode}/{max_epi} \tEpisode Reward: {episode_reward}")
                    #current_episode_avg_reward[self.episode % 10] = episode_reward
                self.episode += 1
                episode_reward = 0

                                   
            actions = self._select_action()

            next_state, reward, terminated, truncated, info = self.step(actions, external_glu)
            self.alpha_agent.observe(reward, self.env.reward_mode=="local")
            self.beta_agent.observe(reward, self.env.reward_mode=="local")
            self.delta_agent.observe(reward, self.env.reward_mode=="local")

            if self.env.reward_mode == "global":
                episode_reward += reward
            elif self.env.reward_mode == "local":
                episode_reward += np.average(reward)

            '''
            if (terminated or truncated) and (int(self.episode % 10) == 0):
                if (np.average(current_episode_avg_reward) > np.average(best_episode_avg_reward)):
                    best_episode_avg_reward = np.copy(current_episode_avg_reward)
                    self._param_save(param_location=param_location, param_suffix=param_suffix)
                self._param_save(param_location=param_location, param_suffix=f"{param_suffix}_final")
            '''
            state = next_state

            #if eps_linear:
                #self.eps = max(self.min_eps, self.max_eps - (self.max_eps - self.min_eps) * (frame_idx / max_frame))
            #else:
                #self.eps = self.max_eps * (self.eps_decay ** self.episode)
            #wandb.log({"Epsilon": self.eps})

            if int(self.episode) == int(max_epi+1):
                print("The train reached the maximum episode. The train has ended.")
                break
        
    def step(self, actions, external_glucose):
        next_state, reward, terminated, truncated, info = self.env.step(actions, external_glucose)
        if self.is_train:
            for i in range(self.islet_num):
                if self.reward_mode == "local":
                    cell_reward = reward[i]
                else:
                    cell_reward = reward
                
        return next_state, reward, terminated, truncated, info

    def _param_save(self, param_location, param_suffix=None):
        abs_path = os.path.dirname(os.path.abspath(param_location)) 
        if not os.path.exists(param_location):
            os.makedirs(param_location)

        #if param_suffix:
            #torch.save(self.alpha_cell_online_dqn.state_dict(), f'{param_location}/alpha_cell_{param_suffix}.pth')
            #torch.save(self.beta_cell_online_dqn.state_dict(), f'{param_location}/beta_cell_{param_suffix}.pth')
            #torch.save(self.delta_cell_online_dqn.state_dict(), f'{param_location}/delta_cell_{param_suffix}.pth')
        #else:
            #torch.save(self.alpha_cell_online_dqn.state_dict(), f'{param_location}/alpha_cell.pth')
            #torch.save(self.beta_cell_online_dqn.state_dict(), f'{param_location}/beta_cell.pth')
            #torch.save(self.delta_cell_online_dqn.state_dict(), f'{param_location}/delta_cell.pth')

        print(f"Parameter saved. Location is '{abs_path}'")

    def _select_action(self):
        action_list = []

        for islet_i in range(self.islet_num):
            alpha_cell_action, beta_cell_action, delta_cell_action = self._get_action(islet_i)
            action_list.append([alpha_cell_action, beta_cell_action, delta_cell_action])

        return np.array(action_list)

    def _get_action(self, i):
        alpha_action = self.alpha_agent.choose(i)
        beta_action = self.beta_agent.choose(i)
        delta_action = self.delta_agent.choose(i)
        return alpha_action, beta_action, delta_action