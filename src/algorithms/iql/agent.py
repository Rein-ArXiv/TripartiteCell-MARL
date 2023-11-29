import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from memory import ReplayBuffer
from model import DDQN

class IQL:
    def __init__(self, env, is_train=None, cuda=False, cuda_num=None):
        self.env = env
        self.state_dim = env.observation_space.shape[1]
        self.action_dim = env.cell_interaction_type_num
        self.islet_num = env.islet_num
        self.reward_mode = env.reward_mode

        assert is_train != None, "Please select execution mode. [is_train=True, is_train=False]"
        self.is_train = is_train

        if cuda:
            self.device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        print(f"Using {self.device}.")

    def train(self, batch_size=256, target_update_freq=2000, gamma=0.95, memory_capacity=2e5, max_epi=1000, eps_linear=True,\
        param_location="../../../parameters/iql"):

        assert self.is_train, "Initialized in test mod."

        self.batch_size = batch_size
        self.max_eps = 1.0
        self.min_eps = 0.01
        self.eps = self.max_eps
        self.eps_decay = 1 - (1 / max_epi * 5)
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.max_norm = 0.5

        self.alpha_cell_loss_list = []
        self.beta_cell_loss_list = []
        self.delta_cell_loss_list = []
        
        self.alpha_transition = list([] for i in range(self.islet_num))
        self.beta_transition = list([] for i in range(self.islet_num))
        self.delta_transition = list([] for i in range(self.islet_num))

        self.alpha_cell_online_dqn = DDQN(self.state_dim, self.action_dim).to(self.device)
        self.alpha_cell_target_dqn = DDQN(self.state_dim, self.action_dim).to(self.device)
        self.beta_cell_online_dqn = DDQN(self.state_dim, self.action_dim).to(self.device)
        self.beta_cell_target_dqn = DDQN(self.state_dim, self.action_dim).to(self.device)
        self.delta_cell_online_dqn = DDQN(self.state_dim, self.action_dim).to(self.device)
        self.delta_cell_target_dqn = DDQN(self.state_dim, self.action_dim).to(self.device)

        self.alpha_cell_target_dqn.load_state_dict(self.alpha_cell_online_dqn.state_dict())
        self.beta_cell_target_dqn.load_state_dict(self.beta_cell_online_dqn.state_dict())
        self.delta_cell_target_dqn.load_state_dict(self.delta_cell_online_dqn.state_dict())

        self.alpha_cell_target_dqn.eval()
        self.beta_cell_target_dqn.eval()
        self.delta_cell_target_dqn.eval()

        self.alpha_cell_optimizer = optim.AdamW(self.alpha_cell_online_dqn.parameters(), lr=1e-4)
        self.beta_cell_optimizer = optim.AdamW(self.beta_cell_online_dqn.parameters(), lr=1e-4)
        self.delta_cell_optimizer = optim.AdamW(self.delta_cell_online_dqn.parameters(), lr=1e-4)
        
        self.alpha_memory = ReplayBuffer(self.state_dim, memory_capacity)
        self.beta_memory = ReplayBuffer(self.state_dim, memory_capacity)
        self.delta_memory = ReplayBuffer(self.state_dim, memory_capacity)


        self.update_count = 0
        self.episode = 0

        wandb.init(project="Tripartite Cell IQL")

        max_frame = int(max_epi * self.env.max_time)
        terminated = True
        truncated = True
        
        print("--Train Start--")
        for frame_idx in range(max_frame):
            if terminated or truncated:
                state, info = self.env.reset()
                terminated = False
                truncated = False
                
                if (self.episode != 0):
                    wandb.log({"Train Episode Reward": episode_reward})
                    print(f"Training Episode: {self.episode}/{max_epi} \tTotal reward: {episode_reward} \tEpsiilon:{self.eps}")
                
                self.episode += 1
                episode_reward = 0

                                   
            actions = self._select_action(state)
            next_state, reward, terminated, truncated, info = self.step(actions)
            if self.env.reward_mode == "global":
                episode_reward += reward
            elif self.env.reward_mode == "local":
                episode_reward += np.average(reward)

            if (terminated or truncated) and (int(self.episode % 10) == 0):
                self._param_save(param_location)   
            
            state = next_state

            
            if len(self.alpha_memory) > self.batch_size:
                alpha_cell_loss, beta_cell_loss, delta_cell_loss = self.update_model()
                wandb.log({"Alpha cell loss": alpha_cell_loss.item(),
                            "Beta cell loss": beta_cell_loss.item(),
                            "Delta cell loss": delta_cell_loss.item()})
                self.update_count += 1

                if self.update_count % self.target_update_freq == 0:
                    self._target_network_update()
                    self.update_count = 0

            if eps_linear:
                self.eps = max(self.min_eps, self.max_eps - (self.max_eps - self.min_eps) * (frame_idx / max_frame))
            else:
                self.eps = self.max_eps * (self.eps_decay ** self.episode)
            wandb.log({"Epsilon": self.eps})

            if int(self.episode) == int(max_epi+1):
                print("Train reached max episode. Train End")
                break
        
    def step(self, actions):
        next_state, reward, terminated, truncated, info = self.env.step(actions)
        if self.is_train:
            for i in range(self.islet_num):
                if self.reward_mode == "local":
                    cell_reward = reward[i]
                else:
                    cell_reward = reward
                self.alpha_transition[i] += [cell_reward, next_state[i], terminated or truncated]
                self.beta_transition[i] += [cell_reward, next_state[i], terminated or truncated]
                self.delta_transition[i] += [cell_reward, next_state[i], terminated or truncated]
                
                self.alpha_memory.push(*self.alpha_transition[i])
                self.beta_memory.push(*self.beta_transition[i])
                self.delta_memory.push(*self.delta_transition[i])

        return next_state, reward, terminated, truncated, info
    

    def _param_save(self, param_location, param_suffix=None):
        abs_path = os.path.dirname(os.path.abspath(param_location)) 
        if not os.path.exists(param_location):
            os.makedirs(param_location)

        if param_suffix:
            torch.save(self.alpha_cell_online_dqn.state_dict(), f'{param_location}/alpha_cell_{param_suffix}.pth')
            torch.save(self.beta_cell_online_dqn.state_dict(), f'{param_location}/beta_cell_{param_suffix}.pth')
            torch.save(self.delta_cell_online_dqn.state_dict(), f'{param_location}/delta_cell_{param_suffix}.pth')
        else:
            torch.save(self.alpha_cell_online_dqn.state_dict(), f'{param_location}/alpha_cell.pth')
            torch.save(self.beta_cell_online_dqn.state_dict(), f'{param_location}/beta_cell.pth')
            torch.save(self.delta_cell_online_dqn.state_dict(), f'{param_location}/delta_cell.pth')

        print(f"Parameter Saved. Location is '{abs_path}'")

    def test(self, param_location=None, param_suffix=None, action_log_path=None):
        assert not self.is_train, "Initialized in train mode."
        episode_reward = 0
        
        self.alpha_cell_online_dqn = DDQN(self.state_dim, self.action_dim).to(self.device)
        self.beta_cell_online_dqn = DDQN(self.state_dim, self.action_dim).to(self.device)
        self.delta_cell_online_dqn = DDQN(self.state_dim, self.action_dim).to(self.device)

        self.param_load(param_location=param_location, param_suffix=param_suffix)
        
        print("--- Test Start ---")
        state, info = self.env.reset()
        terminated = False
        truncated = False
        score = 0

        while not (terminated or truncated):
            actions = self._select_action(state)
            next_state, reward, terminated, truncated, info = self.step(actions)
            
            if self.env.reward_mode == "global":
                score += reward
            elif self.env.reward_mode == "local":
                score += np.average(reward)

            state = next_state
        
        print(f"Score: {score}")
        self.env.close()
    
    def update_model(self):
        alpha_sample_list = self.alpha_memory.sample(self.batch_size)
        beta_sample_list = self.beta_memory.sample(self.batch_size)
        delta_sample_list = self.delta_memory.sample(self.batch_size)
        
        alpha_cell_loss = self._loss(alpha_sample_list, "alpha")
        beta_cell_loss = self._loss(beta_sample_list, "beta")
        delta_cell_loss = self._loss(delta_sample_list, "delta")
        
        self.alpha_cell_optimizer.zero_grad()
        self.beta_cell_optimizer.zero_grad()
        self.delta_cell_optimizer.zero_grad()

        alpha_cell_loss.backward()  
        beta_cell_loss.backward()
        delta_cell_loss.backward()
        
        nn.utils.clip_grad_norm_(self.alpha_cell_online_dqn.parameters(), self.max_norm)
        nn.utils.clip_grad_norm_(self.beta_cell_online_dqn.parameters(), self.max_norm)
        nn.utils.clip_grad_norm_(self.delta_cell_online_dqn.parameters(), self.max_norm)
        
        self.alpha_cell_optimizer.step()
        self.beta_cell_optimizer.step()
        self.delta_cell_optimizer.step()
        
        self.alpha_cell_loss_list.append(alpha_cell_loss.item())
        self.beta_cell_loss_list.append(beta_cell_loss.item())
        self.delta_cell_loss_list.append(delta_cell_loss.item())
        
        return alpha_cell_loss, beta_cell_loss, delta_cell_loss
                
    def param_load(self, param_location=None, param_suffix=None):
        if param_location is None:
            param_location = str(f'../../../parameters/iql') # default location

        if param_suffix:
            self.alpha_cell_online_dqn.load_state_dict(torch.load(f"{param_location}/alpha_cell_{param_suffix}.pth"))
            self.beta_cell_online_dqn.load_state_dict(torch.load(f"{param_location}/beta_cell_{param_suffix}.pth"))
            self.delta_cell_online_dqn.load_state_dict(torch.load(f"{param_location}/delta_cell_{param_suffix}.pth"))

        else:
            self.alpha_cell_online_dqn.load_state_dict(torch.load(f"{param_location}/alpha_cell.pth"))
            self.beta_cell_online_dqn.load_state_dict(torch.load(f"{param_location}/beta_cell.pth"))
            self.delta_cell_online_dqn.load_state_dict(torch.load(f"{param_location}/delta_cell.pth"))

    def _loss(self, sample_list, cell_type):
        state = torch.FloatTensor(sample_list["state"]).to(self.device)
        next_state = torch.FloatTensor(sample_list["next_state"]).to(self.device)
        reward = torch.FloatTensor(sample_list["reward"].reshape(-1, 1)).to(self.device)
        cell_action = torch.LongTensor(sample_list["action"]).reshape(-1, 1).to(self.device)
        terminated_truncated = torch.FloatTensor(sample_list["terminated_truncated"].reshape(-1, 1)).to(self.device)
        
        if cell_type == "alpha":
            curr_q_value = self.alpha_cell_online_dqn(state).gather(1, cell_action)
            next_q_value = self.alpha_cell_target_dqn(next_state).gather(1, self.alpha_cell_online_dqn(next_state).argmax(dim=1, keepdim=True)).detach()
        elif cell_type == "beta":
            curr_q_value = self.beta_cell_online_dqn(state).gather(1, cell_action)
            next_q_value = self.beta_cell_target_dqn(next_state).gather(1, self.beta_cell_online_dqn(next_state).argmax(dim=1, keepdim=True)).detach()
        elif cell_type == "delta":
            curr_q_value = self.delta_cell_online_dqn(state).gather(1, cell_action)
            next_q_value = self.delta_cell_target_dqn(next_state).gather(1, self.delta_cell_online_dqn(next_state).argmax(dim=1, keepdim=True)).detach()    
        
        mask = 1 - terminated_truncated
        target_q_value = (reward + self.gamma * next_q_value * mask).to(self.device)
        loss = F.smooth_l1_loss(curr_q_value, target_q_value)
        return loss
    
    def _select_action(self, state):
        action_list = []
        alpha_cell_action_list = []
        beta_cell_action_list = []
        delta_cell_action_list= []
        
        if self.is_train:
            if self.eps > np.random.random():
                actions = self.env.action_space.sample()
                for islet_i in range(self.islet_num):
                    alpha_cell_action, beta_cell_action, delta_cell_action = actions[islet_i]
                    
                    alpha_cell_action_list.append(alpha_cell_action)
                    beta_cell_action_list.append(beta_cell_action)
                    delta_cell_action_list.append(delta_cell_action)

                    action_list.append([alpha_cell_action, beta_cell_action, delta_cell_action])
                    
            else:
                for islet_i in range(self.islet_num):
                    alpha_cell_action = self._get_action(self.alpha_cell_online_dqn, state[islet_i])
                    beta_cell_action = self._get_action(self.beta_cell_online_dqn, state[islet_i])
                    delta_cell_action = self._get_action(self.delta_cell_online_dqn, state[islet_i])
                    
                    alpha_cell_action_list.append(alpha_cell_action)
                    beta_cell_action_list.append(beta_cell_action)
                    delta_cell_action_list.append(delta_cell_action)

                    action_list.append([alpha_cell_action, beta_cell_action, delta_cell_action])
            
            for i in range(self.islet_num):
                self.alpha_transition[i] = [state[i], alpha_cell_action_list[i]]
                self.beta_transition[i] = [state[i], beta_cell_action_list[i]]
                self.delta_transition[i] = [state[i], delta_cell_action_list[i]]

        else:
            for islet_i in range(self.islet_num):
                alpha_cell_action = self._get_action(self.alpha_cell_online_dqn, state[islet_i])
                beta_cell_action = self._get_action(self.beta_cell_online_dqn, state[islet_i])
                delta_cell_action = self._get_action(self.delta_cell_online_dqn, state[islet_i])

                action_list.append([alpha_cell_action, beta_cell_action, delta_cell_action])

        return np.array(action_list)
    
    def _get_action(self, ddqn, state):
        action = ddqn((torch.from_numpy(state)).to(self.device, dtype=torch.float)).argmax()
        action = action.detach().cpu().numpy()
        return action
        
    def _get_q_value(self, ddqn, state):
        q_value = ddqn((torch.from_numpy(state)).to(self.device, dtype=torch.float)).detach().cpu()
        return q_value

    def _target_network_update(self):
        self.alpha_cell_target_dqn.load_state_dict(self.alpha_cell_online_dqn.state_dict())
        self.beta_cell_target_dqn.load_state_dict(self.beta_cell_online_dqn.state_dict())
        self.delta_cell_target_dqn.load_state_dict(self.delta_cell_online_dqn.state_dict())