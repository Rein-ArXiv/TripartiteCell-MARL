import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from IPython.display import clear_output

from memory import ReplayBuffer, Recording
from model import DQN

# 추가할 것 - parameter prefix, plot 
# 개별 학습, 전체 학습

class DQNAgent:
    def __init__(self, env, batch_size=64, target_update_freq=2000, gamma=0.99, memory_capacity=3e5, log_prefix=0, log_location=None, train=True, cuda_num=0):
        self.env = env
        self.state_dim = self.env.observation_space.shape[1]
        self.action_dim = 3
        self.batch_size = batch_size
        self.islet_num = env.islet_num
        self.time_interval = env.time_interval
        self.memory = ReplayBuffer(self.env, self.islet_num, self.state_dim, memory_capacity)
        self.max_eps = 1.0
        self.min_eps = 0.01
        self.eps = self.max_eps
        self.eps_decay = 2000
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
        self.max_norm = 0.5
        self.islet_dqn = {}
        #self.record = env.record
        self.log_prefix = log_prefix
        self.log_location = log_location

        
        for i in range(self.islet_num):
            islet_name = f"islet_{i}"
        
        
        self.alpha_beta_online_dqn = list()
        self.alpha_beta_target_dqn = list()
        self.alpha_delta_online_dqn = list()
        self.alpha_delta_target_dqn = list()
        self.beta_alpha_online_dqn = list()
        self.beta_alpha_target_dqn = list()
        self.beta_delta_online_dqn = list()
        self.beta_delta_target_dqn = list()
        self.delta_alpha_online_dqn = list()
        self.delta_alpha_target_dqn = list()
        self.delta_beta_online_dqn = list()
        self.delta_beta_target_dqn = list()
        
        self.alpha_beta_optimizer = list()
        self.alpha_delta_optimizer = list()
        self.beta_alpha_optimizer = list()
        self.beta_delta_optimizer = list()
        self.delta_alpha_optimizer = list()
        self.delta_beta_optimizer = list()
        
        for i in range(self.islet_num):
            self.alpha_beta_online_dqn.append(DQN(self.env, self.state_dim, self.action_dim).to(self.device))
            self.alpha_beta_target_dqn.append(DQN(self.env, self.state_dim ,self.action_dim).to(self.device))
        
            self.alpha_delta_online_dqn.append(DQN(self.env, self.state_dim, self.action_dim).to(self.device))
            self.alpha_delta_target_dqn.append(DQN(self.env, self.state_dim, self.action_dim).to(self.device))

            self.beta_alpha_online_dqn.append(DQN(self.env, self.state_dim, self.action_dim).to(self.device))
            self.beta_alpha_target_dqn.append(DQN(self.env, self.state_dim, self.action_dim).to(self.device))

            self.beta_delta_online_dqn.append(DQN(self.env, self.state_dim, self.action_dim).to(self.device))
            self.beta_delta_target_dqn.append(DQN(self.env, self.state_dim, self.action_dim).to(self.device))

            self.delta_alpha_online_dqn.append(DQN(self.env, self.state_dim, self.action_dim).to(self.device))
            self.delta_alpha_target_dqn.append(DQN(self.env, self.state_dim, self.action_dim).to(self.device))

            self.delta_beta_online_dqn.append(DQN(self.env, self.state_dim, self.action_dim).to(self.device))
            self.delta_beta_target_dqn.append(DQN(self.env, self.state_dim, self.action_dim).to(self.device))
        
        
            self.alpha_beta_target_dqn[i].load_state_dict(self.alpha_beta_online_dqn[i].state_dict())
            self.alpha_delta_target_dqn[i].load_state_dict(self.alpha_delta_online_dqn[i].state_dict())
            self.beta_alpha_target_dqn[i].load_state_dict(self.beta_alpha_online_dqn[i].state_dict())
            self.beta_delta_target_dqn[i].load_state_dict(self.beta_delta_online_dqn[i].state_dict())
            self.delta_alpha_target_dqn[i].load_state_dict(self.delta_alpha_online_dqn[i].state_dict())
            self.delta_beta_target_dqn[i].load_state_dict(self.delta_beta_online_dqn[i].state_dict())
        
            self.alpha_beta_target_dqn[i].eval()
            self.alpha_delta_target_dqn[i].eval()
            self.beta_alpha_target_dqn[i].eval()
            self.beta_delta_target_dqn[i].eval()
            
            self.delta_alpha_target_dqn[i].eval()
            self.delta_beta_target_dqn[i].eval()
        
            self.alpha_beta_optimizer.append(optim.AdamW(self.alpha_beta_online_dqn[i].parameters(), lr=1e-3))
            self.alpha_delta_optimizer.append(optim.AdamW(self.alpha_delta_online_dqn[i].parameters(), lr=1e-3))

            self.beta_alpha_optimizer.append(optim.AdamW(self.beta_alpha_online_dqn[i].parameters(), lr=1e-3))
            self.beta_delta_optimizer.append(optim.AdamW(self.beta_delta_online_dqn[i].parameters(), lr=1e-3))

            self.delta_alpha_optimizer.append(optim.AdamW(self.delta_alpha_online_dqn[i].parameters(), lr=1e-3))
            self.delta_beta_optimizer.append(optim.AdamW(self.delta_beta_online_dqn[i].parameters(), lr=1e-3))
        
        self.transition = list()
        self.is_train = train
        
    def step(self, actions):
        next_state, reward, terminated, truncated, info = self.env.step(actions)
        if self.is_train:
            self.transition += [reward, next_state, terminated or truncated]
            self.memory.push(*self.transition)
            
        return next_state, reward, terminated, truncated
    
    def update_model(self):
        sample_list = self.memory.sample(self.batch_size)
        ab_loss_list = []
        ad_loss_list = []
        ba_loss_list = []
        bd_loss_list = []
        da_loss_list = []
        db_loss_list = []
        
        for islet_i in range(self.islet_num):
            alpha_beta_loss, alpha_delta_loss, beta_alpha_loss, beta_delta_loss, delta_alpha_loss, delta_beta_loss = self._loss(sample_list, islet_i)
            self.alpha_beta_optimizer[islet_i].zero_grad()
            self.alpha_delta_optimizer[islet_i].zero_grad()
            self.beta_alpha_optimizer[islet_i].zero_grad()
            self.beta_delta_optimizer[islet_i].zero_grad()
            self.delta_alpha_optimizer[islet_i].zero_grad()
            self.delta_beta_optimizer[islet_i].zero_grad()

            alpha_beta_loss.backward()
            alpha_delta_loss.backward()
            beta_alpha_loss.backward()
            beta_delta_loss.backward()
            delta_alpha_loss.backward()
            delta_beta_loss.backward()

            nn.utils.clip_grad_norm_(self.alpha_beta_online_dqn[islet_i].parameters(), self.max_norm)
            nn.utils.clip_grad_norm_(self.alpha_delta_online_dqn[islet_i].parameters(), self.max_norm)
            nn.utils.clip_grad_norm_(self.beta_alpha_online_dqn[islet_i].parameters(), self.max_norm)
            nn.utils.clip_grad_norm_(self.beta_delta_online_dqn[islet_i].parameters(), self.max_norm)
            nn.utils.clip_grad_norm_(self.delta_alpha_online_dqn[islet_i].parameters(), self.max_norm)
            nn.utils.clip_grad_norm_(self.delta_beta_online_dqn[islet_i].parameters(), self.max_norm)

            self.alpha_beta_optimizer[islet_i].step()
            self.alpha_delta_optimizer[islet_i].step()
            self.beta_alpha_optimizer[islet_i].step()
            self.beta_delta_optimizer[islet_i].step()
            self.delta_alpha_optimizer[islet_i].step()
            self.delta_beta_optimizer[islet_i].step()
        
            ab_loss_list.append(alpha_beta_loss.item())
            ad_loss_list.append(alpha_delta_loss.item())
            ba_loss_list.append(beta_alpha_loss.item())
            bd_loss_list.append(beta_delta_loss.item())
            da_loss_list.append(delta_alpha_loss.item())
            db_loss_list.append(delta_beta_loss.item())
            
        return (ab_loss_list,
                ad_loss_list,
                ba_loss_list,
                bd_loss_list,
                da_loss_list,
                db_loss_list)
                
    def train(self, max_frame=int(2e6), max_epi=1000):
        if not os.path.exists(f'../parameters/{self.log_prefix}'):
            os.makedirs(f'../parameters/{self.log_prefix}')
        
        update_count = 0
        episode = 0
        terminated = True
        truncated = True
        total_reward = 0
                
        print("--Train Start--")
        for frame_idx in tqdm(range(max_frame)):
            if terminated or truncated:
                state = self.env.reset()
                terminated = False
                truncated = False
                episode += 1
                total_reward = 0
                                    
            actions = self._select_action(state)
            
            next_state, reward, terminated, truncated = self.step(actions)
            
            total_reward += reward

            if (terminated or truncated) and (int(episode % 10) == 0):
                #self.env.record.log(log_prefix=self.log_prefix, location=self.log_location)
                        
                for islet_i in range(self.islet_num):
                    torch.save(self.alpha_beta_online_dqn[islet_i].state_dict(), \
                               f'../parameters/{self.log_prefix}/DDQN-TriPartiteCell_alpha_beta_islet_{islet_i+1}_epi_{episode}_interval_{self.time_interval}.pth')
                    torch.save(self.alpha_delta_online_dqn[islet_i].state_dict(), \
                               f'../parameters/{self.log_prefix}/DDQN-TriPartiteCell_alpha_delta_islet_{islet_i+1}_epi_{episode}_interval_{self.time_interval}.pth')
                    torch.save(self.beta_alpha_online_dqn[islet_i].state_dict(), \
                               f'../parameters/{self.log_prefix}/DDQN-TriPartiteCell_beta_alpha_islet_{islet_i+1}_epi_{episode}_interval_{self.time_interval}.pth')
                    torch.save(self.beta_delta_online_dqn[islet_i].state_dict(), \
                               f'../parameters/{self.log_prefix}/DDQN-TriPartiteCell_beta_delta_islet_{islet_i+1}_epi_{episode}_interval_{self.time_interval}.pth')
                    torch.save(self.delta_alpha_online_dqn[islet_i].state_dict(), \
                               f'../parameters/{self.log_prefix}/DDQN-TriPartiteCell_delta_alpha_islet_{islet_i+1}_epi_{episode}_interval_{self.time_interval}.pth')
                    torch.save(self.delta_beta_online_dqn[islet_i].state_dict(), \
                               f'../parameters/{self.log_prefix}/DDQN-TriPartiteCell_delta_beta_islet_{islet_i+1}_epi_{episode}_interval_{self.time_interval}.pth')
                                        
            state = next_state

            if len(self.memory) > self.batch_size:
                loss = self.update_model()
                #losses.append(loss)
                update_count += 1
                self.eps = max(self.min_eps, self.eps - (self.max_eps - self.min_eps) / self.eps_decay)

                if update_count % self.target_update_freq == 0:
                    self._target_network_update()
            
            if int(episode) == int(max_epi+1):
                print("Train reached max episode. Train End")
                break
            
    def test(self, param_log_prefix=None, share=True, param_location=None, test_log_prefix=None, log_location=None, max_epi=1000, inter=1):
        self.is_train = False
        self.eps = 0.
        
        actions_record = []
        if log_location==None:
            log_location = f"../record/test/{test_log_prefix}/"
            
        if not os.path.exists(log_location):
            os.makedirs(log_location)
        
        if param_log_prefix is None:
            param_log_prefix = self.log_prefix
        
        self.param_load(location=param_location, prefix=param_log_prefix, epi=max_epi, share=share)
        
        print("--- Test Start ---")
        state = self.env.reset()
        terminated = False
        truncated = False
        score = 0
        
        alpha_hormone = []
        beta_hormone = []
        delta_hormone = []
        glucoses = []
        
        actions_record = []
        record_file_path = "action_record.txt"
        for i in range(self.islet_num):
            actions_record.append([])
        
        while not (terminated or truncated):
            alpha_hormone.append(state[0][0])
            beta_hormone.append(state[0][1])
            delta_hormone.append(state[0][2])
            glucoses.append(state[0][3])
            
            actions = self._select_action(state)
            for i, action_i in enumerate(actions):
                actions_record[i].append(action_i)
            next_state, reward, terminated, truncated = self.step(actions)
            score += reward
            state = next_state
            
        with open(record_file_path, 'w') as file:
            for i, action_i in enumerate(actions_record):
                action_line = f'{i}-th cell action\n'
                file.write(action_line)
                for one_action in action_i:
                    line = ' '.join(map(str, one_action)) + '\n'
                    file.write(line)
                file.write('\n')
            
        plt.figure(figsize=(16,8))
        plt.subplot(121)
        plt.plot(glucoses)
        plt.subplot(122)
        plt.plot(alpha_hormone)
        plt.plot(beta_hormone)
        plt.plot(delta_hormone)
        plt.savefig(f"{log_location}result.png")
        
        #self.env.record.log(log_prefix=test_log_prefix, location=log_location, plot=True)
        print(f"Score: {score}")
        self.env.close()
    
    # 수정할 것
    def param_load(self, location=None, prefix=None, share=True, epi=1000, inter=1):
        
        if location is None:
            location = str(f'../parameters/{prefix}/')
        
        name = str('DDQN-TriPartiteCell_')
        
        for islet_i in range(self.islet_num):
            if share is True:
                i = 1
            else:
                i = islet_i+1

            self.alpha_beta_online_dqn[islet_i].load_state_dict(torch.load(location + name + f'alpha_beta_islet_{i}_epi_{epi}_interval_{inter}.pth'))
            self.alpha_delta_online_dqn[islet_i].load_state_dict(torch.load(location + name + f'alpha_delta_islet_{i}_epi_{epi}_interval_{inter}.pth'))
            self.beta_alpha_online_dqn[islet_i].load_state_dict(torch.load(location + name + f'beta_alpha_islet_{i}_epi_{epi}_interval_{inter}.pth'))
            self.beta_delta_online_dqn[islet_i].load_state_dict(torch.load(location + name + f'beta_delta_islet_{i}_epi_{epi}_interval_{inter}.pth'))
            self.delta_alpha_online_dqn[islet_i].load_state_dict(torch.load(location + name + f'delta_alpha_islet_{i}_epi_{epi}_interval_{inter}.pth'))
            self.delta_beta_online_dqn[islet_i].load_state_dict(torch.load(location + name + f'delta_beta_islet_{i}_epi_{epi}_interval_{inter}.pth'))
        
            
    def _loss(self, sample_list, islet_i):
        state = torch.FloatTensor(sample_list[islet_i]["state"]).to(self.device)
        next_state = torch.FloatTensor(sample_list[islet_i]["next_state"]).to(self.device)
            
        reward = torch.FloatTensor(sample_list[islet_i]["reward"].reshape(-1, 1)).to(self.device)
        
        alpha_beta_action = torch.LongTensor(sample_list[islet_i]["alpha_beta_action"]).reshape(-1, 1).to(self.device)
        alpha_delta_action = torch.LongTensor(sample_list[islet_i]["alpha_delta_action"]).reshape(-1, 1).to(self.device)
        beta_alpha_action = torch.LongTensor(sample_list[islet_i]["beta_alpha_action"]).reshape(-1, 1).to(self.device)
        beta_delta_action = torch.LongTensor(sample_list[islet_i]["beta_delta_action"]).reshape(-1, 1).to(self.device)
        delta_alpha_action = torch.LongTensor(sample_list[islet_i]["delta_alpha_action"]).reshape(-1, 1).to(self.device)
        delta_beta_action = torch.LongTensor(sample_list[islet_i]["delta_beta_action"]).reshape(-1, 1).to(self.device)
        
        terminated_truncated = torch.FloatTensor(sample_list[islet_i]["terminated_truncated"].reshape(-1, 1)).to(self.device)
        
        curr_alpha_beta_q_value = self.alpha_beta_online_dqn[islet_i](state).gather(1, alpha_beta_action)
        curr_alpha_delta_q_value = self.alpha_delta_online_dqn[islet_i](state).gather(1, alpha_delta_action)
        curr_beta_alpha_q_value = self.beta_alpha_online_dqn[islet_i](state).gather(1, beta_alpha_action)
        curr_beta_delta_q_value = self.beta_delta_online_dqn[islet_i](state).gather(1, beta_delta_action)
        curr_delta_alpha_q_value = self.delta_alpha_online_dqn[islet_i](state).gather(1, delta_alpha_action)
        curr_delta_beta_q_value = self.delta_beta_online_dqn[islet_i](state).gather(1, delta_beta_action)

        next_alpha_beta_q_value = self.alpha_beta_target_dqn[islet_i](next_state).gather(1, self.alpha_beta_online_dqn[islet_i](next_state).argmax(dim=1, keepdim=True)).detach()
        next_alpha_delta_q_value = self.alpha_delta_target_dqn[islet_i](next_state).gather(1, self.alpha_delta_online_dqn[islet_i](next_state).argmax(dim=1, keepdim=True)).detach()
        next_beta_alpha_q_value = self.beta_alpha_target_dqn[islet_i](next_state).gather(1, self.beta_alpha_online_dqn[islet_i](next_state).argmax(dim=1, keepdim=True)).detach()
        next_beta_delta_q_value = self.beta_delta_target_dqn[islet_i](next_state).gather(1, self.beta_delta_online_dqn[islet_i](next_state).argmax(dim=1, keepdim=True)).detach()
        next_delta_alpha_q_value = self.delta_alpha_target_dqn[islet_i](next_state).gather(1, self.delta_alpha_online_dqn[islet_i](next_state).argmax(dim=1, keepdim=True)).detach()
        next_delta_beta_q_value = self.delta_beta_target_dqn[islet_i](next_state).gather(1, self.delta_beta_online_dqn[islet_i](next_state).argmax(dim=1, keepdim=True)).detach()
            
        
        mask = 1 - terminated_truncated
        

        target_alpha_beta_q_value = (reward + self.gamma * next_alpha_beta_q_value * mask).to(self.device)
        target_alpha_delta_q_value = (reward + self.gamma * next_alpha_delta_q_value * mask).to(self.device)
        target_beta_alpha_q_value = (reward + self.gamma * next_beta_alpha_q_value * mask).to(self.device)
        target_beta_delta_q_value = (reward + self.gamma * next_beta_delta_q_value * mask).to(self.device)
        target_delta_alpha_q_value = (reward + self.gamma * next_delta_alpha_q_value * mask).to(self.device)
        target_delta_beta_q_value = (reward + self.gamma * next_delta_beta_q_value * mask).to(self.device)   
            
        alpha_beta_loss = F.smooth_l1_loss(curr_alpha_beta_q_value, target_alpha_beta_q_value)
        alpha_delta_loss = F.smooth_l1_loss(curr_alpha_delta_q_value, target_alpha_delta_q_value)
        beta_alpha_loss = F.smooth_l1_loss(curr_beta_alpha_q_value, target_beta_alpha_q_value)
        beta_delta_loss = F.smooth_l1_loss(curr_beta_delta_q_value, target_beta_delta_q_value)
        delta_alpha_loss = F.smooth_l1_loss(curr_delta_alpha_q_value, target_delta_alpha_q_value)
        delta_beta_loss = F.smooth_l1_loss(curr_delta_beta_q_value, target_delta_beta_q_value)
        
        return alpha_beta_loss, alpha_delta_loss, beta_alpha_loss, beta_delta_loss, delta_alpha_loss, delta_beta_loss
    
    def _select_action(self, state):
        action_list = []
        ab_actions = []
        ad_actions = []
        ba_actions = []
        bd_actions = []
        da_actions = []
        db_actions = []
        
        if self.eps > np.random.random():
            actions = self.env.action_space.sample()
            for islet_i in range(self.islet_num):
                alpha_beta_action, alpha_delta_action, beta_alpha_action, beta_delta_action, delta_alpha_action, delta_beta_action = actions[islet_i]
                action_list.append([alpha_beta_action, alpha_delta_action, beta_alpha_action, beta_delta_action, delta_alpha_action, delta_beta_action])
                ab_actions.append(alpha_beta_action)
                ad_actions.append(alpha_delta_action)
                ba_actions.append(beta_alpha_action)
                bd_actions.append(beta_delta_action)
                da_actions.append(delta_alpha_action)
                db_actions.append(delta_beta_action)
        else:
            for islet_i in range(self.islet_num):
                alpha_beta_action = self._get_action(self.alpha_beta_online_dqn[islet_i], state[islet_i])
                alpha_delta_action = self._get_action(self.alpha_delta_online_dqn[islet_i], state[islet_i])
                beta_alpha_action = self._get_action(self.beta_alpha_online_dqn[islet_i], state[islet_i])
                beta_delta_action = self._get_action(self.beta_delta_online_dqn[islet_i], state[islet_i])
                delta_alpha_action = self._get_action(self.delta_alpha_online_dqn[islet_i], state[islet_i])
                delta_beta_action = self._get_action(self.delta_beta_online_dqn[islet_i], state[islet_i])
                
                ab_actions.append(alpha_beta_action)
                ad_actions.append(alpha_delta_action)
                ba_actions.append(beta_alpha_action)
                bd_actions.append(beta_delta_action)
                da_actions.append(delta_alpha_action)
                db_actions.append(delta_beta_action)
                
                action_list.append([alpha_beta_action, alpha_delta_action, beta_alpha_action, beta_delta_action, delta_alpha_action, delta_beta_action])
            
        if self.is_train:
            self.transition = [state, ab_actions, ad_actions, ba_actions, bd_actions, da_actions, db_actions]
            
        return action_list
    
    def _get_action(self, dqn, state):
        action = dqn((torch.from_numpy(state)).to(self.device, dtype=torch.float)).argmax()
        action = action.detach().cpu().numpy()
        return action
        
    
    def _target_network_update(self):
        for i in range(self.islet_num):
            self.alpha_beta_target_dqn[i].load_state_dict(self.alpha_beta_online_dqn[i].state_dict())
            self.alpha_delta_target_dqn[i].load_state_dict(self.alpha_delta_online_dqn[i].state_dict())
            self.beta_alpha_target_dqn[i].load_state_dict(self.beta_alpha_online_dqn[i].state_dict())
            self.beta_delta_target_dqn[i].load_state_dict(self.beta_delta_online_dqn[i].state_dict())
            self.delta_alpha_target_dqn[i].load_state_dict(self.delta_alpha_online_dqn[i].state_dict())
            self.delta_beta_target_dqn[i].load_state_dict(self.delta_beta_online_dqn[i].state_dict())
    
    def _log(self, log_location=None):
        self.record.log(log_prefix=self.log_prefix, location=log_location)
                        
    def _plot(self,
              episode_idx,
              glucoses,
              rewards,
              alpha_beta_action,
              alpha_delta_action,
              beta_alpha_action,
              beta_delta_action,
              delta_alpha_action,
              delta_beta_action,
              alpha_phase,
              beta_phase,
              delta_phase,
              alpha_amp,
              beta_amp,
              delta_amp,
              alpha_hormone,
              beta_hormone,
              delta_hormone):
        clear_output(True)
        plt.figure(figsize=(20,12))
        plt.subplot(231)
        plt.title('episode_idx %s. glucoses' % (episode_idx))
        plt.plot(glucoses)
        plt.subplot(232)
        plt.title('action_hist')
        plt.hist([alpha_beta_action, alpha_delta_action, beta_alpha_action, beta_delta_action, delta_alpha_action, delta_beta_action],
                 bins = 6, label=['alpha_beta', 'alpha_delta', 'beta_alpha', 'beta_delta', 'delta_alpha', 'delta_beta'])
        plt.legend(loc='upper left')
        plt.subplot(233)
        plt.title('rewards / episode')
        plt.plot(rewards)
        plt.subplot(234)
        plt.title('phase')
        plt.plot((1+np.cos(alpha_phase))/2, label='alpha')
        plt.plot((1+np.cos(beta_phase))/2, label='beta')
        plt.plot((1+np.cos(delta_phase))/2, label='delta')
        plt.legend()
        plt.subplot(235)
        plt.title('amplitude')
        plt.plot(alpha_amp, label='alpha')
        plt.plot(beta_amp, label='beta')
        plt.plot(delta_amp, label='delta')
        plt.legend()
        plt.subplot(236)
        plt.title('hormone')
        plt.plot(alpha_hormone, label='alpha')
        plt.plot(beta_hormone, label='beta')
        plt.plot(delta_hormone, label='delta')
        plt.legend()
        plt.savefig(f"image/episode_{episode_idx}.png")
        plt.show()