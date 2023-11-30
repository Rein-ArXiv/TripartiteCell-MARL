import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim=4, capacity=2e5):
        capacity = int(capacity)
        state_dim = int(state_dim)
        
        self._state_buffer = np.zeros([capacity, state_dim], dtype=np.float32)
        self._action_buffer = np.zeros([capacity], dtype=np.float32)
        self._next_state_buffer = np.zeros([capacity, state_dim], dtype=np.float32)
        self._reward_buffer = np.zeros([capacity], dtype=np.float32)
        self._terminated_truncated_buffer = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._position = 0
        self._size = 0
        
    def push(self, state, action, reward, next_state, terminated_truncated):
        self._state_buffer[self._position] = state
        self._action_buffer[self._position] = action
        self._next_state_buffer[self._position] = next_state
        self._reward_buffer[self._position] = reward

        self._terminated_truncated_buffer[self._position] = terminated_truncated
        self._position = (self._position + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
        
    def sample(self, batch_size):
        idxs = np.random.choice(self._size, size=batch_size, replace=False)
        sampling = dict()
        sampling['state'] = self._state_buffer[idxs]
        sampling['action'] = self._action_buffer[idxs]
        sampling['next_state'] = self._next_state_buffer[idxs]
        sampling['reward'] = self._reward_buffer[idxs]
        sampling['terminated_truncated'] = self._terminated_truncated_buffer[idxs]
        return sampling
    
    def __len__(self):
        return self._size
    
