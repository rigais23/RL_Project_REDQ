import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.max_size = max_size
        self.pointer = 0 # Pointer to the index where the next transition will be stored
        self.size = 0 # Buffer capacity

        # Pre-allocate memory for speed
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        # Determine device for PyTorch tensors
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
    
    def add(self, state, action, reward, next_state, done):
        '''
        Add a NEW transtion to the buffer.
        If the buffer is full, overwrite the oldest.
        '''
        # Add the data into the arrays
        self.state[self.pointer] = state
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.next_state[self.pointer] = next_state
        self.done[self.pointer] = done

        # Update the pointer and size
        self.pointer = (self.pointer +1) % self.max_size
        self.size = min(self.size +1, self.max_size)


    def sample(self, batch_size):
        '''
        Randomly sample a batch of data from the buffer.
        Returns the tensors on the correct device.
        '''
        # Randomly sample indices
        indices = np.random.randint(0, self.size, size = batch_size)

        # Convert to PyTorch tensors and move to the corresponding device
        states = torch.FloatTensor(self.state[indices]).to(self.device)
        actions = torch.FloatTensor(self.action[indices]).to(self.device)
        rewards = torch.FloatTensor(self.reward[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_state[indices]).to(self.device)
        dones = torch.FloatTensor(self.done[indices]).to(self.device)

        return states, actions, rewards, next_states, dones