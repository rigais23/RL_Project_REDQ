import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from src.networks import Actor, Critic
from src.buffer import ReplayBuffer



import torch.nn.functional as F # compute directly withot the need to create an object


class SAC:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device
        # Actor and Critic networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim, max_action).to(device)
        # Target Critic network
        self.critic_target = Critic(state_dim, action_dim, max_action).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict()) # Copy weights from critic to critic_target
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = 3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = 3e-4)
        # Temperature (entropy regularization coefficient)
        self.alpha = 0.2

    
    def select_action(self, state):
        '''
        Given a state, select an action according to the current policy.
        '''
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Add batch dimension
            action, _ = self.actor.sample(state_tensor)
            return action.cpu().numpy()[0] # Return as numpy array without batch dimension


    def train(self, replay_buffer, batch_size, discount_factor = 0.99, tau = 0.005):
        '''
        Train the Actor and Critic networks using a batch of data from the replay buffer.
        '''
        # Get data from the replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # CRITIC UPDATE
        with torch.no_grad():
            # Sample next actions
            next_actions, next_logs_probs = self.actor.sample(next_states)
            # Get target Q-values from the target critic
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            min_target_Q = torch.min(target_Q1, target_Q2)
            # Target value wth ENTROPY REGULARIZATION (Bellman Equation witj Entropy term)
            target_Q = rewards + discount_factor * (1 - dones) * (min_target_Q - self.alpha * next_logs_probs)
        # Get Q-values from the MAIN CRITIC
        current_Q1, current_Q2 = self.critic(states, actions)
        # Critic Loss: MSE between current Q and target Q
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        

        # ACTOR UPDATE
        new_actions, log_probs = self.actor.sample(states)
        current_Q1, current_Q2 = self.critic(states, new_actions)
        min_current_Q = torch.min(current_Q1, current_Q2)
        # Actor Loss: Maximize Q-value while maximizing entropy
        actor_loss = (self.alpha * log_probs -min_current_Q).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target Update (Polyak Averaging)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)