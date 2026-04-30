import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from src.networks import Actor, Critic
from src.buffer import ReplayBuffer



import torch.nn.functional as F # compute directly without the need to create an object


class REDQ:
    def __init__(self, state_dim, action_dim, max_action, device, num_critics=10, subset_size=2):  

        self.device = device
        self.subset_size = subset_size 
        # Actor and Critic networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim, max_action).to(device)
        # Target Critic network
        self.critic_target = Critic(state_dim, action_dim, max_action).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict()) # Copy weights from critic to critic_target
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = 3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = 3e-4)
        # The target entropy is heuristically set to -dim(A)
        self.target_entropy = -float(action_dim)
        # We optimize log_alpha instead of alpha to ensure alpha is always positive
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        # We still keep a self.alpha property for convenience in equations
        self.alpha = self.log_alpha.exp().item()

    
    def select_action(self, state):
        '''
        Given a state, select an action according to the current policy.
        '''
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Add batch dimension
            action, _ = self.actor.sample(state_tensor)
            return action.cpu().numpy()[0] # Return as numpy array without batch dimension


    def train(self, replay_buffer, batch_size, utd_ratio = 20, discount_factor = 0.99, tau = 0.005):
        '''
        Train the Actor and Critic networks using a batch of data from the replay buffer.
        '''
        for _ in range(utd_ratio): # Perform multiple updates per environnment step
            # Get data from the replay buffer
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            # CRITIC UPDATE
            with torch.no_grad():
                # Sample next actions
                next_actions, next_logs_probs = self.actor.sample(next_states)
                # Get target Q-values from the target critic
                all_targets = self.critic_target(next_states, next_actions)
                random_indices = np.random.choice(all_targets.shape[0], size=self.subset_size, replace=False)                
                target_Q1, target_Q2 = all_targets[random_indices[0]], all_targets[random_indices[1]]
                min_target_Q = torch.min(target_Q1, target_Q2)
                # Target value wth ENTROPY REGULARIZATION (Bellman Equation witj Entropy term)
                target_Q = rewards + discount_factor * (1 - dones) * (min_target_Q - self.alpha * next_logs_probs)
            
            # Get Q-values from the MAIN CRITIC
            all_current = self.critic(states, actions)
            # Critic Loss: MSE between current Q and target Q, FOR ALL THE CRITICS IN THE ENSEMBLE
            critic_loss = 0
            for ind in range(all_targets.shape[0]):
                critic_loss += F.mse_loss(all_current[ind], target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            

            # ACTOR UPDATE
            new_actions, log_probs = self.actor.sample(states)

            # OPTIMIZATION: Freeze Critic --> we do not need to compute the gradient for the critic when we update the actor.
            for param in self.critic.parameters():
                param.requires_grad = False

            current_Qs  = self.critic(states, new_actions)
            mean_current_Q = torch.mean(current_Qs, dim = 0) # get the mean of all the critics in the ensemble
            # Actor Loss: Maximize Q-value while maximizing entropy
            actor_loss = (self.alpha * log_probs - mean_current_Q).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ALPHA UPDATE
            # We use the log_probs from the actor update
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Update the alpha value for the next iteration's Bellman equations
            self.alpha = self.log_alpha.exp().item()

            # OPTIMIZATION --> Unfreeze Critic for the next loop
            for param in self.critic.parameters():
                param.requires_grad = True

            # Target Update (Polyak Averaging)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)