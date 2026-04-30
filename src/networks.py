import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


class individual_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 256):
        super(individual_Critic, self).__init__()
        self.l1 = nn.Linear(in_features = state_dim + action_dim, out_features = hidden_dim)
        self.l2 = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.l3 = nn.Linear(in_features = hidden_dim, out_features = 1)

    def forward(self, sa):
        # sa --> concatenated state & action
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x




class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim = 256, num_critics = 2):
        super(Critic, self).__init__()

        #######
        # SAC #
        #######
        '''
        # Q1 architecture
        self.l1 = nn.Linear(in_features = state_dim + action_dim, out_features = hidden_dim)
        self.l2 = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.l3 = nn.Linear(in_features = hidden_dim, out_features = 1)

        # Q2 architecture
        self.l4 = nn.Linear(in_features = state_dim + action_dim, out_features = hidden_dim)
        self.l5 = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.l6 = nn.Linear(in_features = hidden_dim, out_features = 1)
        '''


        ########
        # REDQ #
        ########
        self.q_critics = nn.ModuleList(
            [individual_Critic(state_dim, action_dim, hidden_dim) for _ in range(num_critics)]
        )




    def forward(self, state, action):
        sa = torch.cat([state, action], dim = 1)
        #######
        # SAC #
        #######
        '''
        # Q1 forward
        x1 = self.l1(sa)
        x1 = nn.ReLU()(x1)
        x1 = self.l2(x1)
        x1 = nn.ReLU()(x1)
        q1 = self.l3(x1)

        #Q2 forward
        x2 = self.l4(sa)
        x2 = nn.ReLU()(x2)
        x2 = self.l5(x2)
        x2 = nn.ReLU()(x2)
        q2 = self.l6(x2)

        return q1, q2
        '''

        ########
        # REDQ #
        ########
        # Pass the state-action pair thorugh every critic in the ensemble and get Q-values
        all_q_values = []
        for q_net in self.q_critics:
            q_value = q_net(sa)
            all_q_values.append(q_value)
        
        # Stack all the tensor into a single tensor. Shape --> [num_critics, batch_size, 1]
        all_q_values = torch.stack(all_q_values, dim=0)
        return all_q_values



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim = 256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(in_features = state_dim, out_features = hidden_dim)
        self.l2 = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        # Layers for mean and log_std of the action distribution
        self.mean_layer = nn.Linear(in_features = hidden_dim, out_features = action_dim)
        self.log_std_layer = nn.Linear(in_features = hidden_dim, out_features = action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.l1(state)
        x = nn.ReLU()(x)

        x = self.l2(x)
        x = nn.ReLU()(x)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        # Clamp log_std to prevent numerical issues
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std
    

    def sample(self, state):
        '''
        Sample an action from the policy given the STATE, and return the ACTION and its LOG_PROBABILITY.
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Create a normal distribution and sample an action
        normal = Normal(mean, std)
        x_t = normal.rsample() # reparameterized sample <--> Reparameterization Trick. 
        y_t = torch.tanh(x_t) # Bound action to [-1, 1]
        action = y_t * self.max_action # Weight the action by the max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6) # Adjustment for tanh transformation
        log_prob = log_prob.sum(dim=1, keepdim=True) # Sum over action

        return action, log_prob
