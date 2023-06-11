from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# policy & value net structure for continuous SAC
class continuousPolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, hidden_layer_num=2) -> None:
        super(continuousPolicyNet, self).__init__()
        self.hidden_layer_num = hidden_layer_num
        self.state_layer = nn.Linear(state_dim, hidden_dim)
        for i in range(hidden_layer_num):
            setattr(self, 'hidden_layer{}'.format(i+1), nn.Linear(hidden_dim, hidden_dim))
        self.layer_mu = nn.Linear(hidden_dim, action_dim)
        self.layer_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.state_layer(x))
        for i in range(self.hidden_layer_num):
            x = F.relu(getattr(self, 'hidden_layer{}'.format(i+1))(x))
        mu = torch.tanh(self.layer_mu(x))
        std = F.softplus(self.layer_std(x))
        return mu, std
    
class continuousValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, hidden_layer_num=2) -> None:
        super(continuousValueNet, self).__init__()
        self.hidden_layer_num = hidden_layer_num
        self.state_layer = nn.Linear(state_dim + action_dim, hidden_dim)
        for i in range(hidden_layer_num):
            setattr(self, 'hidden_layer{}'.format(i+1), nn.Linear(hidden_dim, hidden_dim))
        self.action_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.state_layer(x))
        for i in range(self.hidden_layer_num):
            x = F.relu(getattr(self, 'hidden_layer{}'.format(i+1))(x))
        return self.action_layer(x)
    

class PPOContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, hidden_laryer_num, actor_lr, critic_lr,
                 gamma, lmbda , epochs, eps, device) -> None:
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor = continuousPolicyNet(state_dim, hidden_dim, action_dim, hidden_laryer_num).to(device)
        self.critic = continuousValueNet(state_dim, hidden_dim, action_dim, hidden_laryer_num).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return action.flatten().tolist()   
    
    # compute advantage
    def compute_advantage(gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta[0]
            advantage_list.append([advantage])
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    
    def update(self, transitions):                
        states, actions, rewards, next_states, dones, truncateds = transitions
        
        states = states.clone().detach().to(self.device)
        actions = actions.clone().detach().to(self.device)
        rewards = rewards.clone().detach().to(self.device)
        next_states = next_states.clone().detach().to(self.device)
        dones = dones.clone().detach().to(self.device)
        truncateds = truncateds.clone().detach().to(self.device)
        
        # modify the shape of reward
        rewards = (rewards + 8.0) / 8.0
        
        td_target = rewards + self.gamma * (1 - dones) * (1 - truncateds) *self.critic(next_states, self.actor(next_states))
        td_delta = td_target - self.critic(states, actions)
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu().to(self.device))
        
        mu, std = self.actor(states)
        action_dist = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_prob = action_dist.log_prob(actions)
        
        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dist = torch.distributions.Normal(mu, std)
            log_probs = action_dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.smooth_l1_loss(self.critic(states, actions), td_target.detach())
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()