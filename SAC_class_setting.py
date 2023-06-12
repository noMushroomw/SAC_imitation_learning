from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
import replay_buffer as rb

min_std = -20
max_std = 2

def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# policy & value net structure for continuous SAC
class continuousPolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, hidden_layer_num=2) -> None:
        super(continuousPolicyNet, self).__init__()
        self.hidden_layer_num = hidden_layer_num
        self.state_layer = nn.Linear(state_dim, hidden_dim)
        for i in range(hidden_layer_num):
            setattr(self, 'hidden_layer{}'.format(i+1), nn.Linear(hidden_dim, hidden_dim))
        self.layer_mu = nn.Linear(hidden_dim, action_dim)
        self.layer_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound
        self.apply(weight_init)

    def forward(self, x):
        x = F.relu(self.state_layer(x))
        for i in range(self.hidden_layer_num):
            x = F.relu(getattr(self, 'hidden_layer{}'.format(i+1))(x))
        mu = self.layer_mu(x)
        #std = F.softplus(self.layer_std(x))
        std = self.layer_std(x)
        std = torch.clamp(std, min=min_std, max=max_std)
        std = std.exp()
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # compute log_prob according to the action
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        action = action * self.action_bound
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob
    
class continuousValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, hidden_layer_num=2) -> None:
        super(continuousValueNet, self).__init__()
        self.hidden_layer_num = hidden_layer_num
        self.state_layer = nn.Linear(state_dim + action_dim, hidden_dim)
        for i in range(hidden_layer_num):
            setattr(self, 'hidden_layer{}'.format(i+1), nn.Linear(hidden_dim, hidden_dim))
        self.action_layer = nn.Linear(hidden_dim, 1)
        self.apply(weight_init)
        
    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.state_layer(x))
        for i in range(self.hidden_layer_num):
            x = F.relu(getattr(self, 'hidden_layer{}'.format(i+1))(x))
        return self.action_layer(x)
    

class SACContinuous:
    def __init__(self, replay_buffer, state_dim, hidden_dim, action_dim, hidden_laryer_num, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device) -> None:
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.replay_buffer = replay_buffer
        
        self.actor = continuousPolicyNet(state_dim, hidden_dim, action_dim, action_bound, hidden_laryer_num).to(device)
        self.critic1 = continuousValueNet(state_dim, hidden_dim, action_dim, hidden_laryer_num).to(device)
        self.critic2 = continuousValueNet(state_dim, hidden_dim, action_dim, hidden_laryer_num).to(device)
        self.target_critic1 = continuousValueNet(state_dim, hidden_dim, action_dim, hidden_laryer_num).to(device)
        self.target_critic2 = continuousValueNet(state_dim, hidden_dim, action_dim, hidden_laryer_num).to(device)
        
        # initialize the target net as the same as the original value net
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # using log alpha to stablize the training process
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.target_entropy = target_entropy
        self.tau = tau
        self.gamma = gamma
        self.device = device
        
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0].cpu().detach().numpy()
        return action.flatten().tolist()
    
    def cal_target(self, rewards, next_states, dones, truncateds):
        next_actions, log_prob = self.actor(next_states)
        entropy = -torch.prod(log_prob, dim=1, keepdim=True)
        q1 = self.target_critic1(next_states, next_actions)
        q2 = self.target_critic2(next_states, next_actions)
        #next_q = torch.min(q1, q2) + self.log_alpha.exp() * entropy
        next_q = torch.min(q1, q2) - self.log_alpha.exp() * log_prob
        td_target = rewards + self.gamma * (1 - dones) * (1 - truncateds) * next_q
        return td_target        
            
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1.0 - self.tau))
            
    def weighted_mse_loss(self, input, target, weight):
        weight = weight.to(self.device)
        return torch.mean(weight * (input - target) ** 2)
            
    def update(self, batch_size):
        if isinstance(self.replay_buffer, rb.ReplayBuffer):
            batch = self.replay_buffer.sample(batch_size)
            weight = torch.ones([batch_size, 1])
            idxs = None
        else:
            batch, weight, idxs = self.replay_buffer.sample(batch_size)
        
        states, actions, rewards, next_states, dones, truncateds = batch
        
        rewards = rewards.clone().detach().reshape(batch_size, 1)
        dones = dones.clone().detach().reshape(batch_size, 1)
        truncateds = truncateds.clone().detach().reshape(batch_size, 1)
        
        # modify the shape of reward
        #rewards = (rewards + 8.0) / 8.0
        
        td_target = self.cal_target(rewards, next_states, dones, truncateds)
        critic1_loss = self.weighted_mse_loss(self.critic1(states, actions), td_target.detach(), weight)
        critic2_loss = self.weighted_mse_loss(self.critic2(states, actions), td_target.detach(), weight)

        if isinstance(self.replay_buffer, rb.PrioritizedReplayBuffer):
            self.replay_buffer.update_priorities(idxs, td_target.detach().cpu().numpy())
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # update the actor
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1, q2))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # update the alpha
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
        