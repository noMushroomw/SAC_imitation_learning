import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
import replay_buffer as rb
from model import *
from util import *

class SACContinuous:   
    def __init__(self, replay_buffer, batch_size, buffer_size, env, hidden_dim, hidden_layer_num, actor_lr, critic_lr, alpha_lr, tau, gamma, device,
                 scheduler_open=False, scheduler_size=10000, scheduler_gamma=0.9, log_dir='SAC', min_num=1000, num_episodes=1000):
        
        path = 'logs/' + log_dir
        if not os.path.exists(path):
            os.makedirs(path)
        self.writer = TensorWriter(path)
        
        self.min_num = min_num
        self.num_steps = num_episodes

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_range = [env.action_space.low, env.action_space.high]
        
        if replay_buffer == 'ReplayBuffer':
            self.replay_buffer = rb.ReplayBuffer(self.state_dim, self.action_dim, buffer_size, device)
        elif replay_buffer == 'PrioritizedReplayBuffer':
            self.replay_buffer = rb.PrioritizedReplayBuffer(self.state_dim, self.action_dim, buffer_size, device, eps=1e-7, alpha=0.1, beta=0.1)
        self.batch_size = batch_size
        
        self.actor = continuousPolicyNet(self.state_dim, hidden_dim, self.action_dim, hidden_layer_num).to(device)
        self.critic = continuousTwinValueNet(self.state_dim, hidden_dim, self.action_dim, hidden_layer_num).to(device)
        self.target_critic = continuousTwinValueNet(self.state_dim, hidden_dim, self.action_dim, hidden_layer_num).to(device)
        
        
        # initialize the target net as the same as the original value net
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=scheduler_size, gamma=scheduler_gamma)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=scheduler_size, gamma=scheduler_gamma)

        # using log alpha to stablize the training process
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.log_alpha_scheduler = torch.optim.lr_scheduler.StepLR(self.log_alpha_optimizer, step_size=scheduler_size, gamma=scheduler_gamma)
        self.scheduler_open = scheduler_open
        
        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        self.tau = tau
        self.gamma = gamma
        self.device = device
        
    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 + (self.action_range[1] + self.action_range[0]) / 2.0
        
    def take_action(self, state, deterministic=False):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        if deterministic:
             action = self.actor(state)[2].cpu().detach().numpy()
        else:
            action = self.actor(state)[0].cpu().detach().numpy()
        action = self.rescale_action(action)
        return action.flatten().tolist()
    
    def cal_target(self, rewards, next_states, dones, truncateds):
        next_actions, log_prob, _ = self.actor(next_states)
        entropy = -torch.prod(log_prob, dim=1, keepdim=True)
        q1, q2 = self.target_critic(next_states, next_actions)
        next_q = torch.min(q1, q2) - self.log_alpha.exp() * log_prob
        td_target = rewards + self.gamma * (1 - dones) * (1 - truncateds) * next_q
        return td_target        
            
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1.0 - self.tau))
            
    def weighted_mse_loss(self, input, target, weight):
        weight = weight.to(self.device)
        return torch.mean(weight * (input - target) ** 2)
    
    def scheduler_step(self):
        self.critic_scheduler.step()
        self.actor_scheduler.step()
        self.log_alpha_scheduler.step()
        
            
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
        qf1, qf2 = self.critic(states, actions)
        critic1_loss = self.weighted_mse_loss(qf1, td_target.detach(), weight)
        critic2_loss = self.weighted_mse_loss(qf2, td_target.detach(), weight)

        if isinstance(self.replay_buffer, rb.PrioritizedReplayBuffer):
            self.replay_buffer.update_priorities(idxs, td_target.detach().cpu().numpy())
        
        self.critic_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic_optimizer.step()
        
        # update the actor
        new_actions, log_prob, _ = self.actor(states)
        q1, q2 = self.critic(states, new_actions)
        actor_loss = torch.mean(self.log_alpha.exp() * log_prob - torch.min(q1, q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # update the alpha
        alpha_loss = -torch.mean((log_prob + self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        if self.scheduler_open:
            self.scheduler_step()
        
        self.soft_update(self.critic, self.target_critic)
        
        return {'Actor loss': actor_loss.item(),
                'Critic1 loss': critic1_loss.item(),
            }
        
    '''def train_SAC(self):
        self.actor.train()
        self.critic.train()
        
        for i in range(self.num_steps):
            total_reward = 0
            done, truncated = False, False
            state, _ = self.env.reset()
            while not (done or truncated):
                action = self.take_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.replay_buffer.add((state, action, reward, next_state, done, truncated))
                state = next_state
                total_reward += reward
                
                if self.replay_buffer.real_size > self.min_num:
                    self.update(self.batch_size)'''
                    
                
        
        
    def save_model(self, folder):
        path = 'saved_model/SAC/' + folder
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(self.actor.state_dict(), path + '/actor')
        torch.save(self.critic.state_dict(), path + '/critic')
        
    def load_model(self, folder, device):
        path = 'saved_model/SAC/' + folder
        self.actor.load_state_dict(torch.load(path + '/actor', map_location=device))
        self.critic.load_state_dict(torch.load(path + '/critic', map_location=device))
        self.soft_update(self.critic, self.target_critic)
        