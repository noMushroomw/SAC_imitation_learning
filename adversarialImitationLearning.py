import os
import pickle

import numpy as np
import torch
from torch.nn import functional
from torch.optim import Adam
import math

from architectures.gaussian_policy import ContGaussianPolicy
from architectures.utils import gen_noise
from architectures.value_networks import ContTwinQNet
from architectures.utils import polyak_update
from replay_buffer import ReplayBuffer
from tensor_writer import TensorWriter
from models.mlp_discriminator import Discriminator

dtype = torch.float32


# TODO Discrete
class GailContSAC_SRC:
    def __init__(self, policy_config, value_config, env,target_env, device, expert,expert_model, log_dir="latest_runs",running_mean = None,
                 memory_size=1e5, warmup_games=10, batch_size=64, lr=0.0001, gamma=0.99, tau=0.003, alpha=0.2,
                 ent_adj=False, target_update_interval=1, n_games_til_train=1, n_updates_per_train=1,max_steps = 200,eval_step = 0):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size

        path = 'runs/' + log_dir
        if not os.path.exists(path):
            os.makedirs(path)
        self.writer = TensorWriter(path)

        self.memory_size = memory_size
        self.warmup_games = warmup_games
        self.memory = ReplayBuffer(self.memory_size, self.batch_size)
        self.ilmemory = ReplayBuffer(1000000, 4096)

        self.env = env
        self.target_env = target_env
        self.action_range = (env.action_space.low, env.action_space.high)
        self.policy = ContGaussianPolicy(policy_config, self.action_range).to(self.device)
        self.policy_opt = Adam(self.policy.parameters(), lr=lr)
        self.running_mean = running_mean

        self.expert = expert
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]       
        self.discrim_net = Discriminator(state_dim + state_dim).to(self.device)
        self.discrim_criterion = torch.nn.BCELoss()
        self.optimizer_discrim = torch.optim.Adam(self.discrim_net.parameters(), lr=0.0003)

        self.twin_q = ContTwinQNet(value_config).to(self.device)
        self.twin_q_opt = Adam(self.twin_q.parameters(), lr=lr)
        self.target_twin_q = ContTwinQNet(value_config).to(self.device)
        polyak_update(self.twin_q, self.target_twin_q, 1)

        self.tau = tau
        self.gamma = gamma
        self.n_until_target_update = target_update_interval
        self.n_games_til_train = n_games_til_train
        self.n_updates_per_train = n_updates_per_train
        self.max_steps = max_steps

        self.eval_step = eval_step
        self.expert_model = expert_model
        self.target_env = target_env


        self.alpha = alpha
        self.ent_adj = ent_adj
        if ent_adj:
            self.target_entropy = -len(self.action_range)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = Adam([self.log_alpha], lr=lr)

        self.total_train_steps = 0
    
    def cal_wt(self,s_states, s_actions, s_next_states):
        with torch.no_grad():
            # s_states = torch.as_tensor(s, dtype=torch.float32).to(self.device)
            # s_actions = torch.as_tensor(a, dtype=torch.float32).to(self.device)

            # s_next_states = torch.as_tensor(s_, dtype=torch.float32).to(self.device)
            
            sa_inputs = torch.cat([s_states, s_actions], 1)

            sas_inputs = torch.cat([s_states, s_actions, s_next_states], 1)

            sa_logits = self.expert_model.sa_classifier(sa_inputs.to(self.device)+ gen_noise(1, sa_inputs, self.device))
            sas_logits = self.expert_model.sas_adv_classifier(sas_inputs.to(self.device)+ gen_noise(1, sas_inputs, self.device))

            sa_log_probs = torch.log(torch.softmax(sa_logits, dim=1) + 1e-12)
            sas_log_probs = torch.log(torch.softmax(sas_logits + sa_logits, dim=1) + 1e-12)

            delta_r = sas_log_probs[:, 1] - sas_log_probs[:, 0] - sa_log_probs[:, 1] + sa_log_probs[:, 0]
            w_t = torch.exp(delta_r)

        return w_t

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.as_tensor(state[np.newaxis, :].copy(), dtype=torch.float32).to(self.device)
            if deterministic:
                _, _, action = self.policy.sample(state)
            else:
                action, _, _ = self.policy.sample(state)
            return action.detach().cpu().numpy()[0]

    def train_step(self, states, actions, rewards, next_states, done_masks):
        if not torch.is_tensor(states):
            states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
            rewards = torch.as_tensor(rewards[:, np.newaxis], dtype=torch.float32).to(self.device)
            next_states = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
            done_masks = torch.as_tensor(done_masks[:, np.newaxis], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_states)
            next_q = self.target_twin_q(next_states, next_action)[0]
            v = next_q - self.alpha * next_log_prob
 
            expected_q = rewards + done_masks * self.gamma * v

        # Q backprop
        q_val, pred_q1, pred_q2 = self.twin_q(states, actions)
        q_loss = functional.mse_loss(pred_q1, expected_q) + functional.mse_loss(pred_q2, expected_q)

        self.twin_q_opt.zero_grad()
        q_loss.backward()
        self.twin_q_opt.step()

        # Policy backprop
        s_action, s_log_prob, _ = self.policy.sample(states)
        policy_loss = self.alpha * s_log_prob - self.twin_q(states, s_action)[0]
        policy_loss = policy_loss.mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        if self.ent_adj:
            alpha_loss = -(self.log_alpha * (s_log_prob + self.target_entropy).detach()).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp()

        if self.total_train_steps % self.n_until_target_update == 0:
            polyak_update(self.twin_q, self.target_twin_q, self.tau)

        return {'Loss/Policy Loss': policy_loss,
                'Loss/Q Loss': q_loss,
                'Stats/Avg Q Val': q_val.mean(),
                'Stats/Avg Q Next Val': next_q.mean(),
                'Stats/Avg Alpha': self.alpha.item() if self.ent_adj else self.alpha}

    def train(self, num_games, deterministic=False):
        self.policy.train()
        self.twin_q.train()
        acc_gen, acc_expert = 0,0
        for i in range(num_games):
            total_reward = 0
            total_reward_from_dis = 0
            dis_step = 0
            n_steps = 0
            done = False
            state = self.env.reset()
            state = self.running_mean(state)
            while not done:
                if self.total_train_steps <= self.warmup_games:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_action(state, deterministic)

                next_state, reward, done, _ = self.env.step(action)
                next_state = self.running_mean(next_state)
                done_mask = 1.0 if n_steps == self.env._max_episode_steps - 1 else float(not done)
                total_reward += reward
                reward = self.expert_reward(state,action,next_state)
                total_reward_from_dis += reward
                self.memory.add(state, action, reward, next_state, done_mask)
                self.ilmemory.add(state, action, reward, next_state, done_mask)
                n_steps += 1
                
                
                state = next_state
                if n_steps>self.max_steps:
                    break

            if i >= self.warmup_games:
                self.writer.add_scalar('Env/Rewards', total_reward, i)
                self.writer.add_scalar('Env/N_Steps', n_steps, i)
                if i % self.n_games_til_train == 0:
                    
                        
                    for _ in range(n_steps * self.n_updates_per_train):
                        self.total_train_steps += 1
                        s, a, r, s_, d = self.memory.sample()
                        train_info = self.train_step(s, a, r, s_, d)
                        dis_step +=1
                        self.writer.add_train_step_info(train_info, i)
                        if _ % 10 == 0:
                            acc_gen, acc_expert = self.update_discriminator()

                        
                    self.writer.write_train_step()
            if not self.eval_step:
                print("index: {}, steps: {}, total_rewards: {}, total_reward_adv: {}, acc gen: {}, acc expert: {}".format(\
                    i, n_steps, total_reward,round(total_reward_from_dis,4),round(acc_gen,4),round(acc_expert,4)))
            else:
                e_r = self.eval(1)
                print("index: {}, steps: {}, total_rewards: {}, total_reward_adv: {}, acc gen: {}, acc expert: {},eval reward: {}".format(\
                    i, n_steps, total_reward,round(total_reward_from_dis,4),round(acc_gen,4),round(acc_expert,4),round(e_r,4)))
            if i%100==0:
                print('eval reward:',self.eval(10))

    def eval(self, num_games, render=False):
        self.policy.eval()
        self.twin_q.eval()
        total_reward = 0
        
        for i in range(num_games):
            state = self.target_env.reset()
            t = 0
            state = self.running_mean(state)
            done = False
            while not done:
                t += 1
                action = self.get_action(state, deterministic=True)
                next_state, reward, done, _ = self.target_env.step(action)
                next_state = self.running_mean(next_state)
                total_reward += reward
                state = next_state
                if t>self.max_steps:
                    break

        return total_reward/num_games
    def update_discriminator(self):
        for _ in range(1):
            state, action, r, next_state, d = self.ilmemory.sample()
            index = np.random.randint(0, high=len(self.expert)-1, size=10000)
            self.optimizer_discrim.zero_grad()
            expert_state_actions = torch.from_numpy(self.expert[index]).to(dtype).to(self.device)

            g_o_input = torch.cat([torch.tensor(state,dtype=dtype), torch.tensor(next_state,dtype=dtype)], 1).to(self.device)
            
            g_o = self.discrim_net(g_o_input +  gen_noise(1, g_o_input, self.device))
            e_o = self.discrim_net(expert_state_actions+  gen_noise(1, expert_state_actions, self.device))
            
            wt = self.cal_wt(torch.tensor(state,dtype=dtype),torch.tensor(action,dtype=dtype),torch.tensor(next_state,dtype=dtype))
          
            discrim_loss =  -torch.mean(wt.unsqueeze(1) * torch.log(g_o))+ \
                self.discrim_criterion(e_o, torch.zeros((index.shape[0], 1), device=self.device))
            discrim_loss.backward()
            self.optimizer_discrim.step()
            acc_gen = sum(g_o >0.5)/len(g_o)
            acc_expert = sum(e_o <0.5)/len(e_o)
        return acc_gen.item(), acc_expert.item()


        
    def expert_reward(self,state,action, next_state):
        
        state_action = torch.tensor(np.hstack([state, next_state]), dtype=dtype).to(self.device)
        with torch.no_grad():
            state = torch.tensor(state,dtype= dtype).to(self.device).unsqueeze(0)
            next_state = torch.tensor(next_state,dtype= dtype).to(self.device).unsqueeze(0)
            action = torch.tensor(action,dtype= dtype).to(self.device).unsqueeze(0)
   
            wt = self.cal_wt(state,action,next_state).cpu()
            return -math.log(1e-8 + self.discrim_net(state_action + + gen_noise(1, state_action, self.device))[0].item()) * wt.item()


    def save_model(self, folder_name):
        path = 'saved_weights/' + folder_name
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.policy.state_dict(), path + '/policy')
        torch.save(self.twin_q.state_dict(), path + '/twin_q_net')
        pickle.dump(self.running_mean,
            open(path + '/running_mean', 'wb'))

    # Load model parameters
    def load_model(self, folder_name, device):
        path = 'saved_weights/' + folder_name
        self.policy.load_state_dict(torch.load(path + '/policy', map_location=torch.device(device)))
        self.twin_q.load_state_dict(torch.load(path + '/twin_q_net', map_location=torch.device(device)))

        polyak_update(self.twin_q, self.target_twin_q, 1)
        polyak_update(self.twin_q, self.target_twin_q, 1)
        self.running_mean = pickle.load(open(path + '/running_mean', "rb"))

