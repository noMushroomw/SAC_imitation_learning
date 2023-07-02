import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from model import *
from replay_buffer import ReplayBuffer
from SAC_class_setting import SAC
from util import gen_noise
import pickle

class DARC(SAC):
    '''def __init__(self, policy_config, value_config, sa_config, sas_config, source_env, target_env, device, savefolder,running_mean,
                 log_dir="latest_runs", memory_size=1e5, warmup_games=20, batch_size=64, lr=0.0001, gamma=0.99,
                 tau=0.003, alpha=0.2, ent_adj=False, delta_r_scale=1.0, s_t_ratio=10, noise_scale=1.0,
                 target_update_interval=1, n_games_til_train=1, n_updates_per_train=1,decay_rate = 0.99,max_steps = 200):
        super(DARC, self).__init__(source_env, device, log_dir, None, memory_size, None, batch_size, lr, gamma, tau,
                                   alpha, ent_adj, target_update_interval, None, n_updates_per_train)'''
    def __init__(self, running_mean, replay_buffer, buffer_size, source_env, target_env, hidden_dim, hidden_laryer_num, actor_lr, critic_lr, alpha_lr, classifier_lr, tau, gamma, decay_rate, device):
        super(DARC, self).__init__(self, replay_buffer, buffer_size, source_env, hidden_dim, hidden_laryer_num, actor_lr, critic_lr, alpha_lr, tau, gamma, device)
        self.delta_r_scale = delta_r_scale
        self.s_t_ratio = s_t_ratio
        self.noise_scale = noise_scale

        self.source_env = source_env
        self.target_env = target_env
        self.state_dim = source_env.observation_space.shape[0]
        self.action_dim = source_env.action_space.shape[0]

        self.warmup_games = warmup_games
        self.n_games_til_train = n_games_til_train

        self.sa_classifier = SA_Classifier(self.state_dim, self.action_dim, hidden_dim, hidden_layer_num=0).to(self.device)
        self.sa_classifier_opt = Adam(self.sa_classifier.parameters(), lr=classifier_lr)
        self.sas_adv_classifier = SAS_Classifier(self.state_dim, self.action_dim, hidden_dim, hidden_layer_num=0).to(self.device)
        self.sas_adv_classifier_opt = Adam(self.sas_adv_classifier.parameters(), lr=classifier_lr)
        
        self.running_mean = running_mean
        self.max_steps = max_steps
        self.savefolder = savefolder

        self.source_step = 0
        self.target_step = 0
        self.source_memory = self.memory
        self.target_memory = ReplayBuffer(self.memory_size, self.batch_size)
        
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.policy_opt,step_size=1, gamma=decay_rate)
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.twin_q_opt,step_size=1, gamma=decay_rate)
        self.scheduler_sa_classifier_opt = torch.optim.lr_scheduler.StepLR(self.sa_classifier_opt,step_size=1, gamma=decay_rate)
        self.scheduler_sas_adv_classifier_opt = torch.optim.lr_scheduler.StepLR(self.sas_adv_classifier_opt,step_size=1, gamma=decay_rate)

    def step_optim(self):
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        self.scheduler_sa_classifier_opt.step()
        self.scheduler_sas_adv_classifier_opt.step()

    def train_step(self, s_states, s_actions, s_rewards, s_next_states, s_done_masks, *args):
        t_states, t_actions, _, t_next_states, _, game_count = args
        if not torch.is_tensor(s_states):
            s_states = torch.as_tensor(s_states, dtype=torch.float32).to(self.device)
            s_actions = torch.as_tensor(s_actions, dtype=torch.float32).to(self.device)
            s_rewards = torch.as_tensor(s_rewards[:, np.newaxis], dtype=torch.float32).to(self.device)
            s_next_states = torch.as_tensor(s_next_states, dtype=torch.float32).to(self.device)
            s_done_masks = torch.as_tensor(s_done_masks[:, np.newaxis], dtype=torch.float32).to(self.device)

            t_states = torch.as_tensor(t_states, dtype=torch.float32).to(self.device)
            t_actions = torch.as_tensor(t_actions, dtype=torch.float32).to(self.device)
            t_next_states = torch.as_tensor(t_next_states, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            sa_inputs = torch.cat([s_states, s_actions], 1)
            sas_inputs = torch.cat([s_states, s_actions, s_next_states], 1)
            sa_logits = self.sa_classifier(sa_inputs + gen_noise(self.noise_scale, sa_inputs, self.device))
            sas_logits = self.sas_adv_classifier(sas_inputs + gen_noise(self.noise_scale, sas_inputs, self.device))
            sa_log_probs = torch.log(torch.softmax(sa_logits, dim=1) + 1e-12)
            sas_log_probs = torch.log(torch.softmax(sas_logits + sa_logits, dim=1) + 1e-12)

            delta_r = sas_log_probs[:, 1] - sas_log_probs[:, 0] - sa_log_probs[:, 1] + sa_log_probs[:, 0]
            if game_count >= 20 * self.warmup_games:
                s_rewards = s_rewards + self.delta_r_scale * delta_r.unsqueeze(1)

        train_info = super(DARC, self).train_step(s_states, s_actions, s_rewards, s_next_states, s_done_masks)

        s_sa_inputs = torch.cat([s_states, s_actions], 1)
        s_sas_inputs = torch.cat([s_states, s_actions, s_next_states], 1)
        t_sa_inputs = torch.cat([t_states, t_actions], 1)
        t_sas_inputs = torch.cat([t_states, t_actions, t_next_states], 1)
        
        s_sa_logits = self.sa_classifier(s_sa_inputs + gen_noise(self.noise_scale, s_sa_inputs, self.device))
        s_sas_logits = self.sas_adv_classifier(s_sas_inputs + gen_noise(self.noise_scale, s_sas_inputs, self.device))
        t_sa_logits = self.sa_classifier(t_sa_inputs + gen_noise(self.noise_scale, t_sa_inputs, self.device))
        t_sas_logits = self.sas_adv_classifier(t_sas_inputs + gen_noise(self.noise_scale, t_sas_inputs, self.device))

        loss_function = nn.CrossEntropyLoss()
        label_zero = torch.zeros((s_sa_logits.shape[0],), dtype=torch.int64).to(self.device)
        label_one = torch.ones((t_sa_logits.shape[0],), dtype=torch.int64).to(self.device)
        classify_loss = loss_function(s_sa_logits, label_zero)
        classify_loss += loss_function(t_sa_logits, label_one)
        classify_loss += loss_function(s_sas_logits, label_zero)
        classify_loss += loss_function(t_sas_logits, label_one)

        self.sa_classifier_opt.zero_grad()
        self.sas_adv_classifier_opt.zero_grad()
        classify_loss.backward()
        self.sa_classifier_opt.step()
        self.sas_adv_classifier_opt.step()

        s_sa_acc = 1 - torch.argmax(s_sa_logits, dim=1).double().mean()
        s_sas_acc = 1 - torch.argmax(s_sas_logits, dim=1).double().mean()
        t_sa_acc = torch.argmax(t_sa_logits, dim=1).double().mean()
        t_sas_acc = torch.argmax(t_sas_logits, dim=1).double().mean()

        train_info['Loss/Classify Loss'] = classify_loss
        train_info['Stats/Avg Delta Reward'] = delta_r.mean()
        train_info['Stats/Avg Source SA Acc'] = s_sa_acc
        train_info['Stats/Avg Source SAS Acc'] = s_sas_acc
        train_info['Stats/Avg Target SA Acc'] = t_sa_acc
        train_info['Stats/Avg Target SAS Acc'] = t_sas_acc
        return train_info

    def train(self, num_games, deterministic=False):
        self.policy.train()
        self.twin_q.train()
        self.sa_classifier.train()
        self.sas_adv_classifier.train()
        for i in range(num_games):
            source_reward, source_step = self.simulate_env(i, "source", deterministic)

            if i < self.warmup_games or i % self.s_t_ratio == 0:
                target_reward, target_step = self.simulate_env(i, "target", deterministic)
                self.writer.add_scalar('Target Env/Rewards', target_reward, i)
                self.writer.add_scalar('Target Env/N_Steps', target_step, i)
                print("TARGET: index: {}, steps: {}, total_rewards: {}".format(i, target_step, target_reward))

            if i >= self.warmup_games:
                self.writer.add_scalar('Source Env/Rewards', source_reward, i)
                self.writer.add_scalar('Source Env/N_Steps', source_step, i)
                if i % self.n_games_til_train == 0:
                    for _ in range(source_step * self.n_updates_per_train):
                        self.total_train_steps += 1
                        s_s, s_a, s_r, s_s_, s_d = self.source_memory.sample()
                        t_s, t_a, t_r, t_s_, t_d = self.target_memory.sample()
                        train_info = self.train_step(s_s, s_a, s_r, s_s_, s_d, t_s, t_a, t_r, t_s_, t_d, i)
                        self.writer.add_train_step_info(train_info, i)
                    self.writer.write_train_step()
                if i %100 == 0:
                    print('src',self.eval_src(10))
                    print('tgt',self.eval_tgt(10))
                    self.save_model(str(i))




            print("SOURCE: index: {}, steps: {}, total_rewards: {}".format(i, source_step, source_reward))

    def simulate_env(self, game_count, env_name, deterministic):
        if env_name == "source":
            env = self.source_env
            memory = self.source_memory
        elif env_name == "target":
            env = self.target_env
            memory = self.target_memory
        else:
            raise Exception("Env name not recognized")

        total_rewards = 0
        n_steps = 0
        done = False
        state = env.reset()
        state = self.running_mean(state)
        while not done:
            if game_count <= self.warmup_games:
                action = env.action_space.sample()
            else:
                action = self.get_action(state, deterministic)
            next_state, reward, done, _ = env.step(action)
            next_state = self.running_mean(next_state)
            done_mask = 1.0 if n_steps == env._max_episode_steps - 1 else float(not done)
            if n_steps == self.max_steps:
                done = True

            memory.add(state, action, reward, next_state, done_mask)

            if env_name == "source":
                self.source_step += 1
            elif env_name == "target":
                self.target_step += 1
            n_steps += 1
            total_rewards += reward
            state = next_state
        return total_rewards, n_steps

    def save_model(self, folder_name):
        import os
        # super(DARC, self).save_model(folder_name)
        path = os.path.join('saved_weights/'+self.savefolder, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.policy.state_dict(), path + '/policy')
        torch.save(self.twin_q.state_dict(), path + '/twin_q_net')

        torch.save(self.sa_classifier.state_dict(), path + '/sa_classifier')
        torch.save(self.sas_adv_classifier.state_dict(), path + '/sas_adv_classifier')
        # torch.save(self.running_mean.state_dict(), path + '/running_mean')
        pickle.dump(self.running_mean,
                    open(path + '/running_mean', 'wb'))

    # Load model parameters
    def load_model(self, folder_name, device):
        super(DARC, self).load_model(folder_name, device)
        path = 'saved_weights/' + folder_name
        self.sa_classifier.load_state_dict(torch.load(path + '/sa_classifier', map_location=torch.device(device)))
        self.sas_adv_classifier.load_state_dict(
            torch.load(path + '/sas_adv_classifier', map_location=torch.device(device)))
        self.running_mean = pickle.load(open(path + '/running_mean', "rb"))

    def eval_src(self, num_games, render=False):
        self.policy.eval()
        self.twin_q.eval()
        reward_all = 0
        
        for i in range(num_games):
            state = self.source_env.reset()
            state = self.running_mean(state)
            done = False
            total_reward = 0
            step = 0
            while not done:
                if render:
                    self.env.render()
                action = self.get_action(state, deterministic=False)
                next_state, reward, done, _ = self.source_env.step(action)
                next_state = self.running_mean(next_state)
                total_reward += reward
                state = next_state
                if step == self.max_steps:
                    done = True
                step += 1
                
            reward_all += total_reward
        return reward_all/num_games
    
    def eval_tgt(self, num_games, render=False):
        self.policy.eval()
        self.twin_q.eval()
        reward_all = 0
        for i in range(num_games):
            step = 0
            state = self.target_env.reset()
            state = self.running_mean(state)
            done = False
            total_reward = 0
            while not done:
                if render:
                    self.env.render()
                action = self.get_action(state, deterministic=False)
                next_state, reward, done, _ = self.target_env.step(action)
                next_state = self.running_mean(next_state)
                total_reward += reward
                state = next_state
                if step == self.max_steps:
                    done = True
                step += 1
            reward_all += total_reward
        return reward_all/num_games
    
