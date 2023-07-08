from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter


# smooth the plot of agent output
def smooth_figure(data, window_size):
    cumulative_sum = np.cumsum(np.insert(data, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(data[:window_size-1])[::2] / r
    end = (np.cumsum(data[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# train on policy agent (ppo e.g.)
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'truncateds': []}
                state, _ = env.reset()
                done, truncated = False, False
                while not (done or truncated):
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    transition_dict['truncateds'].append(truncated)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

# train off policy agent (sac e.g.)
def train_off_policy_agent(env, agent, num_episodes, minimal_size, batch_size):
    return_list = []
    running_mean = ZFilter(env.observation_space.shape[0], clip=5)
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, _ = env.reset()
                state = running_mean(state)
                done = False
                truncated = False
                while not (done or truncated):
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    next_state = running_mean(next_state)
                    agent.replay_buffer.add((state, action, reward, next_state, done, truncated))
                    state = next_state
                    episode_return += reward
                    if agent.replay_buffer.real_size > minimal_size:
                        train_info = agent.update(batch_size)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
        print(train_info)
    return return_list

def train_darc(agent, num_games, deterministic=False):
        agent.actor.train()
        agent.critic.train()
        agent.sa_classifier.train()
        agent.sas_adv_classifier.train()
        for i in range(num_games):
            source_reward, source_step = agent.simulate_env(i, "source", deterministic)

            if i < agent.warmup_games or i % agent.s_t_ratio == 0:
                target_reward, target_step = agent.simulate_env(i, "target", deterministic)
                agent.writer.add_scalar('Target Env/Rewards', target_reward, i)
                agent.writer.add_scalar('Target Env/N_Steps', target_step, i)
                print("TARGET: index: {}, steps: {}, total_rewards: {}".format(i, target_step, target_reward))

            if i >= agent.warmup_games:
                agent.writer.add_scalar('Source Env/Rewards', source_reward, i)
                agent.writer.add_scalar('Source Env/N_Steps', source_step, i)
                if i % agent.n_games_til_train == 0:
                    for _ in range(source_step):
                        #agent.total_train_steps += 1
                        train_info = agent.update(i)
                        agent.writer.add_train_step_info(train_info, i)
                    agent.writer.write_train_step()
                if i %100 == 0:
                    print('src',agent.eval_src(10))
                    print('tgt',agent.eval_tgt(10))
                    agent.save_model(str(i))

            print("SOURCE: index: {}, steps: {}, total_rewards: {}".format(i, source_step, source_reward))


# sample expert data
def sample_expert_data(env, agent, num_episodes, threshold):
    states, actions = [], []
    episode = 0
    while episode < num_episodes:
        epi_reward = 0
        epi_states, epi_actions = [], []
        state, _ = env.reset()
        done, truncated = False, False
        while not (done or truncated):
            action = agent.take_action(state)
            epi_states.append(state)
            epi_actions.append(action)
            next_state, reward, done, truncated, _ = env.step(action)
            epi_reward += reward
            state = next_state
            
        if epi_reward > threshold:
            states.extend(epi_states)
            actions.extend(epi_actions)
            episode += 1
            
    return states, actions

def gen_noise(scale, tensor, device):
    return scale * torch.randn(tensor.shape).to(device)


class TensorWriter(SummaryWriter):
    def __init__(self, path):
        super(TensorWriter, self).__init__(path)
        self.train_info_buffer = []
        self.train_iteration = None

    def add_train_step_info(self, train_info, i):
        self.train_info_buffer.append(train_info)
        self.train_iteration = i

    def write_train_step(self):
        keys = self.train_info_buffer[0].keys()

        for k in keys:
            total = 0
            for i in range(len(self.train_info_buffer)):
                total += self.train_info_buffer[i][k]
            self.add_scalar(k, total / len(self.train_info_buffer), self.train_iteration)
        self.train_info_buffer = []
        self.train_iteration = None


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)
        self.fix = False

    def __call__(self, x, update=True):
        if update and not self.fix:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

