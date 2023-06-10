from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


# smooth the plot of agent output
def smooth_figure(data, window_size):
    cumulative_sum = np.cumsum(np.insert(data, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(data[:window_size-1])[::2] / r
    end = (np.cumsum(data[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# train on policy agent (ppo e.g.)
def train_on_policy(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                truncated = False
                while not (done or truncated):
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    agent.replay_buffer.add(state, action, reward, next_state, done, truncated)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update()
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode':'%d' % (num_episodes/10 * i +i_episode+1), 'return':'%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

# train off policy agent (dqn e.g.)
## need to be modified transdict and update
def train_off_policy_agent(env, agent, num_episodes, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                truncated = False
                while not (done or truncated):
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    agent.replay_buffer.add((state, action, reward, next_state, done, truncated))
                    state = next_state
                    episode_return += reward
                    if agent.replay_buffer.real_size > minimal_size:
                        agent.update(batch_size)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

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

