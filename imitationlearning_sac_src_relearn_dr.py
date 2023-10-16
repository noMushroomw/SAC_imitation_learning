import gym
import argparse

from models.darc import DARC
from environments.broken_joint import BrokenJointEnv
import os
from zfilter import *
import torch
import gym
# import gymnasium as gym

from models.darc import DARC
from models.sac import ContSAC
from environments.broken_joint import BrokenJointEnv,BrokenJointEnv2
from utils import *
from envs import *
from datetime import datetime
from zfilter import *
import argparse
import gym
import os
import sys
import pickle
import time
from models.gailsac import GailContSAC
from models.src_gailsac import GailContSAC_SRC
from models.src_gailsac_classifier import GailContSAC_SRC_Classifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils_gail import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from torch import nn
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent,eval_model






parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.01, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=8000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')


parser.add_argument('--max-expert-state-num', type=int, default=50000, metavar='N',
                    help='maximal number of main iterations (default: 50000)')




parser.add_argument('--save-model', type=str, default="",
                    help='name of Mujoco environement')
parser.add_argument('--train-steps', type=int, default=2000,
                    help='name of Mujoco environement')
parser.add_argument('--episode-length', type=int, default=1000,
                    help='name of Mujoco environement')
parser.add_argument('--save-file-name', type=str, default='',
                    help='name of Mujoco environement')
parser.add_argument('--optim-epochs', type=int, default=3,
                    help='name of Mujoco environement')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='name of Mujoco environement')
parser.add_argument('--update-freq', type=int, default=50,
                    help='name of Mujoco environement')
parser.add_argument('--noise-scale', type=float, default=1e-4,
                    help='name of Mujoco environement')

parser.add_argument('--env-name', type=str, default="Reacher-v2",
                    help='name of Mujoco environement')

# broken
parser.add_argument('--broken', type=int, default=1,
                    help='whether broken env')
parser.add_argument('--break_src', type=int, default=0,
                    help='whether break src env or target env')
parser.add_argument('--break_joint', type=int, default=0,
                    help='break which joint')
parser.add_argument('--normalize', type=int, default=1,
                    help='break which joint')

# variety
parser.add_argument('--variety-name', type=str, default="d",
                    help='name of Mujoco environement')
parser.add_argument('--variety-degree', type=float, default=0.5,
                    help='name of Mujoco environement')

parser.add_argument('--lower-bound', type=float, default=0.1,
                    help='name of Mujoco environement')
parser.add_argument('--upper-bound', type=float, default=10,
                    help='name of Mujoco environement')
parser.add_argument('--reward-type', type=int, default=2,
                    help='name of Mujoco environement')

args = parser.parse_args()
dtype = torch.float32
torch.set_default_dtype(dtype)

env_name = args.env_name
variety_name = args.variety_name
degree = args.variety_degree
train_steps = args.train_steps


currentDateAndTime = datetime.now()
date = currentDateAndTime.strftime("%Y:%M:%D").split(':')[-1].replace('/','-')
save_model_path = args.save_file_name
save_model_path += date
save_model_path += '_'
save_model_path += str(args.episode_length)
save_model_path += '_'





if args.broken == 0:
    save_model_path += variety_name
    save_model_path += '_'
    save_model_path += str(degree)
    save_model_path += '_'
    save_model_path += str(env_name)
    source_env = get_source_env(env_name)
    if variety_name == 'g':
        target_env = get_new_gravity_env(degree,env_name)
    elif variety_name == 'd':
        target_env = get_new_density_env(degree, env_name)
    elif variety_name == 'f':
        target_env = get_new_friction_env(degree, env_name)
else:
    save_model_path += str(args.break_src) 
    save_model_path += '_' 
    save_model_path += str(args.break_joint)
    save_model_path += '_'
    save_model_path += str(env_name)
    if args.break_src == 1:
        source_env = BrokenJointEnv(gym.make(env_name), [args.break_joint])
        target_env = BrokenJointEnv(gym.make(env_name), [])
    else:
        source_env = BrokenJointEnv(gym.make(env_name), [])
        target_env = BrokenJointEnv(gym.make(env_name), [args.break_joint])   

state_dim = source_env.observation_space.shape[0]
action_dim = source_env.action_space.shape[0]

policy_config = {
    "input_dim": [state_dim],
    "architecture": [{"name": "linear1", "size": 256},
                     {"name": "linear2", "size": 256},
                     {"name": "split1", "sizes": [action_dim, action_dim]}],
    "hidden_activation": "relu",
    "output_activation": "none"
}
value_config = {
    "input_dim": [state_dim + action_dim],
    "architecture": [{"name": "linear1", "size": 256},
                     {"name": "linear2", "size": 256},
                     {"name": "linear2", "size": 1}],
    "hidden_activation": "relu",
    "output_activation": "none"
}
sa_config = {
    "input_dim": [state_dim + action_dim],
    "architecture": [{"name": "linear1", "size": 64},
                     {"name": "linear2", "size": 2}],
    "hidden_activation": "relu",
    "output_activation": "none"
}

sas_config = {
    "input_dim": [state_dim * 2 + action_dim],
    "architecture": [{"name": "linear1", "size": 64},
                     {"name": "linear2", "size": 2}],
    "hidden_activation": "relu",
 
    "output_activation": "none"
}

def lr_decay(actor_optim,cirtic_optim, total_steps,max_steps):
    if total_steps > 7000:
        lr_a_now = args.learning_rate * (1 - (total_steps-7000) /(max_steps))
        for p in actor_optim.param_groups:
            p['lr'] = lr_a_now
        for p in cirtic_optim.param_groups:
            p['lr'] = lr_a_now
    return actor_optim,cirtic_optim




running_state = ZFilter((state_dim,), clip=5)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.set_device(args.gpu_index)
model = DARC(policy_config, value_config, sa_config, sas_config, source_env, target_env, device, ent_adj=True,
             n_updates_per_train=1,max_steps = args.episode_length,batch_size=256,savefolder = None,running_mean=running_state)
savepath = args.save_model
model.load_model(savepath,device)
model.running_mean.fix = True
num_steps = 0
expert_traj = []
total_reward = 0
from tqdm import tqdm
for i_episode in tqdm(range(100000)):

    state = source_env.reset()
    if args.normalize == 1:
        state = model.running_mean(state)
    reward_episode = 0

    for t in range(args.episode_length):
        action = model.get_action(state, deterministic=True)
        next_state, reward, done, _ = source_env.step(action)
        if args.normalize == 1:
            next_state = model.running_mean(next_state)
        reward_episode += reward
        num_steps += 1

        expert_traj.append(np.hstack([state, next_state]))
        

        if done or num_steps >= args.max_expert_state_num:
            break


        state = next_state
    if num_steps >= args.max_expert_state_num:
        break
    total_reward += reward_episode

    # print('Episode {}\t reward: {:.2f} \t step : {}'.format(i_episode, reward_episode,t))
print('eval on src',total_reward/(i_episode))
expert_traj = np.stack(expert_traj)

total_reward = 0
num_steps = 0
for i_episode in tqdm(range(50)):
    state = target_env.reset()
    if args.normalize == 1:
        state = model.running_mean(state)
    reward_episode = 0
    for t in range(args.episode_length):
        action = model.get_action(state, deterministic=True)
        next_state, reward, done, _ = target_env.step(action)
        if args.normalize == 1:
            next_state = model.running_mean(next_state)
        reward_episode += reward
        num_steps += 1
        if done or num_steps >= args.max_expert_state_num:
            break
        state = next_state
    if num_steps >= args.max_expert_state_num:
        break
    total_reward += reward_episode
print('eval on tgt',total_reward/(i_episode))




dtype = torch.float32
torch.set_default_dtype(dtype)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.set_device(args.gpu_index)

"""environment"""
is_disc_action = False
# running_state = ZFilter((state_dim,), clip=5)

# """seeding"""
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
model.running_mean.fix = True
IL = GailContSAC_SRC_Classifier(policy_config, value_config,  source_env,target_env, device,expert_traj,model, ent_adj=True, n_updates_per_train=1,gamma = args.gamma,
                 batch_size=256,lr = args.lr,\
                    max_steps = args.episode_length,\
                        running_mean=model.running_mean,\
                            warmup_games=50, \
                                eval_step = False,noise_scale = args.noise_scale,\
                                    update_freq = args.update_freq,
                                    if_normalize=args.normalize,\
                                clamp_reward_lower_bound = args.lower_bound, clamp_reward_upper_bound = args.upper_bound,\
                               reward_type = args.reward_type)

IL.train(6000, deterministic=False)
