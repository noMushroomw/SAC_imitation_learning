import gym
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os, re, copy, time, random, datetime, argparse
import numpy as np


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


def polyak_update(network, target_network, tau):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(tau * param.data + target_param.data * (1.0 - tau))


def gen_noise(scale, tensor, device):
    return scale * torch.randn(tensor.shape).to(device)

'''
  Generate different type of dynamics mismatch.
  @python version : 3.6.4
'''

'''
  Utils functions and some configs.
  @python version : 3.6.8
'''




nowTime = datetime.datetime.now().strftime('%y-%m-%d%H:%M:%S')
parser = argparse.ArgumentParser(description="Process running arguments")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"




def generate_xml_path():
    import gym, os
    xml_path = os.path.join(gym.__file__[:-11], 'envs/mujoco/assets')

    assert os.path.exists(xml_path)

    return xml_path


gym_xml_path = generate_xml_path()


def record_data(file, content):
    with open(file, 'a+') as f:
        f.write('{}\n'.format(content))


def check_path(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except FileExistsError:
        pass

    return path


def update_xml(index, env_name):
    xml_name = parse_xml_name(env_name)
    os.system('cp xml_path/{0}/{1} {2}/{1}}'.format(index, xml_name, gym_xml_path))

    time.sleep(0.2)


def parse_xml_name(env_name):
    if 'walker' in env_name.lower():
        xml_name = "walker2d.xml"
    elif 'hopper' in env_name.lower():
        xml_name = "hopper.xml"
    elif 'halfcheetah' in env_name.lower():
        xml_name = "half_cheetah.xml"
    elif "ant" in env_name.lower():
        xml_name = "ant.xml"
    elif "reacher" in env_name.lower():
        xml_name = "reacher.xml"
    else:
        raise RuntimeError("No available environment named \'%s\'" % env_name)

    return xml_name


def update_source_env(env_name):
    xml_name = parse_xml_name(env_name)

    os.system(
        'cp xml_path/source_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def update_target_env_gravity(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "gravity" in line:
                pattern = re.compile(r"gravity=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "gravity=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def update_target_env_density(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "density" in line:
                pattern = re.compile(r'(?<=density=")\d+\.?\d*')
                a = pattern.findall(line)
                current_num = float(a[0])
                replace_num = current_num * variety_degree
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def update_target_env_friction(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "friction" in line:
                pattern = re.compile(r"friction=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "friction=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)




def get_new_gravity_env(variety, env_name):
    update_target_env_gravity(variety, env_name)
    env = gym.make(env_name)

    return env


def get_source_env(env_name="Walker2d-v2"):
    update_source_env(env_name)
    env = gym.make(env_name)

    return env


def get_new_density_env(variety, env_name):
    update_target_env_density(variety, env_name)
    env = gym.make(env_name)

    return env


def get_new_friction_env(variety, env_name):
    update_target_env_friction(variety, env_name)
    env = gym.make(env_name)

    return env


import numpy as np

# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/


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

