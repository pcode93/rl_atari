import gym
import random
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple, deque

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

from wrappers import wrap_dqn

class Net(nn.Module):
    def __init__(self, n_actions):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.value = nn.Linear(3136, 512)
        self.value_head = nn.Linear(512, n_actions)
        
        self.advantage = nn.Linear(3136, 512)
        self.advantage_head = nn.Linear(512, n_actions)
        

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        
        value = self.value_head(F.relu(self.value(x)))
        advantage = self.advantage_head(F.relu(self.advantage(x)))
        
        return value + (advantage - advantage.mean())


env = wrap_dqn(gym.make('PongNoFrameskip-v4'))


q_func = Net(6)
q_func.load_state_dict(torch.load(sys.argv[1]))
q_func.cuda()

def var(x):
    x = np.array(x).reshape(1, 4, 84, 84)
    x = torch.from_numpy(x)
    
    return Variable(x).type(torch.FloatTensor).cuda()

def select_action(x):
    if random.random() < 0.02:
        return env.action_space.sample()
    else:
        return np.argmax(q_func(var(x)).data.cpu().numpy(), 1)[0]


import time

observation = env.reset()

while True:
    env.render()

    time.sleep(0.025)

    previous_obs = observation
    action = select_action(observation)

    observation, reward, done, info = env.step(action)  

    if done:
        observation = env.reset()
