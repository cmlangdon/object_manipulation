from torch.utils.data import TensorDataset, DataLoader
import myosuite
import gym
import numpy as np
import torch.nn as nn
import torch
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
import seaborn as sns
plt.rcParams["axes.grid"] = False
matplotlib.rcParams['axes.linewidth'] = .75
from Models.actor import *
from Models.world import *
from stable_baselines3 import PPO


device = 'cpu'

# LOAD MODELS
world_network  = torch.load('/Users/cl1704/PycharmProjects/object_manipulation/FingerReach/Data/world_network.pth',map_location='cpu',weights_only=False)
actor_network  = torch.load('/Users/cl1704/PycharmProjects/object_manipulation/FingerReach/Data/actor_network.pth',map_location='cpu',weights_only=False)


env = gym.make('myoHandReachFixed-v0',render_mode  = 'human')

for episode in range(50):
    env.reset()
    for step in range(1000):

            env.mj_render()
            o = env.get_obs()
            a =actor_network(torch.tensor(o).float())

            next_o, r, done, ifo = env.step(a.detach().cpu().numpy())
env.close()