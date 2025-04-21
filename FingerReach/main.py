import myosuite
import gym
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
import seaborn as sns
N_EPISODES = 1000
from mujoco import viewer


from stable_baselines3 import DDPG
env = gym.make('myoHandReachFixed-v0',render_mode  = 'human')

for episode in range(50):
    env.reset()
    for step in range(100):

            env.mj_render()
            o = env.get_obs()
            a =np.array(torch.normal(mean=1, std=0.1,size=[39]))
            next_o, r, done, ifo = env.step(a)
env.close()