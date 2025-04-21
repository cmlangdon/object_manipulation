import myosuite
import gym
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
N_EPISODES = 1000
from mujoco import viewer


from stable_baselines3 import DDPG
env = gym.make('myoHandReachFixed-v0')

import torch.nn as nn


class WorldNet(nn.Module):
    def __init__(self, state_dim, action_dim, target_dim):
        super(WorldNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.target_dim = target_dim
        # Take state and action and ouput new state
        self.f = torch.nn.Sequential(nn.Linear(self.state_dim + self.action_dim, 16, bias=True),
                                     nn.ReLU6(),
                                     nn.Linear(16, 32, bias=True),
                                     nn.ReLU6(),
                                     nn.Linear(32, 64, bias=True),
                                     nn.ReLU6(),
                                     nn.Linear(64, 128, bias=True),
                                     nn.ReLU6(),
                                     nn.Linear(128, 64, bias=True),
                                     nn.ReLU6(),
                                     nn.Linear(64, 32, bias=True),
                                     nn.ReLU6(),
                                     nn.Linear(32, 16, bias=True),
                                     nn.ReLU6(),
                                     nn.Linear(16, 8, bias=True),
                                     nn.ReLU6(),
                                     nn.Linear(8, self.state_dim, bias=True),
                                     )
        self.mlp_reward = nn.Linear(self.state_dim, self.target_dim, bias=True)

    def forward(self, u):
        # Initial state (position and velocity)
        x = torch.zeros(u.shape[0], u.shape[1] + 1, 115)
        x[:, 0, :] = u[:, 0, :115]
        for t in range(1, u.shape[1] + 1):
            x[:, t, :] = x[:, t - 1, :] + self.f(torch.hstack([x[:, t - 1, :], u[:, t - 1, -39:]]))

        # state_prediction = self.mlp_state(x)
        reward_prediction = -torch.sqrt(torch.sum(self.mlp_reward(x[:, 1:, :]) ** 2, dim=2, keepdim=True))

        return torch.concatenate([x[:, :-1, :], reward_prediction], dim=2)


# GENERATE TRAINING DATASET FOR WORLD MODEL
NUM_EPISODES = 1000
MAX_STEPS = 100
ACTION_DIM = 39
STATE_DIM = 115
TARGET_DIM = 15
# Loop over episodes
SHIFT = 0
inputs = torch.zeros(NUM_EPISODES,MAX_STEPS-SHIFT,STATE_DIM + ACTION_DIM)
labels = torch.zeros(NUM_EPISODES,MAX_STEPS-SHIFT,STATE_DIM + 1)

for episode in range(NUM_EPISODES):
    # Reset environment
    state = env.reset()
    action = np.array(torch.normal(mean=.2, std=0.1,size=[ACTION_DIM]))

    for step in range(MAX_STEPS):
        # Take step in environment to get new state and reward
        new_state, reward, _, _ = env.step(action)
        # Add data
        if step>=SHIFT:
            inputs[episode,step-SHIFT ,:] = torch.tensor(np.hstack([state,action])).float()
            labels[episode,step-SHIFT,:] = torch.tensor(np.hstack([state,reward])).float()
        # Set current state to new state
        state = new_state
# z-score


world_network = WorldNet(state_dim=STATE_DIM,action_dim=ACTION_DIM,target_dim=TARGET_DIM)
#world_network  = torch.load('world_network.pth')

#world_network.train()
# Initialize optimizer and wrap training data as PyTorch dataset
optimizer = torch.optim.Adam(world_network.parameters(), lr=.001,weight_decay = 0)
my_dataset = TensorDataset(inputs, labels)  #
my_dataloader = DataLoader(my_dataset, batch_size=256,shuffle=True)

NUM_EPOCHS = 10000
# Training loop
loss_history = []

for i in range(NUM_EPOCHS):
    epoch_loss = 0
    for batch_idx, (u_batch, z_batch) in enumerate(my_dataloader):
        optimizer.zero_grad()
        x_batch = world_network.forward(u_batch)
        z_batch_centered  = (z_batch - torch.mean(z_batch,dim=[0,1],keepdim=True))/torch.std(z_batch,dim=[0,1],keepdim=True)
        x_batch_centered = (x_batch - torch.mean(x_batch,dim=[0,1],keepdim=True))/torch.std(x_batch,dim=[0,1],keepdim=True)
        loss = torch.nn.MSELoss()(x_batch_centered, z_batch_centered)
        epoch_loss += loss.item() / NUM_EPOCHS
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(world_network.parameters(), max_norm=.1)

        optimizer.step()

    if i % 50 == 0:
        outputs = world_network.forward(inputs)
        print('Epoch: {}/{}.............'.format(i, NUM_EPOCHS), end=' ')
        labels_centered  = labels - torch.mean(labels,dim=1,keepdim=True)
        print("mse_z: {:.8f}".format(torch.nn.MSELoss()(outputs,labels).item()/ torch.nn.MSELoss()(torch.zeros_like(labels_centered), labels_centered).item()))
        torch.save(world_network, '../Notebooks/world_network.pth')
        loss_history.append(epoch_loss)
