
import torch.nn as nn
import torch



class WorldNet(nn.Module):
    def __init__(self, actor_network,device):
        super(WorldNet, self).__init__()
        self.STATE_DIM = 115
        self.ACTION_DIM = 39
        self.TARGET_DIM = 15
        self.N_STEPS = 100
        self.device = device
        self.actor_network = actor_network

        # Take state and action and ouput new state
        self.f = torch.nn.Sequential(nn.Linear(self.STATE_DIM+self.ACTION_DIM, 256, bias=True),
                                     nn.Tanh(),
                                     nn.Linear(256, 128, bias=True),
                                     nn.Tanh(),
                                     nn.Linear(128, 128, bias=True),
                                     nn.Tanh(),
                                     nn.Linear(128, 128, bias=True),
                                     nn.Tanh(),
                                     nn.Linear(128, self.STATE_DIM, bias=True),
                                     )
        self.mlp_reward = nn.Linear(self.STATE_DIM, self.TARGET_DIM, bias=True)

    def forward(self, u):
        '''
        :param u: batch_size  x 6+2 (batch of initial_states)
        :return: Sequence of states and rewards
        '''
        # Initial state (position, velocity, goal)
        x = torch.zeros(u.shape[0], 1, self.STATE_DIM).to(self.device)
        x[:, 0, :] = u

        for t in range(1, self.N_STEPS + 1):
            x_new = x[:, t - 1, :] + self.f(torch.hstack([x[:, t - 1, :], self.actor_network(x[:, t - 1, :])]))
            x = torch.cat((x, x_new.unsqueeze_(1)), 1)

        # Calculate reward for each episode and time.
        reward_prediction = -torch.sqrt(torch.sum(self.mlp_reward(x[:, 1:, :]) ** 2, dim=2, keepdim=True))
        return torch.concatenate([x[:, :-1, :], reward_prediction], dim=2)


