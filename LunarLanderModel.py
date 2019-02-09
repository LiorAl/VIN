import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import gym

def get_VIN_kwargs(gym_env):
    env = gym.make(gym_env)


    VIN_kwargs = {'K'                     : 30,  # Number of Value Iterations
                  'Input_Channels'        : 3,   # Number of channels in input layer -rgb
                  'First_Hidden_Channels' : (32, 64, 64),  # Number of channels in first hidden layer
                  'Q_Channels'            : env.action_space.n,  # Number of channels in q layer (~actions) in VI-module
                  'attention'             : 450,
                  'num_actions'           : env.action_space.n,
                  'critic_features'       : 150}
    env.close()
    return VIN_kwargs


class VIN(nn.Module):
    def __init__(self, num_actions, Input_Channels, First_Hidden_Channels, Q_Channels, K, attention, critic_features):
        super(VIN, self).__init__()
        self.l_i = Input_Channels
        self.l_h = First_Hidden_Channels
        self.l_q = Q_Channels
        self.K   = K
        self.num_actions = num_actions
        self.attention = attention
        self.critic_features = critic_features
        self._recurrent = False

        # Input CNN filters #####
        self.h1 = nn.Conv2d(self.l_i, self.l_h[0], kernel_size=3, stride=1, padding=1,
                            bias=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(self.l_h[0])

        self.h2 = nn.Conv2d(self.l_h[0], self.l_h[1], kernel_size=3, stride=1, padding=1,
                            bias=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(self.l_h[1])

        self.h3 = nn.Conv2d(self.l_h[1], self.l_h[2], kernel_size=3, stride=1, padding=1,
                            bias=True)
        self.maxpool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(self.l_h[2])

        # VI Module #####
        self.r = nn.Conv2d(
            in_channels=self.l_h[2],
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        self.q = nn.Conv2d(
            in_channels=1,
            out_channels=self.l_q,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
            bias=False)

        self.fc = nn.Linear(in_features=self.attention, out_features=self.num_actions, bias=False)
        self.w = Parameter(
            torch.zeros(self.l_q, 1, 5, 5), requires_grad=True)
        self.sm = nn.Softmax(dim=1)
        self.critic_value = nn.Linear(in_features=self.critic_features, out_features=1)

        # # Use GPU if available
        # if device == 'cuda':
        #     self.cuda()


    @property
    def output_size(self):
        return self.attention

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        if self._recurrent:
            return self.l_q
        return 1

    @property
    def is_recurrent(self):
        return self._recurrent

    def forward(self, X, obs):
        h = F.relu(self.bn1(self.maxpool1(self.h1(X))))
        h = F.relu(self.bn2(self.maxpool2(self.h2(h))))
        h = F.relu(self.bn3(self.maxpool3(self.h3(h))))

        r = self.r(h)
        q = self.q(r)
        v, _ = torch.max(q, dim=1, keepdim=True)
        for i in range(0, self.K - 1):
            q = F.conv2d(
                torch.cat([r, v], 1),
                torch.cat([self.q.weight, self.w], 1),
                stride=1,
                padding=2)
            v, _ = torch.max(q, dim=1, keepdim=True)

        q = F.conv2d(
            torch.cat([r, v], 1),
            torch.cat([self.q.weight, self.w], 1),
            stride=1,
            padding=2)

        # slice_s1 = S1.long().expand(config.imsize, 1, config.l_q, q.size(0))
        # slice_s1 = slice_s1.permute(3, 2, 1, 0)
        # q_out = q.gather(2, slice_s1).squeeze(2)
        #
        # slice_s2 = S2.long().expand(1, config.l_q, q.size(0))
        # slice_s2 = slice_s2.permute(2, 1, 0)
        # q_out = q_out.gather(2, slice_s2).squeeze(2)

        # logits = self.fc(q.view(1, -1))
        v, _ = torch.max(q, dim=1, keepdim=True)
        critic = self.critic_value(v.view(-1, self.critic_features))

        return critic, q.view(-1, self.attention)
