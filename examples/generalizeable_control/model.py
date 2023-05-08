import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
sys.path.insert(1, root_dir)
sys.path.insert(1, os.path.join(root_dir, 'externals', 'pytorch_a2c_ppo_acktr_gail'))

from a2c_ppo_acktr.model import NNBase

class CNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # input is 2x6x6
        self.conv1 = nn.Conv2d(input_dim, 25, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(25, 50, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(50, 50, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(50, 50, kernel_size=3, padding=1) 
        self.conv5 = nn.Conv2d(50, 100, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(100, 100, kernel_size=3, padding=1)  
        self.conv8 = nn.Conv2d(100, 50, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(50, 1, kernel_size=3, padding=1) 
        self.batchnorm1 = nn.BatchNorm2d(50)
        self.batchnorm2 = nn.BatchNorm2d(100)
        self.batchnorm3 = nn.BatchNorm2d(1)
        
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) # SMALL
        x = self.batchnorm1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)) # MEDIUM
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.batchnorm2(x)
        x = F.relu(self.conv8(x))
        x = 0.5*torch.tanh(self.batchnorm3(self.conv9(x))) + .1
        x = self.flatten(x)
        return x

class UniversalController(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=36):
        super(UniversalController, self).__init__(recurrent, num_inputs, hidden_size)
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = CNN(num_inputs)

        self.train()

    def forward(self, inputs, rnn_hxs=None, masks=None):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_actor = self.actor(x)

        return hidden_actor, rnn_hxs
