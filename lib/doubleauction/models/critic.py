

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=64, hidden2=64):
        super(Critic, self).__init__()
        self.bn1 = nn.BatchNorm1d(nb_states)
        self.fc1 = nn.Linear(nb_states, hidden1)
        
        self.bn2 = nn.BatchNorm1d(hidden1+nb_actions)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        
        self.bn3 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, s, a):
        out = self.fc1(self.bn1(s))
        out = self.relu(out)

        out = self.fc2(self.bn2(torch.cat([out,a],1)))
        out = self.relu(out)
        out = self.fc3(self.bn3(out))
        return out