
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=64, hidden2=64, init_w=3e-3):
        super(Actor, self).__init__()
        self.bn1 = nn.BatchNorm1d(nb_states)
        self.fc1 = nn.Linear(nb_states, hidden1)
        
        self.bn2 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        
        self.bn3 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        
        self.relu = nn.ReLU()   
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        out = self.fc1(self.bn1(x))
        out = self.relu(out)
        out = self.fc2(self.bn2(out))
        out = self.relu(out)
        out = self.fc3(self.bn3(out))
        out = self.softplus(out)
        
        return out
    
    