

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=64, hidden2=64, hidden3=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 1)

        self.relu = nn.ReLU()
    
    def forward(self, s, a):
        out = self.fc1(s)
        out = self.relu(out)

        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        
        out = self.fc4(out)
    
        return out