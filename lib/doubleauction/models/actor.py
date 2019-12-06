import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=64, hidden2=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        
        self.fc3 = nn.Linear(hidden2, nb_actions)
        
        self.relu = nn.ReLU()   
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
#         out = self.softplus(out)
        
        return out
