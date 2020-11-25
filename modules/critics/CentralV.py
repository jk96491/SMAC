import torch.nn as nn
import torch.nn.functional as F

class CentralV_Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(CentralV_Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q