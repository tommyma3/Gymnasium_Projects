import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn((1, state_dim))
    output = net(state)
    print(output)