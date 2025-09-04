import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
def load_pretrained_dqn(path: str, device: str = "cpu") -> nn.Module:
    """Load a pretrained DQN network from the given path."""
    policy_net = DQN(n_observations=4, n_actions=2).to(device)  # must re-create architecture
    policy_net.load_state_dict(torch.load(path, map_location=device))
    policy_net.eval()
    return policy_net