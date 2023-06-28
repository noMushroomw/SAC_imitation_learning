import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

min_std = -20
max_std = 2
    

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

# policy & value net structure for continuous SAC

# policy net -- Gaussian distribution
class continuousPolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, hidden_layer_num=2) -> None:
        super(continuousPolicyNet, self).__init__()
        self.hidden_layer_num = hidden_layer_num
        self.state_layer = nn.Linear(state_dim, hidden_dim)
        for i in range(hidden_layer_num):
            setattr(self, 'hidden_layer{}'.format(i+1), nn.Linear(hidden_dim, hidden_dim))
        self.layer_mu = nn.Linear(hidden_dim, action_dim)
        self.layer_std = nn.Linear(hidden_dim, action_dim)
        self.apply(weight_init)

    def forward(self, x):
        x = F.relu(self.state_layer(x))
        for i in range(self.hidden_layer_num):
            x = F.relu(getattr(self, 'hidden_layer{}'.format(i+1))(x))
        mu = self.layer_mu(x)
        #std = F.softplus(self.layer_std(x))
        log_std = self.layer_std(x).clamp(min=min_std, max=max_std)
        std = log_std.exp()
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # compute log_prob according to the action
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mu
    
# value net -- twin Q network
class continuousTwinValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, hidden_layer_num=2) -> None:
        super(continuousTwinValueNet, self).__init__()
        self.hidden_layer_num = hidden_layer_num
        
        # Q1 architecture
        self.state_layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        for i in range(hidden_layer_num):
            setattr(self, 'hidden_layer_1{}'.format(i+1), nn.Linear(hidden_dim, hidden_dim))
        self.action_layer_1 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.state_layer_2 = nn.Linear(state_dim + action_dim, hidden_dim)
        for i in range(hidden_layer_num):
            setattr(self, 'hidden_layer_2{}'.format(i+1), nn.Linear(hidden_dim, hidden_dim))
        self.action_layer_2 = nn.Linear(hidden_dim, 1)
        
        self.apply(weight_init)
        
    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        
        x1 = F.relu(self.state_layer_1(x))
        for i in range(self.hidden_layer_num):
            x1 = F.relu(getattr(self, 'hidden_layer_1{}'.format(i+1))(x1))
        x1 = self.action_layer_1(x1)
        
        x2 = F.relu(self.state_layer_2(x))
        for i in range(self.hidden_layer_num):
            x2 = F.relu(getattr(self, 'hidden_layer_2{}'.format(i+1))(x2))
        x2 = self.action_layer_2(x2)
        return x1, x2
    
    
# imitation learning discriminator
class Discriminator:
    def __init__(self, state_dim, hidden_dim, action_dim, hidden_layer_num=2) -> None:
        super(continuousPolicyNet, self).__init__()
        self.hidden_layer_num = hidden_layer_num
        self.state_layer = nn.Linear(state_dim, hidden_dim)
        for i in range(hidden_layer_num):
            setattr(self, 'hidden_layer{}'.format(i+1), nn.Linear(hidden_dim, hidden_dim))
        self.action_layer = nn.Linear(hidden_dim, 1)
        self.apply(weight_init)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        
        x = F.tanh(self.state_layer_1(x))
        for i in range(self.hidden_layer_num):
            x = F.tanh(getattr(self, 'hidden_layer_1{}'.format(i+1))(x))
        x = self.action_layer(x)
        
        return F.sigmoid(x)

# DARC classifier
## SA classifier: linear[s+a, 64], relu, linear[64, 2]
## SAS classifier: linear[s+s+a', 64], relu, linear[64, 2]
class Classifier(nn.Module):
    def __init__(self, state_dim, hidden_dim, hidden_layer_num=2) -> None:
        super(Classifier, self).__init__()
        self.hidden_layer_num = hidden_layer_num
        self.state_layer = nn.Linear(state_dim, hidden_dim)
        for i in range(hidden_layer_num):
            setattr(self, 'hidden_layer{}'.format(i+1), nn.Linear(hidden_dim, hidden_dim))
        self.action_layer = nn.Linear(hidden_dim, 2)
        self.apply(weight_init)

    def forward(self, s1, a, s2=None):
        if s2 is None:
            x = torch.cat([s1, a], dim=1)
        else:
            x = torch.cat([s1, s2, a], dim=1)
        
        x = F.relu(self.state_layer_1(x))
        for i in range(self.hidden_layer_num):
            x = F.relu(getattr(self, 'hidden_layer_1{}'.format(i+1))(x))
        x = self.action_layer(x)
        
        return x