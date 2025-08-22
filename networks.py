
import torch
import torch.nn as nn
from utils import mlp
class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256), log_std_bounds=(-20,2), action_low=None, action_high=None):
        super().__init__()
        self.net = mlp([obs_dim, *hidden], nn.ReLU, nn.ReLU)
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
        self.lmin, self.lmax = log_std_bounds
        self.register_buffer('a_low', torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer('a_high', torch.as_tensor(action_high, dtype=torch.float32))
        self.register_buffer('a_scale', (self.a_high - self.a_low)/2.0)
        self.register_buffer('a_mean',  (self.a_high + self.a_low)/2.0)
    def forward(self, o):
        h = self.net(o)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.lmin, self.lmax)
        std = torch.exp(log_std)
        return mu, std
    @torch.no_grad()
    def act_mean(self, o):
        mu, _ = self.forward(o); y = torch.tanh(mu)
        return self.a_mean + self.a_scale * y
    def sample(self, o):
        mu, std = self.forward(o)
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()              
        y = torch.tanh(u)               
        a = self.a_mean + self.a_scale * y
        logp = dist.log_prob(u).sum(-1, keepdim=True)
        logp -= torch.log(1 - y.pow(2) + 1e-6).sum(-1, keepdim=True)
        logp -= torch.log(self.a_scale).sum().view(1,1)
        return a, logp
    
    
class QCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.q = mlp([obs_dim+act_dim, *hidden, 1], nn.ReLU, nn.Identity)
    def forward(self, o, a): return self.q(torch.cat([o,a], dim=-1))