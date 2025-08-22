import torch
import numpy as np
from utils import to_tensor
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.obs2 = np.zeros((size, obs_dim), np.float32)
        self.acts = np.zeros((size, act_dim), np.float32)
        self.rews = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.ptr = 0; self.size = 0; self.max_size = size; self.device = device
    def store(self, o, a, r, o2, d):
        i = self.ptr
        self.obs[i], self.acts[i], self.rews[i], self.obs2[i], self.done[i] = o, a, r, o2, d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=to_tensor(self.obs[idx], self.device),
            obs2=to_tensor(self.obs2[idx], self.device),
            acts=to_tensor(self.acts[idx], self.device),
            rews=to_tensor(self.rews[idx], self.device).unsqueeze(-1),
            done=to_tensor(self.done[idx], self.device).unsqueeze(-1),
        )