import torch
from networks import SquashedGaussianActor, QCritic
import numpy as np
from typing import Tuple
from utils import to_tensor
import torch.nn.functional as F


class SACConfig:
    env_id: str = "LunarLanderContinuous-v3"
    seed: int = 0
    steps: int = 200_000
    start_steps: int = 5_000
    update_after: int = 5_000
    update_every: int = 1
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden: Tuple[int,int] = (256,256)
    target_entropy_scale: float = 1.0
    eval_episodes: int = 5
    video_folder: str = "videos"
    video_name: str = "sac_lander_demo"
class SACAgent:
    def __init__(self, obs_space, act_space, cfg: SACConfig, device):
        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(act_space.shape))
        self.cfg, self.device = cfg, device
        a_low, a_high = np.asarray(act_space.low, np.float32), np.asarray(act_space.high, np.float32)
        self.actor = SquashedGaussianActor(obs_dim, act_dim, cfg.hidden, action_low=a_low, action_high=a_high).to(device)
        self.q1 = QCritic(obs_dim, act_dim, cfg.hidden).to(device)
        self.q2 = QCritic(obs_dim, act_dim, cfg.hidden).to(device)
        self.q1_t = QCritic(obs_dim, act_dim, cfg.hidden).to(device)
        self.q2_t = QCritic(obs_dim, act_dim, cfg.hidden).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.pi_opt  = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt  = torch.optim.Adam(self.q1.parameters(),   lr=cfg.critic_lr)
        self.q2_opt  = torch.optim.Adam(self.q2.parameters(),   lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.a_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = - cfg.target_entropy_scale * act_dim
    @property
    def alpha(self): return self.log_alpha.exp()
    def select(self, o, deterministic=False):
        o = to_tensor(o, self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic: a = self.actor.act_mean(o)
            else: a, _ = self.actor.sample(o)
        return a.cpu().numpy()[0]
    def update(self, batch):
        o, o2, a, r, d = batch['obs'], batch['obs2'], batch['acts'], batch['rews'], batch['done']
        with torch.no_grad():
            a2, logp2 = self.actor.sample(o2)
            q1_t = self.q1_t(o2, a2); q2_t = self.q2_t(o2, a2)
            q_targ = torch.min(q1_t, q2_t) - self.alpha * logp2
            backup = r + self.cfg.gamma * (1 - d) * q_targ
        q1 = self.q1(o, a); q2 = self.q2(o, a)
        lq1 = F.mse_loss(q1, backup); lq2 = F.mse_loss(q2, backup)
        self.q1_opt.zero_grad(set_to_none=True); lq1.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(set_to_none=True); lq2.backward(); self.q2_opt.step()
        api, logpi = self.actor.sample(o)
        qpi = torch.min(self.q1(o, api), self.q2(o, api))
        lpi = (self.alpha * logpi - qpi).mean()
        self.pi_opt.zero_grad(set_to_none=True); lpi.backward(); self.pi_opt.step()
        lalpha = -(self.log_alpha * (logpi.detach() + self.target_entropy)).mean()
        self.a_opt.zero_grad(set_to_none=True); lalpha.backward(); self.a_opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.data.mul_(1 - self.cfg.tau); pt.data.add_(self.cfg.tau * p.data)
            for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.data.mul_(1 - self.cfg.tau); pt.data.add_(self.cfg.tau * p.data)
        return dict(lq1=lq1.item(), lq2=lq2.item(), lpi=lpi.item(), alpha=self.alpha.item())