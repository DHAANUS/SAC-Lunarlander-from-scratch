import torch

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from sac-skeleton import SACConfig, SACAgent
from replaybuffer import ReplayBuffer
from eval import eval_policy
from record-demo import record_one_episode
from utils import set_seed
import os

def train_and_record():
    cfg = SACConfig()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_env(): return gym.make(cfg.env_id)
    env = make_env()
    obs_space, act_space = env.observation_space, env.action_space
    assert len(act_space.shape) == 1, "Continuous Box action space required."

    agent = SACAgent(obs_space, act_space, cfg, device)
    buf = ReplayBuffer(obs_space.shape[0], act_space.shape[0], size=300_000, device=device)

    o, _ = env.reset(seed=cfg.seed)
    ep_ret, ep_len = 0.0, 0
    print(f"Device: {device} | Env: {cfg.env_id} | Obs: {obs_space.shape} | Act: {act_space.shape}")

    for t in range(1, cfg.steps + 1):
        a = act_space.sample() if t < cfg.start_steps else agent.select(o)
        o2, r, term, trunc, _ = env.step(a)
        d = float(term or trunc)
        buf.store(o, a, r, o2, d)
        o = o2; ep_ret += float(r); ep_len += 1
        if d:
            print(f"[Step {t:>7}] EpRet: {ep_ret:7.1f} | EpLen: {ep_len:4d}")
            o, _ = env.reset(); ep_ret, ep_len = 0.0, 0

        if t >= cfg.update_after and t % cfg.update_every == 0:
            info = agent.update(buf.sample(cfg.batch_size))
            if t % 2000 == 0:
                print(f"[Upd {t:>7}] alpha={info['alpha']:.3f} | Lq1={info['lq1']:.3f} | Lq2={info['lq2']:.3f} | Lpi={info['lpi']:.3f}")

        if t % 10_000 == 0:
            mean_ret, std_ret = eval_policy(make_env, agent, episodes=cfg.eval_episodes)
            print(f"== Eval @ {t:>7} steps: mean={mean_ret:.1f} ± {std_ret:.1f}")

    env.close()
    print("Training done. Recording deterministic demo...")
    folder = record_one_episode(agent, cfg, out_folder=cfg.video_folder, name_prefix=cfg.video_name)
    print(f"Saved video(s) to: {os.path.abspath(folder)}")
train_and_record()