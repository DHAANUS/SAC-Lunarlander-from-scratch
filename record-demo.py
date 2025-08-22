import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from SAC-skeleton import SACConfig
def record_one_episode(agent, cfg: SACConfig, out_folder="videos", name_prefix="sac_lander_demo"):
    os.makedirs(out_folder, exist_ok=True)
    env = gym.make(cfg.env_id, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=out_folder, name_prefix=name_prefix, episode_trigger=lambda ep: True)
    o, _ = env.reset()
    done = False
    while not done:
        a = agent.select(o, deterministic=True)
        o, r, term, trunc, _ = env.step(a)
        done = term or trunc
    env.close()
    return out_folder