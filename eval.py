import numpy as np
def eval_policy(env_fn, agent, episodes=5, seed=None):
    env = env_fn()
    if seed is not None: env.reset(seed=seed)
    rets = []
    for _ in range(episodes):
        o, _ = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            o, r, term, trunc, _ = env.step(agent.select(o, deterministic=True))
            ep_ret += float(r); done = term or trunc
        rets.append(ep_ret)
    env.close()
    return float(np.mean(rets)), float(np.std(rets))
