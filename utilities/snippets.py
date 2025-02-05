from env import TSCEnv
from typing import List
import itertools


def run_a_step(env: TSCEnv, n_obs, on_training: bool, store_experience: bool, learn: bool):  # n_obs中一个agent的观察也是list, 里面的元素要么是 N维的array, 要么是 (1, N)的tensor
    n_action = []
    for agent in env.n_agent:
        action = agent.pick_action(n_obs, on_training)
        n_action.append(action)
    n_next_obs, n_rew, n_done, info = env.step(n_action)

    if store_experience:
        for idx in range(env.n):
            env.n_agent[idx].store_experience(n_obs[idx], n_action[idx], n_rew[idx], n_next_obs[idx], n_done[idx])

    if learn:
        for agent in env.n_agent:
            agent.learn()

    return n_next_obs, n_rew, n_done, info


def run_an_episode(env: TSCEnv, config: dict, on_training: bool, store_experience: bool, learn: bool):
    n_obs = env.reset()  # n个agent的观察
    n_done = [False]
    info = {}

    # current_episode_step_idx 从0到20到40到60
    for config['current_episode_step_idx'] in itertools.count(start=0, step=config['action_interval']):  # config['current_episode_step_idx']是这里改变的
        if config['current_episode_step_idx'] >= config['num_step'] or all(n_done):
            break

        n_next_obs, n_rew, n_done, info = run_a_step(env, n_obs, on_training, store_experience, learn)
        n_obs = n_next_obs

    # print('how many step?', config['current_episode_step_idx'])  # 这些step都能跑完的
    return info

