from env import TSCEnv
import itertools
import numpy as np
from collections import deque


class RewardNormalizer:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0
        self.mean_diff = 0
        self.var = 0

    def update(self, reward) -> None:
        self.n += 1
        last_mean = self.mean
        self.mean += (reward - last_mean) / self.n
        self.mean_diff += (reward - last_mean) * (reward - self.mean)
        self.var = self.mean_diff / self.n if self.n > 1 else 0

    def normalize(self, reward):
        if self.n < 2:
            return reward
        std = self.var ** 0.5
        return (reward - self.mean) / (std + 1e-8)  # Add a small constant to prevent division by zero


class base_wrapper:
    def __init__(self, env: TSCEnv, max_move_num: int, on_training) -> None:
        self.env = env
        self.n_agent = env.n_agent
        self.n = env.n

        self.max_move_num = max_move_num
        self.reward_normalizer = RewardNormalizer()  # 为每一个环境写一个 normalizer

        # 之前 shape=(5, 12, 7)  # 5是有几帧, 12是move数, 7是特征维度

        self._idx = list(range(max_move_num))
        self.time_len = 2  # 2 is better than 5 in our envs
        self.total_obs = deque(maxlen=self.time_len)
        self.on_training = on_training

    def reset(self):
        # Reset the environment and process the observation
        n_obs = self.env.reset()
        n_processed_obs = [self._process_obs(i, obs) for i, obs in enumerate(n_obs)]
        while len(self.total_obs) < self.time_len:
            self.total_obs.append(n_processed_obs)

        np.random.shuffle(self._idx)  # 一次仿真就 shuffle 一次就可以了
        return self.aug_observation()

    def step(self, n_action):
        # 构建单路口 action 的动作
        n_obs, n_rew, n_done, info = self.env.step(n_action)  # 达到最大仿真步就结束

        n_processed_obs = [self._process_obs(i, obs) for i, obs in enumerate(n_obs)]
        self.total_obs.append(n_processed_obs)

        normal_rewards = []
        for r in n_rew:
            self.reward_normalizer.update(r)
            r = self.reward_normalizer.normalize(r)
            normal_rewards.append(r)
        return self.aug_observation(), normal_rewards, n_done, info

    def _process_obs(self, inter_idx, obs):  # 单帧的
        """每个move的数据结构是这样的
            [occupancy, is_straight, is_left, is_right, lane_numbers, is_now_phase, is_next_phase]
            一般有8个move
        """
        inter = self.env.n_intersection[inter_idx]
        num_move = len(inter.n_roadlink)
        inter_2_cur_next_phase, inlane_2_vehicle_dist = obs
        cur_phase, next_phase = inter_2_cur_next_phase

        cur_moves = inter.n_phase[cur_phase].n_available_roadlink_idx
        next_moves = inter.n_phase[next_phase].n_available_roadlink_idx
        process_obs = []
        for move_id in range(num_move):
            n_startlane_id = inter.n_roadlink[move_id].n_startlane_id
            lane_idxs = [inter.n_in_lane_id.index(i) for i in n_startlane_id]
            inlane_v_num = [(inlane_2_vehicle_dist[i] < 150).sum() for i in lane_idxs]
            # 只考虑入射车道
            occupancy = sum(inlane_v_num) / 50
            direction_flags = self._direction_to_flags(inter.n_roadlink[move_id].move_type)
            lane_numbers = len(n_startlane_id) / 5  # 车道数 (默认不会超过 5 个车道)
            is_now_phase = int(move_id in cur_moves)
            is_next_phase = int(move_id in next_moves)
            # 将其添加到 obs 中
            process_obs.append(
                [occupancy, *direction_flags, lane_numbers, is_now_phase, is_next_phase])  # 某个 movement 对应的信息

        # 不是四岔路, 进行补全
        empty = [0] * len(process_obs[0])
        for _ in range(self.max_move_num - len(process_obs)):  # 12个movement
            process_obs.append(empty)
        # process_obs中顺序乱了 也没事, 反正是选择下一相位的
        return process_obs

    def _process_normalized_reward(self, vehicle_state, max_waiting_time=60) -> int:
        pass

    def _direction_to_flags(self, direction):  # one-hot 向量
        return [
            1 if direction == 'go_straight' else 0,
            1 if direction == 'turn_left' else 0,
            1 if direction == 'turn_right' else 0
        ]

    def aug_observation(self) -> list:
        n_obs = []
        for i in range(self.n):  # i是第几个路口
            stacked_obs = np.array([t[i] for t in self.total_obs], dtype=np.float32)
            if self.on_training:
                stacked_obs = self._shuffle(stacked_obs)
                stacked_obs = self._flow_scale(stacked_obs)
            n_obs.append(stacked_obs)
        return n_obs

    def _shuffle(self, observation):  # 默认用
        """对 obs 中每一个时刻进行打乱顺序。例如原始是：
            array([[[0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2]],

                   [[3, 3, 3],
                    [4, 4, 4],
                    [5, 5, 5]],

                   [[6, 6, 6],
                    [7, 7, 7],
                    [8, 8, 8]]])

        转换之后变为：
            array([[[0, 0, 0],
                    [2, 2, 2],
                    [1, 1, 1]],

                   [[3, 3, 3],
                    [5, 5, 5],
                    [4, 4, 4]],

                   [[6, 6, 6],
                    [8, 8, 8],
                    [7, 7, 7]]])
        """
        # Apply the shuffle index to each 2D slice in the 3D array
        return observation[:, self._idx]  # 进行乱序

    def _flow_scale(self, observation):  # 默认用
        """将 obs 的 flow 同时变大或是变小，乘上同一个数字, 希望 agent 关注相对数量, 而不是绝对数量
        """
        # Generate a random scaling factor
        _ratio = 0.8 + 0.4 * np.random.rand()  # noise range is 0.8-1.2

        # Apply the scaling factor to the first column of each 2D slice
        observation[:, :, 0] *= _ratio
        return observation


def run_an_episode(env: TSCEnv, config: dict, on_training: bool, max_move_num: int):
    env = base_wrapper(env, max_move_num, on_training)

    n_obs = env.reset()  # n个agent的观察
    for i in env.n_agent:
        i.reset()

    n_done = [False]
    info = {}

    # current_episode_step_idx 从0到20到40到60
    for config['current_episode_step_idx'] in itertools.count(start=0, step=config['action_interval']):  # config['current_episode_step_idx']是这里改变的
        if config['current_episode_step_idx'] >= config['num_step'] or all(n_done):
            break

        n_action = []  # 相位
        n_binary_action = []  # 选择或保留
        n_logprob_a = []
        for agent in env.n_agent:
            deterministic = not on_training
            binary_action, action, logprob_a = agent.select_action(n_obs, deterministic)
            n_action.append(action)
            n_binary_action.append(binary_action)
            n_logprob_a.append(logprob_a)
        n_next_obs, n_rew, n_done, info = env.step(n_action)

        if on_training:
            for idx in range(env.n):
                env.n_agent[idx].put_data(n_obs[idx], n_binary_action[idx], n_rew[idx], n_next_obs[idx], n_logprob_a[idx], n_done[idx])

            for agent in env.n_agent:
                if agent.can_learn():
                    # print('begin learn!')
                    agent.train()

        n_obs = n_next_obs

    return info
