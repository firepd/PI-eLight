from env.intersection import Intersection
from env.TSC_env import TSCEnv
import numpy as np


class PiRule:  # 有几个路口就会被调用几次
    def __init__(self, config, env: TSCEnv, idx):
        self.config = config
        self.env = env  # type: TSCEnv
        self.idx = idx

        self.inter = env.n_intersection[idx]  # type: Intersection
        self.action_space = env.n_action_space[idx]
        self.num_phase = self.action_space.n
        self.current_phase = 0

        agent_config = self.config[self.config['cur_agent']]  # type: dict
        self.feature_list = agent_config['observation_feature_list']
        self.min_inlane_2_num, self.max_inlane_2_num = 100000, -1
        self.min_outlane_2_num, self.max_outlane_2_num = 100000, -1
        self.min_inlane_2_num_waiting, self.max_inlane_2_num_waiting = 100000, -1
        self.max_move_value = -1
        self.max_norm_move_value = -1

    def reset(self):
        pass

    def update_state(self, obs):
        inlane_v_num, outlane_v_num, inlane_v_wait_num, inlane_v_dist, outlane_v_dist = obs
        max_1, min_1 = inlane_v_num.max(), inlane_v_num.min()
        max_2, min_2 = outlane_v_num.max(), outlane_v_num.min()
        max_3, min_3 = inlane_v_wait_num.max(), inlane_v_wait_num.min()
        if max_1 > self.max_inlane_2_num:
            self.max_inlane_2_num = max_1
        if min_1 < self.min_inlane_2_num:
            self.min_inlane_2_num = min_1
        if max_2 > self.max_outlane_2_num:
            self.max_outlane_2_num = max_2
        if min_2 < self.min_outlane_2_num:
            self.min_outlane_2_num = min_2
        if max_3 > self.max_inlane_2_num_waiting:
            self.max_inlane_2_num_waiting = max_3
        if min_3 < self.min_inlane_2_num_waiting:
            self.min_inlane_2_num_waiting = min_3

    def pick_action(self, n_obs, on_training):
        obs = n_obs[self.idx]
        self.update_state(obs)

        num_move = len(self.inter.n_roadlink)
        move_values = np.zeros(num_move)
        for move_id in range(num_move):
            move_values[move_id] = self._test_policy(obs, move_id)

        phase_values = self._aggregate_for_each_phase(move_values)

        action = phase_values.argmax()
        self.current_phase = action
        return action

    def _test_policy(self, obs, move_id):  # 把 pilight找到的程序放这里
        inlane_2_num_vehicle, outlane_2_num_vehicle, inlane_2_num_waiting_vehicle, inlane_2_vehicle_dist, outlane_2_vehicle_dist = obs
        value = [0]
        n_lanelink_id = self.inter.n_roadlink[move_id].n_lanelink_id
        for lane_link in n_lanelink_id:
            start_lane_name, end_lane_name = lane_link[0], lane_link[1]
            index = self.inter.n_in_lane_id.index(start_lane_name)

            value[0] += (inlane_2_vehicle_dist[index] < 120).sum()

            index = self.inter.n_out_lane_id.index(end_lane_name)
            value[0] -= outlane_2_num_vehicle[index]

        return value[0]

    def _aggregate_for_each_phase(self, move_values):
        phase_values = np.zeros(self.num_phase)
        for phase_id in range(self.num_phase):
            n_roadlink_idx = self.inter.n_phase[phase_id].n_available_roadlink_idx  # 每个phase对应的 roadlink_index
            phase_values[phase_id] = move_values[n_roadlink_idx].sum()
        return phase_values

