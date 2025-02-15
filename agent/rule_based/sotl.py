from agent import BaseAgent
import torch
import numpy as np


class SOTL(BaseAgent):
    """
    Seung-Bae Cools, Carlos Gershenson, and Bart D’Hooghe. 2013. Self-organizing traffic lights: A realistic simulation.
        In Advances in applied self-organizing systems. Springer, 45–55.
    """
    def __init__(self, config, env, idx):
        super(SOTL, self).__init__(config, env, idx)

        # the minimum duration of time of one phase
        self.t_min = 10
        # some thresholds to deal with phase requests
        self.min_green_vehicle = 20
        self.max_red_vehicle = 0
        # phase 2 passable lane
        self.phase_2_passable_lane = np.array(self.inter.phase_2_passable_lane_idx)  # 01的张量

    def reset(self):
        pass

    # 这个算法的特征是 inlane_2_num_waiting_vehicle
    def pick_action(self, n_obs, on_training):
        obs = n_obs[self.idx][0]
        action = self.inter.current_phase
        if self.inter.current_phase_time >= self.t_min:
            # num of waiting vehicles on lanes w.r.t. current phase
            # 目前 phase所对应的lane有多少等待车辆
            num_green_vehicle = np.sum(obs * self.phase_2_passable_lane[self.current_phase:(self.current_phase+1), :])
            # num of waiting vehicles on other lanes
            # 其他的lane有多少等待车辆
            num_red_vehicle = np.sum(obs * (1 - self.phase_2_passable_lane[self.current_phase:(self.current_phase+1), :]))

            if num_green_vehicle <= self.min_green_vehicle and num_red_vehicle > self.max_red_vehicle:
                action = (action + 1) % self.num_phase

        self.current_phase = action
        return self.current_phase
