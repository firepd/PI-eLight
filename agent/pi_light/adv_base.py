import numpy as np
from env.intersection import Intersection
from env.TSC_env import TSCEnv


class Handler:
    def __init__(self, all_data):
        # inlane_2_num_vehicle, outlane_2_num_vehicle, inlane_2_num_waiting_vehicle, inlane_2_vehicle_dist, outlane_2_vehicle_dist = all_data
        self.inlane_v_num, self.outlane_v_num, self.inlane_v_wait_num, self.inlane_v_dist, self.outlane_v_dist = all_data
        self.in_idx, self.out_idx = None, None  # 入射车道的ID

    def set_lane_id(self, in_idx, out_idx):
        self.in_idx, self.out_idx = in_idx, out_idx

    def in_v_num(self):
        return self.inlane_v_num[self.in_idx]

    def out_v_num(self):
        return self.outlane_v_num[self.out_idx]  # 出射车道id

    def in_wait_num(self):
        return self.inlane_v_wait_num[self.in_idx]

    def in_close_num(self, line):  # 带参数的特征  10~100
        return (self.inlane_v_dist[self.in_idx] < line).sum()

    def out_close_num(self, line):  # 带参数的特征  10~100
        return (self.outlane_v_dist[self.out_idx] < line).sum()
    #
    def drive_num(self, time_to_arrive):
        speed_limit = 25
        acceleration = 2.0

        v_prime = self.speed_dict[self.in_idx] + acceleration * time_to_arrive  # 加速10秒
        v_prime[v_prime > speed_limit] = speed_limit
        ac_t = (v_prime - self.speed_dict[self.in_idx]) / acceleration  # 加速时间
        v_t = time_to_arrive - ac_t  # 平稳行驶时间
        dis = (self.speed_dict[self.in_idx] + v_prime) / 2 * ac_t + v_prime * v_t
        num = (dis >= self.inlane_v_dist[self.in_idx]).sum()  # 开过那条线的数量
        return num


class PiPolicy:  # 有几个路口就会被调用几次
    def __init__(self, config, env: TSCEnv, idx, p_Mode):
        self.config = config
        self.env = env  # type: TSCEnv
        self.idx = idx
        self.ProgramMode = p_Mode

        self.inter = env.n_intersection[idx]  # type: Intersection
        self.action_space = env.n_action_space[idx]
        self.num_phase = self.action_space.n
        self.current_phase = 0

        agent_config = self.config[self.config['cur_agent']]  # type: dict
        self.feature_list = agent_config['observation_feature_list']

        self.func_code, self.assig_code = '', ''

    def inject_code(self, code: str):
        if 'threshold[0]' in code:
            last_newline_index = code.rfind('\n')
            func_code, assig_code = code[:last_newline_index], code[last_newline_index + 1:]
            self.func_code, self.assig_code = func_code, assig_code
        else:
            self.func_code = code
            self.assig_code = ''

    def reset(self):
        pass

    # 24/12/4: 我推测是一个move对应多个 lanelink 一个入射lane会指向多个出射lane
    def pick_action(self, n_obs, on_training):
        obs = n_obs[self.idx]
        assert len(obs) == len(self.feature_list)

        num_move = len(self.inter.n_roadlink)
        exec(self.func_code, globals())  # 函数代码

        hand = Handler(obs)
        if self.ProgramMode in ['two', 'share']:  # 'two'有bug
            threshold = [3]
            exec(self.assig_code)  # 赋值代码
            cur_func = afunc if self.ProgramMode == 'share' else cfunc
            move_values = np.zeros(num_move)
            for move_id in range(num_move):
                move_values[move_id] = self._get_value_for_move2(cur_func, hand, move_id)
            phase_values = self._aggregate_for_each_phase(move_values)
            cur_value = phase_values[self.current_phase]  # 维持现有相位
            # print('cur:', cur_value, 'thresh:', threshold[0])
            if cur_value > threshold[0]:
                return self.current_phase

        move_values = np.zeros(num_move)
        for move_id in range(num_move):
            move_values[move_id] = self._get_value_for_move2(afunc, hand, move_id)

        phase_values = self._aggregate_for_each_phase(move_values)
        action = phase_values.argmax()
        self.current_phase = action
        return action

    def _get_value_for_move(self, hand, move_id):  # 入射和出射独立考虑
        pass  # 注定要有两个程序

    def _get_value_for_move2(self, func, hand, move_id):  # 入射和出射成对考虑
        value = 0
        n_lanelink_id = self.inter.n_roadlink[move_id].n_lanelink_id
        for lane_link in n_lanelink_id:
            start_lane_name, end_lane_name = lane_link[0], lane_link[1]
            in_index = self.inter.n_in_lane_id.index(start_lane_name)
            out_index = self.inter.n_out_lane_id.index(end_lane_name)
            hand.set_lane_id(in_index, out_index)
            value += func(hand)

        # todo 开源时候删掉
        # num_lanelink = len(self.inter.n_roadlink[move_id].n_lanelink_id)
        # num_startlane = len(self.inter.n_roadlink[move_id].n_startlane_id)
        # value = value / num_lanelink * num_startlane
        # 不同的路口可能move数不一样，这里归一化一下
        value = value / self.inter.road_link_num
        return value

    def _aggregate_for_each_phase(self, move_values):
        phase_values = np.zeros(self.num_phase)
        for phase_id in range(self.num_phase):
            n_roadlink_idx = self.inter.n_phase[phase_id].n_available_roadlink_idx  # 每个phase对应的 roadlink_index
            phase_values[phase_id] = move_values[n_roadlink_idx].sum()
        return phase_values

