from .road import Road
from .road_link import RoadLink
from .phase import Phase
from typing import List
from utilities.utils import list_with_unique_element


class Intersection:
    def __init__(self, inter_idx, inter_id, inter_dict, env):
        # print('一个路口:', inter_idx, inter_id)
        self.inter_idx = inter_idx  # id
        self.inter_id = inter_id    # string
        self.inter_dict = inter_dict
        self.env = env
        self.eng = env.eng
        self.yellow_phase = 0  # roadnet.json 中黄灯一般是第一个动作, 要么包含右转的availableRoadLinks, 要么啥都不包含, 持续时间很短
        self._yellow_time = 3
        self.current_phase = 0
        self.current_phase_time = 0
        self.during_yellow = False

        self.n_road = []  # type: List[Road]       # 一般有8条路
        self.n_in_road = []  # type: List[Road]    # 4条入射路
        self.n_out_road = []  # type: List[Road]   # 4条出射路
        self.n_lane_id = []  # type: List[str]     # 8*3 或者 8*2个车道的名字
        self.n_in_lane_id = []  # type: List[str]  # 4*3 或者 4*2个车道的名字
        self.n_out_lane_id = []  # type: List[str] # 4*3 或者 4*2个车道的名字

        # scan all road
        for road_id in self.inter_dict['roads']:  # 该路口涉及到的路，一般有8条
            road = self.env.id2road[road_id]
            self.n_road.append(road)

        self.n_roadlink = []  # type: List[RoadLink]  # 一般有12个roadLink
        self.n_num_lanelink = []  # type: List[int]  # 每个roadlink的车道链接的数量
        for roadlink_dict in self.inter_dict['roadLinks']:  # todo 这里可以排除右转的
            roadlink = RoadLink(roadlink_dict, self)
            self.n_roadlink.append(roadlink)
            self.n_num_lanelink.append(len(roadlink.n_lanelink_id))

        # scan all in_road and out_road by roadlink
        for roadlink in self.n_roadlink:  # 每个roadlink涉及到的入射和出射道路
            self.n_in_road.append(self.env.id2road[roadlink.startroad_id])
            self.n_out_road.append(self.env.id2road[roadlink.endroad_id])
        self.n_in_road = list_with_unique_element(self.n_in_road)  # 有重复的Road
        self.n_out_road = list_with_unique_element(self.n_out_road)

        # 得到出射路和入射路的 lane id
        for road in self.n_road:
            self.n_lane_id.extend(road.n_lane_id)
        for in_road in self.n_in_road:
            self.n_in_lane_id.extend(in_road.n_lane_id)  # 包括右转的入射吧
        for out_road in self.n_out_road:
            self.n_out_lane_id.extend(out_road.n_lane_id)  # 包括左转的入射吧

        # 那个引擎中, 动作是 0: 黄灯, 1: 正常动作1, 2: 正常动作2
        # 该路口的动作空间, 0: 正常动作1, 1: 正常动作2
        # len(self.n_phase)就是动作空间
        self.n_phase = []  # type: List[Phase]
        available_movement = set()
        for phase_idx, phase_dict in enumerate(self.inter_dict['trafficLight']['lightphases']):
            # print(phase_dict['availableRoadLinks'])
            if phase_idx == 0:  # 排除 yellow
                turn_right = phase_dict['availableRoadLinks']
                continue
            phase_dict['availableRoadLinks'] = [i for i in phase_dict['availableRoadLinks'] if i not in turn_right]  # 排除右转的
            self.n_phase.append(Phase(phase_idx, phase_dict, self))
            available_movement |= set(phase_dict['availableRoadLinks'])

        self.road_link_num = len(available_movement)   # 排除了右转以后所有的movement id

        # print(inter_id, '每个move有几个lanelink:', self.n_num_lanelink)  # roadLink就是movement
        # phase_num_roadlink = [len(i.n_available_roadlink_idx) for i in self.n_phase]
        # print(inter_id, '每个phase包括了几个 move:', phase_num_roadlink)

        self.n_neighbor_idx = [self.inter_idx]  # this will be determined in TSCEnv once all intersections are scanned
        self.phase_2_passable_lane_idx = self._get_phase_2_passable_lane_idx()  # 0-1mask 济南是[9, 12] 杭州是[6, 8]
        self.phase_2_passable_lanelink_idx = self._get_phase_2_passable_lanelink_idx()  # 0-1mask 济南是(9, 36) 杭州是(6, 16)

    def _get_phase_2_passable_lane_idx(self):
        phase_2_passable_lane_idx = []
        for phase_idx in range(len(self.n_phase)):
            n_lane = [0 for _ in range(len(self.n_in_lane_id))]
            for pass_lane_id in self.n_phase[phase_idx].n_available_startlane_id:  # 每个phase的入射车道的str名字
                lane_idx = self.n_in_lane_id.index(pass_lane_id)
                n_lane[lane_idx] = 1
            phase_2_passable_lane_idx.append(n_lane)
        return phase_2_passable_lane_idx

    def _get_phase_2_passable_lanelink_idx(self):
        phase_2_passable_lanelink_idx = []
        for phase_idx in range(len(self.n_phase)):
            n_lanelink = []
            for i, roadlink in enumerate(self.n_roadlink):
                if i in self.n_phase[phase_idx].n_available_roadlink_idx:
                    n_lanelink.extend([1 for _ in range(self.n_num_lanelink[i])])
                else:
                    n_lanelink.extend([0 for _ in range(self.n_num_lanelink[i])])
            phase_2_passable_lanelink_idx.append(n_lanelink)
        return phase_2_passable_lanelink_idx

    def step(self, action, interval):
        # assert self.n_phase[action].phase_idx != self.yellow_phase
        if self.during_yellow:
            if self.current_phase_time < self._yellow_time:
                self.current_phase_time += interval
            else:  # 等待时间达到了3秒, 切换为新动作
                self.during_yellow = False
                self.eng.set_tl_phase(self.inter_id, self.n_phase[action].phase_idx)  # action 不等于 self.n_phase[action].phase_idx
                self.current_phase = action
                self.current_phase_time = interval
        elif action == self.current_phase:  # 动作一致 自增
            self.current_phase_time += interval
        else:  # 动作不一样就切换为黄灯
            self.during_yellow = True
            self.eng.set_tl_phase(self.inter_id, self.yellow_phase)  # 切换为黄灯 tiny-light 没写上的代码
            self.current_phase_time = interval

    def reset(self):   # 这里就会改变引擎中的相位 环境初始化被调用
        self.current_phase = 0  # 一开始是第一个动作
        self.current_phase_time = 0
        self.during_yellow = False
        self.eng.set_tl_phase(self.inter_id, self.n_phase[self.current_phase].phase_idx)

    def __str__(self):
        return self.inter_id
