import re
import time
import random
from math import log, sqrt
from typing import List
import itertools
from skopt import gp_minimize
from skopt.space import Real, Integer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from env import TSCEnv
from agent.pi_light.adv_program import Bale
from agent.pi_light.adv_base import PiPolicy
from agent.pi_light.adv_utils import *


def pi_run_a_step(env: TSCEnv, n_obs: List):
    n_action = []
    for agent in env.n_agent:
        action = agent.pick_action(n_obs, None)
        n_action.append(action)
    n_next_obs, n_rew, n_done, info = env.step(n_action)
    return n_next_obs, n_rew, n_done, info, n_action[0]


def run_an_episode(env: TSCEnv, config: dict):
    n_obs = env.reset()  # n个agent的观察
    n_done = [False]
    action_sequence = []
    info = {}

    for config['current_episode_step_idx'] in itertools.count(start=0, step=config['action_interval']):
        if config['current_episode_step_idx'] >= config['num_step'] or all(n_done):
            break
        n_next_obs, n_rew, n_done, info, action = pi_run_a_step(env, n_obs)
        n_obs = n_next_obs
        action_sequence.append(str(action))  # note 动作的重复真不能说明什么, 后续相关代码被删除

    travel_time = info['world_2_average_travel_time'][0]  # 全局的，这个值越小越好
    return travel_time


def get_reward(travel_time):
    assert travel_time > 20
    return 200 / travel_time


class MCTS_synthesizer:  # 一个公式要建立一颗树
    def __init__(self, env: TSCEnv, config):
        self.env = env
        self.n_agent = env.n_agent  # type: List[PiPolicy]
        self.config = config
        agent_config = self.config[self.config['cur_agent']]  # type: dict
        self.feature_list = agent_config['observation_feature_list']

        self.weight = 0.5
        self.total_episode = 0
        self.root = None
        self.library = Library()  # 保存搜索到的两段代码字符串, 复杂度, 分数
        self.optimizer = Optimizer(self.get_running_metric)
        self.visited_bank = Memory()
        self.cache = {}

    def distribute(self, best=False):  # 把合成的程序交给每个agent
        if best:
            code = self.library.query_best()
        else:
            code = self.library.query_top_40()
        print('selected codes:')
        print(code)
        for i in self.n_agent:
            i.inject_code(code)

    def init_start_programs(self):
        empty_bale = Bale()
        program_list = empty_bale.get_valid_expansions()
        self.root.evaled = True
        self.total_episode += len(program_list)
        for bale in program_list:
            self.visited_bank.check_code_duplicate(bale.output_code())  # 肯定没有重复
            travel_time = self.evaluate(bale)  # 可能有常数, 如果有参数, 代码会被修改
            reward = get_reward(travel_time)
            mcts_node = MCTS_Node(self.root, bale, weight=self.weight)
            mcts_node.evaled = True
            mcts_node.update(reward)
            self.root.visits += 1
            self.root.children.append(mcts_node)
            self.library.add(bale.output_code(), 1, travel_time)

    def begin_search(self, train_episodes):
        self.root = MCTS_Node(None, None, weight=self.weight)  # root除了它的儿子, 其他都是空的
        self.init_start_programs()

        start = time.time()
        for i in range(1, train_episodes + 1):
            if i % 15 == 0:
                time_spend = round(time.time() - start, 2)
                print(f'time spent:{time_spend}; {i} program evaluated')

            bottom_node, metric = self.search_one()
            if bottom_node is not None:
                bale = bottom_node.bale  # type: Bale
                self.library.add(bale.output_code(), bale.get_complexity(), metric)

        self.library.get_pareto_frontier()
        print('total number of episode:', self.total_episode)

    def search_one(self):
        node = self.root
        epsilon = 0.1

        while node.evaled and len(node.children) > 0:
            if np.random.rand() < epsilon:
                node = node.random_select()
            else:
                node = node.select()

        # 得到的node要么是没有评估过的 (自然没有扩展), 要么是评估过的 (但没有扩展)
        if node.evaled:
            node = node.expand(self.visited_bank)
            if node is None:
                return None, 10000

        # 到这里的节点是没有评估过的
        metric = self.evaluate(node.bale)  # 评估和优化参数是同时进行的
        node.evaled = True

        # 检查动作序列是否重复了
        # 检查带常数项的表达是否重复了

        bottom_node = node
        reward = get_reward(metric)
        # backpropagation
        while node is not None:
            node.update(reward)
            node = node.parent

        return bottom_node, metric

    def evaluate(self, bale):
        code = bale.output_code()  # 字符串
        num_const = code.count('$') // 2  # 两个程序的参数和
        if num_const > 0:  # 输出把常数替换掉的两端程序
            code, opt_step = self.optimizer.optimize_const(code)  # 常数肯定能优化成功
            bale.replace_code(code)
            self.total_episode += opt_step
        self.total_episode += 1
        travel_time = self.get_running_metric(code)
        return travel_time

    def get_running_metric(self, code: str):
        if code in self.cache:
            return self.cache[code]

        for i in self.n_agent:
            i.inject_code(code)
        travel_time = run_an_episode(self.env, self.config)
        self.cache[code] = travel_time
        return travel_time


class MCTS_Node:
    def __init__(self, parent, bale: Bale, weight=None):
        self.parent = parent
        self.children = []
        self.value = 0  # value就是自己或者儿子最大的reward
        self.visits = 0
        self.weight = weight
        self.bale = bale  # 这个程序包含入射和出射的
        self.evaled = False

    def random_select(self):
        return random.sample(self.children, k=1)[0]

    def select(self):
        weights = [(c.value + self.weight * sqrt(log(self.visits) / (c.visits + 0.1))) for c in self.children]
        max_idx = np.argmax(weights)
        return self.children[max_idx]

    def expand(self, visited_bank):
        possible_expands = self.bale.get_valid_expansions()
        for i in possible_expands:
            codes = i.output_code()
            if visited_bank.check_code_duplicate(codes):  # 扩张时检查重复性, 不行就换一个
                continue
            son = MCTS_Node(self, i, weight=self.weight)
            self.children.append(son)
        if len(self.children) > 0:
            child = random.sample(self.children, 1)[0]
            return child
        else:
            # print('没有儿子可以扩展')
            self.value = -1
            self.visits = 10000
            return None

    def update(self, reward):
        self.visits += 1
        self.value = max(self.value, reward)  # 不同之处

    def __repr__(self):
        return f'MCTS node, value:{self.value}, visits:{self.visits}, evaled:{self.evaled}'


class Optimizer:
    def __init__(self, run_func):  # 以及优化常数
        self.run_func = run_func
        # condition_param_space是int
        self.condition_param_space = {'in_v_num': [0, 40], 'out_v_num': [0, 20],
                                      'in_wait_num': [0, 40], 'in_close_num': [0, 20], 'out_close_num': [0, 20]}
        self.condition_default_value = 10
        # 距离的阈值是int
        self.dist_line_range = {'in_close_num': [5, 200], 'out_close_num': [5, 200]}
        self.line_default_value = {'in_close_num': 150, 'out_close_num': 10}

    def gen_range(self, code: str):  # 按参数的出现顺序, 得到每个参数的范围和默认值
        kind_feat_name = re.findall('\$(\w+):(\w+)\$', code)  # 可能有多个 $cond:inlane_2_num_waiting_$
        param_ranges = []
        default_values = []
        for kind, feat_name in kind_feat_name:  # [('cond', 'inlane_2_num_waiting_vehicle'), ('weight', 'inlane_2_num_waiting_vehicle')]
            if kind == 'cond':
                param_range = self.condition_param_space[feat_name]
                param_range = Integer(param_range[0], param_range[1], prior='uniform')
                default_value = self.condition_default_value + int(np.random.rand() * 5)
            elif kind == 'line':
                param_range = self.dist_line_range[feat_name]
                param_range = Integer(param_range[0], param_range[1], prior='uniform')
                default_value = self.line_default_value[feat_name]
            elif kind == 'a':
                # 早期的没有 /road_link_num
                # param_range = Integer(2, 27, prior='uniform')
                # default_value = 6
                param_range = Real(0.15, 2.5, prior='uniform')
                default_value = 0.25
            else:
                assert 0
            param_ranges.append(param_range)
            default_values.append(default_value)
        return param_ranges, default_values

    def optimize_const(self, code):
        param_ranges, default_values = self.gen_range(code)
        self.parts_list = self.decompose(code)
        opt_step = self.get_opt_step(len(default_values))
        result = gp_minimize(self._evaluate, dimensions=param_ranges, n_calls=opt_step, n_initial_points=3,
                             x0=default_values)
        return self.assemble(result.x), opt_step  # 返回所有 $$都被替换成数字的代码

    def _evaluate(self, params):
        full_code = self.assemble(params)
        travel_time = self.run_func(full_code)
        return travel_time

    def get_opt_step(self, num_const):
        setting = {1: 4, 2: 7, 3: 10}
        return setting.get(num_const, 12)

    def decompose(self, code: str):
        return re.split('\$\w+:\w+\$', code)

    def assemble(self, values):
        full_code = concatenate(self.parts_list, values)
        return full_code


def concatenate(code_parts, values):
    result = ''
    for i in range(min(len(code_parts), len(values))):
        result += code_parts[i] + str(values[i])

    # 处理剩余的部分，如果有的话
    if len(code_parts) > len(values):
        result += ''.join(x for x in code_parts[len(values):])
    elif len(values) > len(code_parts):
        result += ''.join(str(x) for x in values[len(code_parts):])
    return result


