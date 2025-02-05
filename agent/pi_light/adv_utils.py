import numpy as np


def indent_code(code: str, depth: int):
    if depth == 0:
        return code
    indentation = depth * 4
    code_lines = code.strip().split('\n')
    new = []
    for i in code_lines:
        new.append(indentation * ' ' + i)
    return '\n'.join(new)


class Library:
    def __init__(self):
        self.programs = []  # [(入射代码, 出射代码)]
        self.complexity = []
        self.metric = []
        self.best_metric = np.Inf  # 默认指标越小越好 如果改了要修改代码
        self.best_program = None
        self.top_40_in_pareto = None

    def add(self, program, complexity, metric):
        if '$' in program[0] or '$' in program[1]:
            assert 0
        self.programs.append(program)
        self.complexity.append(complexity)
        self.metric.append(metric)
        if metric < self.best_metric:
            self.best_metric = metric
            self.best_program = program
            self.report()

    def query_best(self):
        return self.best_program

    def query_top_40(self):
        assert self.top_40_in_pareto is not None
        return self.top_40_in_pareto

    def report(self):
        print('best metric: %.2f' % self.best_metric)
        print(indent_code(self.best_program, 1))
    
    def get_pareto_frontier(self):
        cost = np.stack([self.complexity, self.metric], axis=1)
        cost[:, 1] = cost[:, 1] * 1  # metric越小越好
        indicators = is_pareto_efficient_simple(cost)

        pareto_optimal_program = []
        pareto_complex, pareto_metric = [], []
        for i, is_optimal in enumerate(indicators):
            if is_optimal:
                pareto_optimal_program.append(self.programs[i])
                pareto_complex.append(self.complexity[i])
                pareto_metric.append(round(self.metric[i], 3))

        self.report_frontier(pareto_complex, pareto_metric)
        index = top_45_percent_index(pareto_metric)
        self.top_40_in_pareto = pareto_optimal_program[index]
        return pareto_optimal_program

    def report_frontier(self, pareto_complex, pareto_metric):
        pareto_complex, pareto_metric = np.array(pareto_complex), np.array(pareto_metric)
        sorted_idx = pareto_complex.argsort()  # 复杂度由小到大
        pareto_complex, pareto_metric = pareto_complex[sorted_idx], pareto_metric[sorted_idx]
        print('pareto frontier!')
        print('complexity:', pareto_complex)
        print('metric:', pareto_metric)


# code from https://devpress.csdn.net/python/630451ce7e66823466199b78.html
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def top_45_percent_index(metric_list):
    arr = np.array(metric_list)
    sorted_indices = np.argsort(arr)  # 由小到大
    num_top_45_percent = int(len(arr) * 0.45)
    index = sorted_indices[num_top_45_percent]
    return index


class Memory:
    def __init__(self):
        self.code_data = set()

    def check_code_duplicate(self, code):
        if code in self.code_data:
            return True
        else:
            self.code_data.add(code)
            return False




