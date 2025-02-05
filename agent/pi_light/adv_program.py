import numpy as np
from itertools import combinations, product
from copy import deepcopy


max_block_breadth = 3
max_depth = 2  # 默认2
max_num_if = 3  # 最多有3个if和 if-else
Feat_list = ['in_v_num', 'out_v_num', 'in_wait_num', 'in_close_num', 'out_close_num']
Parameterized = ['in_close_num', 'out_close_num']  # 带参数的特征
ProgramMode = 'one'  # one - AAAI那种; two - 开悟比赛那种; share - 两个程序共享 有一个额外阈值
# print('ProgramMode:', ProgramMode)


def code_indent(code: str, depth: int):
    if depth == 0:
        return code
    indentation = depth * 4
    code_lines = code.strip().split('\n')
    new = []
    for i in code_lines:
        new.append(indentation * ' ' + i)
    return '\n'.join(new)


class Condition:
    def __init__(self, feat_name: str, compare_op: str):
        self.feat_name = feat_name
        self.compare_op = compare_op

    def flip(self):
        pairs = {'>=': '<', '>':'<=', '<=':'>', '<':'>='}
        flipped = pairs[self.compare_op]
        return Condition(self.feat_name, flipped)

    def to_str(self):
        left_code = f'hand.{self.feat_name}'
        para_code = f'($line:{self.feat_name}$)' if self.feat_name in Parameterized else '()'
        placeholder = f'$cond:{self.feat_name}$'
        return left_code + para_code + self.compare_op + placeholder


class Instruction:  # 一行代码
    def __init__(self, feat_name: str):
        self.feat_name = feat_name

    def get_python_code(self, depth) -> object:
        op = '+=' if 'in' in self.feat_name.split('_') else '-='
        left_code = f'value {op} hand.{self.feat_name}'
        para_code = f'($line:{self.feat_name}$)' if self.feat_name in Parameterized else '()'
        return code_indent(left_code + para_code, depth)

    def length(self):
        return 1


class If:
    def __init__(self, condition: Condition):
        self.condition = condition
        self.if_sub_program = Block()

    def get_python_code(self, depth):
        if_code = 'if ' + self.condition.to_str() + ':'
        sub_code = self.if_sub_program.get_python_code(depth + 1)
        codes = code_indent(if_code, depth) + '\n' + sub_code
        return codes

    def length(self):
        return 1 + self.if_sub_program.length()

    def height(self):
        return self.if_sub_program.height() + 1

    def num_condition(self):
        return self.if_sub_program.num_condition() + 1


class Block:  # 可以放一行行的代码也可以放if-else
    def __init__(self):
        self.data = []

    def get_python_code(self, depth):
        codes = [component.get_python_code(depth) for component in self.data]
        code = '\n'.join(codes)  # 如果codes是 [], 能得到 ''
        return code

    def add(self, component):  # component只能是单行代码\if\if-else
        self.data.append(component)  # 只能按顺序加, 不能跳着加

    def can_add_component(self):
        return True if len(self.data) < max_block_breadth else False

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):  # block[0] = 1
        self.data[index] = value

    def length(self):  # 代码总体长度
        lens = [component.length() for component in self.data]
        return sum(lens)

    def height(self):
        heights = [i.height() for i in self.data if type(i) != Instruction]
        if len(heights) == 0:
            return 0
        return max(heights)

    def can_build_higher_bug(self):  # 即使达到了最大高度, 其他不高的也是可以扩展的
        return True if self.height() < max_depth else False

    def num_condition(self):
        return sum([i.num_condition() for i in self.data if type(i) != Instruction])


class SubProgram:
    def __init__(self, inlane):
        self.block = Block()
        if inlane:
            self.feat_list = [i for i in Feat_list if 'in' in i.split('_')]
        else:
            self.feat_list = [i for i in Feat_list if 'out' in i.split('_')]

    def get_valid_expansions(self):
        to_return = []
        # 找到所有没有满的block, 给他最后塞一个指令
        wait_add_block_pos = []  # 以字符串结尾的路径
        self.findall_block_wait_expand(self.block, wait_add_block_pos, [])
        for pos in wait_add_block_pos:
            for feat in self.feat_list:
                new_program = deepcopy(self)
                block = new_program.navigate_to_pos(pos)
                instruction = Instruction(feat)
                block.add(instruction)
                to_return.append(new_program)

        # 找到单条的指令, 把他们变成if-then
        ins_path = []  # 以数字结尾的
        if self.block.num_condition() < max_num_if:
            self.findall_ins_expand2_if(self.block, ins_path, [])
        # IF模块, 随机一个条件, 然后把当前的一行当作If中的子动作
        for path in ins_path:
            pos, idx = path[:-1], path[-1]
            random_conditions = self.get_random_conditions()
            for r in random_conditions:
                new_program = deepcopy(self)
                block = new_program.navigate_to_pos(pos)
                if_comp = If(r)
                if_comp.if_sub_program.add(block[idx])
                block[idx] = if_comp
                to_return.append(new_program)

        # ElSE 模块 加不加差不多

        return to_return

    def findall_block_wait_expand(self, block, to_append: list, ancestor_sequence: list):
        if block.can_add_component():
            to_append.append(ancestor_sequence)  # to_append可能为空, 代表了最上面的block
        for i, compon in enumerate(block):
            if type(compon) is If:
                new_ancestor = ancestor_sequence + [i] + ['if_sub_program']
                self.findall_block_wait_expand(compon.if_sub_program, to_append, new_ancestor)

    def findall_ins_expand2_if(self, block, to_append: list, ancestor_sequence: list):  # ancestor_sequence在一个函数内部不要被改变了
        if len(ancestor_sequence) >= max_depth*2:  # 不让他深度超过2
            return
        flag = False
        for i, compon in enumerate(block):
            if type(compon) is Instruction and flag is False:  # 一个block只找一个
                flag = True
                to_append.append(ancestor_sequence + [i])
            if type(compon) is If:
                new_ancestor = ancestor_sequence + [i] + ['if_sub_program']
                self.findall_ins_expand2_if(compon.if_sub_program, to_append, new_ancestor)

    def navigate_to_pos(self, path):  # 根据path找到新增的类的具体位置
        if len(path) == 0:
            return self.block
        cur_obj = self.block[path[0]]  # 第一个肯定是数字
        for i in path[1:]:
            if type(i) is str:
                cur_obj = getattr(cur_obj, i)  # 字符串能得到 block
            elif type(i) is int:
                cur_obj = cur_obj[i]  # 数字能得到 指令或者 if
            else:
                assert 0, i
        return cur_obj

    def get_random_conditions(self):
        to_return = []
        for feat in self.feat_list:
            for op in ['<=', '>']:
                cond = Condition(feat, op)
                to_return.append(cond)
        return to_return

    def get_random_instruction(self):
        to_return = []
        for feat in self.feat_list:
            instruct = Instruction(feat)
            to_return.append(instruct)
        return to_return

    def length(self):
        return self.block.length()

    def get_python_code(self, depth):
        return self.block.get_python_code(depth)


class Program:
    def __init__(self, a_func):  # 包括出射和入射程序
        self.a_func = a_func  # 算action_value还是current_value
        self.in_program = SubProgram(True)
        self.out_program = SubProgram(False)

    def length(self):
        return self.in_program.length() + self.out_program.length()

    def get_python_code(self):
        prev = 'def afunc(hand):\n' if self.a_func else 'def cfunc(hand):\n'
        parts = [code_indent('value = 0', 1), self.in_program.get_python_code(1), self.out_program.get_python_code(1), code_indent('return value', 1)]
        parts = [i for i in parts if i != '']
        return prev + '\n'.join(parts)

    def get_valid_expansions(self):
        if self.in_program.length() == 0:
            high_priority = []
            sub_programs = SubProgram(True).get_valid_expansions()
            for sub in sub_programs:
                new_p = Program(self.a_func)
                new_p.in_program = sub
                high_priority.append(new_p)
            return high_priority

        to_return = []
        in_expansions = self.in_program.get_valid_expansions()
        for j in in_expansions:
            new_p = self.copy(in_p=False)
            new_p.in_program = j
            to_return.append(new_p)

        out_expansions = self.out_program.get_valid_expansions()
        for j in out_expansions:
            new_p = self.copy(out_p=False)
            new_p.out_program = j
            to_return.append(new_p)

        return to_return

    def copy(self, in_p=True, out_p=True):
        new_p = Program(self.a_func)
        if in_p:
            new_p.in_program = deepcopy(self.in_program)
        if out_p:
            new_p.out_program = deepcopy(self.out_program)
        return new_p


class Bale:
    def __init__(self):
        self.a_program = Program(a_func=True)   # 所有相位竞争的程序
        self.c_program = Program(a_func=False)  # 维持现有相位的程序
        self.code = None

    def output_code(self):
        if self.code is None:
            a_code = self.a_program.get_python_code()
            if ProgramMode == 'one':
                return a_code
            elif ProgramMode == 'share':
                return a_code + f'\nthreshold[0]=$a:0$'

            c_code = self.c_program.get_python_code()
            full_code = a_code + '\n' + c_code
            if self.c_program.length() != 0:  # c_program内部有代码才加阈值
                full_code = full_code + f'\nthreshold[0]=$a:0$'
            return full_code
        else:
            return self.code

    def replace_code(self, code):  # 常数占位符替换掉
        self.code = code

    def get_valid_expansions(self):
        to_return = []
        a_expansions = self.a_program.get_valid_expansions()
        for a_e in a_expansions:
            new_bale = self.copy(a=False)
            new_bale.a_program = a_e
            to_return.append(new_bale)

        if self.a_program.length() == 0 or ProgramMode != 'two':
            return to_return

        # 运行到这里的代码是 a_program.length() > 0 and ProgramMode 是 two的
        if self.c_program.length() == 0:
            high_priority = []
            c_expansions = self.c_program.get_valid_expansions()
            for c_e in c_expansions:
                new_bale = self.copy(a=True)
                new_bale.c_program = c_e
                high_priority.append(new_bale)
            return high_priority

        c_expansions = self.c_program.get_valid_expansions()
        for c_e in c_expansions:
            new_bale = self.copy(c=False)
            new_bale.c_program = c_e
            to_return.append(new_bale)

        return to_return

    def copy(self, a=True, c=True):
        new_bale = Bale()
        if a:
            new_bale.a_program = deepcopy(self.a_program)
        if c:
            new_bale.c_program = deepcopy(self.c_program)
        return new_bale

    def get_complexity(self):
        return self.c_program.length() + self.a_program.length()


if __name__ == '__main__':
    print('初始化:')
    bale_list = Bale().get_valid_expansions()
    print('有几个:', len(bale_list))
    for i in bale_list:
        code = i.output_code()
        print('代码:')
        print(code)

    print('扩展1')
    expansions = bale_list[0].get_valid_expansions()
    for i in expansions:
        print('代码:')
        print(i.output_code())

    print('扩展2')
    new = expansions[-1].get_valid_expansions()
    for i in new:
        code = i.output_code()
        print('代码:')
        print(code)

    print('扩展3')
    new = new[-1].get_valid_expansions()
    for i in new:
        code = i.output_code()
        print('代码:')
        print(code)

    # print("="*50)  # 为了检查有没有改变原来的类
    # for i in bale_list:
    #     print('代码:')
    #     print(i.output_code())

