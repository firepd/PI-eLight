import numpy as np
import time
import torch.nn.functional as F
import torch.nn as nn
import torch

# nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 7), padding=0)  # N*1*12*K -> N*64*12*1
# nn.Conv2d(64, 128, kernel_size=(12, 1), padding=0)
# nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True)
# nn.Linear(64, 64)
# nn.Linear(64, 2)

# 参数量
# 128*7+ 65*128*12+4*192*64+4*64*64*2+4*64*2+65*64+65*2

# FLOPS
# conv_2d_FLOPS(1,64,12,7,(7,)) + conv_2d_FLOPS(64,128,12,1,(12,)) + LSTM_FLops(2, 128, 64) + LSTM_FLops(2, 64, 64) + Linear_Flops(64, 64) + Linear_Flops(64, 2)

def conv_2d_FLOPS(C_in, C_out, H_in, W_in, kernel):
    return (2*C_in+1) * C_out * H_in * W_in * np.prod(kernel)

def LSTM_FLops(seq_len, C_in, H):
    return seq_len*(4*(C_in*H+H*H) + 4*H)

def Linear_Flops(F_in, F_out):
    return (2*F_in + 1) * F_out


class ERNN(nn.Module):
    def __init__(self, observation_space, features_dim: int = 64):  # 利用 RNN 提取信息
        super().__init__()
        self.net_shape = observation_space  # 每个 movement 的特征数量, 8 个 movement, 就是 (N, 8, K)
        # 这里 N 表示由 N 个 frame 堆叠而成
        self.move_num = observation_space[1]

        self.view_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, self.net_shape[-1]), padding=0),  # N*1*8*K -> N*64*8*1
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(self.move_num, 1), padding=0),  # N*64*8*1 -> N*128*1*1 (BatchSize, N, 128, 1, 1)
            nn.ReLU(),
        )  # 每一个 junction matrix 提取的特征
        view_out_size = self._get_conv_out(self.net_shape)
        # print('view_out_size:', view_out_size)  # 128

        self.extract_time_info = nn.LSTM(
            input_size=view_out_size,
            hidden_size=features_dim,
            num_layers=2,
            batch_first=True
        )

    def _get_conv_out(self, shape):
        o = self.view_conv(torch.zeros(1, 1, *shape[1:]))
        return int(np.prod(o.size()))

    def forward(self, observations):
        batch_size = observations.size()[0]  # (BatchSize, N, 8, K)
        observations = observations.view(-1, 1, self.move_num, self.net_shape[-1])  # (BatchSize*N, 1, 8, K)
        conv_out = self.view_conv(observations).view(batch_size, self.net_shape[0], -1)  # (BatchSize*N, 256) --> (BatchSize, N, 256)

        lstm_out, _ = self.extract_time_info(conv_out)
        return lstm_out[:, -1]


class Actor(nn.Module):
    def __init__(self, fe: ERNN, state_dim, action_dim, net_width):
        super().__init__()
        self.fe = fe
        self.l1 = nn.Linear(state_dim, net_width)
        # self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

    def forward(self, state):
        feat = self.fe(state)
        n = torch.relu(self.l1(feat))
        # n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim=0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob


class Critic(nn.Module):
    def __init__(self, fe, state_dim, net_width):
        super().__init__()
        self.fe = fe
        self.C1 = nn.Linear(state_dim, net_width)
        # self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        feat = self.fe(state)
        v = torch.relu(self.C1(feat))
        # v = torch.relu(self.C2(v))
        v = self.C3(v)
        return v


if __name__ == '__main__':
    model = ERNN((2, 12, 7))
    data = torch.rand(128, 2, 12, 7)
    start = time.time()
    for i in range(500):
        out = model(data)
    print('cost time:', time.time() - start)
    print(out.shape)

    data = torch.rand(128, 2, 12, 7)
    out = model(data)
    print(out)

    # data = torch.rand(128, 2, 15, 7)
    # actor = Actor(model, 64, 2, 64)
    # out = actor.pi(data, softmax_dim=1)
    # print(out)
    #
    # data = torch.zeros(128, 2, 15, 7)
    # actor = Actor(model, 64, 2, 64)
    # out = actor.pi(data, softmax_dim=1)
    # print(out)