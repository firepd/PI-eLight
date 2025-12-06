from agent import BaseAgent
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utilities.utils import copy_model_params
from torch.distributions.categorical import Categorical
from replay_buffer import ReplayBuffer
import copy
from agent.drl_based.mplight import share_layer


class Q_Net(nn.Module):
    def __init__(self, num_lanelink, num_phase, phase_2_passable_lanelink):
        super(Q_Net, self).__init__()

        self.Q1 = _Network(num_lanelink, num_phase, phase_2_passable_lanelink)
        self.Q2 = _Network(num_lanelink, num_phase, phase_2_passable_lanelink)

    def forward(self, s):  # 输入是s 不是s + a
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        return q1, q2


class Policy_Net(nn.Module):
    def __init__(self, num_lanelink, num_phase, phase_2_passable_lanelink):
        super(Policy_Net, self).__init__()
        self.P = _Network(num_lanelink, num_phase, phase_2_passable_lanelink)

    def forward(self, s):  # 输出一个概率向量
        logits = self.P(s)
        probs = F.softmax(logits, dim=1)
        return probs


class _Network(torch.nn.Module):  # 采用FRAP的网络结构
    def __init__(self, num_lanelink, num_phase, phase_2_passable_lanelink):
        super(_Network, self).__init__()
        self.num_lanelink = num_lanelink
        self.num_phase = num_phase
        self.phase_2_passable_lanelink = phase_2_passable_lanelink
        self.lanelink_2_applicable_phase = phase_2_passable_lanelink.permute(1, 0)  # num_lanelink * num_phase
        self.phase_competition_mask = self._get_phase_competition_mask(phase_2_passable_lanelink)  # num_phase * (num_phase - 1)
        self.dim_embedding = 4
        self.dim_hidden_repr = 16
        self.dim_conv_repr = 20  # section 4.3.5

        self.phase_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=self.dim_embedding),
            torch.nn.ReLU(),
        )  # Ws, bs in Eq. 3
        self.num_vehicle_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=self.dim_embedding),
            torch.nn.ReLU(),
        )  # Wv, bv in Eq. 3
        self.lanelink_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.dim_embedding * 2, out_features=self.dim_hidden_repr),
            torch.nn.ReLU(),
        )  # Wh, bh in Eq. 4

        self.relation_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=self.dim_embedding)  # sec. 4.3.3

        self.conv_cube = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.dim_hidden_repr * 2, out_channels=self.dim_conv_repr, kernel_size=(1, 1)),
            torch.nn.ReLU(),
        )
        self.conv_relation = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.dim_embedding, out_channels=self.dim_conv_repr, kernel_size=(1, 1)),
            torch.nn.ReLU(),
        )
        self.tail_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.dim_conv_repr, out_channels=self.dim_conv_repr, kernel_size=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=self.dim_conv_repr, out_channels=1, kernel_size=(1, 1)),
        )

    def forward(self, obs):
        batch_size = obs[0].shape[0]
        b_current_passable_lanelink = torch.matmul(obs[0].float(), self.phase_2_passable_lanelink.float())  # batch * num_lanelink
        b_num_waiting_vehicle = obs[1]  # batch * num_lanelink

        # Eq. 3
        lanelink_embedding_wrt_phase = self.phase_embedding(b_current_passable_lanelink.unsqueeze(-1))  # batch * num_lanelink * embedding_dim
        lanelink_embedding_wrt_num_vehicle = self.num_vehicle_embedding(b_num_waiting_vehicle.unsqueeze(-1))  # batch * num_lanelink * embedding_dim
        # Eq. 4
        lanelink_embedding = torch.cat([lanelink_embedding_wrt_phase, lanelink_embedding_wrt_num_vehicle], dim=2)  # batch * num_lanelink * (2 *embedding_dim)
        lanelink_embedding = self.lanelink_embedding(lanelink_embedding)  # batch * num_lanelink * dim_hidden_repr
        lanelink_embedding = lanelink_embedding.permute(0, 2, 1)  # batch * dim_hidden_repr * num_lanelink

        # Eq. 5
        phase_embedding = torch.matmul(lanelink_embedding, self.lanelink_2_applicable_phase.float())  # batch * dim_hidden_repr * num_phase
        phase_embedding = phase_embedding.permute(0, 2, 1)  # batch * num_phase * dim_hidden_repr
        phase_embedding_cube = self._get_phase_embedding_cube(batch_size, phase_embedding)  # batch * (dim_hidden_repr * 2) * num_phase * (num_phase - 1)
        # Eq. 6
        phase_conv = self.conv_cube(phase_embedding_cube)  # batch * dim_conv_repr * num_phase * (num_phase - 1)

        # Eq. 7
        relation_embedding = self.relation_embedding(self.phase_competition_mask).permute(2, 0, 1).unsqueeze(0)  # (batch=1) * embedding_dim * num_phase * (num_phase - 1)
        relation_conv = self.conv_relation(relation_embedding)  # batch * dim_conv_repr * num_phase * (num_phase - 1)

        # Eq. 8
        combined_feature = phase_conv * relation_conv  # batch * dim_conv_repr * num_phase * (num_phase - 1)
        before_merge = self.tail_layer(combined_feature)  # batch * 1 * num_phase * (num_phase - 1)
        q_values = torch.sum(before_merge, dim=3).squeeze(1)  # batch * num_phase
        return q_values

    def _get_phase_embedding_cube(self, batch_size, phase_embedding):
        phase_embedding_cube = torch.zeros((batch_size, 32, self.num_phase, self.num_phase - 1))
        for phase_idx in range(self.num_phase):
            continuous_jdx = 0
            for phase_jdx in range(self.num_phase):
                if phase_idx == phase_jdx:
                    continue
                phase_embedding_comb = torch.cat([
                    phase_embedding[:, phase_idx, :],
                    phase_embedding[:, phase_jdx, :]
                ], dim=1)
                phase_embedding_cube[:, :, phase_idx, continuous_jdx] = phase_embedding_comb
                continuous_jdx += 1
        return phase_embedding_cube

    def _get_phase_competition_mask(self, phase_2_passable_lanelink):
        mask = torch.zeros((self.num_phase, self.num_phase - 1), dtype=torch.int64)
        for phase_idx in range(self.num_phase):
            continuous_jdx = 0
            for phase_jdx in range(self.num_phase):
                if phase_idx == phase_jdx:
                    continue
                for lanelink_idx in range(self.num_lanelink):
                    if phase_2_passable_lanelink[phase_idx][lanelink_idx] == phase_2_passable_lanelink[phase_jdx][lanelink_idx] == 1:
                        mask[phase_idx][continuous_jdx] = 1
                continuous_jdx += 1
        return mask


class SAC(BaseAgent):
    def __init__(self, config, env, idx):
        super(SAC, self).__init__(config, env, idx)
        if 'manhattan' in config['inter_name'].lower():
            self.cur_agent['buffer_size'] = 20000   # 防止DRL用完内存
        self.replay_buffer = ReplayBuffer(self.cur_agent['buffer_size'],
                                          self.cur_agent['batch_size'],
                                          self.obs_shape,
                                          self.config['device'])
        self.tau = 0.005
        self.gamma = 0.95

        phase_2_passable_lanelink = torch.tensor(self.inter.phase_2_passable_lanelink_idx)
        num_lanelink, num_phase = sum(self.inter.n_num_lanelink), len(self.inter.n_phase)
        lr = self.cur_agent['learning_rate']
        # ===================================================================================

        self.actor = Policy_Net(num_lanelink, num_phase, phase_2_passable_lanelink)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.q_critic = Q_Net(num_lanelink, num_phase, phase_2_passable_lanelink)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=lr)

        self.q_critic_target = copy.deepcopy(self.q_critic)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        # if idx > 0: # 默认不share
        #     share_layer(self.actor.P, env.n_agent[0].actor.P)
        #     self.actor_optimizer = env.n_agent[0].actor_optimizer
        #     share_layer(self.q_critic.Q1, env.n_agent[0].q_critic.Q1)
        #     share_layer(self.q_critic.Q2, env.n_agent[0].q_critic.Q2)
        #     share_layer(self.q_critic_target.Q1, env.n_agent[0].q_critic_target.Q1)
        #     share_layer(self.q_critic_target.Q2, env.n_agent[0].q_critic_target.Q2)
        #     self.q_critic_optimizer = env.n_agent[0].q_critic_optimizer

        self.alpha = 0.2  # adaptive_alpha = True
        # We use 0.6 because the recommended 0.98 will cause alpha explosion.
        self.target_entropy = 0.6 * (-np.log(1 / num_phase))  # H(discrete)>0
        self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)
        self.H_mean = 0

        self.network_local = self.actor  # 和迁移代码兼容

    def reset(self):
        self.current_phase = 0
        self.replay_buffer.reset()

    def pick_action(self, n_obs, on_training):
        obs = n_obs[self.idx]  # list
        # Mplight obs[0].shape, obs[1].shape torch.Size([1, 6]) torch.Size([1, 16]) 第一个one-hot,第二个float
        assert obs[0].shape[0] == 1, 'should be mini-batch with size 1'
        self.actor.eval()
        with torch.no_grad():
            probs = self.actor(obs)  # [1, 8]
            if not on_training:
                action = torch.argmax(probs, dim=1).cpu().item()
            else:
                action = Categorical(probs).sample().item()
        self.actor.train()
        self.current_phase = action
        return self.current_phase

    def store_experience(self, obs, action, reward, next_obs, done):
        self.replay_buffer.store_experience(obs, action, reward, next_obs, done)

    def _time_to_learn(self):
        return self.replay_buffer.current_size >= self.cur_agent['batch_size']

    def learn(self):
        if not self._time_to_learn():
            return 0

        s, a, r, s_next, dw = self.replay_buffer.sample_experience()
        dw = dw.float()

        # ------------------------------------------ Train Critic ----------------------------------------#
        '''Compute the target soft Q value'''
        with torch.no_grad():
            next_probs = self.actor(s_next)  # [b, a_dim]
            next_log_probs = torch.log(next_probs + 1e-8)  # [b, a_dim]
            next_q1_all, next_q2_all = self.q_critic_target(s_next)  # [b, a_dim]
            min_next_q_all = torch.min(next_q1_all, next_q2_all)
            v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1, keepdim=True)  # [b,1]
            # (- self.alpha * next_log_probs) 中选择某个动作的概率越小，会给这个Q更大的bonus
            # next_probs * Q可以得到V
            target_Q = r + (1 - dw) * self.gamma * v_next

        '''Update soft Q net'''
        q1_all, q2_all = self.q_critic(s)  # [b,a_dim]
        q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a)  # [b,1]
        q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # ------------------------------------------ Train Actor ----------------------------------------#
        for params in self.q_critic.parameters():
            # 冻结q_critic的网络
            params.requires_grad = False

        probs = self.actor(s)  # [b, a_dim]
        log_probs = torch.log(probs + 1e-8)  # [b, a_dim]
        with torch.no_grad():
            q1_all, q2_all = self.q_critic(s)  # [b, a_dim]
        min_q_all = torch.min(q1_all, q2_all)

        a_loss = torch.sum(probs * (self.alpha * log_probs - min_q_all), dim=1, keepdim=True)  # [b,1]
        # probs * (self.alpha * log_probs) 鼓励探索
        # probs * (- min_q_all) 选Q高的

        self.actor_optimizer.zero_grad()
        a_loss.mean().backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters():
            params.requires_grad = True
        # ------------------------------------------ Train Alpha ----------------------------------------#
        with torch.no_grad():
            self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
        alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)
        # 差值为正（H_mean > target）：策略比预期更随机，需要减小 alpha
        # 差值为负（H_mean < target）：策略比预期更确定，需要增大 alpha。

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp().item()
        # ------------------------------------------ 更新 Target Net ----------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

