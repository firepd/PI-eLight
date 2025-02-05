import numpy as np
import torch
import copy
import math
import random
from torch.distributions import Categorical

from env.intersection import Intersection
from env.TSC_env import TSCEnv

from agent.universal_light.model import Actor, Critic, ERNN


class PPOAgent:
    def __init__(self, config, env: TSCEnv, idx, max_move_num):  # 用于离散动作
        self.config = config
        self.env = env  # type: TSCEnv
        self.idx = idx
        self.cur_agent = self.config[self.config['cur_agent']]  # type: dict

        self.inter = env.n_intersection[idx]  # type: Intersection
        action_space = env.n_action_space[idx]
        self.num_phase = action_space.n
        self.current_phase = 0
        # 以上跟环境有关

        # Init hyperparameters for PPO agent
        self.T_horizon = 3000 * 2
        self.gamma = 0.99  # Discounted Factor
        self.lambd = 0.9  # GAE Factor
        self.clip_rate = 0.2  # PPO Clip rate
        self.K_epochs = 10  # PPO update times
        self.net_width = 64  # Hidden net width
        self.lr = 1e-4  # Learning rate
        self.batch_size = 128
        self.entropy_coef = 0  # Entropy coefficient of Actor
        self.adv_normalization = True  # Advantage normalization

        obs_shape = (2, max_move_num, 7)
        features_dim = 64
        '''Build Actor and Critic'''
        if idx == 0:
            self.feature_extractor = ERNN(obs_shape, features_dim)
            self.actor = Actor(self.feature_extractor, features_dim, 2, self.net_width)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
            self.critic = Critic(self.feature_extractor, features_dim, self.net_width)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        else:
            self.feature_extractor = env.n_agent[0].feature_extractor
            self.actor = env.n_agent[0].actor
            self.actor_optimizer = env.n_agent[0].actor_optimizer
            self.critic = env.n_agent[0].critic
            self.critic_optimizer = env.n_agent[0].critic_optimizer

        '''Build Trajectory holder'''
        self.s_hoder = np.zeros((self.T_horizon, *obs_shape), dtype=np.float32)  # self.T_horizon相当于replay buffer大小
        self.a_hoder = np.zeros((self.T_horizon, 1), dtype=np.int64)
        self.r_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.s_next_hoder = np.zeros((self.T_horizon, *obs_shape), dtype=np.float32)
        self.logprob_a_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.done_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)
        self.cur_idx = 0  # replay buff 中的指针

    def reset(self):
        self.current_phase = 0

    def select_action(self, n_obs, deterministic):
        obs = n_obs[self.idx]

        obs = torch.Tensor(obs).unsqueeze(0)
        with torch.no_grad():
            pi = self.actor.pi(obs, softmax_dim=1)
            # if np.random.rand() < 0.003:
            #     print(pi)
            pi = pi.squeeze(0)
            if deterministic:
                action = torch.argmax(pi).item()
                pi_a = None
            else:
                m = Categorical(pi)
                action = m.sample().item()
                pi_a = pi[action].item()

        if action == 1:
            self.current_phase = (self.current_phase + 1) % self.num_phase  # 实际相位
        return action, self.current_phase, pi_a

    def put_data(self, obs, binary_action, r, obs_next, logprob_a, done):
        # binary_action  # 0: stay, 1: change

        self.s_hoder[self.cur_idx] = obs
        self.a_hoder[self.cur_idx] = binary_action
        self.r_hoder[self.cur_idx] = r
        self.s_next_hoder[self.cur_idx] = obs_next
        self.logprob_a_hoder[self.cur_idx] = logprob_a
        self.done_hoder[self.cur_idx] = done
        self.cur_idx += 1

    def can_learn(self):
        return True if self.cur_idx == self.T_horizon else False

    def train(self):
        '''Prepare PyTorch data from Numpy data'''
        s = torch.from_numpy(self.s_hoder)
        a = torch.from_numpy(self.a_hoder)
        r = torch.from_numpy(self.r_hoder)
        s_next = torch.from_numpy(self.s_next_hoder)
        old_prob_a = torch.from_numpy(self.logprob_a_hoder)
        done = torch.from_numpy(self.done_hoder)
        self.cur_idx = 0

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)

            '''dw(dead and win) 一直是false'''
            deltas = r + self.gamma * vs_ - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, done in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float()
            td_target = adv + vs
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  # sometimes helps

        """PPO update"""
        # Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))

        for _ in range(self.K_epochs):
            # Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm)
            s, a, td_target, adv, old_prob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone()

            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))

                '''actor update'''
                prob = self.actor.pi(s[index], softmax_dim=1)
                prob_a = prob.gather(1, a[index])
                ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a[index]))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2)

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
                self.actor_optimizer.step()

                '''critic update'''
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

