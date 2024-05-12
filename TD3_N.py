import copy
import numpy as np
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes=[256,256]):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q


class Vnet(nn.Module):
    def __init__(self, state_dim, hidden_sizes=[256,256]):
        super(Vnet, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, state):
        v = F.relu(self.l1(state))
        v = F.relu(self.l2(v))
        v = self.l3(v)

        return v


class TD3_N(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            device,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            actor_lr=1e-3,
            critic_lr=1e-3,
            value_lr=1e-3,
            hidden_sizes=[256,256],
            alpha=0.001
    ):
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action, hidden_sizes).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.value = Vnet(state_dim, hidden_sizes).to(self.device)
        self.value_target = copy.deepcopy(self.value)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)

        self.critic1 = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)

        self.critic2 = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)


        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            #next_action = self.actor_target(next_state)
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

            target_V = self.value_target(next_state)
            target_V = reward + not_done * self.discount * target_V

        current_V = self.value(state)
        value_Q = reward + not_done * self.discount * self.value(next_state)
        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)

        value_loss = F.mse_loss(current_V, target_V)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        # value_loss.backward(retain_graph=True)
        self.value_optimizer.step()

        critic1_loss = F.mse_loss(current_Q1, target_Q) + self.alpha * F.mse_loss(value_Q, current_Q1)
        critic2_loss = F.mse_loss(current_Q2, target_Q) + self.alpha * F.mse_loss(value_Q, current_Q2)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic1.state_dict(), filename + "_critic1")
        torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")
        torch.save(self.critic2.state_dict(), filename + "_critic2")
        torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.value.state_dict(), filename + "_value")
        torch.save(self.value_optimizer.state_dict(), filename + "_value_optimizer")

    def load(self, filename):
        self.critic1.load_state_dict(torch.load(filename + "_critic1"))
        self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer"))
        self.critic2.load_state_dict(torch.load(filename + "_critic2"))
        self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.value.load_state_dict(torch.load(filename + "_value"))
        self.value_optimizer.load_state_dict(torch.load(filename + "_value_optimizer"))
