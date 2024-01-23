import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_sizes=[400, 300]):
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
	def __init__(self, state_dim, action_dim, hidden_sizes=[400, 300]):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0]+action_dim, hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], 1)


	def forward(self, state, action):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(torch.cat([q, action], 1)))
		return self.l3(q)

class Vnet(nn.Module):
	def __init__(self, state_dim, hidden_sizes=[400, 300]):
		super(Vnet, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], 1)

	def forward(self, state):
		v = F.relu(self.l1(state))
		v = F.relu(self.l2(v))
		v = self.l3(v)

		return v


class DDPG_N(object):
	def __init__(
		self,
		device,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		actor_lr=1e-3,
		critic_lr=1e-3,
		value_lr=1e-3,
		hidden_sizes=[400, 300],
		alpha=0.005
	):
		self.device =device
		self.actor = Actor(state_dim, action_dim, max_action, hidden_sizes).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

		self.critic = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

		self.value = Vnet(state_dim, hidden_sizes).to(self.device)
		self.value_target = copy.deepcopy(self.value)
		self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)

		self.discount = discount
		self.tau = tau
		self.alpha = alpha


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state))
			target_Q = reward + not_done * self.discount * target_Q

			# Compute the target V value
			target_V = self.value_target(next_state)
			target_V = reward + not_done * self.discount * target_V

		# Get current Q estimate
		current_Q = self.critic(state, action)

		current_V = self.value(state)

		value_Q = reward + not_done * self.discount * self.value(next_state)

		value_loss = F.mse_loss(current_V, target_V)

		self.value_optimizer.zero_grad()
		value_loss.backward()
		# value_loss.backward(retain_graph=True)
		self.value_optimizer.step()

		# Compute critic loss

		critic_loss = F.mse_loss(current_Q, target_Q) + self.alpha * F.mse_loss(value_Q, current_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()
		
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
		torch.save(self.value.state_dict(), filename + "_value")
		torch.save(self.value_optimizer.state_dict(), filename + "_value_optimizer")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.value.load_state_dict(torch.load(filename + "_value"))
		self.value_optimizer.load_state_dict(torch.load(filename + "_value_optimizer"))