import time
import gym 
import numpy as np 
import torch

from torch import nn
from torch.nn import functional as F


def unpack_batch(batch):
  return batch.state, batch.action, batch.next_state, batch.reward, batch.done


class Timer:

	def __init__(self):
		self._start_time = time.time()
		self._step_time = time.time()
		self._step = 0

	def reset(self):
		self._start_time = time.time()
		self._step_time = time.time()
		self._step = 0

	def set_step(self, step):
		self._step = step
		self._step_time = time.time()

	def time_cost(self):
		return time.time() - self._start_time

	def steps_per_sec(self, step):
		sps = (step - self._step) / (time.time() - self._step_time)
		self._step = step
		self._step_time = time.time()
		return sps


def eval_policy(policy, eval_env, eval_episodes=10):
	"""
	Eval a policy
	"""
	max_steps = 99
	avg_reward = 0.
	action_ar = np.eye(eval_env.action_space.n)
	state_ar = np.eye(eval_env.n_width * eval_env.n_height).reshape(eval_env.n_width, eval_env.n_height, -1)

	for _ in range(eval_episodes):
		eval_env.reset()
		state, done = eval_env.random_reset(), False
		best_steps = np.abs(np.array(eval_env.start) - np.array(eval_env.ends[0])).sum()
		print('start:{a},end:{b}'.format(a=eval_env.start,b=eval_env.ends[0]))
		# print('eval_state:', state, state.shape, np.array(state))
		# print(np.array(state))
		step = 0
		while not done:
			state_one_hot = np.concatenate((state_ar[state[0], state[1]], state[2:]), -1)
			# action_one_hot = action_ar[action]
			action = policy.select_action(state_one_hot, explore=True)
			# print(action, action.dtype)
			state, reward, done, _ = eval_env.step(action)
			# print('state:{a},reward:{b},done:{c}'.format(a=state,b=reward,c=done))
			# avg_reward += reward
			step += 1
		episode_reward = (max_steps - step) / (max_steps - best_steps)
		print(f'step:{step}, r:{episode_reward}')
		avg_reward += episode_reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward



def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)



class MLP_Phi(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim, hidden_depth, output_dim):
		super().__init__()
		self.trunk = mlp(state_dim * action_dim, hidden_dim, output_dim, hidden_depth)
		self.apply(weight_init)
		self.state_dim = state_dim
		self.action_dim = action_dim
	def forward(self, s_a):
		s = s_a[...,:self.state_dim].argmax(-1)
		a = s_a[...,self.state_dim:].argmax(-1)
		s_a_onehot = F.one_hot(s*self.action_dim + a, self.state_dim * self.action_dim).float()
		return self.trunk(s_a_onehot)

class MLP(nn.Module):
	def __init__(self,
								input_dim,
								hidden_dim,
								output_dim,
								hidden_depth,
								output_mod=None):
		super().__init__()
		self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
											output_mod)
		self.apply(weight_init)

	def forward(self, x):
		return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
	if hidden_depth == 0:
		mods = [nn.Linear(input_dim, output_dim)]
	else:
		mods = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
		for i in range(hidden_depth - 1):
			mods += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
		mods.append(nn.Linear(hidden_dim, output_dim))
	if output_mod is not None:
		mods.append(output_mod)
	trunk = nn.Sequential(*mods)
	return trunk

def to_np(t):
	if t is None:
		return None
	elif t.nelement() == 0:
		return np.array([])
	else:
		return t.cpu().detach().numpy()