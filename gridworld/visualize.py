import numpy as np
import torch
import gym
import argparse
import os
import pickle
from tensorboardX import SummaryWriter

from utils import util, buffer
from agent.sac import sac_agent
from agent.vlsac import vlsac_agent
from agent.ctrlsac import ctrlsac_agent
from agent.diffsrsac import diffsrsac_agent
from agent.spedersac import spedersac_agent
from agent.spedersac.spedersac_agent import pca_transform
from agent.spedersac.spedersac_iragent import VI_IRL_Agent
from agent.spedersac import spedersac_iragent
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import colors
from scipy.special import logsumexp
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import FixedLocator


def save_kwargs(kwargs, path):
  if hasattr(kwargs, 'action_space'):
    action_space = kwargs['action_space']
    print('action_space:', action_space)
    if isinstance(action_space, gym.spaces.Discrete):
      kwargs['action_space_type'] = 'Discrete'
      kwargs['action_space_n'] = action_space.n
    elif isinstance(action_space, gym.spaces.Box):
      kwargs['action_space_type'] = 'Box'
      kwargs['action_space_shape'] = (action_space.low.min(), action_space.high.max(), action_space.shape)
    else:
      raise ValueError('Unsupported action space type.')
    kwargs['action_space'] = None
  with open(path, 'wb') as f:
    pickle.dump(kwargs, f)
  if 'action_space_n' in kwargs:
    kwargs['action_space'] = action_space

def load_kwargs(path):
  with open(path, 'rb') as f:
    kwargs = pickle.load(f)
  # print('action_space_type:', kwargs['action_space_type'])
  if 'action_space_type' in kwargs.keys():
    
    if kwargs['action_space_type'] == 'Discrete':
      kwargs['action_space'] = gym.spaces.Discrete(kwargs['action_space_n'])
      del kwargs['action_space_n']
    elif kwargs['action_space_type'] == 'Box':
      kwargs['action_space'] = gym.spaces.Box(low=kwargs['action_space_shape'][0], high=kwargs['action_space_shape'][1], shape=kwargs['action_space_shape'][2])
      del kwargs['action_space_shape']
    # delete unnecessary keys
    del kwargs['action_space_type']
  return kwargs
  



def eval_policy(args, env, agent):
  """
  Eval a policy
  """
  avg_reward = 0.
  print(agent.device)
  for _ in range(50):
    # print(env.n_width, env.n_height)
    state, done = env.random_reset(), False
    # print('start:{a},end:{b}'.format(a=env.start,b=env.ends[0]))
    while not done:
      action = agent.select_action(state)
      state = np.array([state])
      probs = agent.actor.evaluate(torch.FloatTensor(state).to(agent.device)).exp().detach().cpu().numpy()
      print('state:', state, 'action:', action, 'probs:', probs)
      state, reward, done, _ = env.step(action)
def visualize_state_action_map_by_task(Q, args, env, agent):
  n_state = env.n_height * env.n_width
  n_task = env.n_height * env.n_width
  fig, axes = plt.subplots(3, 3, figsize=(10,10))
  axes = axes.flatten()
  for i in range(n_task):
    reorganize_q = reorganize(Q[...,i:i+1])[0]
    axes[i].imshow(reorganize_q, cmap='viridis')
  plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
  fig.suptitle('State-Action Map by Task')
  for ax in axes:
    ax.axis('off')
    mark_action_name(ax, n_state)
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/state_action_map_by_task.png')
  print(f'{fig_path}/state_action_map_by_task.png')
  plt.close()

def state_drift_onehot(state, all_states):
  assert state.shape[-1] == 2
  assert all_states.shape[-1] == 2
  distance = (state.reshape(-1,1,2) - all_states.reshape(1,-1,2))**2
  prob = np.exp(-distance.sum(-1))
  prob /= prob.sum()
  return prob


def save_buffer_VI(args, env, agent):
  max_timesteps = 90000
  buffer_size = 90000
  random_reset_freq = 70  
  n_task = 9
  replay_buffer = buffer.ReplayBuffer(agent.state_dim+n_task, agent.action_dim, buffer_size)
  n_state = env.n_height * env.n_width
  P = np.load('P.npy')
  R = np.load('r.npy')
  alpha = 1
  discount = 0.5

  Q, V, error = spedersac_agent.value_iteration(P, R, alpha=alpha, discount=discount)
  print(Q)
  visualize_state_action_map_by_task(Q, args, env, agent)
  print(f'error: {error}')
  
  episode_reward = 0
  episode_timesteps = 0
  episode_num = 0
  state_ar = np.eye(n_state).reshape(env.n_width, env.n_height, -1)
  action_ar = np.eye(env.action_space.n)
  env.reset()
  state, done = env.random_reset(), False
  for t in range(max_timesteps):
    episode_timesteps += 1
    task_id = np.argmax(state[2:])
    state_idx = state[0] * env.n_width + state[1]
    logits = Q[state_idx, :, task_id]
    dist = torch.distributions.Categorical(logits=torch.FloatTensor(logits))
    action = dist.sample().item()
    next_state, reward, done, _ = env.step(action)
    done_bool = float(done) if t < max_timesteps else 0
    state_onehot = np.concatenate((state_ar[state[0], state[1]], state[2:]), -1)
    # print(state_onehot)
    action_onehot = action_ar[action]
    next_state_onehot = np.concatenate((state_ar[next_state[0], next_state[1]], next_state[2:]), -1)
    replay_buffer.add(state_onehot, action_onehot, next_state_onehot, reward, done_bool)
    state = next_state
    episode_reward += reward
    if done: 
      # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
      print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
      # Reset environment
      if ((episode_num+1) % (random_reset_freq) == 0):
        state, done = env.random_reset(), False
        print('start:{a}, end:{b}'.format(a=env.start,b=env.ends[0]))
      else:
        state, done = env.reset(), False
      episode_reward = 0
      episode_timesteps = 0
      episode_num += 1 
  torch.save(replay_buffer.state_dict(), f'./replay_buffer_VI.pkl')


def save_buffer(args, env, agent, model_path):
  max_timesteps = 10000
  buffer_size = 10000
  random_reset_freq = 70
  n_task = env.n_height * env.n_width
  replay_buffer = buffer.ReplayBuffer(2+n_task, agent.action_dim, buffer_size)
  n_state = env.n_height * env.n_width
  state, done = env.reset_to(np.random.randint(0,n_state), args.task_idx), False
  episode_reward = 0
  episode_timesteps = 0
  episode_num = 0
  max_length = env._max_episode_steps
  state_ar = np.eye(n_state).reshape(env.n_width, env.n_height, -1)
  for t in range(int(max_timesteps)):

    episode_timesteps += 1

    # Select action according to policy
    state_one_hot = np.concatenate((state_ar[state[0], state[1]], state[2:]), -1)
    action = agent.select_action(state_one_hot, explore=True)

    # Perform action
    next_state, reward, done, _ = env.step(action) 
    done_bool = float(done) if episode_timesteps < max_length else 0

    # Store data in replay buffer
    replay_buffer.add(state, np.array([action]), next_state, reward, done_bool)

    state = next_state
    episode_reward += reward
  
    if done: 
      # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
      print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
      # Reset environment
      if ((episode_num+1) % (random_reset_freq) == 0):
        state, done = env.random_reset_start(), False
        print('start:{a}, end:{b}'.format(a=env.start,b=env.ends[0]))
      else:
        state, done = env.reset(), False

      episode_reward = 0
      episode_timesteps = 0
      episode_num += 1 
  torch.save(replay_buffer.state_dict(), f'{model_path}/replay_buffer_task{args.task_idx}.pkl')
  print(f'{model_path}/replay_buffer_task{args.task_idx}.pkl')

def save_buffer_random(args, env, agent):
  max_timesteps = 90000
  buffer_size = 90000
  random_reset_freq = 70
  n_task = env.n_height * env.n_width
  replay_buffer = buffer.ReplayBuffer(agent.state_dim+n_task, agent.action_dim, buffer_size)
  n_state = env.n_height * env.n_width
  state, done = env.random_reset(), False
  episode_reward = 0
  episode_timesteps = 0
  episode_num = 0
  max_length = env._max_episode_steps
  state_ar = np.eye(n_state).reshape(env.n_width, env.n_height, -1)
  action_ar = np.eye(env.action_space.n)
  env.reset()
  state, done = env.random_reset(), False
  for t in range(int(max_timesteps)):

    episode_timesteps += 1

    # Select action according to policy
    action = np.random.randint(0, env.action_space.n)

    # Perform action
    next_state, reward, done, _ = env.step(action) 
    done_bool = float(done) if episode_timesteps < max_length else 0

    state_onehot = np.concatenate((state_ar[state[0], state[1]], state[2:]), -1)
    # print(state_onehot)
    action_onehot = action_ar[action]
    next_state_onehot = np.concatenate((state_ar[next_state[0], next_state[1]], next_state[2:]), -1)
    # Store data in replay buffer
    replay_buffer.add(state_onehot, action_onehot, next_state_onehot, reward, done_bool)

    state = next_state
    episode_reward += reward
  
    if done: 
      # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
      print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
      # Reset environment
      if ((episode_num+1) % (random_reset_freq) == 0):
        state, done = env.random_reset(), False
        print('start:{a}, end:{b}'.format(a=env.start,b=env.ends[0]))
      else:
        state, done = env.reset(), False

      episode_reward = 0
      episode_timesteps = 0
      episode_num += 1 
  torch.save(replay_buffer.state_dict(), f'./replay_buffer_random.pkl')
  print(f'./replay_buffer_random.pkl')

def save_buffer_random_labyrinth(args, env, agent):
  trans_prob = np.load('trans_probs.npy')
  n_state = 127
  n_action = 4
  n_task = 3
  time_step = 90000
  state = np.random.randint(0, n_state)
  replay_buffer = buffer.ReplayBuffer(n_state+n_task, n_action, time_step)
  state_ar = np.eye(n_state)
  action_ar = np.eye(n_action)
  task_id_ar = np.eye(n_task)
  for i in range(time_step):
    action = np.random.randint(0, n_action)
    next_state = np.argmax(trans_prob[state, action])
    state_onehot = np.concatenate((state_ar[state], task_id_ar[0]), -1)
    action_onehot = action_ar[action]
    next_state_onehot = np.concatenate((state_ar[next_state], task_id_ar[0]), -1)
    replay_buffer.add(state_onehot, action_onehot, next_state_onehot, 0, 0)
    state = next_state
    print(state, action, next_state)
  torch.save(replay_buffer.state_dict(), f'./replay_buffer_random_labyrinth.pkl')



def visualize_phi_by_state(args, env, agent):
  plt.rcParams.update({'font.size': 15})
  if agent.state_dim != 1:
    print('s_n:', env.n_height)
    print('a_n:', env.action_space.n)
    grid_x, grid_y = np.meshgrid(np.arange(env.n_width), np.arange(env.n_height))
    states = np.concatenate([grid_x.reshape(-1,1), grid_y.reshape(-1,1)], axis=-1)
  elif agent.state_dim == 1:
    states = np.arange(env.observation_space.n).reshape(-1,1)
  actions = np.arange(env.action_space.n).reshape(-1,1)
  action_name = ['U', 'D', 'R', 'L']
  colors = ['red', 'purple', 'blue', 'green']
  fig, axes = plt.subplots(env.n_width,env.n_height, figsize=(10,10))
  axes = axes.flatten()
  x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
  for i in range(states.shape[0]):
    ax = axes[(env.n_height-states[i,1]-1)*env.n_width+states[i,0]]
    # 1D
    # ax = axes[i]
    # x = int(states[i]%env.n_width)
    # y = int(states[i]//env.n_width)
    # ax = axes[(env.n_height-y-1)*env.n_width+x]
    print(ax)
    input_sa = np.concatenate([np.repeat(states[i].reshape(1,-1),actions.shape[0],0), actions], axis=-1)
    state_map = agent.phi(torch.tensor(input_sa).float()).detach().numpy()
    state_map = PCA(n_components=2).fit_transform(state_map)
    x_min, x_max, y_min, y_max = min(x_min, state_map[:,0].min()), max(x_max, state_map[:,0].max()),\
      min(y_min, state_map[:,1].min()), max(y_max, state_map[:,1].max())
    for j in range(actions.shape[0]):
      ax.scatter(state_map[j,0], state_map[j,1], s=60, c=colors[j])
      ax.text(state_map[j,0], state_map[j,1], action_name[j])
    ax.set_title('State {i}'.format(i=states[i]))
    if env.grids.get_type(states[i][0], states[i][1]) == 1:
      ax.set_title('State {i}'.format(i=states[i]), color='red')
  for i in range(states.shape[0]):
    axes[i].set_xlim(x_min-0.1, x_max+0.1)
    axes[i].set_ylim(y_min-0.1, y_max+0.1)
  plt.subplots_adjust(wspace=0.5, hspace=0.5)
  fig.suptitle('Phi by state')
  plt.xlabel('Feature dim 1')
  plt.ylabel('Feature dim 2')

  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/phi_state.png')
  print(f'{fig_path}/phi_state.png')
  plt.close()
  return

def visualize_phi(args, env, agent):
  plt.rcParams.update({'font.size': 15})
  if agent.state_dim != 1:
    print('s_n:', env.height)
    print('a_n:', env.action_space.n)
    # states = np.arange(env.reset().shape[0])
    grid_x, grid_y = np.meshgrid(np.arange(env.n_width), np.arange(env.n_height))
    states = np.concatenate([grid_x.reshape(-1,1), grid_y.reshape(-1,1)], axis=-1)
  elif agent.state_dim == 1:
    states = np.arange(env.observation_space.n).reshape(-1,1)
  # print(states)
  actions = np.repeat(np.arange(env.action_space.n).reshape(1,-1), states.shape[0], axis=0)
  cmaps = ['Reds', 'Purples', 'Blues', 'Greens']
  fig, axes = plt.subplots(2,2, figsize=(10,10))
  axes = axes.flatten()
  x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf

  phi_matrix = np.zeros((env.action_space.n, states.shape[0], args.feature_dim))
  for i in range(env.action_space.n):
    action_map = agent.phi(torch.tensor(np.concatenate([states, actions[:,i:i+1]], axis=-1)).float()).detach().numpy()

    phi_matrix[i] = action_map
    action_map = PCA(n_components=2).fit_transform(action_map)
    print('action_{i}'.format(i=i), action_map)
    axes[i].scatter(action_map[:,0], action_map[:,1], s=60, c=np.arange(states.shape[0]), cmap=cmaps[i], vmin=-1)
    axes[i].set_title(f'Action {i}')
    axes[i].set_xlabel('Feature dim 1')
    axes[i].set_ylabel('Feature dim 2')
    x_min, x_max, y_min, y_max = min(x_min, action_map[:,0].min()), max(x_max, action_map[:,0].max()),\
      min(y_min, action_map[:,1].min()), max(y_max, action_map[:,1].max())
    for j in range(action_map.shape[0]):
      axes[i].text(action_map[j,0],action_map[j,1], f'({states[j,0]},{states[j,1]})')
      # 1D
      # axes[i].text(action_map[j,0],action_map[j,1], f'{states[j]}')
  for i in range(env.action_space.n):
    axes[i].set_xlim(x_min-0.1, x_max+0.1)
    axes[i].set_ylim(y_min-0.1, y_max+0.1)
  plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2, wspace=0.5, hspace=0.5)

  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  # plt.show()
  plt.savefig(f'{fig_path}/phi.png')
  print(f'{fig_path}/phi.png')
  plt.close()

  fig, axis = plt.subplots(1,2, figsize=(20,10))
  phi_matrix_flat = phi_matrix.reshape(-1, args.feature_dim)
  # np.save('NN_phi.npy',phi_matrix)
  for i in range(phi_matrix_flat.shape[1]):
    axis[0].plot(phi_matrix_flat[:,i], label=i)
  axis[0].legend()
  phi_matrix_reduced = PCA(n_components=min(phi_matrix_flat.shape)).fit_transform(phi_matrix_flat)
  for i in range(phi_matrix_reduced.shape[1]):
    axis[1].plot(phi_matrix_reduced[:,i], label=i)
  axis[1].legend()
  plt.savefig(f'{fig_path}/phi_line.png')
  print(f'{fig_path}/phi_line.png')
  plt.close()
  with open(f'{fig_path}/w.txt', 'w') as f:
    print('Writing w')
    f.write('W:\n')
    for name, param in agent.w.named_parameters():
      f.write(f'{name}:{param}')
    print('Writing u')
    f.write('U:\n')
    for name, param in agent.critic.named_parameters():
      f.write(f'{name}:{param}')
  return


def visualize_mu(args, env, agent):
  plt.rcParams.update({'font.size': 15})
  # states = np.arange(env.observation_space.n).reshape(-1,1)
  if agent.state_dim != 1:
    states = np.arange(env.reset().shape[0])
    grid_x, grid_y = np.meshgrid(np.arange(env.n_width), np.arange(env.n_height))
    states = np.concatenate([grid_x.reshape(-1,1), grid_y.reshape(-1,1)], axis=-1)
    print(states)
    # states = np.array([[0,0],[0,1]])
#   actions = np.repeat(np.arange(env.action_space.n).reshape(1,-1), states.shape[0], axis=0)
  elif agent.state_dim == 1:
    states = np.arange(env.observation_space.n).reshape(-1,1)
  # print(states)
  state_map = agent.mu(torch.tensor(states).float()).detach().numpy()
  state_map = PCA(n_components=2).fit_transform(state_map)
  plt.scatter(state_map[:,0], state_map[:,1], s=60, c=np.arange(states.shape[0]), cmap='viridis')
  plt.title('Mu')
  plt.xlabel('Feature dim 1')
  plt.ylabel('Feature dim 2')
  plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
  for j in range(state_map.shape[0]):
    plt.text(state_map[j,0],state_map[j,1], f'({states[j,0]},{states[j,1]})')
    # 1D
    # plt.text(state_map[j,0],state_map[j,1], f'{states[j]}')
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/mu.png')
  print(f'{fig_path}/mu.png')
  plt.close()
  return


def visualize_mu_intrapolate(args, env, agent):
  plt.rcParams.update({'font.size': 15})
  # states = np.arange(env.observation_space.n).reshape(-1,1)
  if agent.state_dim != 1:
    states = np.arange(env.reset().shape[0])
    grid_x, grid_y = np.meshgrid(np.arange(30)/15, np.arange(30)/15)
    states = np.concatenate([grid_x.reshape(-1,1), grid_y.reshape(-1,1)], axis=-1)
    print(states)
    # states = np.array([[0,0],[0,1]])
#   actions = np.repeat(np.arange(env.action_space.n).reshape(1,-1), states.shape[0], axis=0)
  elif agent.state_dim == 1:
    states = np.arange(env.observation_space.n).reshape(-1,1)
  # print(states)
  state_map = agent.mu(torch.tensor(states).float()).detach().numpy()
  state_map = PCA(n_components=2).fit_transform(state_map)
  plt.scatter(state_map[:,0], state_map[:,1], s=30, c=np.arange(states.shape[0]), cmap='viridis', alpha=0.5)

  # # Mark the grid
  # if agent.state_dim == 2:
  #   states = np.arange(env.reset().shape[0])
  #   grid_x, grid_y = np.meshgrid(np.arange(env.n_width), np.arange(env.n_height))
  #   states = np.concatenate([grid_x.reshape(-1,1), grid_y.reshape(-1,1)], axis=-1)
  #   print(states)
  # elif agent.state_dim == 1:
  #   states = np.arange(env.observation_space.n).reshape(-1,1)
  # # print(states)
  # state_map = agent.mu(torch.tensor(states).float()).detach().numpy()
  # plt.scatter(state_map[:,0], state_map[:,1], s=30, c='red')

  plt.title('Mu')
  plt.xlabel('Feature dim 1')
  plt.ylabel('Feature dim 2')
  plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
  # for j in range(state_map.shape[0]):
  #   plt.text(state_map[j,0],state_map[j,1], f'({states[j,0]},{states[j,1]})')
    # 1D
    # plt.text(state_map[j,0],state_map[j,1], f'{states[j]}')
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/mu_intrapolate.png')
  print(f'{fig_path}/mu_intrapolate.png')
  plt.close()
  return

def visualize_phi_intrapolate(args, env, agent):
  plt.rcParams.update({'font.size': 15})
  if agent.state_dim != 1:
    print('s_n:', env.height)
    print('a_n:', env.action_space.n)
    # states = np.arange(env.reset().shape[0])
    grid_x, grid_y = np.meshgrid(np.arange(30)/15, np.arange(30)/15)
    states = np.concatenate([grid_x.reshape(-1,1), grid_y.reshape(-1,1)], axis=-1)
  elif agent.state_dim == 1:
    states = np.arange(env.observation_space.n).reshape(-1,1)
  # print(states)
  actions = np.repeat(np.arange(env.action_space.n).reshape(1,-1), states.shape[0], axis=0)
  cmaps = ['Reds', 'Purples', 'Blues', 'Greens']
  fig, axes = plt.subplots(2,2, figsize=(10,10))
  axes = axes.flatten()
  x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
  for i in range(env.action_space.n):
    action_map = agent.phi(torch.tensor(np.concatenate([states, actions[:,i:i+1]], axis=-1)).float()).detach().numpy()
    action_map = PCA(n_components=2).fit_transform(action_map)
    print('action_{i}'.format(i=i), action_map)
    axes[i].scatter(action_map[:,0], action_map[:,1], s=30, c=np.arange(states.shape[0]), cmap=cmaps[i], vmin=-1, alpha=0.5)
    
    # # Mark the grid
    # if agent.state_dim == 2:
    #   states = np.arange(env.reset().shape[0])
    #   grid_x, grid_y = np.meshgrid(np.arange(env.n_width), np.arange(env.n_height))
    #   grid_states = np.concatenate([grid_x.reshape(-1,1), grid_y.reshape(-1,1)], axis=-1)
    # elif agent.state_dim == 1:
    #   grid_states = np.arange(env.observation_space.n).reshape(-1,1)
    # # print(states)
    # action_map = agent.phi(torch.tensor(np.concatenate([grid_states, actions[:,i:i+1]], axis=-1)).float()).detach().numpy()
    # print('action_{i}'.format(i=i), action_map)
    # axes[i].scatter(action_map[:,0], action_map[:,1], s=60, c=np.arange(states.shape[0]), cmap=cmaps[i], vmin=-1)

    axes[i].set_title(f'Action {i}')
    axes[i].set_xlabel('Feature dim 1')
    axes[i].set_ylabel('Feature dim 2')
    x_min, x_max, y_min, y_max = min(x_min, action_map[:,0].min()), max(x_max, action_map[:,0].max()),\
      min(y_min, action_map[:,1].min()), max(y_max, action_map[:,1].max())

  for i in range(env.action_space.n):
    axes[i].set_xlim(x_min-0.1, x_max+0.1)
    axes[i].set_ylim(y_min-0.1, y_max+0.1)
  plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2, wspace=0.5, hspace=0.5)

  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  # plt.show()
  plt.savefig(f'{fig_path}/phi_intrapolate.png')
  print(f'{fig_path}/phi_intrapolate.png')
  plt.close()
  return

def visualize_phi_by_state_intrapolate(args, env, agent):
  plt.rcParams.update({'font.size': 15})
  if agent.state_dim != 1:
    print('s_n:', env.n_height)
    print('a_n:', env.action_space.n)
    grid_x, grid_y = np.meshgrid(np.arange(env.n_width), np.arange(env.n_height))
    states = np.concatenate([grid_x.reshape(-1,1), grid_y.reshape(-1,1)], axis=-1)
  elif agent.state_dim == 1:
    states = np.arange(env.observation_space.n).reshape(-1,1)
  actions = np.arange(30).reshape(-1,1)/10
  fig, axes = plt.subplots(env.n_width,env.n_height, figsize=(10,10))
  axes = axes.flatten()
  x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
  for i in range(states.shape[0]):
    ax = axes[(env.n_height-states[i,1]-1)*env.n_width+states[i,0]]
    # 1D
    # ax = axes[i]
    # x = int(states[i]%env.n_width)
    # y = int(states[i]//env.n_width)
    # ax = axes[(env.n_height-y-1)*env.n_width+x]
    print(ax)
    input_sa = np.concatenate([np.repeat(states[i].reshape(1,-1),actions.shape[0],0), actions], axis=-1)
    state_map = agent.phi(torch.tensor(input_sa).float()).detach().numpy()
    state_map = PCA(n_components=2).fit_transform(state_map)
    x_min, x_max, y_min, y_max = min(x_min, state_map[:,0].min()), max(x_max, state_map[:,0].max()),\
      min(y_min, state_map[:,1].min()), max(y_max, state_map[:,1].max())
    ax.scatter(state_map[:,0], state_map[:,1], s=30, c=np.arange(actions.shape[0]), cmap='viridis', alpha=0.5)
      # ax.text(state_map[j,0], state_map[j,1], action_name[j])
    ax.set_title('State {i}'.format(i=states[i]))
    if env.grids.get_type(states[i][0], states[i][1]) == 1:
      ax.set_title('State {i}'.format(i=states[i]), color='red')
  for i in range(states.shape[0]):
    axes[i].set_xlim(x_min-0.1, x_max+0.1)
    axes[i].set_ylim(y_min-0.1, y_max+0.1)
  plt.subplots_adjust(wspace=0.5, hspace=0.5)
  fig.suptitle('Phi by state')
  plt.xlabel('Feature dim 1')
  plt.ylabel('Feature dim 2')

  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/phi_state_intrapolate.png')
  print(f'{fig_path}/phi_state_intrapolate.png')
  plt.close()
  return


def reorganize(phi_matrix:np.ndarray):
  """Input: (n_state, n_action, n_skill),
    Output:(n_skill, sqrt(n_state*n_action), sqrt(n_state*n_action))"""
  # print(phi_matrix.shape)
  assert phi_matrix.shape[0] == 9
  assert phi_matrix.shape[1] == 4
  width = height = int(np.sqrt(phi_matrix.shape[0]))
  n_action = phi_matrix.shape[1]
  action_width = int(np.sqrt(n_action))
  n_skill = phi_matrix.shape[2]
  # print('width:{} height:{} n_action:{} n_skill:{}'.format(width, height, n_action, n_skill))
  reorganize_grid_size = width * action_width
  new_matrix = np.zeros((n_skill, reorganize_grid_size, reorganize_grid_size))
  for skill in range(n_skill):
    skill_map = []
    for i in range(width):
      row = phi_matrix[i*width:(i+1)*width,:,skill]
      skill_map.append([x.reshape(2,2) for x in row])
    # print(skill_map)
    new_matrix[skill] = np.block(skill_map)
  # print(new_matrix.shape)
  # print(new_matrix)
  return new_matrix
  
def visualize_phi_matrix(args, env, agent):
  
  plt.rcParams.update({'font.size': 15})
  # if agent.state_dim != 1:
  #   print('s_n:', env.height)
  #   print('a_n:', env.action_space.n)
  #   grid_col, grid_row = np.meshgrid(np.arange(env.n_width), np.arange(env.n_height))
  #   states = np.concatenate([grid_row.reshape(-1,1), grid_col.reshape(-1,1)], axis=-1)
  #   # print(states)
  # elif agent.state_dim == 1:
  #   states = np.arange(env.observation_space.n).reshape(-1,1)

  # actions = np.repeat(np.arange(env.action_space.n).reshape(1,-1), states.shape[0], axis=0)
  # print('actions:', actions)
  n_state, n_action, n_task = agent.n_state, agent.n_action, agent.n_task
  # phi_matrix = np.zeros((env.action_space.n, states.shape[0], args.feature_dim))
  states_all, state_action_pairs, task_id_all = get_state_action_pair_onehot(env, n_state, n_action)
  phi_matrix, mu, u1, u2, w = get_phi_mu_u_w(args, env, agent, states_all, state_action_pairs, n_task)

  # np.save(f'{args.model_path}/phi_matrix.npy', phi_matrix)
  # np.save(f'{args.model_path}/mu.npy', mu)
  # print(f'{args.model_path}/phi_matrix.npy')
  # print(f'{args.model_path}/mu.npy')
  # n_action = env.action_space.n
  # n_state = env.n_height * env.n_width
  # phi_matrix = np.load('Phi_estimated.npy').reshape((n_state, n_action, 9))

  # n_state = env.n_height * env.n_width
  n_skill = 64
  action_name = ['U', 'D', 'R', 'L']
  phi_matrix_reorganize = reorganize(phi_matrix)

  z_w = agent.w(torch.eye(n_state)).detach().numpy()
  u1, u2 = agent.critic(torch.eye(n_state))
  z_u1 = u1.detach().numpy()
  z_u2 = u2.detach().numpy()
  fig_on_each_row = 10
  n_row = n_skill//fig_on_each_row + 1
  # action_reorganize = reorganize(np.expand_dims(actions,-1))
  fig, axes = plt.subplots(n_row, fig_on_each_row, figsize=(30,35))
  axes = axes.flatten()
  [x.axis('off') for x in axes]
  norm = colors.Normalize(phi_matrix_reorganize.min(), phi_matrix_reorganize.max())
  imgs = []
  for i in range(n_skill):
    axes[i].imshow(phi_matrix_reorganize[i], cmap='coolwarm', norm=norm)
    axes[i].set_title(f'Skill {i}', fontsize=30)
    axes[i].axis('off')
    mark_action_name(axes[i], n_state)
    print(i, phi_matrix_reorganize[i])
  plt.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.33, wspace=0.05, hspace=0.12)
  # mu_ax = fig.add_axes([0.05, 0.0, 0.7, 0.10])
  # mu_ax.imshow(mu, cmap='viridis')
  # mu_ax.xaxis.set_tick_params(labeltop=True, labelbottom=False, top=True, bottom=False, size=15)
  # mu_ax.yaxis.set_tick_params(labelleft=True, labelright=False, left=True, right=False, size=15)
  # mu_ax.set_xticks(np.arange(0, mu.shape[1], 5), np.arange(0, mu.shape[1], 5), fontsize=30)
  # mu_ax.set_yticks(np.arange(0, mu.shape[0], 1), np.arange(0, mu.shape[0], 1), fontsize=20)
  # u_ax = fig.add_axes([0.05, 0.11, 0.7, 0.10])
  # u_ax.imshow(z_u1, cmap='viridis')
  # u_ax.xaxis.set_tick_params(labeltop=True, labelbottom=False, top=True, bottom=False, size=15)
  # u_ax.yaxis.set_tick_params(labelleft=True, labelright=False, left=True, right=False, size=15)
  # u_ax.set_xticks(np.arange(0, z_u1.shape[1], 5), np.arange(0, z_u1.shape[1], 5), fontsize=30)
  # u_ax.set_yticks(np.arange(0, z_u1.shape[0], 1), np.arange(0, z_u1.shape[0], 1), fontsize=20)
  # w_ax = fig.add_axes([0.05, 0.22, 0.7, 0.10])
  # w_ax.imshow(z_w, cmap='viridis')
  # w_ax.xaxis.set_tick_params(labeltop=True, labelbottom=False, top=True, bottom=False, size=15)
  # w_ax.yaxis.set_tick_params(labelleft=True, labelright=False, left=True, right=False, size=15)
  # w_ax.set_xticks(np.arange(0, z_w.shape[1], 5), np.arange(0, z_w.shape[1], 5), fontsize=30)
  # w_ax.set_yticks(np.arange(0, z_w.shape[0], 1), np.arange(0, z_w.shape[0], 1), fontsize=20)
  colorbar_ax = fig.add_axes([0.45, 0.35, 0.15, 0.02])
  # colorbar_norm = colors.Normalize(z_w.min(), z_w.max())
  # mpl.colorbar.Colorbar(colorbar_ax, norm=norm, orientation='horizontal', cmap='coolwarm')
  n_segments = 256
  # boundaries = np.linspace(phi_matrix_reorganize.min(), phi_matrix_reorganize.max(), n_segments + 1)
  norm = colors.Normalize(phi_matrix_reorganize.min(), phi_matrix_reorganize.max())
  ticks = np.array([-3, 0, 4])
  cb = ColorbarBase(
      colorbar_ax,
      cmap='coolwarm',
      norm=norm,
      ticks=ticks,
      spacing='proportional',
      orientation='horizontal'
    )
  cb.ax.xaxis.set_major_locator(FixedLocator(ticks))
  cb.ax.tick_params(which='both', length=10)
  cb.set_ticklabels(ticks, fontsize=30)
  # cb.set_ticklabels(['0', '1'], fontsize=12)
  # grid_ax = fig.add_axes([0.77, 0.2, 0.2, 0.2])
  # grid_ax.imshow(np.zeros((env.n_height, env.n_width)), cmap=viridis')
  # grid_ax.axis('off')
  # for j in range(env.n_height):
  #   for k in range(env.n_height):
  #     grid_ax.text(k,j, f'{j*env.n_width+k}',horizontalalignment='center', verticalalignment='center', fontsize=50, c='white')
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'./gridworld/phi_matrix.pdf', dpi=100)
  print(f'./gridworld/phi_matrix.pdf')
  plt.close()


  # PCA
  fig, axes = plt.subplots(n_row, fig_on_each_row, figsize=(30,35))
  axes = axes.flatten()
  [x.axis('off') for x in axes]
  norm = colors.Normalize(phi_matrix_reorganize.min(), phi_matrix_reorganize.max())
  imgs = []
  phi_matrix_reduced = phi_matrix_reorganize.reshape(args.feature_dim, -1).T
  n_component = min(phi_matrix_reduced.shape[0], phi_matrix_reduced.shape[1])
  print('n_component:', n_component)
  pca = PCA(n_components=n_component)
  pca.fit(phi_matrix_reduced)
  phi_matrix_reduced = pca.transform(phi_matrix_reduced).T.reshape(n_component, phi_matrix_reorganize.shape[1], phi_matrix_reorganize.shape[2])

  phi_matrix_reduced[3] *= -1
  norm = colors.Normalize(phi_matrix_reduced.min(), phi_matrix_reduced.max())
  evr = pca.explained_variance_ratio_
  for i in range(n_component):
    im = axes[i].imshow(phi_matrix_reduced[i], cmap='coolwarm', norm=norm)
    print(i, phi_matrix_reduced[i])
    axes[i].set_title(f'PC {i} {evr[i]:.2f}', fontsize=30)
    axes[i].axis('off')
    mark_action_name(axes[i], n_state)
  fig.colorbar(im, ax=axes[35], orientation='vertical', fraction=0.1)
  plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.35, wspace=0.03, hspace=0.2)

  w_rotate = pca.transform(z_w)
  w_rotate[:,3] *= -1
  w_rotate = w_rotate[:,:8]
  w_ax = fig.add_axes([0.05, 0.3, 0.39, 0.20])
  im_w = w_ax.imshow(w_rotate, cmap='viridis')
  fig.colorbar(im_w, ax=w_ax, orientation='vertical', fraction=0.1)
  w_ax.xaxis.set_tick_params(labeltop=True, labelbottom=False, top=True, bottom=False, size=5)
  w_ax.yaxis.set_tick_params(labelleft=True, labelright=False, left=True, right=False, size=5)
  w_ax.set_xticks(np.arange(0, w_rotate.shape[1], 1), np.arange(0, w_rotate.shape[1], 1), fontsize=40)
  w_ax.set_yticks(np.arange(0, w_rotate.shape[0], 1), np.arange(0, w_rotate.shape[0], 1), fontsize=40)
  w_ax.set_title('PC Skill Index', fontsize=50)
  w_ax.set_ylabel('Task Index', fontsize=50)



  # u_rotate = pca.transform(z_u1)
  # mu_rotate = pca.transform(mu)
  # mu_ax = fig.add_axes([0.05, 0.2, 0.7, 0.10])
  # mu_ax.imshow(mu_rotate, cmap='viridis')
  # mu_ax.xaxis.set_tick_params(labeltop=True, labelbottom=False, top=True, bottom=False, size=20)
  # mu_ax.yaxis.set_tick_params(labelleft=True, labelright=False, left=True, right=False, size=20)
  # mu_ax.set_xticks(np.arange(0, mu_rotate.shape[1], 5), np.arange(0, mu_rotate.shape[1], 5), fontsize=30)
  # mu_ax.set_yticks(np.arange(0, mu_rotate.shape[0], 1), np.arange(0, mu_rotate.shape[0], 1), fontsize=30)
  # u_ax = fig.add_axes([0.05, 0.35, 0.7, 0.10])
  # u_ax.imshow(u_rotate, cmap='viridis')
  # u_ax.xaxis.set_tick_params(labeltop=True, labelbottom=False, top=True, bottom=False, size=20)
  # u_ax.yaxis.set_tick_params(labelleft=True, labelright=False, left=True, right=False, size=20)
  # u_ax.set_xticks(np.arange(0, u_rotate.shape[1], 5), np.arange(0, u_rotate.shape[1], 5), fontsize=30)
  # u_ax.set_yticks(np.arange(0, u_rotate.shape[0], 1), np.arange(0, u_rotate.shape[0], 1), fontsize=30)



  n_height = n_width = int(np.sqrt(w_rotate.shape[0]))
  grid_ax = fig.add_axes([0.77, 0.32, 0.18, 0.18])
  grid_ax.imshow(np.zeros((n_height, n_width)), cmap='summer')
  grid_ax.axis('off')
  for j in range(n_height):
    for k in range(n_height):
      grid_ax.text(k,j, f'{j*n_width+k}',horizontalalignment='center', verticalalignment='center', fontsize=50, c='white')
  grid_ax.vlines(np.arange(-0.5, n_width+0.5), -0.5, n_height-0.5, color='black', linewidth=5)
  grid_ax.hlines(np.arange(-0.5, n_height+0.5), -0.5, n_width-0.5, color='black', linewidth=5)
  
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'./gridworld/phi_matrix_PCA.pdf', dpi=100)
  print(f'./gridworld/phi_matrix_PCA.pdf')
  plt.close()

  fig, axes = plt.subplots(1,1, figsize=(10,10))
  axes.plot(evr[:15], marker='o', markersize=20, linewidth=6, color='red')
  axes.set_xlabel('Principal component', fontsize=30)
  axes.set_ylabel('Explained variance ratio', fontsize=30)
  axes.tick_params(axis='both', which='major', labelsize=30)
  plt.subplots_adjust(left=0.3, right=0.8, top=0.8, bottom=0.3)
  plt.savefig(f'./gridworld/phi_PCA_evr.pdf', dpi=100)
  print(f'./gridworld/phi_PCA_evr.pdf')

  return

def show_uw(args, env, agent):

  n_state, n_action, n_task = agent.n_state, agent.n_action, agent.n_task
  # phi_matrix = np.zeros((env.action_space.n, states.shape[0], args.feature_dim))
  states_all, state_action_pairs, task_id_all = get_state_action_pair_onehot(env, n_state, n_action)
  phi_matrix, mu, u1, u2, z_w = get_phi_mu_u_w(args, env, agent, states_all, state_action_pairs, n_task)
  print('z_w:', z_w[:, 0])
  fig, w_ax = plt.subplots(1,1, figsize=(10,3))
  im = w_ax.imshow(z_w, cmap='viridis')
  w_ax.xaxis.set_tick_params(labeltop=True, labelbottom=False, top=True, bottom=False, size=7)
  w_ax.yaxis.set_tick_params(labelleft=True, labelright=False, left=True, right=False, size=7)
  w_ax.set_xticks(np.arange(0, z_w.shape[1], 5), np.arange(0, z_w.shape[1], 5), fontsize=15)
  w_ax.set_yticks(np.arange(0, z_w.shape[0], 1), np.arange(0, z_w.shape[0], 1), fontsize=10)
  # w_ax.set_title('w', fontsize=10)
  w_ax.set_ylabel('Task Index', fontsize=15)
  w_ax.set_title('Skill Index', fontsize=15)
  cb_ax = fig.add_axes([0.93, 0.3, 0.01, 0.5])
  fig.colorbar(im, cax=cb_ax, orientation='vertical', fraction=0.1)
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'./gridworld/w.pdf', dpi=400)
  print(f'./gridworld/w.pdf')
  plt.close()

def visualize_mu_matrix(args, env, agent):
  plt.rcParams.update({'font.size': 15})
  # states = np.arange(env.observation_space.n).reshape(-1,1)
  # if agent.state_dim != 1:
  #   grid_col, grid_row = np.meshgrid(np.arange(env.n_width), np.arange(env.n_height))
  #   states = np.concatenate([grid_row.reshape(-1,1), grid_col.reshape(-1,1)], axis=-1)
  #   print(states)
    # states = np.array([[0,0],[0,1]])
#   actions = np.repeat(np.arange(env.action_space.n).reshape(1,-1), states.shape[0], axis=0)
  # elif agent.state_dim == 1:
  #   states = np.arange(env.observation_space.n).reshape(-1,1)
  print(states)
  state_map = agent.mu(torch.tensor(states).float()).detach().numpy()
  n_skill = 10
  fig, axes = plt.subplots(2, n_skill, figsize=(30,15))
  axes = axes.flatten()
  norm = colors.Normalize(state_map.min(), state_map.max())
  imgs = []
  for i in range(min(state_map.shape[1], n_skill)):
    imgs.append(axes[i].imshow(state_map[:,i].reshape(env.n_height,env.n_width), cmap='viridis'))
    axes[i].set_title(f'Feature{i}')
    axes[i].axis('off')

  n_component = min(state_map.shape[0], state_map.shape[1])
  state_map_reduced = PCA(n_components=n_component).fit_transform(state_map)
  for i in range(n_component):
    imgs.append(axes[i+n_skill].imshow(state_map_reduced[:,i].reshape(env.n_height,env.n_width), cmap='viridis'))
    axes[i+n_skill].set_title(f'PCA {i}')
    axes[i+n_skill].axis('off')

  for i in range(n_component, n_skill):
    imgs.append(axes[i+n_skill].imshow(np.zeros((env.n_height, env.n_width))))
    axes[i+n_skill].axis('off')
  # fig.colorbar(imgs[0], ax=axes, orientation='horizontal', fraction=0.1)
  plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.05, hspace=0)
  fig.suptitle('Mu matrix')
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/mu_matrix.png')
  print(f'{fig_path}/mu_matrix.png')
  plt.close()

def visualize_wu(args, env, agent):
  plt.rcParams.update({'font.size': 15})
  # grid_x, grid_y = np.meshgrid(np.arange(env.n_width), np.arange(env.n_height))
  # states = np.concatenate([grid_x.reshape(-1,1), grid_y.reshape(-1,1)], axis=-1)
  n_task = env.n_height * env.n_width
  z_w = agent.w(torch.eye(n_task)).detach().numpy().transpose().reshape(args.feature_dim, env.n_height, env.n_width)
  u1, u2 = agent.critic(torch.eye(n_task))
  z_u1 = u1.detach().numpy().transpose().reshape(args.feature_dim, env.n_height, env.n_width)
  z_u2 = u2.detach().numpy().transpose().reshape(args.feature_dim, env.n_height, env.n_width)
  print('w:{}'.format(z_w.shape))
  print(z_w)
  fig, axes = plt.subplots(3,args.feature_dim, figsize=(30,20))
  axes = axes.flatten()
  for i in range(args.feature_dim):
    axes[i].imshow(z_w[i], cmap='viridis')
    axes[i].set_title(f'w_{i}')
    axes[i].axis('off')
    axes[i+args.feature_dim].imshow(z_u1[i], cmap='viridis')
    axes[i+args.feature_dim].set_title(f'u1_{i}')
    axes[i+args.feature_dim].axis('off')
    axes[i+2*args.feature_dim].imshow(z_u2[i], cmap='viridis')
    axes[i+2*args.feature_dim].set_title(f'u2_{i}')
    axes[i+2*args.feature_dim].axis('off')
  plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.05, hspace=0.2)
  fig.suptitle('Coefficients')
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/wu.png')
  print(f'{fig_path}/wu.png')
  plt.close()

  z_w = agent.w(torch.eye(n_task)).detach().numpy()
  z_u1 = u1.detach().numpy()
  z_u2 = u2.detach().numpy()
  fig, axes = plt.subplots(3,1, figsize=(30,10))
  axes[0].imshow(z_w, cmap='viridis')
  axes[0].set_title(f'w')
  grid_col, grid_row = np.meshgrid(np.arange(env.n_width), np.arange(env.n_height))
  states_all = np.concatenate([grid_row.reshape(-1,1), grid_col.reshape(-1,1)], axis=-1)
  for i in range(states_all.shape[0]):
    axes[0].text(0,i, f'{states_all[i,1],states_all[i,0]}',horizontalalignment='center', verticalalignment='center', fontsize=10, c='white')
  axes[0].set_xticks(np.arange(0, z_w.shape[1], 5), np.arange(0, z_w.shape[1], 5))
  axes[0].xaxis.set_tick_params(labelbottom=True, labeltop=True, bottom=True, top=True)
  axes[0].set_yticks(np.arange(states_all.shape[0]), np.arange(states_all.shape[0]))

  axes[1].imshow(z_u1, cmap='viridis')
  axes[1].set_title(f'u1')
  axes[1].set_xticks(np.arange(0, z_w.shape[1], 5), np.arange(0, z_w.shape[1], 5))
  axes[1].xaxis.set_tick_params(labelbottom=True, labeltop=True, bottom=True, top=True)
  axes[1].set_yticks(np.arange(states_all.shape[0]), np.arange(states_all.shape[0]))

  axes[2].imshow(z_u2, cmap='viridis')
  axes[2].set_title(f'u2')
  axes[2].set_xticks(np.arange(0, z_w.shape[1], 5), np.arange(0, z_w.shape[1], 5))
  axes[2].xaxis.set_tick_params(labelbottom=True, labeltop=True, bottom=True, top=True)
  axes[2].set_yticks(np.arange(states_all.shape[0]), np.arange(states_all.shape[0]))
  plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.1, hspace=0.4)
  legend = fig.add_axes([0.7, 0.2, 0.4, 0.4])
  legend.imshow(np.ones((env.n_height, env.n_width)), cmap='viridis')
  for j in range(states_all.shape[0]):
    for k in range(states_all.shape[0]):
      legend.text(k,j, f'{j*env.n_width+k}',horizontalalignment='center', verticalalignment='center', fontsize=30, c='white')
  legend.axis('off')
  fig.suptitle('Coefficients')
  plt.savefig(f'{fig_path}/wu_single.png')
  print(f'{fig_path}/wu_single.png')
  plt.close()

def visualize_r_q_P(args, env, agent):
  plt.rcParams.update({'font.size': 15})
  grid_col, grid_row = np.meshgrid(np.arange(env.n_width), np.arange(env.n_height))
  states = np.concatenate([grid_row.reshape(-1,1), grid_col.reshape(-1,1)], axis=-1)
  n_state = env.n_height * env.n_width
  n_task = n_state
  state_action_pairs = np.concatenate([np.repeat(states, env.action_space.n, axis=0), np.tile(np.arange(env.action_space.n).reshape(-1,1), (states.shape[0],1))], axis=-1)
  # print(state_action_pairs)
  phi_matrix = agent.phi(torch.tensor(state_action_pairs).float()).detach().numpy().reshape(n_state, env.action_space.n, args.feature_dim)
  phi_matrix_reorganize = reorganize(phi_matrix)
  n_task = env.n_height * env.n_width
  z_w = agent.w(torch.eye(n_task)).detach().numpy()
  u1, u2 = agent.critic(torch.eye(n_task))
  mu = agent.mu(torch.tensor(states).float()).detach().numpy()
  actions = np.repeat(np.arange(env.action_space.n).reshape(1,-1), states.shape[0], axis=0)
  action_name = ['U', 'D', 'R', 'L']
  action_reorganize = reorganize(np.expand_dims(actions,-1))
  fig, axes = plt.subplots(2, n_task, figsize=(30,10))
  axes = axes.flatten()
  for i in range(n_task):
    # print('z_w:', z_w[i].shape)
    # print('phi_matrix_reorganize:', phi_matrix_reorganize.shape)
    r = np.sum(phi_matrix_reorganize * z_w[i].reshape(-1,1,1), axis=0)
    axes[i].imshow(r, cmap='viridis')
    axes[i].set_title(f'r_{states[i]}')
    axes[i].axis('off')
    for j in range(action_reorganize.shape[1]):
      for k in range(action_reorganize.shape[2]):
        axes[i].text(k,j, f'{action_name[int(action_reorganize[0,j,k])]}',horizontalalignment='center', verticalalignment='center', fontsize=160/n_task)

    q = np.sum(phi_matrix_reorganize * u1[i].detach().numpy().reshape(-1,1,1), axis=0)
    axes[i+n_task].imshow(q, cmap='viridis')
    axes[i+n_task].set_title(f'q1_{states[i]}')
    axes[i+n_task].axis('off')
    for j in range(action_reorganize.shape[1]):
      for k in range(action_reorganize.shape[2]):
        axes[i+n_task].text(k,j, f'{action_name[int(action_reorganize[0,j,k])]}',horizontalalignment='center', verticalalignment='center', fontsize=160/n_task)
    
  plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.05, hspace=0.05)
  fig.suptitle('Reward & Q')
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/r_q.png')
  plt.close()
  print(f'{fig_path}/r_q.png')

  fig, axes = plt.subplots(1, n_state, figsize=(30,10))
  axes = axes.flatten()
  P_all = np.zeros((n_state, 6, 6))
  for i in range(n_state):
    P = np.sum(phi_matrix_reorganize * mu[i].reshape(-1,1,1), axis=0)
    P_all[i] = P
    axes[i].imshow(P, cmap='viridis')
    axes[i].set_title(f'P_{states[i]}')
    axes[i].axis('off')
    for j in range(action_reorganize.shape[1]):
      for k in range(action_reorganize.shape[2]):
        axes[i].text(k,j, f'{action_name[int(action_reorganize[0,j,k])]}',horizontalalignment='center', verticalalignment='center', fontsize=160/n_state)
  plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.05, hspace=0.05)
  fig.suptitle('Transition matrix')
  plt.savefig(f'{fig_path}/P.png')
  plt.close()
  print(f'{fig_path}/P.png')
  P_all = P_all.transpose(1,2,0).reshape(-1, n_state)
  print(P_all)


def ground_truth_P(args, env):
  if os.path.exists('P.npy'):
    return np.load('P.npy')
  grid_col, grid_row = np.meshgrid(np.arange(env.n_width), np.arange(env.n_height))
  states = np.concatenate([grid_row.reshape(-1,1), grid_col.reshape(-1,1)], axis=-1)
  actions_list = np.array([[-1,0],[1,0],[0,1],[0,-1]])
  next_state_one_hot = np.zeros((states.shape[0], actions_list.shape[0], states.shape[0]))
  for i in range(states.shape[0]):
    for j in range(actions_list.shape[0]):
      next_state = states[i] + actions_list[j]
      next_state = np.clip(next_state, 0, [env.n_height-1, env.n_width-1])
      next_state_idx = next_state[0] * env.n_width + next_state[1]
      next_state_one_hot[i,j,next_state_idx] = 1
  np.save('P.npy', next_state_one_hot)
  return next_state_one_hot
      
def ground_truth_r(args, env):
  n_task = args.n_task
  if os.path.exists('r.npy'):
    return np.load('r.npy')
  grid_col, grid_row = np.meshgrid(np.arange(env.n_width), np.arange(env.n_height))
  states = np.concatenate([grid_row.reshape(-1,1), grid_col.reshape(-1,1)], axis=-1)
  actions_list = np.array([[-1,0],[1,0],[0,1],[0,-1]])
  r_one_hot = np.zeros((states.shape[0], actions_list.shape[0], n_task))
  for i in range(states.shape[0]):
    for j in range(actions_list.shape[0]):
      next_state = states[i] + actions_list[j]
      next_state = np.clip(next_state, 0, [env.n_height-1, env.n_width-1])
      next_state_idx = next_state[0] * env.n_width + next_state[1]
      r_one_hot[i,j,next_state_idx] = 1    
  np.save('r.npy', r_one_hot)
  return r_one_hot


def mark_action_name(ax, n_state):
  if os.path.exists('action_reorganize.npy'):
    action_reorganize = np.load('action_reorganize.npy')
  else:
    actions = np.repeat(np.arange(env.action_space.n).reshape(1,-1), n_state, axis=0)
    action_reorganize = reorganize(np.expand_dims(actions,-1))
    np.save('action_reorganize.npy', action_reorganize)
  action_name = ['U', 'D', 'R', 'L']
  for j in range(action_reorganize.shape[1]):
    for k in range(action_reorganize.shape[2]):
      ax.text(k,j, f'{action_name[int(action_reorganize[0,j,k])]}',horizontalalignment='center', verticalalignment='center', fontsize=30)

def get_state_action_pairs(env):
  grid_col, grid_row = np.meshgrid(np.arange(env.n_width), np.arange(env.n_height))
  states = np.concatenate([grid_row.reshape(-1,1), grid_col.reshape(-1,1)], axis=-1)
  n_state = env.n_height * env.n_width
  state_action_pairs = np.concatenate([np.repeat(states, env.action_space.n, axis=0), np.tile(np.arange(env.action_space.n).reshape(-1,1), (states.shape[0],1))], axis=-1)
  task_id_all = np.eye(n_state)
  return states, state_action_pairs, task_id_all

def get_state_action_pair_onehot(env, n_state, n_action):
  states = np.eye(n_state)
  # n_state = env.n_height * env.n_width
  actions = np.eye(n_action)
  state_action_pairs = np.concatenate([np.repeat(states, n_action, axis=0), np.tile(actions, (states.shape[0],1))], axis=-1)
  task_id_all = np.eye(n_state)
  return states, state_action_pairs, task_id_all


def get_phi_mu_u_w(args, env, agent, states, state_action_pairs, n_task):
  # if hasattr(agent.phi, 'matrix'):
  #   return get_phi_mu_u_w_matrix(args, env, agent, states, state_action_pairs)
  n_state = states.shape[0]
  n_action = state_action_pairs.shape[0]//n_state
  # print('state-action pair:', state_action_pairs.shape)
  phi_matrix = agent.phi(torch.tensor(state_action_pairs).float()).detach().numpy().reshape(n_state, n_action, args.feature_dim)

  # phi_matrix_not_flatten = agent.phi(torch.tensor(state_action_pairs).float()).detach().numpy()
  # print('covariance:', phi_matrix_not_flatten.T@phi_matrix_not_flatten)
  mu = agent.mu(torch.tensor(states).float()).detach().numpy()/n_state
  u1, u2 = agent.critic(torch.eye(n_task))
  u1 = u1.detach().numpy()
  u2 = u2.detach().numpy()
  w = agent.w(torch.eye(n_task)).detach().numpy()
  return phi_matrix, mu, u1, u2, w

def get_phi_mu_u_w_matrix(args, env, agent, states, state_action_pairs):
  n_state = env.n_height * env.n_width
  # phi_matrix = agent.phi(torch.tensor(state_action_pairs))
  # phi_matrix = agent.phi.matrix.detach().numpy().reshape(n_state, env.action_space.n, args.feature_dim)
  phi_matrix = agent.phi.embedding.weight.detach().numpy().reshape(n_state, env.action_space.n, args.feature_dim)
  print('phi:', phi_matrix.shape)
  # mu = agent.mu.matrix.detach().numpy()
  mu = agent.mu.embedding.weight.detach().numpy()/n_state
  print('mu:', mu.shape)
  u1, u2 = agent.critic(torch.eye(env.n_height*env.n_width))
  u1 = u1.detach().numpy()
  u2 = u2.detach().numpy()
  w = agent.w(torch.eye(env.n_height*env.n_width)).detach().numpy()
  return phi_matrix, mu, u1, u2, w



def get_v(args, env, agent, states, task_id_onehot, q, alpha):
  n_state, n_action = agent.n_state, agent.n_action
  pi_input = np.concatenate([states, task_id_onehot.reshape(1,-1).repeat(n_state, axis=0)], axis=-1)
  action_log_pi = agent.actor.evaluate_matrix(torch.tensor(pi_input).float()).detach().numpy()
  assert action_log_pi.shape == (n_state, n_action)
  V = np.sum(np.exp(action_log_pi) * (q - alpha * action_log_pi), axis=-1)
  return V

def check_r(args, env, agent):
  np.set_printoptions(precision=2)
  alpha = 1
  discount = 0.5
  n_state, n_action, n_task = agent.n_state, agent.n_action, agent.n_task
  n_height = int(np.sqrt(n_state))
  # states, state_action_pairs, task_id_all = get_state_action_pairs(env)
  states, state_action_pairs, task_id_all = get_state_action_pair_onehot(env, n_state, n_action)
  phi_matrix, mu, u1, u2, w = get_phi_mu_u_w(args, env, agent, states, state_action_pairs, n_task)
  phi_operand1 = phi_matrix.reshape(n_state, n_action, 1, args.feature_dim)
  mu_operand2 = mu.transpose()
  P = phi_operand1 @ mu_operand2
  P = P.reshape(n_state, n_action, n_state)
  # normalized_P = P/P.sum(-1, keepdims=True)
  # real_P = ground_truth_P(args, env)
  real_r = ground_truth_r(args, env)
  reorganize_r = reorganize(real_r)
  phiw_r_matrix = phi_matrix@w.T
  pred_r_matrix = np.zeros((n_state, n_action, n_task))
  fig, axes = plt.subplots(n_height, n_height*2, figsize=(30,20))
  axes = axes.flatten()
  for i in range(n_task):
    q1 = phi_matrix@u1[i]
    q2 = phi_matrix@u2[i]
    q = np.minimum(q1, q2)
    assert q.shape == (n_state, n_action)
    V = get_v(args, env, agent, states, task_id_all[i], q1, alpha)
    Ev = P@V
    assert Ev.shape == (n_state, n_action)
    pred_r = q - discount * Ev
    pred_r_matrix[:,:,i] = pred_r
    phiw_r = phiw_r_matrix[...,i]
    print('pred_r:', pred_r)
    print('r:', phiw_r)
    print('mean relative error:', np.abs((pred_r-phiw_r)/phiw_r).mean())
    reorganize_pred_r = reorganize(pred_r.reshape(n_state, n_action, 1))
    reorganize_phiw_r = reorganize(phiw_r.reshape(n_state, n_action, 1))
    r_min = np.min([reorganize_pred_r[0].min(), reorganize_phiw_r[0].min(), reorganize_r[i].min()])
    r_max = np.max([reorganize_pred_r[0].max(), reorganize_phiw_r[0].max(), reorganize_r[i].max()])
    norm = colors.Normalize(r_min, r_max)
    # axes[i*3].imshow(reorganize_pred_r[0], cmap='viridis', norm=norm)
    axes[i*2].imshow(reorganize_phiw_r[0], cmap='summer')
    axes[i*2].vlines(np.arange(0-0.5, n_height*2+0.5, 2), ymin=-0.5, ymax=n_height*2-0.5, color='black', linewidth=8)
    axes[i*2].hlines(np.arange(0-0.5, n_height*2+0.5, 2), xmin=-0.5, xmax=n_height*2-0.5, color='black', linewidth=8)
    axes[i*2+1].imshow(reorganize_r[i], cmap='summer')
    axes[i*2+1].vlines(np.arange(0-0.5, n_height*2+0.5, 2), ymin=-0.5, ymax=n_height*2-0.5, color='black', linewidth=8)
    axes[i*2+1].hlines(np.arange(0-0.5, n_height*2+0.5, 2), xmin=-0.5, xmax=n_height*2-0.5, color='black', linewidth=8)
  # axes[0].set_title('pred_r', fontsize=50)
  for i in range(3):
    axes[2*i].set_title('Predict R', fontsize=70)
    axes[2*i+1].set_title('Real R', fontsize=70)
  # cos_phiw_real = get_cos(phiw_r_matrix, real_r)
  # cos_pred_real = get_cos(pred_r_matrix, real_r)
  for axis in axes:
    axis.axis('off')
    # mark_action_name(axis, n_state)
  plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.05, hspace=0.05)
  # fig.suptitle('pred_r & r, cos(phiw, real):{:.2f}, cos(pred, real):{:.2f}'.format(cos_phiw_real, cos_pred_real), fontsize=50)
  fig_path = f'{args.model_path.replace("model", "figure")}'
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'./gridworld/pred_r_r.pdf',dpi=400)
  print(f'./gridworld/pred_r_r.pdf')
  plt.close()



def check_r_linechart(args, env, agent):
  np.set_printoptions(precision=2)
  # n_state = env.n_height * env.n_width
  # n_task = n_state
  alpha = 1
  discount = 0.5
  n_state, n_action, n_task = agent.n_state, agent.n_action, agent.n_task
  # states, state_action_pairs, task_id_all = get_state_action_pairs(env)
  states, state_action_pairs, task_id_all = get_state_action_pair_onehot(env, n_state, n_action)
  phi_matrix, mu, u1, u2, w = get_phi_mu_u_w(args, env, agent, states, state_action_pairs, n_task)
  phi_operand1 = phi_matrix.reshape(n_state, n_action, 1, args.feature_dim)
  mu_operand2 = mu.transpose()
  # P = phi_matrix @ (mu.T)
  P = phi_operand1 @ mu_operand2
  # normalized_P = P/P.sum(-1, keepdims=True)
  # real_P = ground_truth_P(args, env)
  # real_P = np.load('trans_probs.npy')
  # real_P = real_P.reshape(n_state, n_action, 1, n_state)
  real_r = ground_truth_r(args, env).reshape(n_state, n_action, n_task, 1)
  # real_P = ground_truth_P(args, env).reshape(n_state, env.action_space.n, 1, n_state)
  
  phiw_r_matrix = phi_matrix@w.T
  # state_only_r = phiw_r_matrix.mean(1).flatten()
  phiw_r_matrix = phiw_r_matrix.reshape(n_state, n_action, n_task, 1) 

  # aggregate_r = (P*phiw_r_matrix).sum(0).sum(0).flatten()
  # np.save(f'{args.model_path}/aggregate_r.npy', aggregate_r)
  # real_Pr = (real_P*real_r).sum(0).sum(0)
  fig, axes = plt.subplots(1, 1, figsize=(70,20))
  # axes.plot(aggregate_r[:n_state], label='0', linewidth=5)
  # print(aggregate_r[:n_state], np.argsort(aggregate_r[:n_state]))
  # axes.plot(aggregate_r[n_state:2*n_state], label='1', linewidth=5)
  # print(aggregate_r[n_state:2*n_state], np.argsort(aggregate_r[n_state:2*n_state]))
  # axes.plot(aggregate_r[2*n_state:], label='2', linewidth=5)
  # print(aggregate_r[2*n_state:], np.argsort(aggregate_r[2*n_state:]))
  axes.plot(phiw_r_matrix.flatten(), label='pred_r', linewidth=5)
  axes.plot(real_r.flatten(), label='ground_truth', linewidth=5)
  reward_trajectory = [0,2,6,13,28,57,116]
  # for x in reward_trajectory:
  #   axes.vlines(x, aggregate_r.min(), aggregate_r.max(), 'r', 'dashed', linewidth=5)
  #   axes.text(x, aggregate_r.max()*1.4, f'{x}', horizontalalignment='center', verticalalignment='top', fontsize=50)
  axes.legend(fontsize=60)
  axes.set_xlabel('state-action-task pair', fontsize=50)
  axes.set_ylabel('reward', fontsize=50)
  axes.tick_params(axis='both', which='major', labelsize=40)
  # cos_phiw_real = get_cos(phiw_r_matrix, real_r)
  corrcoef = np.corrcoef(phiw_r_matrix.flatten(), real_r.flatten())[0,1]
  fig.suptitle('pred_r & r, corrcoef:{:.2f}'.format(corrcoef), fontsize=50)
  fig_path = f'{args.model_path.replace("model", "figure")}'
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/pred_r_r_lineplot.png')
  print(f'{fig_path}/pred_r_r_lineplot.png')
  plt.close()

def get_cos(A, B):
  assert A.shape == B.shape
  a = A.flatten()
  b = B.flatten()
  return np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))

def check_r_vi(args, env, agent):
  R = agent.R.detach().cpu().numpy()
  n_state = env.n_height * env.n_width
  n_task = R.shape[-1]
  real_r = ground_truth_r(args, env)
  cos = get_cos(R, real_r)
  corrcoef = np.corrcoef(R.flatten(), real_r.flatten())[0,1]
  fig, axes = plt.subplots(1, 1, figsize=(50,20))
  axes.plot(R.flatten(), label='pred_r', linewidth=5)
  axes.plot(real_r.flatten(), label='ground_truth', linewidth=5)
  axes.legend()
  fig.suptitle('pred_r & r, cos:{:.2f}, corrcoef:{:.2f}'.format(cos, corrcoef), fontsize=50)
  fig_path = f'{args.model_path.replace("model", "figure")}'
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/pred_r_r_vi.png')
  print(f'{fig_path}/pred_r_r_vi.png')
  plt.close()
  fig, axes = plt.subplots(3, 6, figsize=(30,20))
  axes = axes.flatten()
  for i in range(R.shape[-1]):
    reorganize_R = reorganize(R[...,i:i+1])
    reorganize_real_r = reorganize(real_r[...,i:i+1])
    r_min = np.min([reorganize_R[0].min(), reorganize_real_r[0].min()])
    r_max = np.max([reorganize_R[0].max(), reorganize_real_r[0].max()])
    norm = colors.Normalize(r_min, r_max)
    axes[i*2].imshow(reorganize_R[0], cmap='viridis', norm=norm)
    axes[i*2+1].imshow(reorganize_real_r[0], cmap='viridis', norm=norm)
  axes[0].set_title('pred_r', fontsize=50)
  axes[1].set_title('r', fontsize=50)
  for axis in axes:
    axis.axis('off')
    mark_action_name(axis, n_state)
  plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.05, hspace=0.05)
  fig.suptitle('pred_r & r Value Iteration', fontsize=50)
  fig_path = f'{args.model_path.replace("model", "figure")}'
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/pred_r_r_vi_matrix.png')
  print(f'{fig_path}/pred_r_r_vi_matrix.png')
  plt.close()
  P = ground_truth_P(args, env)
  print(real_r.shape, R.shape, P.shape)
  P = P.reshape(n_state, env.action_space.n, 1, n_state)
  real_r = real_r.reshape(n_state, env.action_space.n, n_task, 1)
  R = R.reshape(n_state, env.action_space.n, n_task, 1)
  state_only_real_r = (P*real_r).sum(0).sum(0)
  state_only_R = (P*R).sum(0).sum(0)
  fig, axes = plt.subplots(1, 1, figsize=(50,20))
  axes.plot(state_only_R.flatten(), label='pred_r', linewidth=5)
  axes.plot(state_only_real_r.flatten(), label='ground_truth', linewidth=5)
  axes.legend()
  cos = get_cos(state_only_R, state_only_real_r)
  corrcoef = np.corrcoef(state_only_R.flatten(), state_only_real_r.flatten())[0,1]
  fig.suptitle('pred_r & r, cos:{:.2f}, corrcoef:{:.2f}'.format(cos, corrcoef), fontsize=50)
  fig_path = f'{args.model_path.replace("model", "figure")}'
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/pred_r_r_vi_state_only.png')
  print(f'{fig_path}/pred_r_r_vi_state_only.png')
  plt.close()
  fig, axes = plt.subplots(3, 6, figsize=(30,20))
  axes = axes.flatten()
  for i in range(n_task):
    print(state_only_R[i])
    axes[2*i].imshow(state_only_R[i].reshape(env.n_height, env.n_width), cmap='viridis')
    axes[2*i+1].imshow(state_only_real_r[i].reshape(env.n_height, env.n_width), cmap='viridis')
  axes[0].set_title('pred_r', fontsize=50)
  axes[1].set_title('r', fontsize=50)
  for axis in axes:
    axis.axis('off')
  plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.05, hspace=0.05)
  fig.suptitle('pred_r & r Value Iteration', fontsize=50)
  fig_path = f'{args.model_path.replace("model", "figure")}'
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/pred_r_r_vi_state_only_matrix.png')
  print(f'{fig_path}/pred_r_r_vi_state_only_matrix.png')
  plt.close()




def check_P(args, env, agent):
  np.set_printoptions(precision=2, suppress=True)
  n_state, n_action, n_task = agent.n_state, agent.n_action, agent.n_task
  # states, state_action_pairs, task_id_all = get_state_action_pairs(env)
  states, state_action_pairs, task_id_all = get_state_action_pair_onehot(env, n_state, n_action)

  alpha = 0.1
  discount = 0.5
  phi_matrix, mu, u1, u2, w = get_phi_mu_u_w(args, env, agent, states, state_action_pairs, n_task)
  # phi_matrix, mu = pca_transform(phi_matrix, mu)

  phi_operand1 = phi_matrix.reshape(n_state, n_action, 1, -1)
  mu_operand2 = mu.transpose()
  P = phi_operand1 @ mu_operand2
  assert P.shape == (n_state, n_action, 1, n_state)
  P = P.reshape(n_state, n_action, n_state)
  normalized_P = P/P.sum(-1, keepdims=True)
  real_P = ground_truth_P(args, env)
  # estimated_P = get_estimate_P(args, env, agent)
  estimated_P = real_P
  cos_predicted_real = get_cos(P, real_P)
  cos_normalized_real = get_cos(normalized_P, real_P)
  fig, axes = plt.subplots(3, 3*3, figsize=(50,20))
  axes = axes.flatten()
  v_min = min(P.min(), estimated_P.min(), real_P.min())
  v_max = max(P.max(), estimated_P.max(), real_P.max())
  norm = colors.Normalize(v_min, v_max)
  for i, p in enumerate([P, estimated_P, real_P]):
    reorganize_p = reorganize(p)
    for s in range(n_task):
      axes[s*3+i].imshow(reorganize_p[s], cmap='viridis', norm=norm)
      mark_action_name(axes[s*3+i], n_state)

  axes[0].set_title('P', fontsize=50)
  axes[1].set_title('estimated P', fontsize=50)
  axes[2].set_title('real P', fontsize=50)
  [axis.axis('off') for axis in axes]
  plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.05, hspace=0.05)
  fig.suptitle('cos(predicted, real):{:.2f}, cos(normalized, real):{:.2f}'.format(cos_predicted_real, cos_normalized_real), fontsize=50)
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/predict_P_P.png')
  print(f'{fig_path}/predict_P_P.png')
  plt.close()
  print('s-a-s:',state_action_pairs)
  print('P:',P)
  print('real_P:', real_P)
  print(np.abs((P-real_P)/P).mean())  
  print('normalized P:', P/P.sum(-1, keepdims=True))

  normalized_P = P/P.sum(-1, keepdims=True)
  KL = np.sum(real_P * np.log(real_P/(np.clip(normalized_P,1e-3,1))), axis=-1)
  print(np.abs((P/P.sum(-1, keepdims=True)-real_P)/P).mean())
  print('KL:', KL)  

  fig, axes = plt.subplots(1, 1, figsize=(30,10))
  axes.plot(P.flatten(), label='P', linewidth=5)
  axes.plot(real_P.flatten(), label='real_P', linewidth=5)
  axes.legend(fontsize=20)
  corrcoef = np.corrcoef(P.flatten(), real_P.flatten())[0,1]
  fig.suptitle(f'P & real_P, corr:{corrcoef}', fontsize=50)
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/predict_P_P_lineplot.png')
  print(f'{fig_path}/predict_P_P_lineplot.png')
  plt.close()


  return

def check_P_corr(args, env, agent):
  np.set_printoptions(precision=3, suppress=True)
  # states, state_action_pairs, task_id_all = get_state_action_pairs(env)
  
  n_state, n_action, n_task = agent.n_state, agent.n_action, agent.n_task
  # print(n_state, n_action, n_task)
  states, state_action_pairs, task_id_all = get_state_action_pair_onehot(env, n_state, n_action)
  alpha = 1
  discount = 0.5
  phi_matrix, mu, u1, u2, w = get_phi_mu_u_w(args, env, agent, states, state_action_pairs, n_task)
  # phi_matrix, mu = pca_transform(phi_matrix, mu)

  phi_operand1 = phi_matrix.reshape(n_state, n_action, 1, -1)
  mu_operand2 = mu.transpose()
  P = phi_operand1 @ mu_operand2
  assert P.shape == (n_state, n_action, 1, n_state)
  P = P.reshape(n_state, n_action, n_state)
  real_P = np.load('trans_probs.npy')
  corr = np.corrcoef(P.flatten(), real_P.flatten())[0,1]
  print('corr:', corr)
  prev_half_corr = np.corrcoef(P[:n_state//2].flatten(), real_P[:n_state//2].flatten())[0,1]
  print('prev_half_corr:', prev_half_corr)
  next_half_corr = np.corrcoef(P[n_state//2:].flatten(), real_P[n_state//2:].flatten())[0,1]
  print('next_half_corr:', next_half_corr)
  # print(P[116, 0, 116], real_P[116, 0, 116])
  fig, axes = plt.subplots(1, 1, figsize=(30,10))
  axes.plot(real_P.flatten(), label='real_P', linewidth=3, alpha=0.8)
  axes.plot(P.flatten(), label='P', linewidth=3, alpha=0.8)

  axes.legend(fontsize=20)
  fig.suptitle(f'P & real_P, corr:{corr}', fontsize=50)
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/predict_P_P_lineplot.png')
  print(f'{fig_path}/predict_P_P_lineplot.png')
  plt.close()
  return

def check_w_u_equation(args, env, agent):
  np.set_printoptions(precision=2)
  states, state_action_pairs, task_id_all = get_state_action_pairs(env)
  n_state = env.n_height * env.n_width
  n_task = n_state
  alpha = 0.1
  discount = 0.5
  phi_matrix, mu, u1, u2, w = get_phi_mu_u_w(args, env, agent, states, state_action_pairs)
  # fig, axes = plt.subplots(1, 1, figsize=(10,10))
  # phi_matrix_flatten = phi_matrix.reshape(n_state*env.action_space.n, args.feature_dim)
  # cov = phi_matrix_flatten.T@phi_matrix_flatten
  # axes.imshow(cov, cmap='viridis')
  # fig.suptitle('Covariance matrix of phi')
  # fig_path = args.model_path.replace('model', 'figure')
  # if not os.path.exists(fig_path):
  #   os.makedirs(fig_path)
  # plt.savefig(f'{fig_path}/cov_phi.png')


  pred_w = np.zeros((n_task, args.feature_dim))
  for i in range(n_task):
    Q1 = phi_matrix @ u1[i]
    Q2 = phi_matrix @ u2[i]
    Q = np.minimum(Q1, Q2)
    V = get_v(args, env, agent, states, task_id_all[i], Q, alpha)
    assert V.shape == (n_state,)

    # print('Q1:', Q1.shape, Q1)  
    # V = np.log(np.mean(np.exp(Q1), axis=1, keepdims=True))
    # V = logsumexp(Q1/alpha, axis=1, keepdims=True) * alpha
    # print('V1_all:', V.shape, V)
    # print('w:', w.shape)
    # print('u1:', u1)
    # print('mu:', mu.shape)
    pred_w[i] = u1[i] - (V@mu)*discount
    pred_w_i = pred_w[i]
    error = np.abs(pred_w_i-w[i])
    error = np.where(np.abs(w[i]) < 1e-2, error, error/np.abs(w[i]))
    print('pred_w:{a},\n w:{b},\n diff:{c}'.format(a=pred_w_i, b=w[i], c=np.mean(error)))
  cos_total = get_cos(pred_w, w)
  fig, axes = plt.subplots(1, 2, figsize=(30,10))
  norm_min = min(w.min(), pred_w.min())
  norm_max = max(w.max(), pred_w.max())
  norm = colors.Normalize(norm_min, norm_max)
  axes[0].imshow(w, cmap='viridis', norm=norm)
  axes[0].set_title('w', fontsize=50)
  axes[1].imshow(pred_w, cmap='viridis', norm=norm)
  axes[1].set_title('pred_w', fontsize=50)  
  plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.05, hspace=0.05)
  fig.suptitle(f'cos(predict, real):{cos_total}', fontsize=50)
  fig_path = f'{args.model_path.replace("model", "figure")}/balance'
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/pred_w_w.png')
  print(f'{fig_path}/pred_w_w.png')
  plt.close()



def check_action_likelihood_vi(args, env, agent):
  np.set_printoptions(precision=2)
  states, state_action_pairs, task_id_all = get_state_action_pairs(env)
  n_state = env.n_height * env.n_width
  n_task = n_state
  alpha = 1
  discount = 0.5
  Q = agent.Q
  assert Q.shape == (n_state, env.action_space.n, n_task)
  action_log_pi = torch.log_softmax(Q/alpha, dim=1).detach().cpu().numpy()
  np.save('action_log_pi_vi.npy', action_log_pi)
  assert action_log_pi.shape == (n_state, env.action_space.n, n_task)
  fig, axes = plt.subplots(3, 3, figsize=(30,20))
  axes = axes.flatten()
  for i in range(n_task):
    reorganize_logpi = reorganize(action_log_pi[...,i:i+1])
    axes[i].imshow(reorganize_logpi[0], cmap='viridis')
  for axis in axes:
    axis.axis('off')
    mark_action_name(axis, n_state)
  plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.05, hspace=0.05)
  fig.suptitle('Action likelihood', fontsize=50)
  fig_path = f'{args.model_path.replace("model", "figure")}'
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/action_likelihood_vi.png')
  print(f'{fig_path}/action_likelihood_vi.png')
  plt.close()


def check_action_likelihood(args, env, agent):
  np.set_printoptions(precision=2)
  states, state_action_pairs, task_id_all = get_state_action_pairs(env)
  n_state = env.n_height * env.n_width
  n_task = n_state
  alpha = 1
  discount = 0.5
  phi_matrix, mu, u1, u2, w = get_phi_mu_u_w(args, env, agent, states, state_action_pairs)
  q1 = phi_matrix @ u1.T
  q2 = phi_matrix @ u2.T
  q = np.minimum(q1, q2)
  action_log_pi = torch.log_softmax(torch.FloatTensor(q/alpha), dim=1).detach().cpu().numpy()
  np.save('action_log_pi.npy', action_log_pi)
  assert action_log_pi.shape == (n_state, env.action_space.n, n_task)
  fig, axes = plt.subplots(3, 3, figsize=(30,20))
  axes = axes.flatten()
  for i in range(n_task):
    reorganize_logpi = reorganize(action_log_pi[...,i:i+1])
    axes[i].imshow(reorganize_logpi[0], cmap='viridis')
  for axis in axes:
    axis.axis('off')
    mark_action_name(axis, n_state)
  plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.05, hspace=0.05)
  fig.suptitle('Action likelihood', fontsize=50)
  fig_path = f'{args.model_path.replace("model", "figure")}'
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/action_likelihood.png')
  print(f'{fig_path}/action_likelihood.png')
  plt.close()

def compare_action_likelihood(args, env, agent):
  n_task = 9
  action_log_pi_64 = np.load('action_log_pi_64.npy')
  action_log_pi_9 = np.load('action_log_pi_9.npy')
  # buffer_path = model_path.replace('irl_model', 'model')
  buffer = torch.load(f'model/{args.env}/{args.alg}/replay_buffer_balance.pkl')
  print(f'Replay buffer loaded from model/{args.env}/{args.alg}/replay_buffer_balance.pkl')
  # action_count = np.zeros((env.n_height*env.n_width, env.action_space.n))
  state_id = (buffer['state'][:,0] * env.n_width + buffer['state'][:,1]).astype(int)
  action = buffer['action'][:,0].astype(int)
  task = np.argmax(buffer['state'][:,2:], axis=1)

  empirical_count = np.zeros((env.n_height*env.n_width, env.action_space.n, n_task))
  for i in range(state_id.shape[0]):
    # print(state_id[i], action[i], task[i])
    empirical_count[state_id[i], action[i], task[i]] += 1
  empirical_f = empirical_count/(empirical_count.sum(axis=1, keepdims=True)+1e-6)
  empirical_log_f = np.log(empirical_f+1e-7)
  corrcoef_64 = np.corrcoef(action_log_pi_64.flatten(), empirical_log_f.flatten())[0,1]
  corrcoef_9 = np.corrcoef(action_log_pi_9.flatten(), empirical_log_f.flatten())[0,1]
  assert action_log_pi_64.shape == action_log_pi_9.shape
  fig, axes = plt.subplots(1, 1, figsize=(30,10))
  axes.plot(action_log_pi_64.flatten(), label='64', linewidth=1.5)
  axes.plot(action_log_pi_9.flatten(), label='9', linewidth=1.5)
  axes.plot(empirical_log_f.flatten(), label='empirical', linewidth=4, alpha=0.3)
  axes.legend(fontsize=20)
  fig.suptitle(f'Action likelihood comparison, corr_64:{corrcoef_64:.2f}, corr_9:{corrcoef_9:.2f}', fontsize=20)
  plt.savefig('action_likelihood_comparison.png')
  print('action_likelihood_comparison.png')
  plt.close()

def check_q(args, env, agent):
  n_state, n_action, n_task = agent.n_state, agent.n_action, agent.n_task
  states, state_action_pairs, task_id_all = get_state_action_pair_onehot(env, n_state, n_action)
  alpha = 1
  discount = 0.5
  phi_matrix, mu, u1, u2, w = get_phi_mu_u_w(args, env, agent, states, state_action_pairs, n_task)
  q1 = phi_matrix @ u1.T
  q2 = phi_matrix @ u2.T
  q = np.minimum(q1, q2)
  assert q.shape == (n_state, n_action, n_task)
  real_Q, real_V = value_iteration(np.load('P.npy'), np.load('r.npy'), discount)
  assert real_Q.shape == (n_state, n_action, n_task)
  corrcoef = np.corrcoef(q.flatten(), real_Q.flatten())[0,1]
  fig, axes = plt.subplots(1, 1, figsize=(30,10))
  axes.plot(q.flatten(), label='pred_q', linewidth=5)
  axes.plot(real_Q.flatten(), label='real_q', linewidth=5)
  axes.legend(fontsize=20)
  fig.suptitle(f'pred_q & real_q, corrcoef:{corrcoef}', fontsize=20)
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/pred_q_real_q.png')
  print(f'{fig_path}/pred_q_real_q.png')
  plt.close()
  
  fig, axes = plt.subplots(3, 6, figsize=(30,20))
  axes = axes.flatten()
  for i in range(n_task):
    reorganize_q = reorganize(q[...,i:i+1])
    reorganize_real_Q = reorganize(real_Q[...,i:i+1])
    q_min = np.min([reorganize_q[0].min(), reorganize_real_Q[0].min()])
    q_max = np.max([reorganize_q[0].max(), reorganize_real_Q[0].max()])
    norm = colors.Normalize(q_min, q_max)
    axes[i*2].imshow(reorganize_q[0], cmap='viridis', norm=norm)
    axes[i*2+1].imshow(reorganize_real_Q[0], cmap='viridis', norm=norm)
  for axis in axes:
    axis.axis('off')
    mark_action_name(axis, n_state)
  plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.05, hspace=0.05)
  fig.suptitle('pred_q & real_q', fontsize=20)
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/pred_q_real_q_matrix.png')
  print(f'{fig_path}/pred_q_real_q_matrix.png')
  plt.close()


  return


def compare_phi_representation():
  phi_matrix_matrix = np.load('Phi.npy')
  phi_matrix_NN = np.load('NN_phi.npy')
  assert phi_matrix_matrix.shape == phi_matrix_NN.shape
  fig, axes = plt.subplots(1, 1, figsize=(50,10))
  axes.plot(phi_matrix_matrix.flatten(), label='Ground Truth', linewidth=3)
  axes.plot(phi_matrix_NN.flatten(), label='NN', linewidth=3)
  axes.legend(fontsize=20)
  fig.suptitle('Phi representation comparison', fontsize=20)
  plt.savefig('phi_representation_comparison.png')
  plt.close()

def get_estimate_P(args, env, agent):
  max_size = 90000
  # n_state = env.n_height * env.n_width
  # n_task = n_state
  # n_action = env.action_space.n
  n_state, n_action, n_task = agent.n_state, agent.n_action, agent.n_task
  count = np.zeros((n_state, n_action, n_state))
  # replay_buffer = buffer.ReplayBuffer(state_dim, action_dim, args.buffer_size)
  buffer_path = model_path.replace('irl_model', 'model')
  replay_buffer = torch.load(f'./model/{args.env}/{args.alg}/replay_buffer_labyrinth.pkl')
  print(f'Replay buffer loaded from ./model/{args.env}/{args.alg}/replay_buffer_labyrinth.pkl')
  state = replay_buffer['state']
  print('state:', state.shape)
  action = replay_buffer['action']
  next_state = replay_buffer['next_state']
  # assert state.shape == (max_size, n_state+n_task)
  # assert action.shape == (max_size, n_action)
  # assert next_state.shape == (max_size, n_state+n_task)
  # state_id = state[:,0] * env.n_width + state[:,1]
  # next_state_id = next_state[:,0] * env.n_width + next_state[:,1]
  state_id = np.argmax(state[:,:n_state],-1)
  action_id = np.argmax(action,-1)
  next_state_id = np.argmax(next_state[:,:n_state],-1)
  for i in range(max_size):
    count[int(state_id[i]), int(action_id[i]), int(next_state_id[i])] += 1
  P = count/count.sum(-1, keepdims=True)
  return P

def SVD_P(args, env, agent):
  n_state, n_action, n_task = 9,4,9
  discount = 0.5
  alpha = 1
  # n_state = env.n_height * env.n_width
  # P = ground_truth_P(args, env)
  P = get_estimate_P(args, env, agent)
  P = np.load('trans_probs.npy')
  P = np.load('P.npy')
  P_2d = P.reshape(-1, n_state)
  U, S, V = np.linalg.svd(P_2d)
  U_truncated = U[:,:n_state]
  print('U_truncated:{a}, S:{b}, V:{c}'.format(a=U_truncated.shape, b=S.shape, c=V.shape))
  Phi = U_truncated@np.sqrt(np.diag(S)) # Phi:[n_state*n_action, n_state]
  Mu = (np.sqrt(np.diag(S))@V).T # Mu:[n_state, n_state]
  model_path = f'model/{args.env}/{args.alg}'
  np.save(f'{model_path}/Phi_estimated.npy', Phi)
  np.save(f'{model_path}/Mu_estimated.npy', Mu)
  print('Phi:{a}, Mu:{b}'.format(a=Phi.shape, b=Mu.shape))
  fig, axes = plt.subplots(1, 1, figsize=(30,30))
  axes.imshow(Phi)
  colorbar_ax = fig.add_axes([0.7, 0.35, 0.3, 0.05])
  colorbar_norm = colors.Normalize(Phi.min(), Phi.max())
  mpl.colorbar.Colorbar(colorbar_ax, norm=colorbar_norm, orientation='horizontal')
  fig.suptitle('Phi matrix')
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/Phi_estimated.png')
  print(f'{fig_path}/Phi_estimated.png')
  plt.close()
  fig, axes = plt.subplots(1, 1, figsize=(30,30))
  axes.imshow(Mu)
  colorbar_ax = fig.add_axes([0.7, 0.35, 0.3, 0.05])
  colorbar_norm = colors.Normalize(Mu.min(), Mu.max())
  mpl.colorbar.Colorbar(colorbar_ax, norm=colorbar_norm, orientation='horizontal')
  fig.suptitle('Mu matrix')
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/Mu_estimated.png')
  print(f'{fig_path}/Mu_estimated.png')
  plt.close()
  # R = ground_truth_r(args, env)
  # R_2d = R.reshape(-1, n_task)
  # W = (np.linalg.pinv(Phi)@R_2d).T # W:[n_task, n_state]
  # Q, vf = value_iteration(P, R, discount=discount, alpha=alpha) # Q:[n_state, n_action, n_task], V:[n_state, n_task]
  # Q_2d = Q.reshape(-1, n_task)
  # print(U_truncated.T@U_truncated)
  # uf = (np.linalg.pinv(Phi)@Q_2d).T # U:[n_task, n_state]
  # predict_U = W + (vf.T@Mu)*discount
  # print('uf:{a}, predict_U:{b}'.format(a=uf, b=predict_U))
  # error = np.abs(uf-predict_U).mean()
  # print('error:', error)
  # print(np.allclose(uf, predict_U))

def value_iteration(P, R, discount=0.5, max_iter=1000, tol=1e-6, alpha=0.1):
  """P:[n_state, n_action, n_state], R:[n_state, n_action, n_task]"""
  if os.path.exists('Q.npy'):
    Q = np.load('Q.npy')
    V = np.load('V.npy')
    return Q, V
  n_state, n_action, _ = P.shape
  n_task = R.shape[-1]
  Q = np.zeros((n_state, n_action, n_task))
  V = np.zeros((n_state, n_task))
  for i in range(max_iter):
    Q = R + discount * (P@V)
    V_new = logsumexp(Q/alpha, axis=1) * alpha
    if np.abs(V_new-V).max() < tol:
      break
    V = V_new
  print(f'Exit iteration after {i} step')
  np.save('Q.npy', Q)
  np.save('V.npy', V)
  return Q, V

def check_Phi_cov(args, env, agent):
  states, state_action_pairs, task_id_all = get_state_action_pairs(env)
  n_state = env.n_height * env.n_width
  n_task = n_state
  phi_matrix, mu, u1, u2, w = get_phi_mu_u_w_matrix(args, env, agent, states, state_action_pairs)
  # phi_matrix = np.load('Phi.npy')
  fig, axes = plt.subplots(1, 1, figsize=(10,10))
  phi_matrix_flatten = phi_matrix.reshape(n_state*env.action_space.n, args.feature_dim)
  cov = phi_matrix_flatten.T@phi_matrix_flatten
  axes.imshow(cov, cmap='viridis')
  fig.suptitle('Covariance matrix of phi')
  fig_path = args.model_path.replace('model', 'figure')
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/cov_phi.png')
  print(f'{fig_path}/cov_phi.png')
  plt.close()

def check_mapback_u(args, env, agent, kwargs):
  states, state_action_pairs, task_id_all = get_state_action_pairs(env)
  n_state = env.n_height * env.n_width
  n_task = n_state
  args.dir = '3x3-f64-phimu_matrix_dim9'
  args.feature_dim = 9
  kwargs['feature_dim'] = 9
  model_path = f'model/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  agent = spedersac_agent.Discrete_SPEDERSACAgent(**kwargs)
  agent.load_state_dict(torch.load(f'{model_path}/ckpt_{args.ckpt_n}.pt'))
  phi_matrix_9, mu_9, u1_9, u2_9, w_9 = get_phi_mu_u_w(args, env, agent, states, state_action_pairs)
  args.dir = '3x3-f64-phimu_matrix_dim64'
  args.feature_dim = 64
  kwargs['feature_dim'] = 64
  model_path = f'model/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  agent = spedersac_agent.Discrete_SPEDERSACAgent(**kwargs)
  agent.load_state_dict(torch.load(f'{model_path}/ckpt_{args.ckpt_n}.pt'))
  phi_matrix_64, mu_64, u1_64, u2_64, w_64 = get_phi_mu_u_w(args, env, agent, states, state_action_pairs)
  # linear regression of phi_9 = phi_64@theta
  phi_matrix_64_flatten = phi_matrix_64.reshape(n_state*env.action_space.n, 64)
  phi_matrix_9_flatten = phi_matrix_9.reshape(n_state*env.action_space.n, 9)
  theta = np.linalg.lstsq(phi_matrix_64_flatten, phi_matrix_9_flatten, rcond=None)[0]
  
  print(theta.shape)
  new_u1_64 = u1_9@(theta.T)
  new_u2_64 = u2_9@(theta.T)
  new_w_64 = w_9@(theta.T)
  new_r = phi_matrix_64@new_w_64.T
  real_r = ground_truth_r(args, env)
  corrcoef = np.corrcoef(new_r.flatten(), real_r.flatten())[0,1]
  fig, axes = plt.subplots(1, 1, figsize=(50,20))
  axes.plot(new_r.flatten(), label='pred_r', linewidth=5)
  axes.plot(real_r.flatten(), label='ground_truth', linewidth=5)
  axes.legend(fontsize=20)
  fig.suptitle('pred_r & r, corrcoef:{:.2f}'.format(corrcoef), fontsize=50)
  fig_path = f'{args.model_path.replace("model", "figure")}'
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  plt.savefig(f'{fig_path}/pred_r_r_mapback.png')
  print(f'{fig_path}/pred_r_r_mapback.png')
  plt.close()
  np.save('inferred_uw_64.npy', [new_u1_64, new_u2_64, new_w_64])

def svd_on_s_a_t(args, env, agent):
  n_state, n_action, n_task = 9, 4, 9
  replay_buffer = buffer.ReplayBuffer(state_dim+n_task, action_dim, args.buffer_size)
  replay_buffer.load_state_dict(torch.load(f'model/{args.env}/{args.alg}/replay_buffer_VI_onehotsa.pkl'))
  print(f'Replay buffer loaded from model/{args.env}/{args.alg}/replay_buffer_VI_onehotsa.pkl')
  # value count
  df = {
      'state': np.argmax(replay_buffer.state[:,:n_state], axis=1),
      'action': np.argmax(replay_buffer.action, axis=1),
      'next_state': np.argmax(replay_buffer.next_state[:,:n_state], axis=1),
      'task': np.argmax(replay_buffer.state[:,n_state:], axis=1)
  }
  import pandas as pd
  df = pd.DataFrame(df)
  df['s_a_idx'] = df['state']*n_action + df['action']
  df = df.groupby(['task']).sum()
  print(df)


EPS_GREEDY = 0.01

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default='0')                     
  parser.add_argument("--alg", default="spedersac")                     # Alg name (sac, vlsac, spedersac, ctrlsac, mulvdrq, diffsrsac, spedersac)
  parser.add_argument("--env", default="RandomWalk-v0")          # Environment name
  parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
  # parser.add_argument("--start_timesteps", default=25e3, type=float)# Time steps initial random policy is used
  # parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
  # parser.add_argument("--max_timesteps", default=1e6, type=float)   # Max time steps to run environment
  # parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
  # parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
  # parser.add_argument("--hidden_dim", default=256, type=int)      # Network hidden dims
  parser.add_argument("--feature_dim", default=64, type=int)      # Latent feature dim
  parser.add_argument("--discount", default=0.5)                 # Discount factor
  # parser.add_argument("--tau", default=0.005)                     # Target network update rate
  # parser.add_argument("--learn_bonus", action="store_true")        # Save model and optimizer parameters
  # parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
  # parser.add_argument("--extra_feature_steps", default=3, type=int)
  parser.add_argument("--ckpt_n", default=1000, type=int)
  parser.add_argument("--mode", default='rl')
  parser.add_argument("--task_idx", default=0, type=int)
  args = parser.parse_args()

  if args.alg == 'mulvdrq':
    import sys
    sys.path.append('agent/mulvdrq/')
    from agent.mulvdrq.train_metaworld import Workspace, cfg
    cfg.task_name = args.env
    cfg.seed = args.seed
    workspace = Workspace(cfg)
    workspace.train()

    sys.exit()

  # env = gym.make(args.env)
  # eval_env = gym.make(args.env)
  # env.seed(args.seed)
  # eval_env.seed(args.seed)
  # max_length = env._max_episode_steps

  # set model path
  model_path = f'model/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  if args.mode == 'irl':
    model_path = f'irl_{model_path}'
  # model_path = model_path.replace('/0','/estimated_P')
  args.model_path = model_path
  # set seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # 
  # print(env.observation_space.n)
  # print(env.action_space.n)
  # if isinstance(env.action_space, gym.spaces.Discrete):
  #   state_dim = 2
  #   action_dim = 1
    # print('Discrete action space:', state_dim+action_dim)
    # max_action = float(env.action_space.n)
  # elif isinstance(env.action_space, gym.spaces.Box):
  #   state_dim = env.observation_space.shape[0]
  #   action_dim = env.action_space.shape[0]
  #   max_action = float(env.action_space.high[0])
  # print(model_path)
  # state_dim = env.height * env.width
  # action_dim = env.action_space.n
  env = None
  if os.path.exists(f'{model_path}/kwargs.pkl'):
    print('OK')
    # kwargs = torch.load(f'{model_path}/kwargs.pt')
    kwargs = load_kwargs(f'{model_path}/kwargs.pkl')
    print(f'Load kwargs from {model_path}/kwargs.pkl')
  else:
    raise ValueError(f'No kwargs found in {model_path}/kwargs.pkl')
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_space": env.action_space,
        "discount": args.discount,
        "tau": args.tau,
        "hidden_dim": args.hidden_dim,
    }

  # Initialize policy
  if args.alg == "sac":
    agent = sac_agent.SACAgent(**kwargs)
  elif args.alg == 'vlsac':
    kwargs['extra_feature_steps'] = args.extra_feature_steps
    kwargs['feature_dim'] = args.feature_dim
    agent = vlsac_agent.VLSACAgent(**kwargs)
  elif args.alg == 'ctrlsac':
    kwargs['extra_feature_steps'] = args.extra_feature_steps
    # hardcoded for now
    kwargs['feature_dim'] = 2048  
    kwargs['hidden_dim'] = 1024
    agent = ctrlsac_agent.CTRLSACAgent(**kwargs)
  elif args.alg == 'diffsrsac':
    agent = diffsrsac_agent.DIFFSRSACAgent(**kwargs)
  elif args.alg == 'spedersac':
    # kwargs['extra_feature_steps'] = 5
    # kwargs['phi_and_mu_lr'] = 0.00001
    # kwargs['phi_hidden_dim'] = 512
    # kwargs['phi_hidden_depth'] = 1
    # kwargs['mu_hidden_dim'] = 512
    # kwargs['mu_hidden_depth'] = 1
    # kwargs['critic_and_actor_lr'] = 3e-4
    # kwargs['critic_and_actor_hidden_dim'] = 256
    # kwargs['feature_dim'] = args.feature_dim
    kwargs['device'] = 'cpu'
    # kwargs['actor_name'] = 'softmax'
    # kwargs['discount'] = 0.5
    kwargs['pretrain_model_path'] = f'{model_path.replace("irl_model", "model")}/ckpt_{args.ckpt_n}.pt'
    # kwargs['alpha'] = 1
    print(kwargs)
    for key, value in kwargs.items():
      setattr(args, key, value)
    # agent = VI_IRL_Agent(**kwargs)
    agent = spedersac_iragent.Inverse_Discrete_SPEDERSACAgent(**kwargs)

    # if isinstance(env.action_space, gym.spaces.Discrete):
    #   print('Discrete_SPEDERSACAgent')
    #   agent = spedersac_agent.Discrete_SPEDERSACAgent(**kwargs)
    # else:
    #   agent = spedersac_agent.SPEDERSACAgent(**kwargs)

    if os.path.exists(f'{model_path}/ckpt_{args.ckpt_n}.pt'):
      agent.load_state_dict(torch.load(f'{model_path}/ckpt_{args.ckpt_n}.pt'))
      # agent.load_state_dict(torch.load('model/SimpleGridWorld-v0/spedersac/0/0/ckpt_4000.pt'))
      print(f'Load model from {model_path}/ckpt_{args.ckpt_n}.pt')
    else:
      print(f'No model found in {model_path}/ckpt_{args.ckpt_n}.pt')
      exit()


  visualize_phi_matrix(args, env, agent)
  show_uw(args, env, agent)
  check_r(args, env, agent)
