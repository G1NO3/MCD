import numpy as np
import torch
import gym
import argparse
import os
import pickle
import visualize

from tensorboardX import SummaryWriter

from utils import util, buffer
from agent.sac import sac_agent
from agent.vlsac import vlsac_agent
from agent.ctrlsac import ctrlsac_agent
from agent.diffsrsac import diffsrsac_agent
from agent.spedersac import spedersac_agent

def print_param(model):
  for name, param in model.named_parameters():
    print(name, param)


EPS_GREEDY = 0.0

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default='0')                     
  parser.add_argument("--alg", default="spedersac")                     # Alg name (sac, vlsac, spedersac, ctrlsac, mulvdrq, diffsrsac, spedersac)
  parser.add_argument("--env", default="HalfCheetah-v4")          # Environment name
  parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--start_timesteps", default=25e3, type=float)# Time steps initial random policy is used
  parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
  parser.add_argument("--max_timesteps", default=1e6, type=float)   # Max time steps to run environment
  parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
  parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
  parser.add_argument("--hidden_dim", default=256, type=int)      # Network hidden dims
  parser.add_argument("--feature_dim", default=2048, type=int)      # Latent feature dim
  parser.add_argument("--discount", default=0.99)                 # Discount factor
  parser.add_argument("--tau", default=0.005)                     # Target network update rate
  parser.add_argument("--learn_bonus", action="store_true")        # Save model and optimizer parameters
  parser.add_argument("--save_model", action="store_false")        # Save model and optimizer parameters
  parser.add_argument("--extra_feature_steps", default=3, type=int)
  parser.add_argument("--random_reset_freq", default=100, type=int)
  parser.add_argument("--buffer_size", default=int(1e6), type=int)
  parser.add_argument("--ckpt_n", default=0, type=int)
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

  env = gym.make(args.env)
  eval_env = gym.make(args.env)
  env.seed(args.seed)
  eval_env.seed(args.seed)
  max_length = env._max_episode_steps

  # setup log 
  log_path = f'log/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  summary_writer = SummaryWriter(log_path)
  # set model path
  model_path = f'model/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  # set seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # 
  # print(env.observation_space.n)
  # print(env.action_space.n)
  if isinstance(env.action_space, gym.spaces.Discrete):
    state_dim = env.reset().shape[0]
    action_dim = 1
    # print('Discrete action space:', state_dim+action_dim)
    max_action = float(env.action_space.n)
    print('max_action:', max_action)
  elif isinstance(env.action_space, gym.spaces.Box):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
  state_dim = env.n_height * env.n_width
  action_dim = env.action_space.n
  kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "action_space": env.action_space,
    "discount": args.discount,
    "tau": args.tau,
    "hidden_dim": args.hidden_dim,
    "device": "cuda:0"
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
    kwargs['extra_feature_steps'] = 5
    kwargs['phi_and_mu_lr'] = 1e-3
    kwargs['phi_hidden_dim'] = 512
    kwargs['phi_hidden_depth'] = 1
    kwargs['mu_hidden_dim'] = 512
    kwargs['mu_hidden_depth'] = 1
    kwargs['critic_and_actor_lr'] = 3e-4
    kwargs['critic_and_actor_hidden_dim'] = 256
    kwargs['feature_dim'] = args.feature_dim
    kwargs['actor_name'] = 'softmax'
    kwargs['n_task'] = env.n_height * env.n_width
    kwargs['discount'] = 0.5
    # kwargs['n_width'] = env.n_width
    # kwargs['n_height'] = env.n_height
    if isinstance(env.action_space, gym.spaces.Discrete):
      agent = spedersac_agent.Discrete_SPEDERSACAgent(**kwargs)
    else:
      agent = spedersac_agent.SPEDERSACAgent(**kwargs)
    # agent.load_state_dict(torch.load(f'{model_path}/ckpt_{args.ckpt_n}.pt'))
  visualize.save_kwargs(kwargs, f'{model_path}/kwargs.pkl')
  print(f'kwargs saved at {model_path}/kwargs.pkl')
  n_task = env.n_height * env.n_width
  replay_buffer = buffer.ReplayBuffer(state_dim+n_task, action_dim, args.buffer_size)
  # Evaluate untrained policy
  evaluations = [util.eval_policy(agent, eval_env)]

  state, done = env.reset(), False
  # print('state:', state)
  episode_reward = 0
  episode_timesteps = 0
  episode_num = 0
  timer = util.Timer()
  # reward_window = buffer.RewardBuffer(30)
  action_ar = np.eye(env.action_space.n)
  state_ar = np.eye(env.n_width * env.n_height).reshape(env.n_width, env.n_height, -1)
  for t in range(int(args.max_timesteps)):

    episode_timesteps += 1

    # Select action randomly or according to policy
    state_one_hot = np.concatenate((state_ar[state[0], state[1]], state[2:]), -1)
    if t < args.start_timesteps:
      action = env.action_space.sample()
    else:
      # action = agent.select_action(state, explore=True)
      # epsilon greedy as mentioned in the CTRL paper
      if np.random.uniform(0, 1) < EPS_GREEDY:
        action = env.action_space.sample()
      else:
        action = agent.select_action(state_one_hot, explore=True)

    # Perform action
    next_state, reward, done, _ = env.step(action) 
    done_bool = float(done) if episode_timesteps < max_length else 0

    # Store data in replay buffer
    # print('state, action, next_state', state, action, next_state)
    
    next_state_one_hot = np.concatenate((state_ar[next_state[0], next_state[1]], next_state[2:]), -1)
    action_one_hot = action_ar[action]
    
    replay_buffer.add(state_one_hot, action_one_hot, next_state_one_hot, reward, done_bool)
    # replay_buffer.add(state, np.array([action]), next_state, reward, done_bool)

    state = next_state
    episode_reward += reward
    
    # Train agent after collecting sufficient data
    if t >= args.start_timesteps:
      info = agent.train(replay_buffer, batch_size=args.batch_size)

    if done: 
      # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
      print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
      # Reset environment
      if ((episode_num+1) % (args.random_reset_freq) == 0):
        state, done = env.random_reset(), False
        print('start:{a}, end:{b}'.format(a=env.start,b=env.ends[0]))
      else:
        state, done = env.reset(), False

      episode_reward = 0
      episode_timesteps = 0
      episode_num += 1 

    # Evaluate episode
    if (t + 1) % args.eval_freq == 0:
      steps_per_sec = timer.steps_per_sec(t+1)
      evaluation = util.eval_policy(agent, eval_env)
      evaluations.append(evaluation)

      if t >= args.start_timesteps:
        info['evaluation'] = evaluation
        for key, value in info.items():
          summary_writer.add_scalar(f'info/{key}', value, t+1)
        summary_writer.add_scalar(f'info/reward', episode_reward, t+1)
        summary_writer.flush()
        # save model
      if args.save_model:
        # print(type(agent.state_dict()))
        # print(type(kwargs))
        torch.save(agent.state_dict(), f'{model_path}/ckpt_{t+1}.pt')
        # torch.save(kwargs, f'{model_path}/kwargs.pt')

        # pickle.dump(kwargs, open(f'{model_path}/kwargs.pkl', 'wb'))
        print(f'Model saved at {model_path}/ckpt_{t+1}.pt')

        # print('Phi:')
        # print_param(agent.mu)
        # print('Mu:')
        # print_param(agent.mu)
        print('Log Alpha:')
        print(agent.log_alpha)
        print('U:')
        print_param(agent.critic)
        # print('Actor:')
        # print_param(agent.actor)
        print('W:')
        print_param(agent.w)


      print('Step {}. Steps per sec: {:.4g}.'.format(t+1, steps_per_sec))

  summary_writer.close()

  print('Total time cost {:.4g}s.'.format(timer.time_cost()))
