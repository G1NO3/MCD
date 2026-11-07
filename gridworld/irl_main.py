import numpy as np
import torch
import gym
import argparse
import os
import pickle
import visualize

from tensorboardX import SummaryWriter

from utils import util, buffer
# from agent.sac import sac_agent
# from agent.vlsac import vlsac_agent
# from agent.ctrlsac import ctrlsac_agent
# from agent.diffsrsac import diffsrsac_agent
# from agent.spedersac import spedersac_agent
from agent.spedersac import spedersac_iragent
import pandas as pd
def print_param(model):
  for name, param in model.named_parameters():
    print(name, param)


EPS_GREEDY = 0.05
torch.set_printoptions(precision=2,threshold=10)
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

  # setup log 
  log_path = f'irl_log/{args.env}/{args.alg}/{args.dir}/{args.seed}/task{args.task_idx}'
  summary_writer = SummaryWriter(log_path)
  # set model path
  model_path = f'irl_model/{args.env}/{args.alg}/{args.dir}/{args.seed}'
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  # set seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)


  # if isinstance(env.action_space, gym.spaces.Discrete):
  #   state_dim = 2
  #   action_dim = 1
  #   max_action = float(env.action_space.n)
  #   print('max_action:', max_action)
  # elif isinstance(env.action_space, gym.spaces.Box):
  #   state_dim = env.observation_space.shape[0]
  #   action_dim = env.action_space.shape[0]
  #   max_action = float(env.action_space.high[0])
  # state_dim = env.n_height * env.n_width
  # action_dim = env.action_space.n
  state_dim = 9
  action_dim = 4
  n_task = 9
  kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    # "action_space": env.action_space,
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
    kwargs['n_task'] = n_task
    kwargs['discount'] = 0.5
    kwargs['pretrain_model_path'] = f'{model_path.replace("irl_model", "model")}/ckpt_{args.ckpt_n}.pt'
    kwargs['alpha'] = 1
    for key, value in kwargs.items():
      setattr(args, key, value)
    # agent = spedersac_iragent.VI_IRL_Agent(**kwargs)
    agent = spedersac_iragent.Inverse_Discrete_SPEDERSACAgent(**kwargs)
    # if isinstance(env.action_space, gym.spaces.Discrete):
    #   agent = spedersac_iragent.Inverse_Discrete_SPEDERSACAgent(**kwargs)
    # else:
    #   agent = spedersac_iragent.Inverse_SPEDERSACAgent(**kwargs)
  
  visualize.save_kwargs(kwargs, f'{model_path}/kwargs.pkl')
  print(f'kwargs saved at {model_path}/kwargs.pkl')

  replay_buffer = buffer.ReplayBuffer(state_dim+n_task, action_dim, args.buffer_size)
  random_buffer = buffer.ReplayBuffer(state_dim+n_task, action_dim, args.buffer_size)
  # buffer_path = model_path.replace('irl_model', 'model')
  replay_buffer.load_state_dict(torch.load(f'model/{args.env}/{args.alg}/replay_buffer_VI_onehotsa.pkl'))
  random_buffer.load_state_dict(torch.load(f'model/{args.env}/{args.alg}/replay_buffer_VI_onehotsa.pkl'))
  # replay_buffer.load_state_dict(torch.load(f'model/{args.env}/{args.alg}/replay_buffer_labyrinth.pkl'))
  # random_buffer.load_state_dict(torch.load(f'model/{args.env}/{args.alg}/replay_buffer_random_labyrinth.pkl'))
  print(f'Replay buffer loaded from model/{args.env}/{args.alg}/replay_buffer_VI_onehotsa.pkl')
  # print(replay_buffer.state.shape)
  # print(f'Replay buffer loaded from model/{args.env}/{args.alg}/replay_buffer_labyrinth.pkl')
  # print(f'Random buffer loaded from model/{args.env}/{args.alg}/replay_buffer_random_labyrinth.pkl')
  # evaluations = [util.eval_policy(agent, eval_env)]
  # if not os.path.exists(f'{model_path}/balance'):
  #   os.makedirs(f'{model_path}/balance')
  for t in range(int(args.max_timesteps)):
    # info = agent.train(replay_buffer, batch_size=args.batch_size)
    info = agent.two_step_train(random_buffer, replay_buffer, batch_size=args.batch_size)

      # save model
    if (t+1) % args.eval_freq == 0:
      # evaluation = util.eval_policy(agent, eval_env)
      # evaluations.append(evaluation)
      # info['evaluation'] = evaluation
      for key, value in info.items():
        summary_writer.add_scalar(f'info/{key}', value, t+1)
      summary_writer.flush()
      if args.save_model:
        torch.save(agent.state_dict(), f'{model_path}/ckpt_{t+1}.pt')
        print(f'Model saved at {model_path}/ckpt_{t+1}.pt')
      # print('Log Alpha:')
      # print(agent.log_alpha)
      # print('U:')
      # print_param(agent.critic)
      # print('W:')
      # print_param(agent.w)

  summary_writer.close()
