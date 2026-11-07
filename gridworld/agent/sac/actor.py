"""
We adapt the code from https://github.com/denisyarats/pytorch_sac
"""


import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

from utils import util


class TanhTransform(pyd.transforms.Transform):
  domain = pyd.constraints.real
  codomain = pyd.constraints.interval(-1.0, 1.0)
  bijective = True
  sign = +1

  def __init__(self, cache_size=1):
    super().__init__(cache_size=cache_size)

  @staticmethod
  def atanh(x):
    return 0.5 * (x.log1p() - (-x).log1p())

  def __eq__(self, other):
    return isinstance(other, TanhTransform)

  def _call(self, x):
    return x.tanh()

  def _inverse(self, y):
    # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
    # one should use `cache_size=1` instead
    return self.atanh(y)

  def log_abs_det_jacobian(self, x, y):
    # We use a formula that is more numerically stable, see details in the following link
    # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
    return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale
    self.base_dist = pyd.Normal(loc, scale)
    transforms = [TanhTransform()]
    super().__init__(self.base_dist, transforms)

  @property
  def mean(self):
    mu = self.loc
    for tr in self.transforms:
        mu = tr(mu)
    return mu


class DiagGaussianActor(nn.Module):
  """torch.distributions implementation of an diagonal Gaussian policy."""
  def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                log_std_bounds):
    super().__init__()

    self.log_std_bounds = log_std_bounds
    self.trunk = util.mlp(obs_dim, hidden_dim, 2 * action_dim,
                            hidden_depth)

    self.outputs = dict()
    self.apply(util.weight_init)

  def forward(self, obs):
    mu, log_std = self.trunk(obs).chunk(2, dim=-1)

    # constrain log_std inside [log_std_min, log_std_max]
    log_std = torch.tanh(log_std)
    log_std_min, log_std_max = self.log_std_bounds
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                  1)

    std = log_std.exp()

    self.outputs['mu'] = mu
    self.outputs['std'] = std

    dist = SquashedNormal(mu, std)
    return dist
  
class DiscreteActor(nn.Module):
    """Torch implementation of a discrete policy using Categorical distribution."""
    
    def __init__(self, obs_dim, action_n, hidden_dim, hidden_depth):
        super().__init__()
        print('obs_dim:{a}, action_n:{b}, hidden_dim:{c}, hidden_depth:{d}'.format(a=obs_dim,b=action_n,c=hidden_dim,d=hidden_depth))
        # Define the MLP that maps the observations to logits for actions
        self.trunk = util.mlp(obs_dim, hidden_dim, action_n, hidden_depth)
        
        # For storing outputs, just like in DiagGaussianActor
        self.outputs = dict()
        
        # Apply weight initialization if necessary (optional, based on your utilities)
        self.apply(util.weight_init)

    def forward(self, obs):
        # Forward pass through the MLP to get logits
        logits = self.trunk(obs)

        # Store the logits in outputs (for debugging/inspection if needed)
        self.outputs['logits'] = logits
        # print('logits:',logits)
        # Create a categorical distribution over the actions
        dist = pyd.Categorical(logits=logits)
        return dist
    
    def evaluate(self, obs, epsilon=1e-8):
        logits = self.trunk(obs)
        probs = F.softmax(logits, dim=-1)
        z = (probs == 0.0) * epsilon
        log_probs = torch.log(probs + z)
        return log_probs

class SoftmaxActor(nn.Module):
    """Torch implementation of a discrete policy using Categorical distribution."""
    
    def __init__(self, phi, critic, action_n, feature_dim, n_task, alpha, device, log_alpha, state_dim, action_dim):
        super().__init__()
        self.phi = phi
        self.critic = critic
        self.action_n = action_n
        self.device = device
        self.feature_dim = feature_dim
        self.free_parameter = nn.Parameter(torch.rand(1), requires_grad=True) 
        self.n_task = n_task
        self.alpha = alpha
        self.log_alpha = log_alpha
        self.state_dim = state_dim
        self.action_dim = action_dim
        print('state_dim:', state_dim)
        # print('actor alpha:', self.alpha.is_leaf)
        # print(action_dim)
        if action_dim == 1:
          self.actions_all = torch.arange(action_n).reshape(-1,1).to(self.device)
        elif action_dim == 4:
          self.actions_all = torch.eye(action_n).to(self.device)

    # def forward(self, state):
    #   assert state.shape[-1] == self.n_task + 2
    #   # cur_state = torch.take_along_dim(state,torch.tensor([[0,1]]).to(self.device),-1)
    #   cur_state = state[...,:2]
    #   # goal_state = torch.take_along_dim(state,torch.tensor([[2,3]]).to(self.device),-1)
    #   # task_id = torch.take_along_dim(state,torch.arange(2,self.n_task).to(self.device),-1)
    #   task_id = state[...,2:]
    #   q_list = torch.zeros(cur_state.shape[0],self.action_n).to(self.device)
    #   for i in range(self.action_n):
    #       action = torch.tensor([i]).repeat(cur_state.shape[0], 1).to(self.device)
    #       z_phi = self.phi(torch.concat([cur_state, action], -1))
    #       # q1, q2 = self.critic(torch.concat([z_phi, batch_goal_state], -1))
    #       u1, u2 = self.critic(task_id)
    #       # u_double = task_id@self.critic + self.critic_bias
    #       # assert u1.shape[-1] == self.feature_dim
    #       # assert u2.shape[-1] == self.feature_dim
    #       q1 = torch.sum(u1*z_phi, -1, keepdim=True)
    #       q2 = torch.sum(u2*z_phi, -1, keepdim=True)
    #       assert q1.shape[-1] == 1
    #       assert q2.shape[-1] == 1
    #       # q_double = torch.sum(u_double*z_phi.unsqueeze(0), -1, keepdim=True)
    #       # q = torch.min(q_double, dim=0, keepdim=True).values
    #       # assert q.shape[-1] == 1
    #       q = torch.min(q1, q2)
    #       # print('q:',q.shape)
    #       q_list[:,i:i+1] = q
    #   dist = torch.distributions.Categorical(logits=q_list/self.alpha)
    #   return dist
    
    def forward(self, state):
      assert state.shape[-1] == self.n_task + self.state_dim
      cur_state = state[...,:self.state_dim]
      task_id = state[...,self.state_dim:]
      state_action_pair = torch.concat([torch.repeat_interleave(cur_state,self.action_n,dim=0),torch.tile(self.actions_all,(cur_state.shape[0],1))],-1)
      z_phi = self.phi(state_action_pair).reshape(cur_state.shape[0],self.action_n,self.feature_dim).detach()
      u1, u2 = self.critic(task_id)
      q1 = torch.sum(u1.unsqueeze(1)*z_phi, -1)
      q2 = torch.sum(u2.unsqueeze(1)*z_phi, -1)
      assert q1.shape[-1] == self.action_n
      assert q2.shape[-1] == self.action_n
      q = torch.min(q1, q2)
      dist = torch.distributions.Categorical(logits=q/self.alpha.detach())
      return dist

    def evaluate(self, state):
      assert state.shape[-1] == self.n_task + self.state_dim
      cur_state = state[...,:2]
      task_id = state[...,2:]
      q_list = torch.zeros(cur_state.shape[0],self.action_n).to(self.device)
      for i in range(self.action_n):
          action = torch.tensor([i]).repeat(cur_state.shape[0], 1).to(self.device)
          z_phi = self.phi(torch.concat([cur_state, action], -1))
          u1, u2 = self.critic(task_id)
          q1 = torch.sum(u1*z_phi, -1, keepdim=True)
          q2 = torch.sum(u2*z_phi, -1, keepdim=True)
          assert q1.shape[-1] == 1
          assert q2.shape[-1] == 1
          q = torch.min(q1, q2)
          q_list[:,i:i+1] = q
      probs = F.softmax(q_list/self.alpha.detach(), dim=-1)
      # probs = F.softmax(q_list/self.alpha, dim=-1)
      z = (probs == 0.0) * 1e-8
      log_probs = torch.log(probs + z)
      return log_probs
    def evaluate_matrix(self, state):
      assert state.shape[-1] == self.n_task + self.state_dim
      cur_state = state[...,:self.state_dim]
      task_id = state[...,self.state_dim:]
      state_action_pair = torch.concat([torch.repeat_interleave(cur_state,self.action_n,dim=0),torch.tile(self.actions_all,(cur_state.shape[0],1))],-1)
      z_phi = self.phi(state_action_pair).reshape(cur_state.shape[0],self.action_n,self.feature_dim).detach()
      u1, u2 = self.critic(task_id)
      q1 = torch.sum(u1.unsqueeze(1)*z_phi, -1)
      q2 = torch.sum(u2.unsqueeze(1)*z_phi, -1)
      assert q1.shape[-1] == self.action_n
      assert q2.shape[-1] == self.action_n
      q = torch.min(q1, q2)
      # log_probs = q - torch.logsumexp(q/self.alpha, dim=-1, keepdim=True)
      probs = F.softmax(q/self.log_alpha.exp(), dim=-1)
      z = (probs == 0.0) * 1e-8
      log_probs = torch.log(probs + z)
      return log_probs


class RandomActor(nn.Module):
    """Torch implementation of a discrete policy using Categorical distribution."""
    
    def __init__(self, action_n, device):
        super().__init__()
        self.action_n = action_n
        self.device = device
        
    def forward(self, state):
        return torch.distributions.Categorical(logits=torch.zeros(state.shape[0], self.action_n).to(self.device))
  
    def evaluate(self, state):
        logits = torch.ones(state.shape[0], self.action_n).to(self.device)
        probs = logits/self.action_n
        return torch.log(probs)