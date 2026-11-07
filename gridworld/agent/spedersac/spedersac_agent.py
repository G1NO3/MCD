import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import os
import gym
from utils import util
import numpy as np
# from utils.util import unpack_batch, RunningMeanStd
from utils.util import unpack_batch
from utils.util import MLP

from agent.sac.sac_agent import SACAgent
from agent.sac.actor import DiagGaussianActor
from agent.sac.actor import DiscreteActor
from agent.sac.actor import SoftmaxActor
from agent.sac.actor import RandomActor
from torchinfo import summary
from scipy.special import logsumexp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_inv_w_u(P, R, phi_matrix, mu_matrix, discount, alpha):
    Phi = phi_matrix.reshape(phi_matrix.shape[0]*phi_matrix.shape[1], -1).detach().cpu().numpy()
    mu_matrix = mu_matrix.detach().cpu().numpy()
    R_flatten0 = R.reshape(R.shape[0]*R.shape[1], -1)
    # w = np.linalg.pinv(Phi.T@Phi)@Phi.T@R_flatten0
    w = np.linalg.pinv(Phi)@R.reshape(-1, R.shape[-1])
    assert w.shape == (Phi.shape[1], R.shape[-1])
    W = w.T
    Q, V = value_iteration(P, R, discount, alpha)
    U = W + discount * V.T@mu_matrix
    return W, U

def value_iteration(P, R, discount=0.5, max_iter=1000, tol=1e-3, alpha=1):
  """P:[n_state, n_action, n_state], R:[n_state, n_action, n_task]"""
  n_state, n_action, _ = P.shape
  n_task = R.shape[-1]
  Q = np.zeros((n_state, n_action, n_task))
  V = np.zeros((n_state, n_task))
  for i in range(max_iter):
    Q = R + discount * (P@V)
    V_new = logsumexp(Q/alpha, axis=1) * alpha
    # V_new = np.mean(Q, axis=1)
    error = np.abs(V_new-V).max()
    if error < tol:
      break
    V = V_new
  print(f'Exit iteration after {i} step')
  return Q, V, error


def pca_transform(phi_matrix, mu):
  phi_matrix = phi_matrix.reshape(phi_matrix.shape[0]*phi_matrix.shape[1], -1)
  phi_mean_feature = phi_matrix.mean(0, keepdims=True)
  phi_centralized = phi_matrix - phi_mean_feature
  cov = phi_centralized.T @ phi_centralized
  eigval, eigvec = np.linalg.eig(cov)
  idx = np.argsort(eigval)[::-1]
  # choose the first 9
  idx = idx[:9]
  eigval = eigval[idx]
  eigvec_transform = eigvec[:,idx]
  phi_reduced = phi_matrix @ eigvec_transform
  # normalized
#   norm_matrix = np.linalg.norm(phi_reduced, axis=0, keepdims=True)
#   phi_reduced = phi_reduced / np.linalg.norm(phi_reduced, axis=0, keepdims=True)
#   phi_reduced = phi_reduced / norm_matrix
  mu_reduced = mu @ eigvec_transform
#   mu_reduced = mu_reduced * norm_matrix
  reconstructed_P = phi_reduced @ mu_reduced.T
  original_P = phi_matrix @ mu.T
  print('reconstruction error:', np.linalg.norm(reconstructed_P - phi_matrix@mu.T))
  print('corrcoef:', np.corrcoef(reconstructed_P.flatten(), original_P.flatten()))
  print('explained variance:', eigval)
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(1,1,figsize=(5,5))
  ax.plot(reconstructed_P.flatten(), label='reconstructed')
  ax.plot(original_P.flatten(), label='original')
  ax.legend(fontsize=20)
  fig.suptitle(f'corr:{np.corrcoef(reconstructed_P.flatten(), original_P.flatten())[0,1]}')
  plt.savefig('reconstruction.png')
  return phi_reduced, mu_reduced

class RFFCritic(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()

        # Q1
        self.l1 = nn.Linear(input_dim, hidden_dim)  # random feature
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

        # Q2
        self.l4 = nn.Linear(input_dim, hidden_dim)  # random feature
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, output_dim)

        self.outputs = dict()

    def forward(self, critic_feed_feature):
        q1 = torch.sin(self.l1(critic_feed_feature))
        q1 = F.elu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.sin(self.l4(critic_feed_feature))
        q2 = F.elu(self.l5(q2))
        q2 = self.l6(q2)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

class MLPCritic(nn.Module):
    
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
    
            # Q1
            self.l1 = nn.Linear(input_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
            self.l3 = nn.Linear(hidden_dim, output_dim)
    
            # Q2
            self.l4 = nn.Linear(input_dim, hidden_dim)
            self.l5 = nn.Linear(hidden_dim, hidden_dim)
            self.l6 = nn.Linear(hidden_dim, output_dim)
    
            self.outputs = dict()
    
        def forward(self, critic_feed_feature):
            q1 = F.elu(self.l1(critic_feed_feature))
            q1 = F.elu(self.l2(q1))
            q1 = self.l3(q1)
    
            q2 = F.elu(self.l4(critic_feed_feature))
            q2 = F.elu(self.l5(q2))
            q2 = self.l6(q2)
    
            self.outputs['psi1'] = q1
            self.outputs['psi2'] = q2
    
            return q1, q2



class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def one_hot(tensor, depth):
    return torch.eye(depth).to(tensor.device)[tensor.long()]

class SingleMatrixEmbedding(nn.Module):
    def __init__(self, n_task, feature_dim):
        super(SingleMatrixEmbedding, self).__init__()
        self.matrix = nn.Embedding(n_task, feature_dim)
        # nn.init.zeros_(self.matrix.weight)
        # self.matrix.weight[:,0] = 1
    def forward(self, task_id: torch.Tensor):
        return self.matrix(task_id.argmax(-1).long())
    
class SingleMatrixCriticEmbedding(nn.Module):
    def __init__(self, n_task, feature_dim):
        super(SingleMatrixCriticEmbedding, self).__init__()
        self.matrix = nn.Embedding(n_task, feature_dim)
        # nn.init.zeros_(self.matrix.weight)
        # self.matrix.weight[:,0] = 1
    def forward(self, task_id: torch.Tensor):
        return self.matrix(task_id.argmax(-1).long()), self.matrix(task_id.argmax(-1).long())
    
class SingleMatrixFixedPhi(nn.Module):
    def __init__(self, state_dim, n_action, feature_dim):
        super(SingleMatrixFixedPhi, self).__init__()
        self.state_dim = state_dim
        self.n_action = n_action
        self.matrix = nn.Embedding(state_dim*n_action, feature_dim)
        # nn.init.zeros_(self.matrix.weight)
        
    def forward(self, state_action_pair):
        s = state_action_pair[...,:self.state_dim].argmax(-1)
        a = state_action_pair[...,self.state_dim:].argmax(-1)
        # assert state_action_pair.shape[-1] == 3
        idx = s*self.n_action + a
        return self.matrix(idx.long())
class SingleMatrixFixedMu(nn.Module):
    def __init__(self, state_dim, feature_dim):
        super(SingleMatrixFixedMu, self).__init__()
        self.state_dim = state_dim
        self.matrix = nn.Embedding(state_dim, feature_dim)
        # nn.init.zeros_(self.matrix.weight)
    def forward(self, state):
        # assert state.shape[-1] == 2
        s = state.argmax(-1)
        idx = s
        return self.matrix(idx.long())

class SingleMatrix(nn.Module):
    """
    Linear theta
    <phi(s, a), theta> = r
    """

    def __init__(
            self,
            input_dim,
            output_dim,
    ):
        super(SingleMatrix, self).__init__()
        self.l = nn.Linear(input_dim, output_dim)
        # self.weight = nn.Parameter(torch.zeros(input_dim, output_dim), requires_grad=True)
        # self.bias = nn.Parameter(torch.zeros(output_dim), requires_grad=True)

    def forward(self, task_id):
        z_w = self.l(task_id)
        # z_w = task_id @ self.weight + self.bias
        return z_w
class SingleMatrixCritic(nn.Module):
    """
    Linear theta
    <phi(s, a), u> = Q
    """

    def __init__(
            self,
            input_dim,
            output_dim,
    ):
        super(SingleMatrixCritic, self).__init__()
        self.l1 = nn.Linear(input_dim, output_dim)
        self.l2 = nn.Linear(input_dim, output_dim)
        # self.weight1 = nn.Parameter(torch.zeros(input_dim, output_dim), requires_grad=True)
        # self.bias1 = nn.Parameter(torch.zeros(output_dim), requires_grad=True)
        # self.weight2 = nn.Parameter(torch.zeros(input_dim, output_dim), requires_grad=True)
        # self.bias2 = nn.Parameter(torch.zeros(output_dim), requires_grad=True)

    def forward(self, task_id):
        u1 = self.l1(task_id)
        u2 = self.l2(task_id)
        # u1 = task_id @ self.weight1 + self.bias1
        # u2 = task_id @ self.weight2 + self.bias2
        return u1, u2


class SPEDERSACAgent(SACAgent):
    """
    SAC with VAE learned latent features
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            action_space,
            phi_and_mu_lr=-1,
            # 3e-4 was originally proposed in the paper, but seems to results in fluctuating performance
            phi_hidden_dim=-1,
            phi_hidden_depth=-1,
            mu_hidden_dim=-1,
            mu_hidden_depth=-1,
            critic_and_actor_lr=-1,
            critic_and_actor_hidden_dim=-1,
            discount=0.99,
            target_update_period=2,
            tau=0.005,
            alpha=0.1,
            auto_entropy_tuning=True,
            hidden_dim=1024,
            feature_tau=0.005,
            feature_dim=2048,  # latent feature dim
            use_feature_target=True,
            extra_feature_steps=1,
            device=device
    ):

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            tau=tau,
            alpha=alpha,
            discount=discount,
            target_update_period=target_update_period,
            auto_entropy_tuning=auto_entropy_tuning,
            hidden_dim=hidden_dim,
        )


        self.feature_dim = feature_dim
        self.feature_tau = feature_tau
        self.use_feature_target = use_feature_target
        self.extra_feature_steps = extra_feature_steps
        self.action_space = action_space

        self.phi = MLP(input_dim=state_dim + action_dim,
                       output_dim=feature_dim,
                       hidden_dim=phi_hidden_dim,
                       hidden_depth=phi_hidden_depth).to(device)

        if use_feature_target:
            self.phi_target = copy.deepcopy(self.phi)

        self.mu = MLP(input_dim=state_dim,
                      output_dim=feature_dim,
                      hidden_dim=mu_hidden_dim,
                      hidden_depth=mu_hidden_depth).to(device)

        self.theta = Theta(output_dim=feature_dim).to(device)

        if isinstance(action_space, gym.spaces.Box):
            self.actor = DiagGaussianActor(
                obs_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=critic_and_actor_hidden_dim,
                hidden_depth=2,
                log_std_bounds=[-5., 2.],
            ).to(device)
        elif isinstance(action_space, gym.spaces.Discrete):
            self.actor = DiscreteActor(
                obs_dim=state_dim,
                action_n=action_space.n,
                hidden_dim=critic_and_actor_hidden_dim,
                hidden_depth=2,
            ).to(device)

        self.feature_optimizer = torch.optim.Adam(
            list(self.phi.parameters()) + list(self.mu.parameters()) + list(self.theta.parameters()),
            weight_decay=0, lr=phi_and_mu_lr)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                weight_decay=0, lr=critic_and_actor_lr,
                                                betas=[0.9, 0.999])  # lower lr for actor/alpha
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=critic_and_actor_lr, betas=[0.9, 0.999])

        self.critic = RFFCritic(output_dim=feature_dim+state_dim//2, hidden_dim=critic_and_actor_hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 weight_decay=0, lr=critic_and_actor_lr, betas=[0.9, 0.999])

    def feature_step(self, batch, s_random, a_random, s_prime_random):
        """
        Loss implementation
        """

        state, action, next_state, reward, _ = unpack_batch(batch)

        z_phi = self.phi(torch.concat([state, action], -1))
        z_phi_random = self.phi(torch.concat([s_random, a_random], -1))

        z_mu_next = self.mu(next_state)
        z_mu_next_random = self.mu(s_prime_random)

        assert z_phi.shape[-1] == self.feature_dim
        assert z_mu_next.shape[-1] == self.feature_dim
        # print('z_phi:',z_phi.shape)
        # print('z_mu:',z_mu_next.shape)
######Weird: why not only take the trace?
        # model_loss_pt1 = -2 * z_phi @ z_mu_next.T  # check if need to sum
        model_loss_pt1 = -2 * torch.diag(z_phi @ z_mu_next.T)
        # print(model_loss_pt1)
        model_loss_pt2_a = z_phi_random @ z_mu_next_random.T
        # model_loss_pt2 = model_loss_pt2_a @ model_loss_pt2_a.T
        model_loss_pt2 = model_loss_pt2_a ** 2

        model_loss_pt1_summed = 1. / torch.numel(model_loss_pt1) * torch.sum(model_loss_pt1)
        model_loss_pt2_summed = 1. / torch.numel(model_loss_pt2) * torch.sum(model_loss_pt2)

        model_loss = model_loss_pt1_summed + model_loss_pt2_summed
        r_loss = 0.5 * F.mse_loss(self.theta(z_phi), reward).mean()

        loss = model_loss + r_loss  # + prob_loss

        self.feature_optimizer.zero_grad()
        loss.backward()
        # print gradient
        print('Feature gradients:')
        for name, param in self.phi.named_parameters():
            if param.requires_grad:
                print(name, torch.linalg.norm(param.grad, ord=2).item(), tuple(param.shape), end=' ')
        print()
        self.feature_optimizer.step()

        return {
            'total_loss': loss.item(),
            'model_loss': model_loss.item(),
            'r_loss': r_loss.item(),
            # 'prob_loss': prob_loss.item(),
        }

    def update_feature_target(self):
        for param, target_param in zip(self.phi.parameters(), self.phi_target.parameters()):
            target_param.data.copy_(self.feature_tau * param.data + (1 - self.feature_tau) * target_param.data)

    def critic_step(self, batch):
        """
        Critic update step
        """
        state, action, next_state, reward, done = unpack_batch(batch)
        cur_state = torch.take_along_dim(state,torch.tensor([[0,1]]).to(self.device),-1)
        cur_next_state = torch.take_along_dim(next_state,torch.tensor([[0,1]]).to(self.device),-1)
        goal_state = torch.take_along_dim(state,torch.tensor([[2,3]]).to(self.device),-1)
        goal_next_state = torch.take_along_dim(next_state,torch.tensor([[2,3]]).to(self.device),-1)
        # print('state:{a}, action:{b}, next_state:{c}, reward:{d}, done:{e}'.format(a=state.shape, b=action.shape, c=next_state.shape, d=reward.shape, e=done.shape))
        with torch.no_grad():
            dist = self.actor(next_state)
            if isinstance(self.action_space, gym.spaces.Box):
                next_action = dist.rsample()
            else:   
                next_action = dist.sample().unsqueeze(-1)
            # next_action = dist.sample().unsqueeze(-1)
            # print('next_action:',next_action.shape) 
            next_action_log_pi = dist.log_prob(next_action).sum(-1, keepdim=True)
            # print('next_action_log_pi:',next_action_log_pi.shape)
            z_phi = self.phi(torch.concat([cur_state, action], -1))
            z_phi_next = self.phi(torch.concat([cur_next_state, next_action], -1))

            next_q1, next_q2 = self.critic_target(torch.concat([z_phi_next, goal_next_state], -1))
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_action_log_pi
            target_q = reward + (1. - done) * self.discount * next_q

        q1, q2 = self.critic(z_phi)
        q1_loss = F.mse_loss(target_q, q1)
        q2_loss = F.mse_loss(target_q, q2)
        q_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        print('Critic gradients:')
        for name, param in self.critic.named_parameters():
            if param.requires_grad:
                print(name, torch.linalg.norm(param.grad, ord=2).item(), tuple(param.shape), end=' ')
        print()
        self.critic_optimizer.step()

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1': q1.mean().item(),
            'q2': q2.mean().item()
        }

    def update_actor_and_alpha(self, batch):
        """
        Actor update step
        """
        dist = self.actor(batch.state)
        if isinstance(self.action_space, gym.spaces.Box):
            action = dist.rsample()
        else:
            action = dist.sample().unsqueeze(-1)
        # action = dist.sample().unsqueeze(-1)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        print('logits:',dist.logits)
        print('prob:',dist.probs)
        print('log_prob:',log_prob)
        z_phi = self.phi(torch.concat([batch.state, action], -1))

        q1, q2 = self.critic(z_phi)
        q = torch.min(q1, q2)

        actor_loss = ((self.alpha) * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        print('Actor gradients:')
        for name, param in self.actor.named_parameters():
            if param.requires_grad:
                print(name, torch.linalg.norm(param.grad, ord=2).item(), tuple(param.shape), end=' ')
        print()
        self.actor_optimizer.step()

        info = {'actor_loss': actor_loss.item()}

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            info['alpha_loss'] = alpha_loss
            info['alpha'] = self.alpha

        return info

    def train(self, buffer, batch_size):
        """
        One train step
        """
        self.steps += 1

        # Feature step
        for _ in range(self.extra_feature_steps + 1):
            batch_1 = buffer.sample(batch_size)
            batch_2 = buffer.sample(batch_size)
            s_random, a_random, s_prime_random, _, _ = unpack_batch(batch_2)

            feature_info = self.feature_step(batch_1, s_random, a_random, s_prime_random)

            # Update the feature network if needed
            if self.use_feature_target:
                self.update_feature_target()

        # Critic step
        critic_info = self.critic_step(batch_1)

        # Actor and alpha step
        actor_info = self.update_actor_and_alpha(batch_1)

        # Update the frozen target models
        self.update_target()

        return {
            **feature_info,
            **critic_info,
            **actor_info,
        }



class Discrete_SPEDERSACAgent(SACAgent):
    """
    SAC with VAE learned latent features
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            action_space,
            phi_and_mu_lr=-1,
            # 3e-4 was originally proposed in the paper, but seems to results in fluctuating performance
            phi_hidden_dim=-1,
            phi_hidden_depth=-1,
            mu_hidden_dim=-1,
            mu_hidden_depth=-1,
            critic_and_actor_lr=-1,
            critic_and_actor_hidden_dim=-1,
            discount=0.99,
            target_update_period=2,
            tau=0.005,
            alpha=0.1,
            auto_entropy_tuning=True,
            hidden_dim=1024,
            feature_tau=0.005,
            feature_dim=2048,  # latent feature dim
            use_feature_target=True,
            extra_feature_steps=1,
            device=device,
            actor_name=None,
            n_task=9,
            n_width=3,
            n_height=3,
            pretrain_model_path=None
    ):

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            tau=tau,
            alpha=alpha,
            discount=discount,
            target_update_period=target_update_period,
            auto_entropy_tuning=auto_entropy_tuning,
            hidden_dim=hidden_dim,
            device=device
        )
        self.feature_dim = feature_dim
        self.feature_tau = feature_tau
        self.use_feature_target = use_feature_target
        self.extra_feature_steps = extra_feature_steps
        self.action_space = action_space
        self.action_dim = action_dim
        self.n_task = n_task
        self.n_width = n_width
        self.n_height = n_height
        self.n_action = action_space.n
        # grid_col, grid_row = torch.meshgrid(torch.arange(0, self.n_width), torch.arange(0, self.n_height), indexing='xy')
        # self.states_all = torch.concat([grid_row.reshape(-1, 1), grid_col.reshape(-1, 1)], -1).float().to(device)
        # self.actions_all = torch.arange(self.action_space.n).reshape(-1, 1).float().to(device)
        self.states_all = torch.eye(self.n_width*self.n_height).to(device)
        self.actions_all = torch.eye(self.n_action).to(device)
        self.state_action_pairs = torch.concat([torch.repeat_interleave(self.states_all, self.action_space.n, dim=0), torch.tile(self.actions_all, (self.states_all.shape[0], 1))], -1).to(device)
        self.task_id_all = torch.eye(self.n_task).to(self.device)

        self.phi = MLP(input_dim=state_dim + action_dim,
                       output_dim=feature_dim,
                       hidden_dim=phi_hidden_dim,
                       hidden_depth=phi_hidden_depth).to(device)
        if use_feature_target:
            self.phi_target = copy.deepcopy(self.phi)
        self.mu = MLP(input_dim=state_dim,
                      output_dim=feature_dim,
                      hidden_dim=mu_hidden_dim,
                      hidden_depth=mu_hidden_depth).to(device)
        ### only for fixed mu
        # self.phi = SingleMatrixFixedPhi(n_width=self.n_width, n_action=self.action_space.n,
        #                         feature_dim=self.feature_dim).to(device)
        # if use_feature_target:
        #     self.phi_target = copy.deepcopy(self.phi)
        # self.mu = SingleMatrixFixedMu(n_width=self.n_width, feature_dim=self.feature_dim).to(device)

        for name, param in self.mu.named_parameters():
            print(name, param.shape)

        # self.w = SingleMatrix(input_dim=n_task, output_dim=feature_dim).to(device)
        self.w = SingleMatrixEmbedding(n_task=n_task, feature_dim=feature_dim).to(device)

        self.feature_optimizer = torch.optim.Adam(
            list(self.phi.parameters()) + list(self.mu.parameters()) + list(self.w.parameters()),
            weight_decay=0, lr=phi_and_mu_lr)
        
        # self.critic = SingleMatrixCritic(input_dim=n_task, output_dim=feature_dim).to(device)
        self.critic = SingleMatrixCriticEmbedding(n_task=n_task, feature_dim=feature_dim).to(device)
        # self.critic = torch.rand(2, n_task, feature_dim).to(device) # a table for u
        # self.critic.requires_grad = True
        # self.critic_bias = torch.rand(2, 1, feature_dim).to(device)
        # self.critic_bias.requires_grad = True
        self.critic_target = copy.deepcopy(self.critic)
        # self.critic_target_bias = copy.deepcopy(self.critic_target)
        # self.critic = RFFCritic(input_dim=state_dim//2, output_dim=feature_dim, hidden_dim=critic_and_actor_hidden_dim).to(device)
        # self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(list(self.critic.parameters())+list(self.phi.parameters()),
                                                 weight_decay=0, lr=critic_and_actor_lr, betas=[0.9, 0.999])
        self.all_optimizer = torch.optim.Adam(
            list(self.mu.parameters()) + list(self.w.parameters()) + list(self.critic.parameters())+list(self.phi.parameters()),
            weight_decay=0, lr=3*phi_and_mu_lr)
        if isinstance(action_space, gym.spaces.Box):
            self.actor = DiagGaussianActor(
                obs_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=critic_and_actor_hidden_dim,
                hidden_depth=2,
                log_std_bounds=[-5., 2.],
            ).to(device)
        elif isinstance(action_space, gym.spaces.Discrete):
            if actor_name == 'softmax':
                self.actor = SoftmaxActor(phi=self.phi, critic=self.critic, action_n=action_space.n, 
                                          n_task=n_task, feature_dim=feature_dim, device=device,
                                          alpha=self.alpha, log_alpha=self.log_alpha, state_dim=state_dim,
                                          action_dim=action_dim).to(device)
            elif actor_name == 'discrete':
                self.actor = DiscreteActor(
                    obs_dim=state_dim,
                    action_n=action_space.n,
                    hidden_dim=critic_and_actor_hidden_dim,
                    hidden_depth=2,
                ).to(device)
            elif actor_name == 'random':
                self.actor = RandomActor(action_n=action_space.n, device=device).to(device)
        
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
        #                                         weight_decay=0, lr=critic_and_actor_lr,
        #                                         betas=[0.9, 0.999])  # lower lr for actor/alpha
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=critic_and_actor_lr, betas=[0.9, 0.999])
        




    def __str__(self):
        print('Phi:')
        for name, param in self.phi.named_parameters():
            print(name, param.shape)
        print('Mu:')
        for name, param in self.mu.named_parameters():
            print(name, param.shape)
        print('W:')
        for name, param in self.w.named_parameters():
            print(name, param.shape)
        # print(self.w)
        # print(self.w_bias)
        print('Actor:')
        for name, param in self.actor.named_parameters():
            print(name, param.shape)
        print('Critic:')
        for name, param in self.critic.named_parameters():
            print(name, param.shape)
        # print(self.critic)
        # print(self.critic_bias)
        print('Log Alpha:', self.log_alpha)
        return ''

    def select_action(self, state, explore=True):
        # Maybe change the critic network to output four hidden vectors at the same time

        if len(state.shape) == 1:
            state = torch.FloatTensor(np.array([state])).to(self.device)
        else:
            state = torch.FloatTensor(state).to(self.device)
        # print('state:',state)
        # print(self.device)
        # cur_state = torch.take_along_dim(state,torch.tensor([[0,1]]).to(self.device),-1)
        # goal_state = torch.take_along_dim(state,torch.tensor([[2,3]]).to(self.device),-1)
        # z_w = self.w(goal_state)
        # assert z_w.shape[-1] == self.feature_dim
        dist = self.actor(state)
        # dist = self.actor(torch.concat([cur_state, z_w], -1))
        if explore:
            action = dist.sample()
        if isinstance(self.actor, DiagGaussianActor):
            action = dist.rsample() if explore else dist.mean
            assert action.ndim == 2 and action.shape[0] == 1
        elif isinstance(self.actor, DiscreteActor):
            action = dist.sample() if explore else torch.argmax(dist.probs).unsqueeze(0)
            assert action.shape[0] == 1
        elif isinstance(self.actor, SoftmaxActor):
            action = dist.sample() if explore else torch.argmax(dist.probs).unsqueeze(0)
            assert action.shape[0] == 1
        elif isinstance(self.actor, RandomActor):
            action = dist.sample() 
        action = action.clamp(*self.action_range)
        return util.to_np(action[0])

    def feature_step(self, batch, s_random, a_random, s_prime_random):
        """
        Loss implementation
        """

        state, action, next_state, reward, _ = unpack_batch(batch)
        assert state.shape[-1] == self.state_dim + self.n_task
        assert action.shape[-1] == self.action_dim
        assert s_random.shape[-1] == self.state_dim + self.n_task
        assert a_random.shape[-1] == self.action_dim
        assert s_prime_random.shape[-1] == self.state_dim + self.n_task
        assert next_state.shape[-1] == self.state_dim + self.n_task
        assert reward.shape[-1] == 1

        # grid_col, grid_row = np.meshgrid(np.arange(self.n_width), np.arange(self.n_height))
        # states = np.concatenate([grid_row.reshape(-1,1), grid_col.reshape(-1,1)], axis=-1)
        z_mu_all = self.mu(self.states_all)


        cur_state = state[..., :self.state_dim]
        cur_next_state = next_state[..., :self.state_dim]
        task_id = state[..., self.state_dim:]
        cur_s_random = s_random[..., :self.state_dim]
        cur_s_prime_random = s_prime_random[..., :self.state_dim]

        z_phi = self.phi(torch.concat([cur_state, action], -1))
        z_phi_random = self.phi(torch.concat([cur_s_random, a_random], -1))

        z_mu_next = self.mu(cur_next_state)
        z_mu_next_random = self.mu(cur_s_prime_random)

        assert z_phi.shape[-1] == self.feature_dim
        assert z_mu_next.shape[-1] == self.feature_dim
        # print('z_phi:',z_phi.shape)
        # print('z_mu:',z_mu_next.shape)
######Weird: why not only take the trace?
        # model_loss_pt1 = -2 * z_phi @ z_mu_next.T  # check if need to sum
        model_loss_pt1 = -2 * torch.diag(z_phi @ z_mu_next.T)
        # print(model_loss_pt1)
        model_loss_pt2_a = z_phi_random @ z_mu_all.T
        # model_loss_pt2 = model_loss_pt2_a @ model_loss_pt2_a.T
        model_loss_pt2 = model_loss_pt2_a ** 2

        model_loss_pt1_summed = 1. / torch.numel(model_loss_pt1) * torch.sum(model_loss_pt1)
        model_loss_pt2_summed = 1. / torch.numel(model_loss_pt2) * torch.sum(model_loss_pt2)

        model_loss = model_loss_pt1_summed + model_loss_pt2_summed
        # z_w = task_id@self.w + self.w_bias
        z_w = self.w(task_id)
        assert z_w.shape[-1] == self.feature_dim
        assert len(z_w.shape) == 2

        predict_r = torch.sum(z_phi * z_w, dim=-1, keepdim=True)
        w_l1_loss = torch.abs(z_w).mean()
        phi_l1_loss = torch.abs(z_phi).mean()
        mu_l1_loss = torch.abs(z_mu_next).mean()
        assert predict_r.shape[-1] == 1
        r_loss = 0.5 * F.mse_loss(predict_r, reward).mean()

        
        


        # loss = model_loss + r_loss + phi_l1_loss + w_l1_loss + mu_l1_loss  # + prob_loss
        loss = model_loss + r_loss
        self.feature_optimizer.zero_grad()
        loss.backward()
        # print gradient
        # print('Feature gradients:')
        # for name, param in self.phi.named_parameters():
        #     if param.requires_grad:
        #         print(name, torch.linalg.norm(param.grad, ord=2).item(), tuple(param.shape), end=' ')
        # print()
        # for name, param in self.mu.named_parameters():
        #     if param.requires_grad:
        #         print(name, torch.linalg.norm(param.grad, ord=2).item(), tuple(param.shape), end=' ')
        # print()
        # for name, param in self.w.named_parameters():
        #     if param.requires_grad:
        #         print(name, torch.linalg.norm(param.grad, ord=2).item(), tuple(param.shape), end=' ')
        # print()
        #clip gradient
        # for param in self.phi.parameters():
        #     param.grad = torch.clamp(param.grad, -1, 1)
        # for param in self.mu.parameters():
        #     param.grad = torch.clamp(param.grad, -1, 1)
        # for param in self.w.parameters():
        #     param.grad = torch.clamp(param.grad, -1, 1)
        self.feature_optimizer.step()

        return {
            'total_loss': loss.item(),
            'model_loss': model_loss.item(),
            'r_loss': r_loss.item(),
            'w_l1_loss': w_l1_loss.item(),
            'phi_l1_loss': phi_l1_loss.item(),
            'mu_l1_loss': mu_l1_loss.item(),
            # 'prob_loss': prob_loss.item(),
        }

    def update_feature_target(self):
        for param, target_param in zip(self.phi.parameters(), self.phi_target.parameters()):
            target_param.data.copy_(self.feature_tau * param.data + (1 - self.feature_tau) * target_param.data)

    def critic_step(self, batch):
        """
        Critic update step
        """
        state, action, next_state, reward, done = unpack_batch(batch)
        assert state.shape[-1] == self.state_dim + self.n_task
        assert action.shape[-1] == self.action_dim
        assert next_state.shape[-1] == self.state_dim + self.n_task
        assert reward.shape[-1] == 1
        assert done.shape[-1] == 1
        # cur_state = torch.take_along_dim(state,torch.tensor([[0,1]]).to(self.device),-1)
        # next_cur_state = torch.take_along_dim(next_state,torch.tensor([[0,1]]).to(self.device),-1)
        # goal_state = torch.take_along_dim(state,torch.tensor([[2,3]]).to(self.device),-1)
        # task_id = torch.take_along_dim(state,torch.arange(2,self.n_task).to(self.device),-1)
        # goal_next_state = torch.take_along_dim(next_state,torch.tensor([[2,3]]).to(self.device),-1)
        # next_task_id = torch.take_along_dim(next_state,torch.arange(2,self.n_task).to(self.device),-1)
        # print('state:{a}, action:{b}, next_state:{c}, reward:{d}, done:{e}'.format(a=state.shape, b=action.shape, c=next_state.shape, d=reward.shape, e=done.shape))
        cur_state = state[..., :2]
        next_cur_state = next_state[..., :2]
        task_id = state[..., 2:]
        next_task_id = next_state[..., 2:]
        with torch.no_grad():
            z_w = self.w(task_id)
            z_w_next = self.w(next_task_id)  
            # pi_input = torch.concat([cur_next_state, goal_next_state], -1)
            # pi_input = torch.concat([cur_next_state, z_w_next], -1)
            pi_input = state
            next_action_log_pi = self.actor.evaluate_matrix(pi_input) # [batch_size, action_n]
            z_phi = self.phi(torch.concat([cur_state, action], -1))
            next_q = 0
            # next_u_double = next_task_id@self.critic + self.critic_bias # [2, batch_size, feature_dim]
            # assert next_u_double.shape[-1] == self.feature_dim
            # assert next_u_double.shape == (2, state.shape[0], self.feature_dim)

            for i in range(self.action_space.n):
                next_action = torch.tensor([i]).repeat(state.shape[0], 1).to(device)
                z_phi_next = self.phi(torch.concat([next_cur_state, next_action], -1)) # [batch_size, feature_dim]
                next_u1, next_u2 = self.critic_target(next_task_id)
                # next_q1, next_q2 = self.critic(goal_next_state)
                # next_q_double = torch.sum(next_u_double*z_phi_next.unsqueeze(0), -1, keepdim=True) # [2, batch_size, 1]
                next_q1 = torch.sum(next_u1*z_phi_next, -1, keepdim=True)
                next_q2 = torch.sum(next_u2*z_phi_next, -1, keepdim=True)
                # next_q_a = next_q_double.min(0, keepdim=False).values # [batch_size, 1]
                next_q_a = torch.min(next_q1, next_q2)
                next_q += (next_q_a - self.alpha * next_action_log_pi[:, i].unsqueeze(-1)) * next_action_log_pi[:,i].unsqueeze(-1).exp()

            # next_q = (next_action_log_pi.exp() * torch.stack(next_q, dim=-1)).sum(-1, keepdim=True)
            target_q = reward + (1. - done) * self.discount * next_q

            # mu = self.mu(self.states_all)
            # transit_prob = torch.sum(z_phi*mu, -1, keepdim=True)
            # transit_prob = torch.clamp(transit_prob, 1e-3, 1-1e-3)
            # target_u = self.w(task_id) + self.discount * next_q * mu / transit_prob

        u1, u2 = self.critic(task_id)
        # u1_loss = F.mse_loss(target_u, u1)
        # u2_loss = F.mse_loss(target_u, u2)6
        # u_loss = u1_loss + u2_loss
        # u_double = task_id@self.critic + self.critic_bias # [2, batch_size, feature_dim]
        # assert u_double.shape[-1] == self.feature_dim
        # assert q1.shape[-1] == self.feature_dim
        # assert q2.shape[-1] == self.feature_dim
        q1 = torch.sum(u1*z_phi, -1, keepdim=True)
        q2 = torch.sum(u2*z_phi, -1, keepdim=True)
        assert q1.shape[-1] == 1
        assert q2.shape[-1] == 1
        # q_double = torch.sum(u_double*z_phi.unsqueeze(0), -1, keepdim=True) # [2, batch_size, 1]
        # q_loss = F.mse_loss(target_q.expand_as(q_double), q_double)
        # target_q_expand = target_q.expand_as(q_double)
        # q_loss = F.mse_loss(target_q_expand, q_double)
        # q1 = q_double[0]
        # q2 = q_double[1]
        q1_loss = F.mse_loss(target_q, q1)
        q2_loss = F.mse_loss(target_q, q2)
        q_loss = q1_loss + q2_loss
        u1_l1_loss = torch.abs(u1).mean()
        u2_l1_loss = torch.abs(u2).mean()
        # loss = q_loss + (u1_l1_loss + u2_l1_loss)*0.5
        loss = q_loss
        self.critic_optimizer.zero_grad()
        # self.feature_optimizer.zero_grad()
        # q_loss.backward()
        # loss = q_loss + u_loss
        loss.backward()
        # print('Critic gradients:')
        # for name, param in self.critic.named_parameters():
        #     if param.requires_grad:
        #         print(name, torch.linalg.norm(param.grad, ord=2).item(), tuple(param.shape), end=' ')
        # print()
        # print('Feature gradients:')
        # print('w:')
        # for name, param in self.w.named_parameters():
        #     if param.requires_grad:
        #         print(name, torch.linalg.norm(param.grad, ord=2).item(), tuple(param.shape), end=' ')
        # print()
        # print('phi:')
        # for name, param in self.phi.named_parameters():
        #     if param.requires_grad:
        #         print(name, torch.linalg.norm(param.grad, ord=2).item(), tuple(param.shape), end=' ')
        # print()
        # print('mu:')
        # for name, param in self.mu.named_parameters():
        #     if param.requires_grad:
        #         print(name, torch.linalg.norm(param.grad, ord=2).item(), tuple(param.shape), end=' ')
        # print()
        # for param in self.critic.parameters():
        #     param.grad = torch.clamp(param.grad, -1, 1)
        # for param in self.w.parameters():
        #     param.grad = torch.clamp(param.grad, -1, 1)
        self.critic_optimizer.step()
        # self.feature_optimizer.step()

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1': q1.mean().item(),
            'q2': q2.mean().item(),
            'u_l1_loss': (u1_l1_loss+u2_l1_loss).item(),
            # 'u2_loss': u2_loss.item()
        }
    def critic_step_matrix(self, batch):
        """
        Critic update step
        """
        state, action, next_state, reward, done = unpack_batch(batch)
        assert state.shape[-1] == self.state_dim + self.n_task
        assert action.shape[-1] == self.action_dim
        assert next_state.shape[-1] == self.state_dim + self.n_task
        assert reward.shape[-1] == 1
        assert done.shape[-1] == 1
        cur_state = state[..., :self.state_dim]
        next_cur_state = next_state[..., :self.state_dim]
        task_id = state[..., self.state_dim:]
        next_task_id = next_state[..., self.state_dim:]
        z_phi = self.phi(torch.concat([cur_state, action], -1))
        with torch.no_grad():
            pi_input = state
            next_action_log_pi = self.actor.evaluate_matrix(pi_input) # [batch_size, action_n]
            
            state_action_pair = torch.concat([torch.repeat_interleave(next_cur_state,self.n_action,dim=0),torch.tile(self.actions_all,(cur_state.shape[0],1))],-1)
            z_phi_next = self.phi(state_action_pair).reshape(cur_state.shape[0],self.n_action,self.feature_dim).detach()
            next_u1, next_u2 = self.critic_target(next_task_id)
            next_q1 = torch.sum(next_u1.unsqueeze(1)*z_phi_next, -1)
            next_q2 = torch.sum(next_u2.unsqueeze(1)*z_phi_next, -1)
            assert next_q1.shape[-1] == self.n_action
            assert next_q2.shape[-1] == self.n_action
            next_q = torch.min(next_q1, next_q2)
            assert next_q.shape == (cur_state.shape[0], self.n_action)
            next_q_h = torch.sum((next_q - self.alpha * next_action_log_pi) * next_action_log_pi.exp(), -1, keepdim=True)
            assert next_q_h.shape == (cur_state.shape[0], 1)
            target_q = reward + (1. - done) * self.discount * next_q_h
            assert target_q.shape == (cur_state.shape[0], 1)

        u1, u2 = self.critic(task_id)
        q1 = torch.sum(u1*z_phi, -1, keepdim=True)
        q2 = torch.sum(u2*z_phi, -1, keepdim=True)
        assert q1.shape[-1] == 1
        assert q2.shape[-1] == 1
        q1_loss = F.mse_loss(target_q, q1)
        q2_loss = F.mse_loss(target_q, q2)
        q_loss = q1_loss + q2_loss
        u1_l1_loss = torch.abs(u1).mean()
        u2_l1_loss = torch.abs(u2).mean()
        loss = q_loss
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        # self.feature_optimizer.step()

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1': q1.mean().item(),
            'q2': q2.mean().item(),
            'u_l1_loss': (u1_l1_loss+u2_l1_loss).item(),
            # 'u2_loss': u2_loss.item()
        }

    def uw_step_matrix(self):

        # task_id_repeat = task_id.unsqueeze(1).repeat(1, self.n_state, 1)
        target_u = torch.zeros((self.n_task, self.feature_dim)).to(self.device)
        # with torch.no_grad():
        phi_matrix = self.phi(self.state_action_pairs).reshape(self.states_all.shape[0], self.action_space.n, self.feature_dim).detach()
        # assert target_u.shape == (state.shape[0], self.feature_dim)
        mu_matrix = self.mu(self.states_all)/self.states_all.shape[0]
        w = self.w(self.task_id_all)
        u1, u2 = self.critic(self.task_id_all)

        q1 = phi_matrix@u1.T
        q2 = phi_matrix@u2.T
        q = torch.min(q1, q2).transpose(1,2)
        assert q.shape == (self.states_all.shape[0], self.n_task, self.action_space.n)
        pi_input_all = torch.concat([self.states_all.unsqueeze(1).repeat(1,self.n_task,1), self.task_id_all.unsqueeze(0).repeat(self.states_all.shape[0],1,1)], -1)
        assert pi_input_all.shape == (self.states_all.shape[0], self.n_task, self.state_dim + self.n_task)
        action_log_pi_matrix = self.actor.evaluate_matrix(pi_input_all.reshape(-1, self.state_dim+self.n_task)).reshape(self.states_all.shape[0], self.n_task, self.action_space.n)
        V = torch.sum(action_log_pi_matrix.exp() * (q - self.alpha.detach() * action_log_pi_matrix), -1)
        assert V.shape == (self.states_all.shape[0], self.n_task)
        target_u = w + self.discount * (V.T@mu_matrix)
        assert target_u.shape == (self.n_task, self.feature_dim)

        u1_loss = F.mse_loss(target_u, u1)
        u2_loss = F.mse_loss(target_u, u2)
        u_loss = u1_loss + u2_loss
        # print('u1_loss:',u1_loss)
        self.all_optimizer.zero_grad()
        u_loss.backward()
        self.all_optimizer.step()
        return {
            'u1_loss': u1_loss.item(),
            'u2_loss': u2_loss.item(),
            'u1': u1.mean().item(),
            'u2': u2.mean().item()
    }
    def uw_step(self):

        # task_id_repeat = task_id.unsqueeze(1).repeat(1, self.n_state, 1)
        target_u = torch.zeros((self.n_task, self.feature_dim)).to(self.device)
        # with torch.no_grad():
        phi_matrix = self.phi(self.state_action_pairs).reshape(self.states_all.shape[0], self.action_space.n, self.feature_dim).detach()
            

        # assert target_u.shape == (state.shape[0], self.feature_dim)
        mu_matrix = self.mu(self.states_all)/self.states_all.shape[0]
        w = self.w(self.task_id_all)
        u1, u2 = self.critic(self.task_id_all)
        for i, task_id in enumerate(self.task_id_all):
            # u1 = u1.unsqueeze(1).unsqueeze(-1)
            # u2 = u2.unsqueeze(1).unsqueeze(-1)
            # assert u1.shape == (state.shape[0], 1, self.feature_dim, 1)
            q1 = phi_matrix@u1[i]
            q2 = phi_matrix@u2[i]
            q = torch.min(q1, q2)
            assert q.shape == (self.states_all.shape[0], self.action_space.n)
            # assert q2.shape == (state.shape[0], self.n_state, self.action_space.n, 1)
            pi_input = torch.concat([self.states_all, task_id.reshape(1,-1).repeat(self.states_all.shape[0],1)], -1)
            action_log_pi = self.actor.evaluate(pi_input)
            v = torch.sum(action_log_pi.exp() * (q - self.alpha * action_log_pi), -1).detach()
            assert v.shape == (self.states_all.shape[0],)
            target_u[i] = w[i] + self.discount * v@mu_matrix
        u1_loss = F.mse_loss(target_u, u1)
        u2_loss = F.mse_loss(target_u, u2)
        u_loss = u1_loss + u2_loss
        # print('u1_loss:',u1_loss)
        self.all_optimizer.zero_grad()
        u_loss.backward()
        self.all_optimizer.step()
        return {
            'u1_loss': u1_loss.item(),
            'u2_loss': u2_loss.item(),
            'u1': u1.mean().item(),
            'u2': u2.mean().item()
    }

    def update_actor_and_alpha(self, batch):
        """
        Actor update step
        """
        # action = dist.sample().unsqueeze(-1)
        # batch_cur_state = torch.take_along_dim(batch.state,torch.tensor([[0,1]]).to(self.device),-1)
        # batch_goal_state = torch.take_along_dim(batch.state,torch.tensor([[2,3]]).to(self.device),-1)
        batch_cur_state = batch.state[...,:2]
        batch_task_id = batch.state[...,2:]
        # z_w = self.w(batch_task_id)
        # pi_input = torch.concat([batch_cur_state, batch_goal_state], -1)
        # pi_input = torch.concat([batch_cur_state, z_w], -1)
        pi_input = batch.state
        action_log_pi = self.actor.evaluate(pi_input)
        actor_loss = 0
        for i in range(self.action_space.n):
            action = torch.tensor([i]).repeat(batch.state.shape[0], 1).to(device)
            z_phi = self.phi(torch.concat([batch_cur_state, action], -1))
            # q1, q2 = self.critic(torch.concat([z_phi, batch_goal_state], -1))
            u1, u2 = self.critic(batch_task_id)
            assert u1.shape[-1] == self.feature_dim
            assert u2.shape[-1] == self.feature_dim
            q1 = torch.sum(u1*z_phi, -1, keepdim=True)
            q2 = torch.sum(u2*z_phi, -1, keepdim=True)
            q = torch.min(q1, q2)
            assert q.shape[-1] == 1
            actor_loss += ((self.alpha) * action_log_pi[:, i].unsqueeze(-1) - q) * action_log_pi[:,i].unsqueeze(-1).exp()
        
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad()
        # self.feature_optimizer.zero_grad()
        actor_loss.backward()
        # print('Actor gradients:')
        # for name, param in self.actor.named_parameters():
        #     print(name, param)
            # if param.requires_grad:
            #     print(name, torch.linalg.norm(param.grad, ord=2).item(), tuple(param.shape), end=' ')
        # print()
        # for param in self.actor.parameters():
        #     param.grad = torch.clamp(param.grad, -1, 1)
        # for param in self.w.parameters():
        #     param.grad = torch.clamp(param.grad, -1, 1)
        self.actor_optimizer.step()
        # self.feature_optimizer.step()

        info = {'actor_loss': actor_loss.item()}

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            # alpha_loss = (action_log_pi.exp() * (self.alpha * (-action_log_pi - self.target_entropy))).sum(-1,keepdims=True).detach().mean()
            alpha_loss = (self.alpha * (-action_log_pi - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha.grad = torch.clamp(self.log_alpha.grad, -1, 1)
            self.log_alpha_optimizer.step()

            info['alpha_loss'] = alpha_loss
            info['alpha'] = self.alpha

        return info
    
    def state_dict(self):
        return {'actor': self.actor.state_dict(),
				'critic': self.critic.state_dict(),
				'log_alpha': self.log_alpha,
				'phi': self.phi.state_dict(),
				'mu': self.mu.state_dict(),
				'w': self.w.state_dict()}

    def load_state_dict(self, state_dict):
        # print(state_dict.keys())
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.log_alpha = state_dict['log_alpha']
        self.phi.load_state_dict(state_dict['phi'])
        self.mu.load_state_dict(state_dict['mu'])
        self.w.load_state_dict(state_dict['w'])
	
    def train(self, buffer, batch_size):
        """
        One train step
        """
        self.steps += 1

        # Feature step
        for _ in range(self.extra_feature_steps + 1):
            batch_1 = buffer.sample(batch_size)
            batch_2 = buffer.sample(batch_size)
            # print(torch.concat([batch_1.state, batch_1.action, batch_1.next_state, batch_1.reward], -1))
            # print(batch_2)
            s_random, a_random, s_prime_random, _, _ = unpack_batch(batch_2)

            feature_info = self.feature_step(batch_1, s_random, a_random, s_prime_random)

            # Update the feature network if needed
            if self.use_feature_target:
                self.update_feature_target()

        uw_info = self.uw_step_matrix()
        # Critic step
        critic_info = self.critic_step_matrix(batch_1)


        # Actor and alpha step
        # actor_info = self.update_actor_and_alpha(batch_1)
        actor_info = dict()
        actor_info['log_alpha'] = self.log_alpha

        # Update the frozen target models
        self.update_target()

        return {
            **feature_info,
            **critic_info,
            **actor_info,
            **uw_info
        }
