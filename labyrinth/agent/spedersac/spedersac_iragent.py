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
from utils.util import MLP, MLP_Phi

from agent.sac.sac_agent import SACAgent
from agent.sac.actor import DiagGaussianActor
from agent.sac.actor import DiscreteActor
from agent.sac.actor import SoftmaxActor
from torchinfo import summary
from agent.spedersac.spedersac_agent import SingleMatrix, SingleMatrixCritic, \
            SingleMatrixFixedMu, SingleMatrixFixedPhi, SingleMatrixEmbedding, SingleMatrixCriticEmbedding, \
            value_iteration
from sklearn.decomposition import PCA
from agent.spedersac.spedersac_agent import pca_transform, init_inv_w_u
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_printoptions(precision=1, threshold=10000)
np.set_printoptions(precision=1, threshold=10000)
class Inverse_Discrete_SPEDERSACAgent():
    """
    Q-Learning with Discrete Actor
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            action_space=None,
            phi_and_mu_lr = -1,
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

        # super().__init__(
        #     state_dim=state_dim,
        #     action_dim=action_dim,
        #     action_space=action_space,
        #     tau=tau,
        #     alpha=alpha,
        #     discount=discount,
        #     target_update_period=target_update_period,
        #     auto_entropy_tuning=auto_entropy_tuning,
        #     hidden_dim=hidden_dim,
        #     device=device
        # )
        self.feature_dim = feature_dim
        self.feature_tau = feature_tau
        self.use_feature_target = use_feature_target
        self.extra_feature_steps = extra_feature_steps
        self.action_space = action_space
        self.n_action = action_dim
        self.action_dim = action_dim
        self.n_task = n_task
        self.alpha = torch.tensor(alpha).to(device)
        self.log_alpha = torch.tensor(np.log(alpha)).to(device)
        # self.n_width = n_width
        # self.n_height = n_height
        # self.n_state = n_width*n_height
        self.state_dim = state_dim
        self.n_state = state_dim
        self.device = device
        self.steps = 0
        self.discount = discount
        # grid_col, grid_row = torch.meshgrid(torch.arange(0, self.n_width), torch.arange(0, self.n_height), indexing='xy')
        # self.states_all = torch.concat([grid_row.reshape(-1, 1), grid_col.reshape(-1, 1)], -1).float().to(device)
        # self.actions_all = torch.arange(self.action_space.n).reshape(-1, 1).float().to(device)
        # self.state_action_pairs = torch.concat([torch.repeat_interleave(self.states_all, self.action_space.n, dim=0), torch.tile(self.actions_all, (self.states_all.shape[0], 1))], -1).to(device)
        self.task_id_all = torch.eye(self.n_task).to(self.device)

        # pretrained_model = torch.load(pretrain_model_path)
        # print(f'load pretrained model from: {pretrain_model_path}')

        # self.phi = MLP(input_dim=state_dim + action_dim,
        #                output_dim=feature_dim,
        #                hidden_dim=phi_hidden_dim,
        #                hidden_depth=phi_hidden_depth).to(device)
        # print('state_dim:', state_dim)
        self.phi = MLP_Phi(state_dim, action_dim, phi_hidden_dim, phi_hidden_depth, feature_dim).to(device)
        # self.phi.load_state_dict(pretrained_model['phi'])
        # if use_feature_target:
        #     self.phi_target = copy.deepcopy(self.phi)
        self.mu = MLP(input_dim=state_dim,
                      output_dim=feature_dim,
                      hidden_dim=mu_hidden_dim,
                      hidden_depth=mu_hidden_depth).to(device)
        # self.mu.load_state_dict(pretrained_model['mu'])
        self.states_all = torch.eye(self.state_dim).to(device)
        self.actions_all = torch.eye(self.action_dim).to(device)
        self.state_action_pairs = torch.concat([torch.repeat_interleave(self.states_all, self.action_dim, dim=0), torch.tile(self.actions_all, (self.states_all.shape[0], 1))], -1).to(device)
        # self.phi_matrix = self.phi(self.state_action_pairs).reshape(self.states_all.shape[0], self.action_space.n, self.feature_dim).detach()
        # self.mu_matrix = self.mu(self.states_all)/self.states_all.shape[0]
        # self.mu_matrix = self.mu_matrix.detach()

        ###No loading anymore
        # self.phi = SingleMatrixFixedPhi(self.state_dim, self.action_dim, self.feature_dim).to(device)
        # self.phi.load_state_dict(pretrained_model['phi'])
        # self.phi.load_state_dict({'matrix.weight': torch.FloatTensor(np.load('Phi_estimated.npy'))})
        # self.mu = SingleMatrixFixedMu(self.state_dim, self.feature_dim).to(device)
        # self.mu.load_state_dict(pretrained_model['mu'])

        # self.phi_matrix = self.phi(self.state_action_pairs).reshape(self.states_all.shape[0], self.action_space.n, self.feature_dim).detach()
        # self.mu_matrix = self.mu(self.states_all)/self.states_all.shape[0]
        # self.mu_matrix = self.mu_matrix.detach()

        # print('phi_matrix:', self.phi_matrix)
        # print('mu_matrix:', self.mu_matrix)
        # self.mu.load_state_dict({'matrix.weight': torch.FloatTensor(np.load('Mu_estimated.npy'))})
        # self.phi_matrix = self.phi.matrix.weight.reshape(self.states_all.shape[0], self.action_space.n, self.feature_dim).detach()
        # self.mu_matrix = self.mu.matrix.weight.detach()

        # self.phi_pca = PCA(n_components=9)
        # self.phi_matrix_reduced, self.mu_matrix_reduced = pca_transform(self.phi_matrix.cpu().numpy(), self.mu_matrix.cpu().numpy())

        # self.phi = SingleMatrixFixedPhi(self.n_width, self.action_space.n, self.feature_dim).to(device) 
        # self.phi.load_state_dict({'matrix.weight': self.phi_matrix.reshape(-1, self.feature_dim)})
        # self.mu = SingleMatrixFixedMu(self.n_width, self.feature_dim).to(device)
        # self.mu.load_state_dict({'matrix.weight': self.mu_matrix})
        # self.phi_matrix = self.phi.matrix.reshape(self.states_all.shape[0], self.action_space.n, self.feature_dim)
        # self.mu_matrix = self.mu.matrix.reshape(self.n_task, self.feature_dim)

        # self.phi_matrix = torch.FloatTensor(np.load('Phi.npy')).reshape(self.states_all.shape[0], self.action_space.n, self.feature_dim).to(device).detach()
        # self.mu_matrix = torch.FloatTensor(np.load('Mu.npy')).reshape(self.n_task, self.feature_dim).to(device).detach()

        # self.w = SingleMatrix(input_dim=n_task, output_dim=feature_dim).to(device)
        # self.critic = SingleMatrixCritic(input_dim=n_task, output_dim=feature_dim).to(device)

        # w_all, u_all = init_inv_w_u(np.load('P.npy'), np.load('r.npy'), self.phi_matrix, self.mu_matrix, discount, alpha)
        # assert np.isclose(self.phi_matrix.cpu().numpy().reshape(-1,self.feature_dim)@(w_all.T), np.load('r.npy').reshape(-1, self.n_task), atol=1e-3).all()
        self.w = SingleMatrixEmbedding(self.n_task, self.feature_dim).to(device)
        # self.w.load_state_dict({'matrix.weight': torch.FloatTensor(w).to(device)})
        self.critic = SingleMatrixCriticEmbedding(self.n_task, self.feature_dim).to(device)
        # self.critic.load_state_dict({'matrix.weight': torch.FloatTensor(critic_1).to(device)})
        # self.w = nn.Parameter(torch.FloatTensor(w_all), requires_grad=True)
        # self.critic = nn.Parameter(torch.FloatTensor(u_all), requires_grad=True)
        # self.optimizer = torch.optim.Adam([self.w, self.critic], lr=critic_and_actor_lr)
        self.feature_optimizer = torch.optim.Adam(list(self.phi.parameters()) + list(self.mu.parameters()), lr=phi_and_mu_lr)
        self.optimizer = torch.optim.Adam(list(self.critic.parameters()) + list(self.w.parameters()),
                                            lr=critic_and_actor_lr, betas=[0.9, 0.999])
        
        # grid_col, grid_row = torch.meshgrid(torch.arange(0, self.n_width), torch.arange(0, self.n_height), indexing='xy')
        # self.states_all = torch.concat([grid_row.reshape(-1, 1), grid_col.reshape(-1, 1)], -1).float().to(device)
        # self.actions_all = torch.arange(self.action_space.n).reshape(-1, 1).float().to(device)
        # self.state_action_pairs = torch.concat([torch.repeat_interleave(self.states_all, self.action_space.n, dim=0), torch.tile(self.actions_all, (self.states_all.shape[0], 1))], -1).to(device)
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
        else:
            self.actor = SoftmaxActor(phi=self.phi, critic=self.critic, action_n=action_dim, 
                            n_task=n_task, feature_dim=feature_dim, device=device,
                            alpha=self.alpha, log_alpha=self.log_alpha, state_dim=state_dim,
                            action_dim=action_dim).to(device)
        

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
        print('Actor:')
        for name, param in self.actor.named_parameters():
            print(name, param.shape)
        print('Critic:')
        for name, param in self.critic.named_parameters():
            print(name, param.shape)
        print('Log Alpha:', self.log_alpha)
        return ''

    def select_action(self, state, explore=False):
        # Maybe change the critic network to output four hidden vectors at the same time

        if len(state.shape) == 1:
            state = torch.FloatTensor(np.array([state])).to(self.device)
        else:
            state = torch.FloatTensor(state).to(self.device)
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
        action = action.clamp(*self.action_range)
        return util.to_np(action[0])

    def ir_step(self, batch):
        """
        IR update step
        """
        state, action, next_state, reward, done = unpack_batch(batch)
        assert state.shape[-1] == self.state_dim
        assert action.shape[-1] == self.action_dim
        assert next_state.shape[-1] == self.state_dim
        assert reward.shape[-1] == 1
        assert done.shape[-1] == 1
        pred_action_log_pi = self.actor.evaluate_matrix(state)
        bc_loss = F.cross_entropy(pred_action_log_pi, action.squeeze(-1).long())
        # bc_loss = F.nll_loss(pred_action_log_pi, action.squeeze(-1).long())

        # self.critic_optimizer.zero_grad()
        # bc_loss.backward()
        # self.critic_optimizer.step()
        # print('bc_loss:', bc_loss)

        target_u = torch.zeros((self.n_task, self.feature_dim)).to(self.device)

        w = self.w(self.task_id_all)
        u1, u2 = self.critic(self.task_id_all)
        for i, task_id in enumerate(self.task_id_all):
            q1 = self.phi_matrix@u1[i]
            q2 = self.phi_matrix@u2[i]
            q = torch.min(q1, q2)
            assert q.shape == (self.states_all.shape[0], self.action_dim)
            pi_input = torch.concat([self.states_all, task_id.reshape(1,-1).repeat(self.states_all.shape[0],1)], -1)
            action_log_pi = self.actor.evaluate_matrix(pi_input)
            v = torch.sum(action_log_pi.exp() * (q - self.alpha.detach() * action_log_pi), -1)
            assert v.shape == (self.states_all.shape[0],)
            target_u[i] = w[i] + self.discount * v@self.mu_matrix
        u1_loss = F.mse_loss(target_u, u1)
        u2_loss = F.mse_loss(target_u, u2)
        rep_loss = (u1_loss + u2_loss)/2

        loss = 5*bc_loss + rep_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'bc_loss': bc_loss.item(),
            'rep_loss': rep_loss.item(),
        }
    def value_iteration_uw(self, w, alpha, discount, n_iter):
        V = torch.zeros(self.n_state, self.n_task).to(self.device)
        u = torch.zeros(self.n_task, self.feature_dim).to(self.device)
        assert self.mu_matrix.shape == (self.n_state, self.feature_dim)
        assert self.phi_matrix.shape == (self.n_state, self.n_action, self.feature_dim)
        assert w.shape == (self.n_task, self.feature_dim)
        for i in range(n_iter):
            # print((V.T@self.mu_matrix).shape)
            new_u = w + discount * ((V.T)@self.mu_matrix)
            error = torch.max(torch.abs(new_u - u))
            u = new_u
            q = self.phi_matrix@u.T
            assert q.shape == (self.n_state, self.n_action, self.n_task)
            V = torch.logsumexp(q/alpha, dim=1)*alpha
        return u, V, error
    def ir_step_vi(self, batch):
        state, action, next_state, reward, done = unpack_batch(batch)
        assert state.shape[-1] == self.state_dim
        assert action.shape[-1] == self.action_dim
        assert next_state.shape[-1] == self.state_dim
        assert reward.shape[-1] == 1
        assert done.shape[-1] == 1
        w_all = self.w(self.task_id_all)
        u_all, V, error = self.value_iteration_uw(w_all, self.alpha, self.discount, 50)
        state_idx = state[:,0]*self.n_width + state[:,1]
        task_id = torch.argmax(state[:,2:], dim=-1)
        action_idx = action.squeeze(-1).long()
        q = self.phi_matrix@u_all.T
        assert q.shape == (self.n_state, self.action_dim, self.n_task)
        action_logit = q[state_idx.long(), :, task_id.long()]
        bc_loss = F.cross_entropy(action_logit, action_idx)
        loss = bc_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'bc_loss': bc_loss.item(),
            'vi_error': error.item()
        }

    def ir_step_matrix(self, batch):
        state, action, next_state, reward, done = unpack_batch(batch)
        assert state.shape[-1] == self.state_dim + self.n_task
        assert action.shape[-1] == self.action_dim
        assert next_state.shape[-1] == self.state_dim + self.n_task
        assert reward.shape[-1] == 1
        assert done.shape[-1] == 1
        pred_action_log_pi = self.actor.evaluate_matrix(state)
        assert pred_action_log_pi.shape == (state.shape[0], self.action_dim)
        bc_loss = F.cross_entropy(pred_action_log_pi, action)


        w = self.w(self.task_id_all)
        u1, u2 = self.critic(self.task_id_all)

        phi_matrix = self.phi(self.state_action_pairs).reshape(self.states_all.shape[0], self.action_dim, self.feature_dim)
        mu_matrix = self.mu(self.states_all)/self.states_all.shape[0]


        q1 = phi_matrix@u1.T
        q2 = phi_matrix@u2.T
        q = torch.min(q1, q2).transpose(1,2)
        assert q.shape == (self.states_all.shape[0], self.n_task, self.action_dim)
        self.pi_input_all = torch.concat([self.states_all.unsqueeze(1).repeat(1,self.n_task,1), self.task_id_all.unsqueeze(0).repeat(self.states_all.shape[0],1,1)], -1)
        assert self.pi_input_all.shape == (self.states_all.shape[0], self.n_task, self.state_dim + self.n_task)
        action_log_pi_matrix = self.actor.evaluate_matrix(self.pi_input_all.reshape(-1, self.state_dim + self.n_task)).reshape(self.states_all.shape[0], self.n_task, self.action_dim)
        V = torch.sum(action_log_pi_matrix.exp() * (q - self.alpha.detach() * action_log_pi_matrix), -1)
        assert V.shape == (self.states_all.shape[0], self.n_task)
        target_u = w + self.discount * (V.T@mu_matrix)
        assert target_u.shape == (self.n_task, self.feature_dim)

        u1_loss = F.mse_loss(target_u, u1)
        u2_loss = F.mse_loss(target_u, u2)
        rep_loss = (u1_loss + u2_loss)/2
        l2_loss = (torch.square(u1).mean() + torch.square(u2).mean() + torch.square(w).mean())/3
        l1_loss = (torch.abs(w).mean()) * 0.3
        # self.w_optimizer.zero_grad()
        # rep_loss.backward()
        # self.w_optimizer.step()

        loss = bc_loss + rep_loss + l2_loss + l1_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'bc_loss': bc_loss.item(),
            'rep_loss': rep_loss.item(),
        }


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

        z_mu_all = self.mu(self.states_all)

        cur_state = state[..., :self.state_dim]
        cur_next_state = next_state[..., :self.state_dim]
        cur_s_random = s_random[..., :self.state_dim]
        cur_s_prime_random = s_prime_random[..., :self.state_dim]

        z_phi = self.phi(torch.concat([cur_state, action], -1))
        z_phi_random = self.phi(torch.concat([cur_s_random, a_random], -1))

        z_mu_next = self.mu(cur_next_state)
        z_mu_next_random = self.mu(cur_s_prime_random)

        assert z_phi.shape[-1] == self.feature_dim
        assert z_mu_next.shape[-1] == self.feature_dim
        model_loss_pt1 = -2 * torch.diag(z_phi @ z_mu_next.T)
        model_loss_pt2_a = z_phi_random @ z_mu_all.T
        model_loss_pt2 = model_loss_pt2_a ** 2

        model_loss_pt1_summed = 1. / torch.numel(model_loss_pt1) * torch.sum(model_loss_pt1)
        model_loss_pt2_summed = 1. / torch.numel(model_loss_pt2) * torch.sum(model_loss_pt2)

        model_loss = model_loss_pt1_summed + model_loss_pt2_summed

        phi_l1_loss = torch.abs(z_phi).mean()
        mu_l1_loss = torch.abs(z_mu_next).mean()

        loss = model_loss
        self.feature_optimizer.zero_grad()
        loss.backward()
        self.feature_optimizer.step()
        return {
            'total_loss': loss.item(),
            'model_loss': model_loss.item(),
            'phi_l1_loss': phi_l1_loss.item(),
            'mu_l1_loss': mu_l1_loss.item(),
            # 'prob_loss': prob_loss.item(),
        }

    def state_dict(self):
        return {'actor': self.actor.state_dict(),
				'critic': self.critic.state_dict(),
				'log_alpha': self.log_alpha,
				'phi': self.phi.state_dict(),
				'mu': self.mu.state_dict(),
				'w': self.w.state_dict()}

    def load_state_dict(self, state_dict):
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
        batch_1 = buffer.sample(batch_size)
        training_info = self.ir_step_matrix(batch_1)
        return {
            **training_info
        }
    
    def two_step_train(self, off_buffer, expert_buffer, batch_size):
        self.steps += 1
        
        batch_expert = expert_buffer.sample(batch_size)
        batch_expert_2 = expert_buffer.sample(batch_size)
        # for i in range(6):
        batch_1 = off_buffer.sample(batch_size)
        batch_2 = off_buffer.sample(batch_size)
        s_random, a_random, s_prime_random, _, _ = unpack_batch(batch_1)
        # print(s_random.shape)
        training_info_1 = self.feature_step(batch_2, s_random, a_random, s_prime_random)    
        training_info_2 = self.ir_step_matrix(batch_expert)
        return {
            **training_info_1,
            **training_info_2
        }


class VI_IRL_Agent(SACAgent):
    """
    Q-Learning with Discrete Actor
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            action_space,
            # 3e-4 was originally proposed in the paper, but seems to results in fluctuating performance
            phi_and_mu_lr=1e-4,
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
        self.n_state = n_width*n_height
        self.n_action = action_space.n
        self.discount = discount
        # self.alpha = alpha
        self.R = nn.Parameter(torch.zeros(self.n_state, self.n_action, self.n_task).to(self.device), requires_grad=True)
        self.Q = nn.Parameter(torch.zeros(self.n_state, self.n_action, self.n_task).to(self.device), requires_grad=True)
        self.optimizer = torch.optim.Adam([self.R, self.Q], lr=3e-4)
        self.P = torch.tensor(np.load('P.npy')).float().to(self.device)
        print('P:', self.P.shape)

    def value_iteration(self, R, P, discount, alpha, n_iter=1000):
        assert R.shape == (self.n_state, self.n_action, self.n_task)
        assert P.shape == (self.n_state, self.n_action, self.n_state)
        P = P.reshape(self.n_state, self.n_action, 1, self.n_state)
        Q = torch.zeros(self.n_state, self.n_action, self.n_task).to(self.device)
        V = torch.zeros(self.n_state, self.n_task).to(self.device)
        for i in range(n_iter):
            new_Q = R + discount * (P@V).reshape(self.n_state, self.n_action, self.n_task)
            error = torch.max(torch.abs(new_Q - Q))
            Q = new_Q
            max_Q = torch.max(Q, dim=1, keepdim=True)[0]
            V = torch.logsumexp((Q-max_Q)/alpha, dim=1)*alpha + max_Q
        return Q, V, error
    
    def ir_step_vi(self, batch):
        """
        IR update step
        """
        state, action, next_state, reward, done = unpack_batch(batch)
        assert state.shape[-1] == self.state_dim
        assert action.shape[-1] == self.action_dim
        assert next_state.shape[-1] == self.state_dim
        assert reward.shape[-1] == 1
        assert done.shape[-1] == 1
        Q, V, error = self.value_iteration(self.R, self.P, self.discount, self.alpha, 50)
        assert Q.shape == (self.n_state, self.n_action, self.n_task)
        assert V.shape == (self.n_state, self.n_task)
        state_idx = state[:,0]*self.n_width + state[:,1]
        task_id = torch.argmax(state[:,2:], dim=-1)
        action_idx = action.squeeze(-1).long()
        action_logit = Q[state_idx.long(), :, task_id.long()]
        # print('action_logit:', action_logit.shape, action_logit.dtype)
        bc_loss = F.cross_entropy(action_logit, action_idx)
        loss = bc_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'bc_loss': bc_loss.item(),
            'vi_error': error.item()
        }
    
    def ir_step(self, batch):
        """
        IR update step
        """
        state, action, next_state, reward, done = unpack_batch(batch)
        assert state.shape[-1] == self.state_dim
        assert action.shape[-1] == self.action_dim
        assert next_state.shape[-1] == self.state_dim
        assert reward.shape[-1] == 1
        assert done.shape[-1] == 1
        state_idx = state[:,0]*self.n_width + state[:,1]
        task_id = torch.argmax(state[:,2:], dim=-1)
        action_idx = action.squeeze(-1).long()
        action_logit = self.Q[state_idx.long(), :, task_id.long()]
        bc_loss = F.cross_entropy(action_logit, action_idx)


        V = torch.logsumexp(self.Q/self.alpha, dim=1)
        assert V.shape == (self.n_state, self.n_task)
        target_Q = self.R + self.discount * (self.P.reshape(self.n_state, self.n_action, 1, self.n_state)@V).reshape(self.n_state, self.n_action, self.n_task)
        assert target_Q.shape == (self.n_state, self.n_action, self.n_task)
        Q_loss = F.mse_loss(self.Q, target_Q)

        loss = bc_loss + Q_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'bc_loss': bc_loss.item(),
            'Q_loss': Q_loss.item()
        }
    def state_dict(self):
        return {'R': self.R,
                'Q': self.Q}
    def load_state_dict(self, state_dict):
        self.R = state_dict['R']
        self.Q = state_dict['Q']

    def train(self, buffer, batch_size):
        """
        One train step
        """
        self.steps += 1
        batch_1 = buffer.sample(batch_size)
        training_info = self.ir_step(batch_1)
        return {
            **training_info
        }