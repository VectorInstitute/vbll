import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from collections.abc import Callable
import abc
import warnings

from vbll.utils.distributions import Normal, DenseNormal, get_parameterization

def KL(p, q_scale):
    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean ** 2).sum(-1).sum(-1) / q_scale
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)

    return 0.5*(mse_term + trace_term + logdet_term) # currently exclude constant

@dataclass
class VBLLReturn():
    predictive: Normal | DenseNormal # Could return distribution or mean/cov
    train_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    val_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ood_scores: None | Callable[[torch.Tensor], torch.Tensor] = None

class VBLLClassificationD(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 regularization_weight,
                 parameterization='dense',
                 softmax_bound='jensen',
                 return_ood=False,
                 prior_scale=1.,
                 wishart_scale=1.,
                 dof=1.):
        super(VBLLClassificationD, self).__init__()

        self.wishart_scale = wishart_scale
        self.dof = (dof + out_features + 1.)/2.
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_scale = prior_scale

        # noise distribution
        self.noise_mean = nn.Parameter(torch.zeros(out_features), requires_grad = False)
        self.noise_logdiag = nn.Parameter(torch.randn(out_features) - 1)

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features))
        self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) - np.log(in_features))
        if parameterization == 'dense':
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, in_features)/in_features)

        if softmax_bound == 'jensen':
            self.softmax_bound = self.jensen_bound

        self.return_ood = return_ood

    def noise_cov(self):
        return torch.exp(self.noise_logdiag)

    def W_cov(self):
        out = torch.exp(self.W_logdiag)
        if self.W_dist == DenseNormal:
            out = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(out)
        return out

    def W(self):
        return self.W_dist(self.W_mean, self.W_cov())

    def noise(self):
        return Normal(self.noise_mean, self.noise_cov())

    # ----- bounds

    def adaptive_bound(self, x, y):
        # TODO(jamesharrison)
        raise NotImplementedError('Adaptive bound not currently implemented')

    def jensen_bound(self, x, y):
        pred = self.logit_predictive(x)
        linear_term = pred.mean[torch.arange(x.shape[0]), y]
        pre_lse_term = pred.mean + 0.5 * pred.covariance_diagonal
        lse_term = torch.logsumexp(pre_lse_term, dim=-1)
        return linear_term - lse_term

    def montecarlo_bound(self, x, y, n_samples=10):
        sampled_log_sm = F.log_softmax(self.logit_predictive(x).rsample(sample_shape=torch.Size([n_samples])), dim=-1)
        mean_over_samples = torch.mean(sampled_log_sm, dim=0)
        return mean_over_samples[torch.arange(x.shape[0]), y]

    # ----- forward and core ops

    def forward(self, x):
        # TODO(jamesharrison): add assert on shape of x input
        out = VBLLReturn(torch.distributions.Categorical(probs = self.predictive(x)),
                          self._get_train_loss_fn(x),
                          self._get_val_loss_fn(x))
        if self.return_ood: out.ood_scores = self.max_predictive(x)
        return out

    def logit_predictive(self, x):
        return (self.W() @ x[..., None]).squeeze(-1) + self.noise()

    def predictive(self, x, n_samples = 10):
        softmax_samples = F.softmax(self.logit_predictive(x).rsample(sample_shape=torch.Size([n_samples])), dim=-1)
        return torch.clip(torch.mean(softmax_samples, dim=0),min=0.,max=1.)

    def _get_train_loss_fn(self, x):

        def loss_fn(y):
            noise = self.noise()

            kl_term = KL(self.W(), self.prior_scale)
            wishart_term = (self.dof * noise.logdet_precision - 0.5 * self.wishart_scale * noise.trace_precision)

            total_elbo = torch.mean(self.softmax_bound(x, y))
            total_elbo += self.regularization_weight * (wishart_term - kl_term)
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            return -torch.mean(torch.log(self.predictive(x)[torch.arange(x.shape[0]), y]))

        return loss_fn

    # ----- OOD metrics

    def max_predictive(self, x):
        return torch.max(self.predictive(x), dim=-1)[0]

class VBLLClassificationG(nn.Module):
    # TODO(jamesharrison): add dirichlet
    def __init__(self,
                 in_features,
                 out_features,
                 regularization_weight,
                 parameterization='dense',
                 softmax_bound='jensen',
                 return_ood=False,
                 prior_scale=1.,
                 wishart_scale=1.,
                 dof=1.):
        super(VBLLClassificationG, self).__init__()

        self.wishart_scale = wishart_scale
        self.dof = (dof + in_features + 1.)/2.
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_scale = prior_scale

        # noise distribution
        self.noise_mean = nn.Parameter(torch.zeros(in_features), requires_grad = False)
        self.noise_logdiag = nn.Parameter(torch.randn(in_features))

        # last layer distribution
        self.mu_dist = get_parameterization(parameterization)
        self.mu_mean = nn.Parameter(0.1*torch.randn(out_features, in_features))
        self.mu_logdiag = nn.Parameter(torch.randn(out_features, in_features))
        if parameterization == 'dense':
            raise NotImplementedError('Dense embedding cov not implemented for g-vbll')

        if softmax_bound == 'jensen':
            self.softmax_bound = self.jensen_bound

        self.return_ood = return_ood

    def noise_cov(self):
        return torch.exp(self.noise_logdiag)

    def mu_cov(self):
        out = torch.exp(self.mu_logdiag)
        # TODO(jamesharrison): add impl for dense/low rank cov
        return out

    def mu(self):
        return self.mu_dist(self.mu_mean, self.mu_cov())

    def noise(self):
        return Normal(self.noise_mean, self.noise_cov())

    # ----- bounds

    def adaptive_bound(self, x, y):
        # TODO(jamesharrison)
        raise NotImplementedError('Adaptive bound not implemented for g-vbll')

    def jensen_bound(self, x, y):
        linear_pred = self.noise() + self.mu_mean[y]
        linear_term = linear_pred.log_prob(x)
        if isinstance(linear_pred, Normal):
            # Is there a more elegant way to handle this?
            linear_term = linear_term.sum(-1)

        trace_term = (self.mu_cov()[y] / self.noise_cov()).sum(-1)

        pre_lse_term = self.logit_predictive(x)
        lse_term = torch.logsumexp(pre_lse_term, dim=-1)
        return linear_term - 0.5 * trace_term - lse_term

    def montecarlo_bound(self, x, y, n_samples=10):
        # TODO(jamesharrison)
        raise NotImplementedError('Monte carlo bound not implemented for g-vbll')

    # ----- forward and core ops

    def forward(self, x):
        # TODO(jamesharrison): add assert on shape of x input
        out = VBLLReturn(torch.distributions.Categorical(probs = self.predictive(x)),
                          self._get_train_loss_fn(x),
                          self._get_val_loss_fn(x))
        if self.return_ood: out.ood_scores = self.max_predictive(x)
        return out

    def logit_predictive(self, x):
        # likelihood of x under marginalized
        logprob = (self.mu() + self.noise()).log_prob(x.unsqueeze(-2))
        if isinstance(self.mu(), Normal):
            # Is there a more elegant way to handle this?
            logprob = logprob.sum(-1)
        return logprob

    def predictive(self, x):
        return torch.clip(F.softmax(self.logit_predictive(x), dim=-1), min=0., max=1.)

    def _get_train_loss_fn(self, x):

        def loss_fn(y):
            # print('computing loss fn')
            noise = self.noise()
            kl_term = KL(self.mu(), self.prior_scale)
            wishart_term = (self.dof * noise.logdet_precision - 0.5 * self.wishart_scale * noise.trace_precision)

            total_elbo = torch.mean(self.softmax_bound(x, y))
            total_elbo += self.regularization_weight * (wishart_term - kl_term)
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            return -torch.mean(torch.log(self.predictive(x)[np.arange(x.shape[0]), y]))

        return loss_fn

    # ----- OOD metrics

    def max_predictive(self, x):
        return torch.max(self.predictive(x), dim=-1)[0]
