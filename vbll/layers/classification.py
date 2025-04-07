import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from collections.abc import Callable
import abc
import warnings

from vbll.utils.distributions import Normal, DenseNormal, LowRankNormal, get_parameterization

def gaussian_kl(p, q_scale):
    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean ** 2).sum(-1).sum(-1) / q_scale
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)

    return 0.5*(mse_term + trace_term + logdet_term) # currently exclude constant


def gamma_kl(cov_dist, prior_dist):
    kl = torch.distributions.kl.kl_divergence(cov_dist, prior_dist)
    return (kl).sum(-1)


def expected_gaussian_kl(p, q_scale, cov_factor):
    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean ** 2).sum(-1)/ q_scale
    combined_mse_term = (cov_factor * mse_term).sum(-1)
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)

    return 0.5*(combined_mse_term + trace_term + logdet_term) # currently exclude constant


@dataclass
class VBLLReturn():
    predictive: Normal | DenseNormal # Could return distribution or mean/cov
    train_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    val_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ood_scores: None | Callable[[torch.Tensor], torch.Tensor] = None

class DiscClassification(nn.Module):
    """Variational Bayesian Disciminative Classification

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        regularization_weight : float
            Weight on regularization term in ELBO
        parameterization : str
            Parameterization of covariance matrix. Currently supports {'dense', 'diagonal', 'lowrank'}
        softmax_bound : str
            Bound to use for softmax. Currently supports 'jensen'
        return_ood : bool
            Whether to return OOD scores
        prior_scale : float
            Scale of prior covariance matrix
        wishart_scale : float
            Scale of Wishart prior on noise covariance
        dof : float
            Degrees of freedom of Wishart prior on noise covariance
        cov_rank : int
            Rank of low-rank correction used in the LowRankNormal covariance parameterization.
     """
    def __init__(self,
                 in_features,
                 out_features,
                 regularization_weight,
                 parameterization='dense',
                 softmax_bound='jensen',
                 return_ood=False,
                 prior_scale=1.,
                 wishart_scale=1.,
                 dof=1.,
                 cov_rank=3):
        super(DiscClassification, self).__init__()

        self.wishart_scale = wishart_scale
        self.dof = (dof + out_features + 1.)/2.
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_scale = prior_scale * (2. / in_features) # kaiming init/width scaling

        # noise distribution
        self.noise_mean = nn.Parameter(torch.zeros(out_features), requires_grad = False)
        self.noise_logdiag = nn.Parameter(torch.randn(out_features) - 1)

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(2./in_features)) # kaiming init

        self.W_logdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features) - np.log(in_features)) # make output dim invariant
        if parameterization == 'dense':
            self.W_offdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features, in_features)/in_features) # init off dim elements small
        elif parameterization == 'lowrank':
            self.W_offdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features, cov_rank)/in_features) # init off dim elements small
        
        if softmax_bound == 'jensen':
            self.softmax_bound = self.jensen_bound
        else:
            raise NotImplementedError('Only semi-Monte Carlo is currently implemented.')

        self.return_ood = return_ood

    def W(self):
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist == Normal:
            cov = self.W_dist(self.W_mean, cov_diag)
        elif self.W_dist == DenseNormal:
            tril = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            cov = self.W_dist(self.W_mean, tril)
        elif self.W_dist == LowRankNormal:
            cov = self.W_dist(self.W_mean, self.W_offdiag, cov_diag)

        return cov

    def noise(self):
        return Normal(self.noise_mean, torch.exp(self.noise_logdiag))

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

    def predictive(self, x, n_samples = 20):
        softmax_samples = F.softmax(self.logit_predictive(x).rsample(sample_shape=torch.Size([n_samples])), dim=-1)
        return torch.clip(torch.mean(softmax_samples, dim=0),min=0.,max=1.)

    def _get_train_loss_fn(self, x):

        def loss_fn(y):
            noise = self.noise()

            kl_term = gaussian_kl(self.W(), self.prior_scale)
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


class tDiscClassification(nn.Module):
    """
    Variational Bayesian t-Classification

    This version of the VBLL Classification layer also infers noise covariance.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    regularization_weight : float
        Weight on regularization term in ELBO
    parameterization : str
        Parameterization of covariance matrix. Currently supports {'dense', 'diagonal'}
    softmax_bound : str
        Bound to use for softmax. Currently supports 'semimontecarlo'
    prior_scale : float
        Scale of prior covariance matrix
    wishart_scale : float
        Scale of Wishart prior on noise covariance
    dof : float
        Degrees of freedom of Wishart prior on noise covariance
    cov_rank : int
        Rank of low-rank correction used in the LowRankNormal covariance parameterization.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 regularization_weight,
                 parameterization='dense',
                 softmax_bound='reduced_kn',
                 return_ood=False,
                 prior_scale=1.,
                 wishart_scale=1.,
                 dof=2.,
                 cov_rank=None,
                 kn_alpha=None,
                 ):

        super(tDiscClassification, self).__init__()

        self.wishart_scale = wishart_scale
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_dof = dof
        self.prior_rate = 1./wishart_scale
        exp_cov = self.prior_rate/(self.prior_dof - 1)
        self.prior_scale = prior_scale * 2. / (exp_cov * in_features) 

        # variational posterior over noise params
        self.noise_log_dof = nn.Parameter(torch.ones(out_features) * np.log(self.prior_dof))
        self.noise_log_rate = nn.Parameter(torch.ones(out_features) * np.log(self.prior_rate))

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(2./in_features)) # kaiming init

        self.W_logdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features) - 0.5 * np.log(in_features)) # make output dim invariant
        if parameterization == 'diagonal':
            pass
        elif parameterization == 'dense':
            self.W_offdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features, in_features))
        elif parameterization == 'diagonal_natural':
            raise NotImplementedError('diagonal_natural not implemented')
        elif parameterization == 'lowrank':
            raise NotImplementedError('lowrank not implemented')
        else:
            raise ValueError('invalid parameterization')

        if softmax_bound == 'semimontecarlo':
            self.softmax_bound = self.semimontecarlo_bound
        elif softmax_bound == 'reduced_kn':
            self.softmax_bound = self.reduced_kn
            if kn_alpha is None:
                self.alpha = nn.Parameter(0.1 * torch.ones(1))
            else:
                self.alpha = nn.Parameter(torch.ones(1) * kn_alpha)
        else:
            raise NotImplementedError('Only semi-Monte Carlo and reduced_kn are currently implemented.')

        self.return_ood = return_ood

    @property
    def W(self):
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist == Normal:
            cov = self.W_dist(self.W_mean, cov_diag)
        elif self.W_dist == DenseNormal:
            tril = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            cov = self.W_dist(self.W_mean, tril)

        return cov

    @property
    def noise(self):
        noise_dof = torch.exp(self.noise_log_dof)
        noise_rate = torch.exp(self.noise_log_rate)
        return torch.distributions.gamma.Gamma(noise_dof, noise_rate)

    @property
    def noise_prior(self):
        return torch.distributions.gamma.Gamma(self.prior_dof, self.prior_rate)

    # ----- bounds

    def semimontecarlo_bound(self, x, y, n_samples=1):
        # Samples from inverse gamma distribution
        pred = self.logit_predictive(x)
        linear_term = pred.mean[torch.arange(x.shape[0]), y]
        pre_lse_term = pred.mean + 0.5 * pred.covariance_diagonal
        lse_term = torch.logsumexp(pre_lse_term, dim=-1)
        return linear_term - lse_term

    def reduced_kn(self, x, y):
        # Uses the Knowles-Minka bound with alpha = 1/2 - alpha/Sigma
        # https://tminka.github.io/papers/knowles-minka-nips2011.pdf

        Wx = (self.W @ x[..., None]).squeeze(-1)
        cov = (Wx.variance + 1)
        linear_term = Wx.mean[torch.arange(x.shape[0]), y]

        pre_lse_term = Wx.mean + self.alpha * cov
        lse_term = torch.logsumexp(pre_lse_term, dim=-1)

        exp_cov = torch.exp(self.noise_log_rate - self.noise_log_dof + 1)
        exp_prec = torch.exp(self.noise_log_dof - self.noise_log_rate)

        cov_term = cov * (exp_cov/4 + exp_prec * self.alpha ** 2 - self.alpha)
        return linear_term - lse_term - 0.5 * cov_term.sum(-1)

    def forward(self, x):
        out = VBLLReturn(torch.distributions.Categorical(probs = self.predictive(x)),
                          self._get_train_loss_fn(x),
                          self._get_val_loss_fn(x))
        if self.return_ood: out.ood_scores = self.max_predictive(x)
        return out

    def logit_predictive(self, x):
        # sample noise covariance, currently only doing 1 sample
        cov_sample = 1./self.noise.rsample(sample_shape=torch.Size([x.shape[0]]))
        Wx = (self.W @ x[..., None]).squeeze(-1)
        mean = Wx.mean
        pred_cov = (Wx.variance + 1) * cov_sample
        return Normal(mean, torch.sqrt(pred_cov))

    def predictive(self, x, n_samples = 20):
        softmax_samples = F.softmax(self.logit_predictive(x).rsample(sample_shape=torch.Size([n_samples])), dim=-1)
        return torch.clip(torch.mean(softmax_samples, dim=0),min=0.,max=1.)

    def _get_train_loss_fn(self, x):

        def loss_fn(y):
            kl_term = gamma_kl(self.noise, self.noise_prior)
            cov_factor = self.noise.concentration / self.noise.rate
            kl_term += expected_gaussian_kl(self.W, self.prior_scale, cov_factor)

            total_elbo = torch.mean(self.softmax_bound(x, y))
            total_elbo -= self.regularization_weight * kl_term
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            return -torch.mean(torch.log(self.predictive(x)[torch.arange(x.shape[0]), y]))

        return loss_fn

    # ----- OOD metrics

    def max_predictive(self, x):
        return torch.max(self.predictive(x), dim=-1)[0]


class HetClassification(nn.Module):
    """
    Heteroscedastic Variational Bayesian Classification

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    regularization_weight : float
        Weight on regularization term in ELBO
    parameterization : str
        Parameterization of covariance matrix. Currently supports {'dense', 'diagonal', 'lowrank'}
    softmax_bound : str
        Bound to use for softmax. Currently supports 'semimontecarlo'
    prior_scale : float
        Scale of prior covariance matrix/initialization
    prior_scale : float
        Scale of noise prior covariance matrix/initalization
    cov_rank : int
        Rank of low-rank correction used in the LowRankNormal covariance parameterization.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 regularization_weight,
                 parameterization='dense',
                 softmax_bound='reduced_kn',
                 return_ood=False,
                 prior_scale=1.,
                 noise_prior_scale=0.01,
                 cov_rank=None,
                 kn_alpha=None):
        super(HetClassification, self).__init__()

        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_scale = prior_scale * (1. / in_features)
        self.noise_prior_scale = noise_prior_scale * (1. / in_features)

        # noise distribution
        self.noise_mean = nn.Parameter(torch.zeros(out_features), requires_grad = False)

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(2 / in_features))

        self.M_dist = get_parameterization(parameterization) # currently, use same parameterization
        self.M_mean = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(2 / in_features))
        
        self.W_logdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features) - 0.5 * np.log(in_features))
        self.M_logdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features) + 0.5 * np.log(noise_prior_scale/in_features))

        if parameterization == 'diagonal':
            pass
        elif parameterization == 'dense':
            self.W_offdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features, in_features)/in_features)
            self.M_offdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features, in_features)/in_features + 0.5 * np.log(noise_prior_scale))
        elif parameterization == 'dense_precision':
            raise NotImplementedError('dense_precision not implemented')
        elif parameterization == 'lowrank':
            raise NotImplementedError('lowrank not implemented')
        else:
            raise ValueError('invalid parameterization')

        if softmax_bound == 'semimontecarlo':
            self.softmax_bound = self.semimontecarlo_bound
        elif softmax_bound == 'reduced_kn':
            self.softmax_bound = self.reduced_kn
            if kn_alpha is None:
                self.alpha = nn.Parameter(0.1 * torch.ones(1))
            else:
                self.alpha = nn.Parameter(torch.ones(1) * kn_alpha)
        else:
            raise NotImplementedError('Only semi-Monte Carlo and reduced_kn are currently implemented.')

        self.return_ood = return_ood

    @property
    def M(self):
        cov_diag = torch.exp(self.M_logdiag)
        if self.M_dist == Normal:
            cov = self.M_dist(self.M_mean, cov_diag)
        elif (self.M_dist == DenseNormal):
            tril = torch.tril(self.M_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            cov = self.M_dist(self.M_mean, tril)

        return cov

    @property
    def W(self):
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist == Normal:
            cov = self.W_dist(self.W_mean, cov_diag)
        elif (self.W_dist == DenseNormal):
            tril = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            cov = self.W_dist(self.W_mean, tril)
        return cov

    def log_noise(self, x, M):
        return (M @ x[..., None]).squeeze(-1)

    # ----- bounds
    def semimontecarlo_bound(self, x, y):
        # samples noise within logit_predictive        
        pred = self.logit_predictive(x, consistent_variance=False) # no point in sampling consistent variance for loss computation

        linear_term = pred.mean[torch.arange(x.shape[0]), y]
        pre_lse_term = pred.mean + 0.5 * pred.covariance_diagonal
        lse_term = torch.logsumexp(pre_lse_term, dim=-1)
        return linear_term - lse_term

    def reduced_kn(self, x, y):
        # Uses the Knowles-Minka bound with alpha = 1/2 - alpha/Sigma
        # https://tminka.github.io/papers/knowles-minka-nips2011.pdf
        Wx = (self.W @ x[..., None]).squeeze(-1)
        cov = (Wx.variance + 1)
        linear_term = Wx.mean[torch.arange(x.shape[0]), y]

        pre_lse_term = Wx.mean + self.alpha * cov
        lse_term = torch.logsumexp(pre_lse_term, dim=-1)

        log_noise_cov = self.log_noise(x, self.M)
        exp_cov = torch.exp(log_noise_cov.mean + 0.5 * log_noise_cov.scale ** 2)
        exp_prec = torch.exp(-log_noise_cov.mean + 0.5 * log_noise_cov.scale ** 2)

        cov_term = cov * (exp_cov/4 + exp_prec * self.alpha ** 2 - self.alpha)
        return linear_term - lse_term - 0.5 * cov_term.sum(-1)

    def forward(self, x, consistent_variance=False):
        # need to return sampling-based output
        out = VBLLReturn(torch.distributions.Categorical(probs = self.predictive(x, consistent_variance)),
                         self._get_train_loss_fn(x),
                         self._get_val_loss_fn(x, consistent_variance))
        if self.return_ood: out.ood_scores = self.max_predictive(x, consistent_variance)
        return out

    def logit_predictive(self, x, consistent_variance):
        # sample noise (single sample)
        if consistent_variance:
            sigma2 = torch.exp(self.log_noise(x,self.M.rsample()))
        else:
            sigma2 = torch.exp(self.log_noise(x,self.M).rsample())
            
        Wx = (self.W @ x[..., None]).squeeze(-1)
        mean = Wx.mean
        stdev = torch.sqrt((Wx.variance + 1) * sigma2)
        return Normal(mean, stdev)

    def predictive(self, x, consistent_variance, n_samples = 20):
        softmax_samples = F.softmax(self.logit_predictive(x, consistent_variance).rsample(sample_shape=torch.Size([n_samples])), dim=-1)
        return torch.clip(torch.mean(softmax_samples, dim=0),min=0.,max=1.)

    def _get_train_loss_fn(self, x):
        def loss_fn(y):
            log_noise_cov = self.log_noise(x, self.M)

            # compute expected KL
            expect_sigma_inv = torch.exp(-log_noise_cov.mean + 0.5 * log_noise_cov.scale ** 2)
            kl_term_ll = torch.mean(expected_gaussian_kl(self.W, self.prior_scale, expect_sigma_inv))
            kl_term_noise = gaussian_kl(self.M, self.noise_prior_scale)

            total_elbo = torch.mean(self.softmax_bound(x, y))
            total_elbo -= self.regularization_weight * kl_term_ll + kl_term_noise

            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x, consistent_variance):
        def loss_fn(y):
            return -torch.mean(torch.log(self.predictive(x, consistent_variance)[torch.arange(x.shape[0]), y]))

        return loss_fn

    # ----- OOD metrics

    def max_predictive(self, x, consistent_variance):
        return torch.max(self.predictive(x, consistent_variance), dim=-1)[0]
        

class GenClassification(nn.Module):
    """Variational Bayesian Generative Classification

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        regularization_weight : float
            Weight on regularization term in ELBO
        parameterization : str
            Parameterization of covariance matrix. Currently supports 'dense' and 'diagonal'
        softmax_bound : str
            Bound to use for softmax. Currently supports 'jensen'
        return_ood : bool
            Whether to return OOD scores
        prior_scale : float
            Scale of prior covariance matrix
        wishart_scale : float
            Scale of Wishart prior on noise covariance
        dof : float
            Degrees of freedom of Wishart prior on noise covariance
     """
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
        super(GenClassification, self).__init__()

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

    def mu(self):
        # TODO(jamesharrison): add impl for dense/low rank cov
        return self.mu_dist(self.mu_mean, torch.exp(self.mu_logdiag))

    def noise(self):
        return Normal(self.noise_mean, torch.exp(self.noise_logdiag))

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

        trace_term = (self.mu().covariance_diagonal[y] / self.noise().covariance_diagonal).sum(-1)

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
            noise = self.noise()
            kl_term = gaussian_kl(self.mu(), self.prior_scale)
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
