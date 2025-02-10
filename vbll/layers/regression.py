import numpy as np
import torch
from dataclasses import dataclass
from vbll.utils.distributions import Normal, DenseNormal, LowRankNormal, DenseNormalPrec, get_parameterization
from collections.abc import Callable
import torch.nn as nn


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
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    combined_mse_term = (cov_factor * mse_term).sum(-1)
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)

    return 0.5*(combined_mse_term + trace_term + logdet_term) # currently exclude constant


@dataclass
class VBLLReturn():
    predictive: Normal | DenseNormal | torch.distributions.studentT.StudentT
    train_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    val_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ood_scores: None | Callable[[torch.Tensor], torch.Tensor] = None


class Regression(nn.Module):
    """
    Variational Bayesian Linear Regression

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    regularization_weight : float
        Weight on regularization term in ELBO
    parameterization : str
        Parameterization of covariance matrix. Currently supports {'dense', 'diagonal', 'lowrank', 'dense_precision'}
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
                 prior_scale=1.,
                 wishart_scale=1e-2,
                 cov_rank=None,
                 dof=1.):
        super(Regression, self).__init__()

        self.wishart_scale = wishart_scale
        self.dof = (dof + out_features + 1.)/2.
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_scale = prior_scale * (1. / in_features) 

        # noise distribution
        self.noise_mean = nn.Parameter(torch.zeros(out_features), requires_grad = False)
        self.noise_logdiag = nn.Parameter(torch.randn(out_features) * (np.log(wishart_scale)))

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features))
        
        if parameterization == 'diagonal':
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) - 0.5 * np.log(in_features))
        elif parameterization == 'dense':
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) - 0.5 * np.log(in_features))
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, in_features)/in_features)
        elif parameterization == 'dense_precision':
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) + 0.5 * np.log(in_features))
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, in_features)*0.0)
        elif parameterization == 'lowrank':
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) - 0.5 * np.log(in_features))
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, cov_rank)/in_features)

    def W(self):
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist == Normal:
            cov = self.W_dist(self.W_mean, cov_diag)
        elif (self.W_dist == DenseNormal) or (self.W_dist == DenseNormalPrec):
            tril = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            cov = self.W_dist(self.W_mean, tril)
        elif self.W_dist == LowRankNormal:
            cov = self.W_dist(self.W_mean, self.W_offdiag, cov_diag)

        return cov

    def noise(self):
        return Normal(self.noise_mean, torch.exp(self.noise_logdiag))

    def forward(self, x):
        out = VBLLReturn(self.predictive(x),
                         self._get_train_loss_fn(x),
                         self._get_val_loss_fn(x))
        return out

    def predictive(self, x):
        return (self.W() @ x[..., None]).squeeze(-1) + self.noise()

    def _get_train_loss_fn(self, x):

        def loss_fn(y):
            # construct predictive density N(W @ phi, Sigma)
            W = self.W()
            noise = self.noise()
            pred_density = Normal((W.mean @ x[...,None]).squeeze(-1), noise.scale)
            pred_likelihood = pred_density.log_prob(y)

            trace_term = 0.5*((W.covariance_weighted_inner_prod(x.unsqueeze(-2)[..., None])) * noise.trace_precision)

            kl_term = gaussian_kl(W, self.prior_scale)
            wishart_term = (self.dof * noise.logdet_precision - 0.5 * self.wishart_scale * noise.trace_precision)
            total_elbo = torch.mean(pred_likelihood - trace_term)
            total_elbo += self.regularization_weight * (wishart_term - kl_term)
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            # compute log likelihood under variational posterior via marginalization
            logprob = self.predictive(x).log_prob(y).sum(-1) # sum over output dims            
            return -logprob.mean(0) # mean over batch dim

        return loss_fn



class tRegression(nn.Module):
    """
    Variational Bayesian Linear Student-t Regression
    
    This version of the VBLL regression layer also infers noise covariance.

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
                 prior_scale=100.,
                 wishart_scale=100.,
                 cov_rank=None,
                 dof=1.):
        super(tRegression, self).__init__()

        self.wishart_scale = wishart_scale
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_dof = dof
        self.prior_rate = 1./wishart_scale
        self.prior_scale = prior_scale * (2. / in_features) # kaiming init

        # variational posterior over noise params
        self.noise_log_dof = nn.Parameter(torch.ones(out_features) * np.log(self.prior_dof))
        self.noise_log_rate = nn.Parameter(torch.ones(out_features) * np.log(self.prior_rate))

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(2. / in_features))

        self.W_logdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features) - 0.5 * np.log(in_features))
        if parameterization == 'dense':
            self.W_offdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features, in_features)/in_features)
        elif parameterization == 'lowrank':
            self.W_offdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features, cov_rank))
        elif parameterization == 'dense_precision':
            raise NotImplementedError()
            

    @property
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

    @property
    def noise(self):
      noise_dof = torch.exp(self.noise_log_dof)
      noise_rate = torch.exp(self.noise_log_rate)
      return torch.distributions.gamma.Gamma(noise_dof, noise_rate)

    @property
    def noise_prior(self):
      return torch.distributions.gamma.Gamma(self.prior_dof, self.prior_rate)

    def forward(self, x):
        out = VBLLReturn(self.predictive(x),
                         self._get_train_loss_fn(x),
                         self._get_val_loss_fn(x))
        return out

    def predictive(self, x):
        dof = 2 * self.noise.concentration
        Wx = (self.W @ x[..., None]).squeeze(-1)
        mean = Wx.mean
        pred_cov = (Wx.variance + 1) * self.noise.rate / self.noise.concentration
        return torch.distributions.studentT.StudentT(dof, mean, torch.sqrt(pred_cov))

    def _get_train_loss_fn(self, x):

        def loss_fn(y):
            cov_factor = self.noise.concentration / self.noise.rate

            pred_err = (y - (self.W.mean @ x[...,None]).squeeze(-1)) ** 2
            pred_likelihood = (cov_factor * pred_err).sum(-1)

            logdet_term = (torch.digamma(self.noise.concentration) - torch.log(self.noise.rate)).sum(-1)
            trace_term = (self.W.covariance_weighted_inner_prod(x.unsqueeze(-2)[..., None])).sum(-1)

            kl_term = expected_gaussian_kl(self.W, self.prior_scale, cov_factor)
            kl_term += gamma_kl(self.noise, self.noise_prior)

            total_elbo = -0.5 * torch.mean(pred_likelihood + trace_term - logdet_term)
            total_elbo -= self.regularization_weight * kl_term
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            # compute log likelihood under variational posterior via marginalization
            logprob = self.predictive(x).log_prob(y).sum(-1) # sum over output dims
            return -logprob.mean(0) # mean over batch dim

        return loss_fn


class HetRegression(nn.Module):
    """
    Heteroscedastic Variational Bayesian Linear Regression

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
    prior_scale : float
        Scale of prior covariance matrix
    noise_prior_scale : float
        Scale of prior/init for the noise covariance VBLL
    grad_correction_scale : float
        Correction factor for mean estimation, in range [0,1] with 0 corresponding to standard estimator
    """
    def __init__(self,
                 in_features,
                 out_features,
                 regularization_weight,
                 parameterization='dense',
                 prior_scale=1.,
                 noise_prior_scale=0.01,
                 grad_correction_scale=0.):
        super(HetRegression, self).__init__()

        self.regularization_weight = regularization_weight
        self.grad_correction = grad_correction_scale

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_scale = prior_scale * (1. / in_features)
        self.noise_prior_scale = noise_prior_scale * (1. / in_features)

        # noise distribution
        self.noise_mean = nn.Parameter(torch.zeros(out_features), requires_grad = False)

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(2. / in_features))

        self.M_dist = get_parameterization(parameterization) # currently, use same parameterization
        self.M_mean = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(2. / in_features))

        if parameterization == 'diagonal':
            self.W_logdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features) - 0.5 * np.log(in_features))
            self.M_logdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features) + 0.5 * np.log(noise_prior_scale/in_features))

        elif parameterization == 'dense':
            self.W_logdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features) - 0.5 * np.log(in_features))
            self.W_offdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features, in_features)/in_features)

            self.M_logdiag = nn.Parameter(1e-3 * torch.randn(out_features, in_features) + 0.5 * np.log(noise_prior_scale/in_features))
            self.M_offdiag = nn.Parameter(1e-3 * * torch.randn(out_features, in_features, in_features)/in_features)
        elif parameterization == 'dense_precision':
            raise NotImplementedError()

        elif parameterization == 'lowrank':
            raise NotImplementedError()

        else:
            raise ValueError('Invalid cov parameterization')
    @property
    def M(self):
        cov_diag = torch.exp(self.M_logdiag)
        if self.M_dist == Normal:
            cov = self.M_dist(self.M_mean, cov_diag)
        elif (self.M_dist == DenseNormal) or (self.M_dist == DenseNormalPrec):
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

    def forward(self, x, consistent_variance=False):
        # TODO: return mixture distribution as opposed to a single noise sample
        out = VBLLReturn(self.predictive_sample(x, consistent_variance),
                         self._get_train_loss_fn(x),
                         self._get_val_loss_fn(x, consistent_variance))
        return out

    def predictive_sample(self, x, consistent_variance):
        if consistent_variance:
            sigma2 = torch.exp(self.log_noise(x,self.M.rsample()))
        else:
            sigma2 = torch.exp(self.log_noise(x,self.M).rsample())
            
        Wx = (self.W @ x[..., None]).squeeze(-1)
        mean = Wx.mean
        cov = torch.sqrt((Wx.variance + 1) * sigma2)
        return Normal(mean, cov)

    def _get_train_loss_fn(self, x):
        def loss_fn(y):
            W = self.W
            M = self.M
            log_noise_cov = self.log_noise(x, M)
            expect_sigma_inv = torch.exp(-log_noise_cov.mean + 0.5 * log_noise_cov.scale ** 2)
            expect_log_sigma = log_noise_cov.mean

            grad_correction = (expect_sigma_inv.detach()) ** self.grad_correction

            # compute pred density with expected Sigma terms
            err = y - (W.mean @ x[...,None]).squeeze(-1)
            mse_term = (err.pow(2) * expect_sigma_inv).sum(-1)

            logdet_term = (expect_log_sigma).sum(-1)
            trace_term = W.covariance_weighted_inner_prod(x.unsqueeze(-2)[..., None])
            total_elbo = - 0.5 * torch.mean(grad_correction * (mse_term + logdet_term + trace_term))

            # compute expected KL
            kl_term_ll = torch.mean(grad_correction * expected_gaussian_kl(W, self.prior_scale, expect_sigma_inv))
            kl_term_noise = torch.mean(grad_correction * gaussian_kl(M, self.noise_prior_scale))
            total_elbo -= self.regularization_weight * (kl_term_noise + kl_term_ll)
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x, consistent_variance, n_samples = 20):
        def loss_fn(y):
            # sample noise n times
            # compute log likelihood under variational posterior via marginalization with sampled noise
            running_loss = 0.
            for i in range(n_samples):
                running_loss += -torch.mean(self.predictive_sample(x, consistent_variance).log_prob(y).sum(-1)) # sum over output dims
            return running_loss / n_samples

        return loss_fn
