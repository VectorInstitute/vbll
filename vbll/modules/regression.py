import torch
from dataclasses import dataclass
from vbll.utils.distributions as dist

@dataclass
class VBLLReturn():
    predictive: torch.Tensor # Could return distribution or mean/cov
    train_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    val_loss_fn: Callable[[torch.Tensor], torch.Tensor]

class VBLLRegression(nn.Module):
    """Variational Bayesian Linear Regression module.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        regularization_weight (float): Weight of regularization term.
        lastlayer_parameterization (str): Parameterization of last layer covariance.
        noise_parameterization (str): Parameterization of noise covariance.
        prior_scale (float): Scale of prior covariance.
        wishart_scale (float): Scale of Wishart distribution.
        dof (float): Degrees of freedom of Wishart distribution.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 regularization_weight: float,
                 lastlayer_parameterization='dense': str,
                 noise_parameterization='diagonal': str,
                 prior_scale=1.: float,
                 wishart_scale=1.: float,
                 dof=1.: float):

        self.wishart_scale = wishart_scale
        self.dof = (dof + out_features + 1.)/2.
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        prior_mean = nn.Parameter(torch.zeros(in_features, out_features),
                                  requires_grad = False)
        prior_cov = dist.DiagonalCovariance(covariance_scale = prior_scale,
                                       requires_grad = False)
        self.prior = dist.Gaussian(prior_mean, prior_cov)

        # last layer distribution
        posterior_cov = dist.get_cov_parameterization(lastlayer_parameterization)(in_features, out_features)
        self.W = dist.Gaussian(nn.Parameter(torch.randn(out_features, in_features)),
                          posterior_cov)

        # noise distribution
        noise_mean = nn.Parameter(torch.zeros(out_features, 1),
                                  requires_grad = False)
        noise_cov = dist.get_cov_parameterization(noise_parameterization)(out_features, 1)
        self.noise = dist.Gaussian(noise_mean, noise_cov)

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input data.   

        Returns:
            VBLLReturn: Object containing predictive distribution and loss functions.
        """

        return VBLLReturn(self.predictive(x),
                          self._get_train_loss_fn(x)
                          self._get_val_loss_fn(x))

    def predictive(self, x):
        """Predictive distribution.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            Gaussian: Predictive distribution.
        """
        
        Wx = dist.gaussian_vector_product(self.W, x)
        return dist.add_gaussians(Wx, self.noise)

    def _get_train_loss_fn(self, x):

        def loss_fn(y):
            # construct predictive density N(W @ phi, Sigma)
            pred_density = dist.Gaussian(self.W.mean @ x, self.noise)
            pred_likelihood = self.pred_density.log_prob(y) 

            trace_term = 0.5*((self.W.cov.weighted_inner_prod(x)) * self.noise.cov.inverse_trace)

            kl_term = KL(self.W, self.prior)
            wishart_term = (self.dof * self.noise.cov.inverse_logdet - 0.5 * self.wishart_scale * self.noise.cov.inverse_trace)

            total_elbo = torch.mean(pred_likelihood - trace_term)
            total_elbo += self.regularization_weight * (wishart_term - kl_term)
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            # compute log likelihood under variational posterior via marginalization
            return -self.predictive(x).log_prob(y)

        return loss_fn