import torch
from dataclasses import dataclass

@dataclass
class VBLLReturn():
    predictive: torch.Tensor # Could return distribution or mean/cov 
    #loss_fn: Do we want to include the loss_fn in this object? It could be returned independently as well.

class VBLLRegression(nn.Module):

    def __init__(self, in_features, out_features, prior_scale, wishart_scale, dof, dataset_size, loss_fn):

        self.prior_cov_scale = self.cfg.PRIOR_SCALE
        self.scale = self.cfg.WISHART_SCALE
        self.dof = (self.cfg.DOF + self.out_dim + 1.)/2.
        self.dataset_size = self.cfg.DATASET_SIZE

        self._W = nn.Parameter(torch.randn(out_feaures, in_features))
        self.prior_mean = torch.zeros(out_features, in_features)
        self.S = get_parameterization(self.cfg.S_COVARIANCE) 

        self.Sig_inv = get_parameterization(self.cfg.NOISE_COVARIANCE)
        

    def forward(self, x, return_loss_fn = False):
        predicive = x

        if return_loss_fn:
                return VBLLReturn(x, predictive), self._get_loss_fn(x)

        return VBLLReturn(x, predictive)


    def _get_loss_fn(self, features):

        def loss_fn(y):
            pred_likelihood = self.gaussian_ll(features, y) # batch dim
            trace_term = 0.5*((tp(features) @ self.S.mat @ feat) * self.Sig_inv.trace)

            kl_term = self.KL()
            wishart_term = (self.dof * self.Sig_inv.logdet - 0.5*self.scale*self.Sig_inv.trace)

            total_elbo = torch.mean(pred_likelihood - trace_term) + (1./self.dataset_size)*(wishart_term - kl_term)
            return - total_elbo

        return loss_fn

    def KL(self):
        mu_delta = tp(self.prior_mean - self.W)
        
        mse_term = (mu_delta**2).sum(-1).sum(-1)/self.prior_cov_scale
        trace_term = (self.out_dim * self.S.trace/self.prior_cov_scale)
        logdet_term = (self.phi_dim * np.log(self.prior_cov_scale) - self.S.logdet)

        return 0.5*(mse_term + trace_term + self.out_dim * logdet_term) # currently exclude constant

    def gaussian_ll(self, feat, y):
        # TODO move to helper function
        err = (y.unsqueeze(-1) - self.W @ feat) 
        return -0.5*(self.Sig_inv.weighted_inner_prod(err) - self.Sig_inv.logdet) # currently do not include constant


class VBLLRegressionELBO():

    def __init__(self, features, bll):

        self.features = features

        # Weights
        self.weights = bll.W
        self.prior_covariance = bll.S # Confusing not to pass around
        self.noise_covariance = bll.Sig_inv

        # Hyperparams
        self.scale = bll.scale
        self.dataset_size = bll.dataset_size

    def forward(self, y):


    def gaussian_ll(self, features, y): # should be implemented more generally

        err = (y.unsqueeze(-1) - self.weights @ feat) 
        return -0.5*(self.noise_covariance.weighted_inner_prod(err) - self.noise_covariance.logdet) # currently do not include constant


    feat = self.feat(x)



def elbo(model, out):

    # search model to find appropriate layer
    # operate on layer

    return model[layer].elbo(out)

