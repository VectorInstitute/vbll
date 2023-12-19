import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from collections.abc import Callable
import abc
import warnings


def tp(M):
    return M.transpose(-1,-2)

def sym(M):
    return (M + tp(M))/2.

class Normal(torch.distributions.Normal):
    def __init__(self, loc, var):
        super(Normal, self).__init__(loc, var)

    @property
    def chol_covariance(self):
        return torch.diag_embed(self.scale.sqrt())

    @property
    def covariance_diagonal(self):
        return self.scale

    @property
    def covariance(self):
        return torch.diag_embed(self.scale)

    @property
    def precision(self):
        return torch.diag_embed(1./self.scale)

    @property
    def logdet_covariance(self):
        return torch.log(self.scale).sum(-1)

    @property
    def logdet_precision(self):
        return -torch.log(self.scale).sum(-1)

    @property
    def trace_covariance(self):
        return self.scale.sum(-1)

    @property
    def trace_precision(self):
        return 1./self.scale.sum(-1)

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (self.scale.unsqueeze(-1) * (b ** 2)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (1./self.scale.unsqueeze(-1) * (b ** 2)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __add__(self, inp):
        if isinstance(inp, Normal):
            new_cov =  self.scale + inp.scale
            return Normal(self.loc + inp.loc, torch.clip(new_cov, min = 1e-8))
        elif isinstance(inp, torch.Tensor):
            return Normal(self.loc + inp, self.scale)
        else:
            raise NotImplementedError('Distribution addition only implemented for diag covs')

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(inp.unsqueeze(-3), reduce_dim = False)
        return Normal(self.loc @ inp, torch.clip(new_cov, min = 1e-8))

    def squeeze(self, idx):
        return Normal(self.loc.squeeze(idx), self.scale.squeeze(idx))


class DenseNormal(torch.distributions.MultivariateNormal):
    def __init__(self, loc, cholesky):
        super(DenseNormal, self).__init__(loc, scale_tril=cholesky)

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        return self.scale_tril

    @property
    def covariance(self):
        return self.scale_tril @ tp(self.scale_tril)

    @property
    def inverse_covariance(self):
        warnings.warn("Direct matrix inverse for dense covariances is O(N^3), consider using eg inverse weighted inner product")
        return tp(torch.linalg.inv(self.scale_tril)) @ self.scale_tril

    @property
    def logdet_covariance(self):
        return torch.diagonal(self.scale_tril, dim1=-2, dim2=-1).log().sum(-1)

    @property
    def trace_covariance(self):
        return (self.scale_tril**2).sum(-1).sum(-1) # compute as frob norm squared

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((tp(self.scale_tril) @ b)**2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (torch.linalg.solve(self.scale_tril, b)**2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(inp.unsqueeze(-3), reduce_dim = False)
        return Normal(self.loc @ inp, torch.clip(new_cov, min = 1e-8))

    def squeeze(self, idx):
        return DenseNormal(self.loc.squeeze(idx), self.scale_tril.squeeze(idx))
