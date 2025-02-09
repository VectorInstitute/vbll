import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from collections.abc import Callable
import abc
import warnings
from typing import Optional, Union

# Credit to https://github.com/brentyi/fannypack/blob/2888aa5d969824ac1e1a528264674ece3f4703f9/fannypack/utils/_math.py
def cholesky_inverse(u: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """Alternative to `torch.cholesky_inverse()`, with support for batch dimensions.

    Relevant issue tracker: https://github.com/pytorch/pytorch/issues/7500

    Args:
        u (torch.Tensor): Triangular Cholesky factor. Shape should be `(*, N, N)`.
        upper (bool, optional): Whether to consider the Cholesky factor as a lower or
            upper triangular matrix.

    Returns:
        torch.Tensor:
    """
    if u.dim() == 2 and not u.requires_grad:
        return torch.cholesky_inverse(u, upper=upper)
    return torch.cholesky_solve(torch.eye(u.size(-1)).expand(u.size()), u, upper=upper)

# Credit to https://github.com/brentyi/fannypack/blob/2888aa5d969824ac1e1a528264674ece3f4703f9/fannypack/utils/_math.py
def cholupdate(
    L: torch.Tensor,
    x: torch.Tensor,
    weight: Optional[Union[torch.Tensor, float]] = None,
) -> torch.Tensor:
    """Batched rank-1 Cholesky update.

    Computes the Cholesky decomposition of `RR^T + weight * xx^T`.

    Args:
        L (torch.Tensor): Lower triangular Cholesky decomposition of a PSD matrix. Shape
            should be `(*, matrix_dim, matrix_dim)`.
        x (torch.Tensor): Rank-1 update vector. Shape should be `(*, matrix_dim)`.
        weight (torch.Tensor or float, optional): Set to -1 for "downdate". Shape must
            be broadcastable with `(*, matrix_dim)`.

    Returns:
        torch.Tensor: New L matrix. Same shape as L.
    """
    # Expected shapes: (*, dim, dim) and (*, dim)
    batch_dims = L.shape[:-2]
    matrix_dim = x.shape[-1]
    assert x.shape[:-1] == batch_dims
    assert matrix_dim == L.shape[-1] == L.shape[-2]

    # Flatten batch dimensions, and clone for tensors we need to mutate
    L = L.reshape((-1, matrix_dim, matrix_dim))
    x = x.reshape((-1, matrix_dim)).clone()
    L_out_cols = []

    sign: Union[float, torch.Tensor]
    if weight is None:
        sign = L.new_ones((1,))
    elif isinstance(weight, float):
        x = x * np.sqrt(np.abs(weight))
        sign = float(np.sign(weight))
    else:
        x = x * torch.sqrt(torch.abs(weight))
        sign = torch.sign(weight)

    # Cholesky update; mostly copied from Wikipedia:
    # https://en.wikipedia.org/wiki/Cholesky_decomposition
    for k in range(matrix_dim):
        r = torch.sqrt(L[:, k, k] ** 2 + sign * x[:, k] ** 2)
        c = (r / L[:, k, k])[:, None]
        s = (x[:, k] / L[:, k, k])[:, None]

        # We build output column-by-column to avoid in-place modification errors
        L_out_col = torch.zeros_like(L[:, :, k])
        L_out_col[:, k] = r
        L_out_col[:, k + 1 :] = (L[:, k + 1 :, k] + sign * s * x[:, k + 1 :]) / c
        L_out_cols.append(L_out_col)

        # We clone x at each iteration, also to avoid in-place modification errors
        x_next = x.clone()
        x_next[:, k + 1 :] = c * x[:, k + 1 :] - s * L_out_col[:, k + 1 :]
        x = x_next

    # Stack columns together
    L_out = torch.stack(L_out_cols, dim=2)

    # Unflatten batch dimensions and return
    return L_out.reshape(batch_dims + (matrix_dim, matrix_dim))

def get_parameterization(p):
  if p in cov_param_dict:
    return cov_param_dict[p]
  else:
    raise ValueError('Must specify a valid covariance parameterization.')

def tp(M):
    return M.transpose(-1,-2)

def sym(M):
    return (M + tp(M))/2.

class Normal(torch.distributions.Normal):
    def __init__(self, loc, chol):
        super(Normal, self).__init__(loc, chol)

    @property
    def mean(self):
        return self.loc

    @property
    def var(self):
        return self.scale ** 2

    @property
    def chol_covariance(self):
        return torch.diag_embed(self.scale)

    @property
    def covariance_diagonal(self):
        return self.var

    @property
    def covariance(self):
        return torch.diag_embed(self.var)

    @property
    def precision(self):
        return torch.diag_embed(1./self.var)

    @property
    def logdet_covariance(self):
        return 2 * torch.log(self.scale).sum(-1)

    @property
    def logdet_precision(self):
        return -2 * torch.log(self.scale).sum(-1)

    @property
    def trace_covariance(self):
        return self.var.sum(-1)

    @property
    def trace_precision(self):
        return (1./self.var).sum(-1)

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (self.var.unsqueeze(-1) * (b ** 2)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((b ** 2)/self.var.unsqueeze(-1)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __add__(self, inp):
        if isinstance(inp, Normal):
            new_cov =  self.var + inp.var
            return Normal(self.mean + inp.mean, torch.sqrt(torch.clip(new_cov, min = 1e-12)))
        elif isinstance(inp, torch.Tensor):
            return Normal(self.mean + inp, self.scale)
        else:
            raise NotImplementedError('Distribution addition only implemented for diag covs')

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(inp.unsqueeze(-3), reduce_dim = False)
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min = 1e-12)))

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
        return tp(torch.linalg.inv(self.scale_tril)) @ torch.linalg.inv(self.scale_tril)

    @property
    def logdet_covariance(self):
        return 2. * torch.diagonal(self.scale_tril, dim1=-2, dim2=-1).log().sum(-1)

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
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min = 1e-12)))

    def squeeze(self, idx):
        return DenseNormal(self.loc.squeeze(idx), self.scale_tril.squeeze(idx))

class DenseNormalPrec(torch.distributions.MultivariateNormal):
    """A DenseNormal parameterized by the mean and the cholesky decomp of the precision matrix.

    This function also includes a recursive_update function which performs a recursive 
    linear regression update with effecient cholesky factor updates. 
    """
    def __init__(self, loc, cholesky, validate_args=False):
        prec = cholesky @ tp(cholesky)
        super(DenseNormalPrec, self).__init__(loc, precision_matrix=prec, validate_args=validate_args)
        self.tril = cholesky

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        raise NotImplementedError()

    @property
    def covariance(self):
        warnings.warn("Direct matrix inverse for dense covariances is O(N^3), consider using eg inverse weighted inner product")
        # TODO replace with cholesky_inverse
        return cholesky_inverse(self.tril)

    @property
    def inverse_covariance(self):
        return self.precision_matrix

    @property
    def logdet_covariance(self):
        return -2. * torch.diagonal(self.tril, dim1=-2, dim2=-1).log().sum(-1)

    @property
    def trace_covariance(self):
        return (torch.inverse(self.tril)**2).sum(-1).sum(-1) # compute as frob norm squared

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (torch.linalg.solve(self.tril, b)**2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((tp(self.tril) @ b)**2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(inp.unsqueeze(-3), reduce_dim = False)
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min = 1e-12)))

    def squeeze(self, idx):
        return DenseNormalPrecision(self.loc.squeeze(idx), self.tril.squeeze(idx))

    def recursive_update(self, X, y, noise_cov):
        noise_cov = noise_cov.unsqueeze(-1)
        prec = self.inverse_covariance # out_dim * feat_dim * feat_dim
        chol = self.tril

        XTy =  (tp(y) @ X) / noise_cov # out_dim * feat_dim

        # recursively update cholesky
        for i in range(X.shape[0]):
            x = X[i].unsqueeze(-2) / torch.sqrt(noise_cov) # out_dim * feat_dim
            chol = cholupdate(chol, x)

        cov_update = (prec @ self.loc.unsqueeze(-1)) # out_dim * feat dim * 1
        cov_update += XTy.unsqueeze(-1) # out_dim * feat dim * 1
        new_loc = (cholesky_inverse(chol) @ cov_update).squeeze(-1) # out_dim * feat dim
        
        return chol, new_loc


class LowRankNormal(torch.distributions.LowRankMultivariateNormal):
    def __init__(self, loc, cov_factor, diag):
        super(LowRankNormal, self).__init__(loc, cov_factor=cov_factor, cov_diag=diag)

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        raise NotImplementedError()

    @property
    def covariance(self):
        return self.cov_factor @ tp(self.cov_factor) + torch.diag_embed(self.cov_diag)

    @property
    def inverse_covariance(self):
        # TODO(jamesharrison): implement via woodbury
        raise NotImplementedError()

    @property
    def logdet_covariance(self):
        # Apply Matrix determinant lemma
        term1 = torch.log(self.cov_diag).sum(-1)
        arg1 = tp(self.cov_factor) @ (self.cov_factor/self.cov_diag.unsqueeze(-1))
        term2 = torch.linalg.det(arg1 + torch.eye(arg1.shape[-1])).log()
        return term1 + term2

    @property
    def trace_covariance(self):
        # trace of sum is sum of traces
        trace_diag = self.cov_diag.sum(-1)
        trace_lowrank = (self.cov_factor ** 2).sum(-1).sum(-1)
        return trace_diag + trace_lowrank

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        diag_term = (self.cov_diag.unsqueeze(-1) * (b ** 2)).sum(-2)
        factor_term = ((tp(self.cov_factor) @ b) ** 2).sum(-2)
        prod = diag_term + factor_term
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        raise NotImplementedError()

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(inp.unsqueeze(-3), reduce_dim = False)
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min = 1e-12)))

    def squeeze(self, idx):
        return LowRankNormal(self.loc.squeeze(idx), self.cov_factor.squeeze(idx), self.cov_diag.squeeze(idx))


cov_param_dict = {
    'dense': DenseNormal,
    'dense_precision': DenseNormalPrec,
    'diagonal': Normal,
    'lowrank': LowRankNormal
}
