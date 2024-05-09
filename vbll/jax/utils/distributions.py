import numpyro
import jax.numpy as jnp
from jax.scipy.linalg import inv, solve
import warnings

def tp(M):
    return M.transpose(-3,-1,-2)

def sym(M):
    return (M + tp(M))/2.

class Normal(numpyro.distributions.Normal):
    def __init__(self, loc, scale, validate_args=None):
        super(Normal, self).__init__(loc, scale, validate_args=validate_args)

    @property
    def mean(self):
        return self.loc

    @property
    def var(self):
        return self.scale ** 2

    @property
    def chol_covariance(self):
        return jnp.diag(self.scale)

    @property
    def covariance_diagonal(self):
        return self.var

    @property
    def covariance(self):
        return jnp.diag(self.var)

    @property
    def precision(self):
        return jnp.diag(1. / self.var)

    @property
    def logdet_covariance(self):
        return 2 * jnp.log(self.scale).sum()

    @property
    def logdet_precision(self):
        return -self.logdet_covariance

    @property
    def trace_covariance(self):
        return self.var.sum()

    @property
    def trace_precision(self):
        return (1. / self.var).sum()

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        """
        Compute the covariance-weighted inner product between the Normal distribution and a vector b.

        Args:
            b (jnp.ndarray): The vector to compute the inner product with.
            reduce_dim (bool, optional): Whether to reduce the result along the last dimension. Defaults to True.

        Returns:
            jnp.ndarray: The covariance-weighted inner product.
        """
        prod = (self.var[:, None] * (b ** 2)).sum(axis=-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        """
        Compute the precision-weighted inner product between the Normal distribution and a vector b.

        Args:
            b (jnp.ndarray): The vector to compute the inner product with.
            reduce_dim (bool, optional): Whether to reduce the result along the last dimension. Defaults to True.

        Returns:
            jnp.ndarray: The precision-weighted inner product.
        """
        prod = ((b ** 2) / self.var[:, None]).sum(axis=-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __add__(self, inp):
        """
        Add the Normal distribution with another Normal distribution or a scalar value.

        Args:
            inp (Normal or jnp.ndarray or float or int): The input to add to the Normal distribution.

        Returns:
            Normal: The resulting Normal distribution after addition.

        Raises:
            NotImplementedError: If the input type is not supported for addition.
        """
        if isinstance(inp, Normal):
            new_var = self.var + inp.var
            new_scale = jnp.sqrt(jnp.clip(new_var, a_min=1e-12))
            return Normal(self.mean + inp.mean, new_scale)
        elif isinstance(inp, (jnp.ndarray, float, int)):
            return Normal(self.mean + inp, self.scale)
        else:
            raise NotImplementedError('Distribution addition only implemented for diag covs')

    def __matmul__(self, inp):
        """
        Perform matrix multiplication between the Normal distribution and an input matrix.

        Args:
            inp (jnp.ndarray): The input matrix to multiply with the Normal distribution.

        Returns:
            Normal: The resulting Normal distribution after matrix multiplication.
        """
        new_mean = self.loc @ inp
        new_var = self.covariance_weighted_inner_prod(inp, reduce_dim=False)
        new_scale = jnp.sqrt(jnp.clip(new_var, a_min=1e-12))
        return Normal(new_mean, new_scale)

    def squeeze(self, idx):
        """
        Removes a single dimension from the Normal distribution
        """
        # TODO: missing assert required?
        return Normal(jnp.squeeze(self.loc, axis=idx), jnp.squeeze(self.scale, axis=idx))

class DenseNormal(numpyro.distributions.MultivariateNormal):
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
        return tp(inv(self.scale_tril)) @ self.scale_tril

    @property
    def logdet_covariance(self):
        return 2. * jnp.log(jnp.diagonal(self.scale_tril, axis1=-2, axis2=-1)).sum(axis=-1)

    @property
    def trace_covariance(self):
        return (self.scale_tril**2).sum(axis=-1).sum(axis=-1) # compute as the Frob norm squared

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((tp(self.scale_tril) @ b) ** 2).sum(axis=-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (solve(self.scale_tril, b) ** 2).sum(axis=-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1

        new_mean = self.loc @ inp
        new_var = self.covariance_weighted_inner_prod(jnp.expand_dims(inp, -3), reduce_dim=False)
        new_scale = jnp.sqrt(jnp.clip(new_var, a_min=1e-12))

        return Normal(new_mean.squeeze(-1), new_scale.squeeze(-1))

    def squeeze(self, idx):
        return DenseNormal(self.loc.squeeze(axis=idx), self.scale_tril.squeeze(axis=idx))

cov_param_dict = {
    "dense": DenseNormal,
    "diagonal": Normal
}

def get_parameterization(p):
  if p in cov_param_dict:
    return cov_param_dict[p]
  else:
    raise ValueError('Must specify a valid covariance parameterization.')