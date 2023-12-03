import torch
import abc


cov_param_map = {
    'dense': DenseCovariance,
    'diagonal': DiagonalCovariance,
    # 'lowrank': LowRankCovariance
}

def get_cov_parameterization(p):
  if p in cov_param_map:
    return cov_param_map[p]
  else:
    raise ValueError('Must specify a valid covariance parameterization.')

# ----- covariance objects

class Covariance(abc.ABC, nn.Module):
  @abc.abstractmethod
  def cholesky(self):
    raise NotImplementedError

  @abc.abstractmethod
  def mat(self):
    raise NotImplementedError

  @abc.abstractmethod
  def inverse_mat(self):
    raise NotImplementedError

  @abc.abstractmethod
  def logdet(self):
    raise NotImplementedError

  @abc.abstractmethod
  def trace(self):
    raise NotImplementedError

  @abc.abstractmethod
  def weighted_inner_product(self):
    raise NotImplementedError

  @abc.abstractmethod
  def inverse_weighted_inner_product(self):
    raise NotImplementedError

class DenseCovariance(Covariance):
  def __init__(self,
               mat_dim,
               batch_dim=1,
               diag_offset=0.,
               requires_grad=True):

    super(DenseCovariance, self).__init__()
    self.mat_dim = mat_dim
    self.batch_dim = batch_dim

    log_diag_init = torch.randn(batch_dim, mat_dim)
    self.log_diag = nn.Parameter(log_diag_init - diag_offset,
                                requires_grad=requires_grad)

    # by default, scale down off diag terms
    off_diag_init = torch.randn(batch_dim, mat_dim, mat_dim)/mat_dim
    self.off_diag = nn.Parameter(off_diag_init,
                                 requires_grad=requires_grad)


  @property
  def cholesky(self):
    return torch.tril(self.off_diag, diagonal=-1) + torch.diag_embed(torch.exp(self.log_diag))

  @property
  def mat(self):
    # Cholesky param
    return self.cholesky @ tp(self.cholesky)

  @property
  def inverse_mat(self):
    warnings.warn("Direct matrix inverse for dense covariances is O(N^3), consider using eg inverse weighted inner product")
    L_tri_inv = torch.inverse(self.cholesky)
    return tp(L_tri_inv) @ L_tri_inv

  @property
  def logdet(self):
    return 2*self.log_diag.sum(-1)

  @property
  def trace(self):
    return (self.cholesky**2).sum(-1).sum(-1) # compute as frob norm squared

  def weighted_inner_prod(self, b):
    return ((tp(self.cholesky) @ b)**2).sum(-1).sum(-1)

  def inverse_weighted_inner_prod(self, b):
    return (torch.linalg.solve(self.cholesky, b)**2).sum(-1).sum(-1)


class DiagonalCovariance(Covariance):
  def __init__(self,
               mat_dim,
               batch_dim=1,
               diag_offset=0.,
               requires_grad=True,
               covariance_scale=None):

    super(DiagonalCovariance, self).__init__()
    self.mat_dim = mat_dim
    self.batch_dim = batch_dim

    if covariance_scale is None:
      log_diag_init = torch.randn(batch_dim, mat_dim)
    else:
      log_diag_init = np.log(covariance_scale) * torch.ones(batch_dim, mat_dim)

    self.log_diag = nn.Parameter(log_diag_init - diag_offset,
                                requires_grad=requires_grad)

  @property
  def diagonal(self):
    return torch.exp(self.log_diag)

  @property
  def inverse_diagonal(self):
    return torch.exp(-self.log_diag)

  @property
  def diagonal_sqrt(self):
    return torch.exp(0.5 * self.log_diag)

  @property
  def diagonal_inv_sqrt(self):
    return torch.exp(-0.5 * self.log_diag)

  @property
  def cholesky(self):
    return torch.diag_embed(torch.exp(0.5 * self.log_diag))

  @property
  def mat(self):
    return torch.diag_embed(torch.exp(self.log_diag))

  @property
  def inv_mat(self):
    return torch.diag_embed(torch.exp(-self.log_diag))

  @property
  def logdet(self):
    return self.log_diag.sum(-1)

  @property
  def inverse_logdet(self):
    return -self.log_diag.sum(-1)

  @property
  def trace(self):
    return torch.exp(self.log_diag).sum(-1)

  @property
  def trace(self):
    return torch.exp(-self.log_diag).sum(-1)

  def weighted_inner_prod(self, b):
    return ((torch.exp(0.5 * self.log_diag).unsqueeze(-1) * b)**2).sum(-1).sum(-1)

  def inverse_weighted_inner_prod(self, b):
    return ((torch.exp(-0.5 * self.log_diag).unsqueeze(-1) * b)**2).sum(-1).sum(-1)

def KL(p1, p2):
    """Computes KL divergence between Gaussian random variables."""
    # TODO(jamesharrison)
    pass

def gaussian_vector_product(W, x):
  """Computes the Gaussian distribution resulting from the product of a Gaussian
  distributed tensor W with a vector x."""
  # TODO(jamesharrison): check dims
  return Gaussian(W.mean @ x, W.cov.weighted_inner_prod(x))

def gaussian_scalar_product(gaussian, s):
  return Gaussian(gaussian.mean * s, gaussian.cov * s**2)

def add_gaussians(p1, p2):
  """Returns sum of gaussian random variables assuming uncorrelated"""
  # construct combined covariance object:
  # TODO(jamesharrison)
  # combined_cov =
  return Gaussian(p1.mean + p2.mean, combined_cov)

# TODO(jamesharrison): is this a stupid abstraction?
# ----- gaussian objects

class Gaussian(nn.Module):
  def __init__(self, mean, cov):
    self.mean = mean
    self.cov = cov

  def log_prob(self, x):
    """Gaussian log probability"""
    logprob = -0.5 * self.cov.inverse_weighted_inner_prod(x - self.mean)
    logprob -= 0.5 * self.cov.inverse_logdet
    logprob -= 0.5 * self.cov.mat_dim * np.log(2 * np.pi)
    return logprob

  def sample(self):
    # TODO(jamesharrison)
    pass
