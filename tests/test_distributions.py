"""
Tests for VBLL distribution classes.
"""
import pytest
import torch
import numpy as np
from vbll.utils.distributions import Normal, DenseNormal, DenseNormalPrec, LowRankNormal, get_parameterization


class TestNormal:
    """Test the Normal (diagonal covariance) distribution."""
    
    def test_initialization(self, device):
        """Test Normal distribution initialization."""
        mean = torch.randn(5, 3, device=device)
        scale = torch.randn(5, 3, device=device)
        dist = Normal(mean, scale)
        
        assert torch.allclose(dist.mean, mean)
        assert torch.allclose(dist.scale, scale)
        assert torch.allclose(dist.var, scale ** 2)
    
    def test_properties(self, device):
        """Test Normal distribution properties."""
        mean = torch.randn(2, 4, device=device)
        scale = torch.ones(2, 4, device=device)
        dist = Normal(mean, scale)
        
        # Test covariance diagonal
        expected_var = torch.ones(2, 4, device=device)
        assert torch.allclose(dist.covariance_diagonal, expected_var)
        
        # Test trace
        expected_trace = torch.tensor([4.0, 4.0], device=device)
        assert torch.allclose(dist.trace_covariance, expected_trace)
        
        # Test log determinant
        expected_logdet = torch.tensor([0.0, 0.0], device=device)
        assert torch.allclose(dist.logdet_covariance, expected_logdet)
    
    def test_addition(self, device):
        """Test Normal distribution addition."""
        mean1 = torch.randn(3, 2, device=device)
        scale1 = torch.ones(3, 2, device=device)
        dist1 = Normal(mean1, scale1)
        
        mean2 = torch.randn(3, 2, device=device)
        scale2 = torch.ones(3, 2, device=device)
        dist2 = Normal(mean2, scale2)
        
        result = dist1 + dist2
        
        expected_mean = mean1 + mean2
        expected_var = scale1**2 + scale2**2
        expected_scale = torch.sqrt(expected_var)
        
        assert torch.allclose(result.mean, expected_mean)
        assert torch.allclose(result.scale, expected_scale)
    
    def test_matrix_multiplication(self, device):
        """Test Normal distribution matrix multiplication."""
        mean = torch.randn(2, 3, device=device)
        scale = torch.ones(2, 3, device=device)
        dist = Normal(mean, scale)
        
        matrix = torch.randn(3, 1, device=device)
        result = dist @ matrix
        
        expected_mean = mean @ matrix
        expected_var = (scale**2).unsqueeze(-1) * (matrix**2)
        expected_scale = torch.sqrt(expected_var.squeeze(-1))
        
        assert torch.allclose(result.mean, expected_mean)
        assert torch.allclose(result.scale, expected_scale)
    
    def test_covariance_weighted_inner_product(self, device):
        """Test covariance weighted inner product."""
        mean = torch.randn(2, 3, device=device)
        scale = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 2.0]], device=device)
        dist = Normal(mean, scale)
        
        b = torch.randn(2, 3, 1, device=device)
        result = dist.covariance_weighted_inner_prod(b)
        
        expected = (scale**2 * (b**2)).sum(-2).squeeze(-1)
        assert torch.allclose(result, expected)


class TestDenseNormal:
    """Test the DenseNormal (full covariance) distribution."""
    
    def test_initialization(self, device):
        """Test DenseNormal distribution initialization."""
        mean = torch.randn(2, 3, device=device)
        tril = torch.tril(torch.randn(2, 3, 3, device=device))
        dist = DenseNormal(mean, tril)
        
        assert torch.allclose(dist.mean, mean)
        assert torch.allclose(dist.chol_covariance, tril)
    
    def test_covariance_matrix(self, device):
        """Test covariance matrix computation."""
        mean = torch.randn(1, 2, device=device)
        tril = torch.tril(torch.eye(2, device=device))
        dist = DenseNormal(mean, tril)
        
        expected_cov = torch.eye(2, device=device)
        assert torch.allclose(dist.covariance, expected_cov)
    
    def test_trace_covariance(self, device):
        """Test trace of covariance matrix."""
        mean = torch.randn(1, 2, device=device)
        tril = torch.tril(torch.eye(2, device=device))
        dist = DenseNormal(mean, tril)
        
        expected_trace = torch.tensor([2.0], device=device)
        assert torch.allclose(dist.trace_covariance, expected_trace)
    
    def test_logdet_covariance(self, device):
        """Test log determinant of covariance matrix."""
        mean = torch.randn(1, 2, device=device)
        tril = torch.tril(torch.eye(2, device=device))
        dist = DenseNormal(mean, tril)
        
        expected_logdet = torch.tensor([0.0], device=device)
        assert torch.allclose(dist.logdet_covariance, expected_logdet)
    
    def test_matrix_multiplication(self, device):
        """Test DenseNormal matrix multiplication."""
        mean = torch.randn(1, 2, device=device)
        tril = torch.tril(torch.eye(2, device=device))
        dist = DenseNormal(mean, tril)
        
        matrix = torch.randn(2, 1, device=device)
        result = dist @ matrix
        
        expected_mean = mean @ matrix
        expected_var = (tril @ matrix.T).T @ (tril @ matrix.T)
        expected_scale = torch.sqrt(expected_var.squeeze(-1))
        
        assert torch.allclose(result.mean, expected_mean)
        assert torch.allclose(result.scale, expected_scale)


class TestDenseNormalPrec:
    """Test the DenseNormalPrec (precision matrix) distribution."""
    
    def test_initialization(self, device):
        """Test DenseNormalPrec distribution initialization."""
        mean = torch.randn(1, 2, device=device)
        tril = torch.tril(torch.eye(2, device=device))
        dist = DenseNormalPrec(mean, tril)
        
        assert torch.allclose(dist.mean, mean)
        assert torch.allclose(dist.tril, tril)
    
    def test_inverse_covariance(self, device):
        """Test inverse covariance computation."""
        mean = torch.randn(1, 2, device=device)
        tril = torch.tril(torch.eye(2, device=device))
        dist = DenseNormalPrec(mean, tril)
        
        expected_inv_cov = torch.eye(2, device=device)
        assert torch.allclose(dist.inverse_covariance, expected_inv_cov)
    
    def test_logdet_covariance(self, device):
        """Test log determinant of covariance matrix."""
        mean = torch.randn(1, 2, device=device)
        tril = torch.tril(torch.eye(2, device=device))
        dist = DenseNormalPrec(mean, tril)
        
        expected_logdet = torch.tensor([0.0], device=device)
        assert torch.allclose(dist.logdet_covariance, expected_logdet)


class TestLowRankNormal:
    """Test the LowRankNormal distribution."""
    
    def test_initialization(self, device):
        """Test LowRankNormal distribution initialization."""
        mean = torch.randn(2, 3, device=device)
        cov_factor = torch.randn(2, 3, 2, device=device)
        cov_diag = torch.ones(2, 3, device=device)
        dist = LowRankNormal(mean, cov_factor, cov_diag)
        
        assert torch.allclose(dist.mean, mean)
        assert torch.allclose(dist.cov_factor, cov_factor)
        assert torch.allclose(dist.cov_diag, cov_diag)
    
    def test_covariance_matrix(self, device):
        """Test covariance matrix computation."""
        mean = torch.randn(1, 2, device=device)
        cov_factor = torch.randn(1, 2, 1, device=device)
        cov_diag = torch.ones(1, 2, device=device)
        dist = LowRankNormal(mean, cov_factor, cov_diag)
        
        expected_cov = cov_factor @ cov_factor.transpose(-1, -2) + torch.diag_embed(cov_diag)
        assert torch.allclose(dist.covariance, expected_cov)
    
    def test_trace_covariance(self, device):
        """Test trace of covariance matrix."""
        mean = torch.randn(1, 2, device=device)
        cov_factor = torch.zeros(1, 2, 1, device=device)
        cov_diag = torch.ones(1, 2, device=device)
        dist = LowRankNormal(mean, cov_factor, cov_diag)
        
        expected_trace = torch.tensor([2.0], device=device)
        assert torch.allclose(dist.trace_covariance, expected_trace)
    
    def test_matrix_multiplication(self, device):
        """Test LowRankNormal matrix multiplication."""
        mean = torch.randn(1, 2, device=device)
        cov_factor = torch.zeros(1, 2, 1, device=device)
        cov_diag = torch.ones(1, 2, device=device)
        dist = LowRankNormal(mean, cov_factor, cov_diag)
        
        matrix = torch.randn(2, 1, device=device)
        result = dist @ matrix
        
        expected_mean = mean @ matrix
        expected_var = (cov_diag.unsqueeze(-1) * (matrix**2)).sum(-2)
        expected_scale = torch.sqrt(expected_var.squeeze(-1))
        
        assert torch.allclose(result.mean, expected_mean)
        assert torch.allclose(result.scale, expected_scale)


class TestGetParameterization:
    """Test the get_parameterization function."""
    
    def test_valid_parameterizations(self):
        """Test that valid parameterizations are returned correctly."""
        assert get_parameterization('diagonal') == Normal
        assert get_parameterization('dense') == DenseNormal
        assert get_parameterization('dense_precision') == DenseNormalPrec
        assert get_parameterization('lowrank') == LowRankNormal
    
    def test_invalid_parameterization(self):
        """Test that invalid parameterizations raise ValueError."""
        with pytest.raises(ValueError):
            get_parameterization('invalid')


class TestDistributionOperations:
    """Test distribution operations and edge cases."""
    
    def test_squeeze_operation(self, device):
        """Test squeeze operation on distributions."""
        mean = torch.randn(1, 3, device=device)
        scale = torch.randn(1, 3, device=device)
        dist = Normal(mean, scale)
        
        squeezed = dist.squeeze(0)
        assert squeezed.mean.shape == (3,)
        assert squeezed.scale.shape == (3,)
    
    def test_precision_weighted_inner_product(self, device):
        """Test precision weighted inner product."""
        mean = torch.randn(2, 3, device=device)
        scale = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 2.0]], device=device)
        dist = Normal(mean, scale)
        
        b = torch.randn(2, 3, 1, device=device)
        result = dist.precision_weighted_inner_prod(b)
        
        expected = ((b**2) / (scale**2).unsqueeze(-1)).sum(-2).squeeze(-1)
        assert torch.allclose(result, expected)
    
    def test_covariance_clipping(self, device):
        """Test that covariance values are properly clipped."""
        mean = torch.randn(2, 3, device=device)
        scale = torch.tensor([[1e-15, 1e-10, 1.0], [1e-20, 1e-5, 2.0]], device=device)
        dist = Normal(mean, scale)
        
        # Test addition with very small variances
        dist2 = Normal(torch.zeros_like(mean), torch.tensor([[1e-16, 1e-12, 1.0], [1e-18, 1e-8, 1.0]], device=device))
        result = dist + dist2
        
        # Check that variances are clipped to minimum value
        assert torch.all(result.var >= 1e-12)
    
    def test_batch_operations(self, device):
        """Test operations on batched distributions."""
        batch_size = 4
        mean = torch.randn(batch_size, 3, device=device)
        scale = torch.ones(batch_size, 3, device=device)
        dist = Normal(mean, scale)
        
        # Test batch matrix multiplication
        matrix = torch.randn(3, 1, device=device)
        result = dist @ matrix
        
        assert result.mean.shape == (batch_size, 1)
        assert result.scale.shape == (batch_size, 1)
        
        # Test batch addition
        dist2 = Normal(torch.zeros_like(mean), torch.ones_like(scale))
        result_add = dist + dist2
        
        assert result_add.mean.shape == (batch_size, 3)
        assert result_add.scale.shape == (batch_size, 3)

