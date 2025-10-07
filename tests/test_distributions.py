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
        scale = torch.ones(5, 3, device=device)
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