"""
Tests for utility functions and edge cases.
"""
import pytest
import torch
import numpy as np
from vbll.utils.distributions import (
    cholesky_inverse, cholupdate, tp, sym, get_parameterization,
    cov_param_dict
)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_cholesky_inverse(self, device):
        """Test cholesky inverse function."""
        # Test with simple 2x2 matrix
        tril = torch.tril(torch.eye(2, device=device))
        inv = cholesky_inverse(tril)
        expected = torch.eye(2, device=device)
        assert torch.allclose(inv, expected)
        
        # Test with batch dimensions
        batch_size = 3
        tril_batch = torch.tril(torch.eye(2, device=device)).unsqueeze(0).repeat(batch_size, 1, 1)
        inv_batch = cholesky_inverse(tril_batch)
        expected_batch = torch.eye(2, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        assert torch.allclose(inv_batch, expected_batch)
    
    def test_cholupdate(self, device):
        """Test Cholesky update function."""
        # Test with identity matrix
        L = torch.tril(torch.eye(3, device=device))
        x = torch.tensor([1.0, 0.0, 0.0], device=device)
        weight = 1.0
        
        L_new = cholupdate(L, x, weight)
        
        # Check that L_new is lower triangular
        assert torch.allclose(L_new, torch.tril(L_new))
        
        # Check that L_new @ L_new.T gives expected result
        expected = L @ L.T + weight * torch.outer(x, x)
        result = L_new @ L_new.T
        assert torch.allclose(result, expected, rtol=1e-5)
    
    def test_cholupdate_downdate(self, device):
        """Test Cholesky downdate (negative weight)."""
        # Test with identity matrix
        L = torch.tril(torch.eye(3, device=device))
        x = torch.tensor([0.5, 0.0, 0.0], device=device)
        weight = -1.0
        
        L_new = cholupdate(L, x, weight)
        
        # Check that L_new is lower triangular
        assert torch.allclose(L_new, torch.tril(L_new))
    
    def test_tp_function(self, device):
        """Test transpose function."""
        # Test with 2D matrix
        M = torch.randn(3, 4, device=device)
        M_tp = tp(M)
        expected = M.transpose(-1, -2)
        assert torch.allclose(M_tp, expected)
        
        # Test with 3D tensor
        M_3d = torch.randn(2, 3, 4, device=device)
        M_3d_tp = tp(M_3d)
        expected_3d = M_3d.transpose(-1, -2)
        assert torch.allclose(M_3d_tp, expected_3d)
    
    def test_sym_function(self, device):
        """Test symmetric function."""
        # Test with 2D matrix
        M = torch.randn(3, 3, device=device)
        M_sym = sym(M)
        expected = (M + M.T) / 2
        assert torch.allclose(M_sym, expected)
        
        # Test that result is symmetric
        assert torch.allclose(M_sym, M_sym.T)
    
    def test_get_parameterization(self):
        """Test get_parameterization function."""
        # Test valid parameterizations
        assert get_parameterization('diagonal') == cov_param_dict['diagonal']
        assert get_parameterization('dense') == cov_param_dict['dense']
        assert get_parameterization('dense_precision') == cov_param_dict['dense_precision']
        assert get_parameterization('lowrank') == cov_param_dict['lowrank']
        
        # Test invalid parameterization
        with pytest.raises(ValueError):
            get_parameterization('invalid')
    
    def test_cov_param_dict(self):
        """Test covariance parameterization dictionary."""
        expected_keys = {'diagonal', 'dense', 'dense_precision', 'lowrank'}
        assert set(cov_param_dict.keys()) == expected_keys
        
        # Test that all values are classes
        for key, value in cov_param_dict.items():
            assert isinstance(value, type)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_tensors(self, device):
        """Test with empty tensors."""
        # Test with empty batch dimension
        mean = torch.randn(0, 3, device=device)
        scale = torch.randn(0, 3, device=device)
        
        # This should not raise an error
        from vbll.utils.distributions import Normal
        dist = Normal(mean, scale)
        assert dist.mean.shape == (0, 3)
        assert dist.scale.shape == (0, 3)
    
    def test_single_element_tensors(self, device):
        """Test with single element tensors."""
        mean = torch.tensor([1.0], device=device)
        scale = torch.tensor([2.0], device=device)
        
        from vbll.utils.distributions import Normal
        dist = Normal(mean, scale)
        
        # Test operations
        result = dist + dist
        assert result.mean.item() == 2.0
        assert result.scale.item() == np.sqrt(8.0)  # sqrt(2^2 + 2^2)
    
    def test_very_small_values(self, device):
        """Test with very small values."""
        mean = torch.tensor([1e-10], device=device)
        scale = torch.tensor([1e-15], device=device)
        
        from vbll.utils.distributions import Normal
        dist = Normal(mean, scale)
        
        # Test that operations don't produce NaN
        result = dist + dist
        assert torch.isfinite(result.mean)
        assert torch.isfinite(result.scale)
        assert result.scale >= 1e-12  # Check clipping
    
    def test_very_large_values(self, device):
        """Test with very large values."""
        mean = torch.tensor([1e10], device=device)
        scale = torch.tensor([1e5], device=device)
        
        from vbll.utils.distributions import Normal
        dist = Normal(mean, scale)
        
        # Test that operations don't produce inf
        result = dist + dist
        assert torch.isfinite(result.mean)
        assert torch.isfinite(result.scale)
    
    def test_nan_handling(self, device):
        """Test handling of NaN values."""
        mean = torch.tensor([float('nan')], device=device)
        scale = torch.tensor([1.0], device=device)
        
        from vbll.utils.distributions import Normal
        dist = Normal(mean, scale)
        
        # Test that NaN propagates correctly
        assert torch.isnan(dist.mean)
        assert not torch.isnan(dist.scale)
    
    def test_inf_handling(self, device):
        """Test handling of inf values."""
        mean = torch.tensor([float('inf')], device=device)
        scale = torch.tensor([1.0], device=device)
        
        from vbll.utils.distributions import Normal
        dist = Normal(mean, scale)
        
        # Test that inf propagates correctly
        assert torch.isinf(dist.mean)
        assert not torch.isinf(dist.scale)


class TestMathematicalProperties:
    """Test mathematical properties of utility functions."""
    
    def test_cholesky_inverse_properties(self, device):
        """Test mathematical properties of cholesky inverse."""
        # Test that A * A^(-1) = I
        A = torch.randn(3, 3, device=device)
        A = A @ A.T  # Make positive definite
        L = torch.linalg.cholesky(A)
        A_inv = cholesky_inverse(L)
        
        identity = A @ A_inv
        expected_identity = torch.eye(3, device=device)
        assert torch.allclose(identity, expected_identity, rtol=1e-5)
    
    def test_cholupdate_properties(self, device):
        """Test mathematical properties of cholupdate."""
        # Test that cholupdate preserves positive definiteness
        L = torch.tril(torch.eye(3, device=device))
        x = torch.randn(3, device=device)
        weight = 1.0
        
        L_new = cholupdate(L, x, weight)
        
        # Check that L_new @ L_new.T is positive definite
        A_new = L_new @ L_new.T
        eigenvals = torch.linalg.eigvals(A_new)
        assert torch.all(eigenvals.real >= -1e-6)  # Allow small numerical errors
    
    def test_symmetry_properties(self, device):
        """Test symmetry properties."""
        # Test that sym function produces symmetric matrices
        M = torch.randn(4, 4, device=device)
        M_sym = sym(M)
        
        # Check symmetry
        assert torch.allclose(M_sym, M_sym.T)
        
        # Check that sym is idempotent
        M_sym_sym = sym(M_sym)
        assert torch.allclose(M_sym, M_sym_sym)


class TestPerformance:
    """Test performance characteristics."""
    
    def test_memory_efficiency(self, device):
        """Test memory efficiency of operations."""
        # Test with large tensors
        batch_size = 100
        dim = 50
        
        mean = torch.randn(batch_size, dim, device=device)
        scale = torch.ones(batch_size, dim, device=device)
        
        from vbll.utils.distributions import Normal
        dist = Normal(mean, scale)
        
        # Test that operations don't use excessive memory
        result = dist + dist
        assert result.mean.shape == (batch_size, dim)
        assert result.scale.shape == (batch_size, dim)
    
    def test_computational_efficiency(self, device):
        """Test computational efficiency."""
        import time
        
        # Test timing of operations
        mean = torch.randn(100, 50, device=device)
        scale = torch.ones(100, 50, device=device)
        
        from vbll.utils.distributions import Normal
        dist = Normal(mean, scale)
        
        # Time addition operation
        start_time = time.time()
        for _ in range(100):
            result = dist + dist
        end_time = time.time()
        
        # Should complete in reasonable time
        assert (end_time - start_time) < 1.0  # Less than 1 second
    
    def test_batch_operations(self, device):
        """Test efficiency of batch operations."""
        batch_size = 50
        dim = 20
        
        mean = torch.randn(batch_size, dim, device=device)
        scale = torch.ones(batch_size, dim, device=device)
        
        from vbll.utils.distributions import Normal
        dist = Normal(mean, scale)
        
        # Test batch matrix multiplication
        matrix = torch.randn(dim, 1, device=device)
        result = dist @ matrix
        
        assert result.mean.shape == (batch_size, 1)
        assert result.scale.shape == (batch_size, 1)
        
        # Test batch addition
        dist2 = Normal(torch.zeros_like(mean), torch.ones_like(scale))
        result_add = dist + dist2
        
        assert result_add.mean.shape == (batch_size, dim)
        assert result_add.scale.shape == (batch_size, dim)


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_invalid_input_shapes(self, device):
        """Test handling of invalid input shapes."""
        from vbll.utils.distributions import Normal
        
        # Test mismatched shapes
        mean = torch.randn(3, 4, device=device)
        scale = torch.randn(3, 5, device=device)  # Different last dimension
        
        with pytest.raises((RuntimeError, ValueError)):
            dist = Normal(mean, scale)
    
    def test_invalid_parameterization(self):
        """Test handling of invalid parameterization."""
        with pytest.raises(ValueError):
            get_parameterization('nonexistent')
    
    def test_cholupdate_invalid_inputs(self, device):
        """Test cholupdate with invalid inputs."""
        L = torch.tril(torch.eye(3, device=device))
        
        # Test with wrong shape
        x_wrong = torch.randn(4, device=device)  # Wrong dimension
        with pytest.raises(AssertionError):
            cholupdate(L, x_wrong)
    
    def test_matrix_multiplication_shape_errors(self, device):
        """Test matrix multiplication shape errors."""
        from vbll.utils.distributions import Normal
        
        mean = torch.randn(2, 3, device=device)
        scale = torch.ones(2, 3, device=device)
        dist = Normal(mean, scale)
        
        # Test with wrong matrix shape
        matrix_wrong = torch.randn(4, 1, device=device)  # Wrong first dimension
        with pytest.raises(AssertionError):
            result = dist @ matrix_wrong

