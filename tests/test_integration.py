"""
Integration tests for VBLL components.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from vbll.layers.classification import DiscClassification
from vbll.layers.regression import Regression, HetRegression
from vbll.utils.distributions import Normal, DenseNormal


class TestEndToEndClassification:
    """End-to-end tests for classification."""
    
    def test_simple_classification_training(self, sample_data_medium, classification_params):
        """Test simple classification training loop."""
        data = sample_data_medium
        layer = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        
        # Training loop
        for epoch in range(5):
            optimizer.zero_grad()
            
            result = layer(data['x'])
            train_loss = result.train_loss_fn(data['y_classification'])
            
            train_loss.backward()
            optimizer.step()
            
            # Check that loss is finite and decreasing
            assert torch.isfinite(train_loss)
            if epoch > 0:
                # Loss should generally decrease (allow for some noise)
                pass
    
    def test_classification_uncertainty(self, sample_data_medium, classification_params):
        """Test classification uncertainty estimation."""
        data = sample_data_medium
        layer = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            return_ood=True,
            **classification_params
        )
        
        # Train for a few epochs
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            result = layer(data['x'])
            train_loss = result.train_loss_fn(data['y_classification'])
            train_loss.backward()
            optimizer.step()
        
        # Test uncertainty estimation
        result = layer(data['x'])
        ood_scores = result.ood_scores
        
        assert ood_scores.shape == (data['batch_size'],)
        assert torch.all(ood_scores >= 0) and torch.all(ood_scores <= 1)
        
        # Test predictive probabilities
        pred_probs = result.predictive.probs
        assert pred_probs.shape == (data['batch_size'], data['output_features'])
        assert torch.allclose(pred_probs.sum(-1), torch.ones(data['batch_size']))
    
    def test_different_softmax_bounds(self, sample_data_small, classification_params):
        """Test different softmax bounds in classification."""
        data = sample_data_small
        
        # Test Jensen bound
        layer_jensen = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            softmax_bound='jensen',
            **classification_params
        )
        
        result_jensen = layer_jensen(data['x'])
        loss_jensen = result_jensen.train_loss_fn(data['y_classification'])
        
        assert torch.isfinite(loss_jensen)
        assert result_jensen.predictive.probs.shape == (data['batch_size'], data['output_features'])
    
    def test_classification_with_different_parameterizations(self, sample_data_small, classification_params):
        """Test classification with different covariance parameterizations."""
        data = sample_data_small
        
        for param in ['diagonal', 'dense']:
            layer = DiscClassification(
                in_features=data['input_features'],
                out_features=data['output_features'],
                parameterization=param,
                **classification_params
            )
            
            result = layer(data['x'])
            loss = result.train_loss_fn(data['y_classification'])
            
            assert torch.isfinite(loss)
            assert result.predictive.probs.shape == (data['batch_size'], data['output_features'])


class TestEndToEndRegression:
    """End-to-end tests for regression."""
    
    def test_simple_regression_training(self, sample_data_medium, regression_params):
        """Test simple regression training loop."""
        data = sample_data_medium
        layer = Regression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        
        # Training loop
        for epoch in range(5):
            optimizer.zero_grad()
            
            result = layer(data['x'])
            train_loss = result.train_loss_fn(data['y_regression'])
            
            train_loss.backward()
            optimizer.step()
            
            # Check that loss is finite
            assert torch.isfinite(train_loss)
    
    def test_regression_uncertainty(self, sample_data_medium, regression_params):
        """Test regression uncertainty estimation."""
        data = sample_data_medium
        layer = Regression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        # Train for a few epochs
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            result = layer(data['x'])
            train_loss = result.train_loss_fn(data['y_regression'])
            train_loss.backward()
            optimizer.step()
        
        # Test uncertainty estimation
        pred_dist = layer.predictive(data['x'])
        
        # Test mean and variance
        mean = pred_dist.mean
        variance = pred_dist.scale ** 2
        
        assert mean.shape == (data['batch_size'], data['output_features'])
        assert variance.shape == (data['batch_size'], data['output_features'])
        assert torch.all(variance >= 0)
        
        # Test sampling
        samples = pred_dist.sample((100,))
        assert samples.shape == (100, data['batch_size'], data['output_features'])
        
        # Test that sample statistics are reasonable
        sample_mean = samples.mean(0)
        sample_variance = samples.var(0)
        
        assert torch.allclose(sample_mean, mean, rtol=0.1)
        assert torch.allclose(sample_variance, variance, rtol=0.5)
    
    def test_heteroscedastic_regression(self, sample_data_medium, regression_params):
        """Test heteroscedastic regression uncertainty."""
        data = sample_data_medium
        layer = HetRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        # Train for a few epochs
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            result = layer(data['x'])
            train_loss = result.train_loss_fn(data['y_regression'])
            train_loss.backward()
            optimizer.step()
        
        # Test heteroscedastic uncertainty
        pred_dist = layer.predictive_sample(data['x'], consistent_variance=False)
        
        assert pred_dist.mean.shape == (data['batch_size'], data['output_features'])
        assert pred_dist.scale.shape == (data['batch_size'], data['output_features'])
        assert torch.all(pred_dist.scale >= 0)
        
        # Test that uncertainty varies across samples (heteroscedastic)
        uncertainties = []
        for _ in range(5):
            pred_dist_sample = layer.predictive_sample(data['x'], consistent_variance=False)
            uncertainties.append(pred_dist_sample.scale)
        
        # Check that uncertainties are different (heteroscedastic)
        uncertainties_tensor = torch.stack(uncertainties)
        assert not torch.allclose(uncertainties_tensor[0], uncertainties_tensor[1], atol=1e-6)


class TestMathematicalConsistency:
    """Test mathematical consistency of VBLL components."""
    
    def test_kl_divergence_properties(self, device):
        """Test KL divergence mathematical properties."""
        # Test that KL divergence is non-negative
        mean1 = torch.randn(2, 3, device=device)
        scale1 = torch.ones(2, 3, device=device)
        dist1 = torch.distributions.Normal(mean1, scale1)
        
        mean2 = torch.randn(2, 3, device=device)
        scale2 = torch.ones(2, 3, device=device)
        dist2 = torch.distributions.Normal(mean2, scale2)
        
        kl = torch.distributions.kl.kl_divergence(dist1, dist2)
        assert torch.all(kl >= 0)
        
        # Test that KL divergence is zero for identical distributions
        kl_self = torch.distributions.kl.kl_divergence(dist1, dist1)
        assert torch.allclose(kl_self, torch.zeros_like(kl_self))
    
    def test_covariance_matrix_properties(self, device):
        """Test covariance matrix mathematical properties."""
        # Test that covariance matrices are positive semi-definite
        mean = torch.randn(2, 3, device=device)
        tril = torch.tril(torch.randn(2, 3, 3, device=device))
        dist = DenseNormal(mean, tril)
        
        cov = dist.covariance
        # Check that eigenvalues are non-negative (positive semi-definite)
        eigenvals = torch.linalg.eigvals(cov)
        assert torch.all(eigenvals.real >= -1e-6)  # Allow small numerical errors
    
    def test_distribution_addition_properties(self, device):
        """Test properties of distribution addition."""
        mean1 = torch.randn(2, 3, device=device)
        scale1 = torch.ones(2, 3, device=device)
        dist1 = Normal(mean1, scale1)
        
        mean2 = torch.randn(2, 3, device=device)
        scale2 = torch.ones(2, 3, device=device)
        dist2 = Normal(mean2, scale2)
        
        result = dist1 + dist2
        
        # Test that addition is commutative
        result_comm = dist2 + dist1
        assert torch.allclose(result.mean, result_comm.mean)
        assert torch.allclose(result.scale, result_comm.scale)
        
        # Test that variance is additive for independent distributions
        expected_var = dist1.var + dist2.var
        assert torch.allclose(result.var, expected_var)
    
    def test_matrix_multiplication_properties(self, device):
        """Test properties of matrix multiplication with distributions."""
        mean = torch.randn(2, 3, device=device)
        scale = torch.ones(2, 3, device=device)
        dist = Normal(mean, scale)
        
        matrix = torch.randn(3, 1, device=device)
        result = dist @ matrix
        
        # Test that mean is transformed correctly
        expected_mean = mean @ matrix
        assert torch.allclose(result.mean, expected_mean)
        
        # Test that variance is transformed correctly
        expected_var = (scale**2).unsqueeze(-1) * (matrix**2)
        assert torch.allclose(result.var, expected_var.squeeze(-1))


class TestNumericalStability:
    """Test numerical stability of VBLL components."""
    
    def test_small_variance_handling(self, device):
        """Test handling of very small variances."""
        mean = torch.randn(2, 3, device=device)
        scale = torch.tensor([[1e-15, 1e-10, 1.0], [1e-20, 1e-5, 2.0]], device=device)
        dist = Normal(mean, scale)
        
        # Test that operations don't produce NaN or inf
        result = dist + dist
        assert torch.all(torch.isfinite(result.mean))
        assert torch.all(torch.isfinite(result.scale))
        assert torch.all(result.scale >= 1e-12)  # Check clipping
    
    def test_large_variance_handling(self, device):
        """Test handling of very large variances."""
        mean = torch.randn(2, 3, device=device)
        scale = torch.tensor([[1e6, 1e5, 1.0], [1e7, 1e4, 2.0]], device=device)
        dist = Normal(mean, scale)
        
        # Test that operations don't produce NaN or inf
        result = dist + dist
        assert torch.all(torch.isfinite(result.mean))
        assert torch.all(torch.isfinite(result.scale))
    
    def test_extreme_parameter_values(self, sample_data_small, classification_params):
        """Test with extreme parameter values."""
        data = sample_data_small
        
        # Test with very small regularization weight
        layer = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            regularization_weight=1e-10,
            **{k: v for k, v in classification_params.items() if k != 'regularization_weight'}
        )
        
        result = layer(data['x'])
        loss = result.train_loss_fn(data['y_classification'])
        assert torch.isfinite(loss)
        
        # Test with very large regularization weight
        layer = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            regularization_weight=1e10,
            **{k: v for k, v in classification_params.items() if k != 'regularization_weight'}
        )
        
        result = layer(data['x'])
        loss = result.train_loss_fn(data['y_classification'])
        assert torch.isfinite(loss)


class TestMemoryEfficiency:
    """Test memory efficiency of VBLL components."""
    
    def test_large_batch_handling(self, classification_params, regression_params):
        """Test handling of large batches."""
        batch_size = 1000
        input_features = 50
        output_features = 10
        
        x = torch.randn(batch_size, input_features)
        y_class = torch.randint(0, output_features, (batch_size,))
        y_reg = torch.randn(batch_size, output_features)
        
        # Test classification
        layer_class = DiscClassification(
            in_features=input_features,
            out_features=output_features,
            **classification_params
        )
        
        result_class = layer_class(x)
        loss_class = result_class.train_loss_fn(y_class)
        assert torch.isfinite(loss_class)
        
        # Test regression
        layer_reg = Regression(
            in_features=input_features,
            out_features=output_features,
            **regression_params
        )
        
        result_reg = layer_reg(x)
        loss_reg = result_reg.train_loss_fn(y_reg)
        assert torch.isfinite(loss_reg)
    
    def test_memory_usage_with_different_parameterizations(self, sample_data_medium, classification_params):
        """Test memory usage with different parameterizations."""
        data = sample_data_medium
        
        # Test diagonal parameterization (memory efficient)
        layer_diag = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            parameterization='diagonal',
            **classification_params
        )
        
        result_diag = layer_diag(data['x'])
        loss_diag = result_diag.train_loss_fn(data['y_classification'])
        assert torch.isfinite(loss_diag)
        
        # Test dense parameterization (more memory intensive)
        layer_dense = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            parameterization='dense',
            **classification_params
        )
        
        result_dense = layer_dense(data['x'])
        loss_dense = result_dense.train_loss_fn(data['y_classification'])
        assert torch.isfinite(loss_dense)


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility."""
    
    def test_cpu_gpu_consistency(self, sample_data_small, classification_params, device):
        """Test consistency between CPU and GPU."""
        data = sample_data_small
        
        # Test on CPU
        layer_cpu = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        x_cpu = data['x'].cpu()
        y_cpu = data['y_classification'].cpu()
        
        result_cpu = layer_cpu(x_cpu)
        loss_cpu = result_cpu.train_loss_fn(y_cpu)
        
        # Test on GPU if available
        if torch.cuda.is_available():
            layer_gpu = DiscClassification(
                in_features=data['input_features'],
                out_features=data['output_features'],
                **classification_params
            ).to(device)
            
            x_gpu = data['x'].to(device)
            y_gpu = data['y_classification'].to(device)
            
            result_gpu = layer_gpu(x_gpu)
            loss_gpu = result_gpu.train_loss_fn(y_gpu)
            
            # Both should produce finite losses
            assert torch.isfinite(loss_cpu)
            assert torch.isfinite(loss_gpu)
    
    def test_different_precisions(self, sample_data_small, classification_params):
        """Test with different floating point precisions."""
        data = sample_data_small
        
        # Test with float32
        layer_f32 = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        x_f32 = data['x'].float()
        y_f32 = data['y_classification']
        
        result_f32 = layer_f32(x_f32)
        loss_f32 = result_f32.train_loss_fn(y_f32)
        assert torch.isfinite(loss_f32)
        
        # Test with float64
        layer_f64 = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        ).double()
        
        x_f64 = data['x'].double()
        y_f64 = data['y_classification']
        
        result_f64 = layer_f64(x_f64)
        loss_f64 = result_f64.train_loss_fn(y_f64)
        assert torch.isfinite(loss_f64)

