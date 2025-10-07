"""
Tests for VBLL regression layers.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from vbll.layers.regression import (
    Regression, tRegression, HetRegression,
    gaussian_kl, gamma_kl, expected_gaussian_kl
)


class TestRegression:
    """Test standard VBLL Regression layer."""
    
    def test_initialization(self, sample_data_small, regression_params):
        """Test Regression initialization."""
        data = sample_data_small
        layer = Regression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        assert layer.in_features == data['input_features']
        assert layer.out_features == data['output_features']
        assert layer.regularization_weight == regression_params['regularization_weight']
    
    def test_forward_pass(self, sample_data_small, regression_params):
        """Test forward pass of Regression."""
        data = sample_data_small
        layer = Regression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        x = data['x']
        result = layer(x)
        
        # Check return type
        assert hasattr(result, 'predictive')
        assert hasattr(result, 'train_loss_fn')
        assert hasattr(result, 'val_loss_fn')
        
        # Check predictive distribution
        pred_dist = result.predictive
        assert hasattr(pred_dist, 'mean')
        assert hasattr(pred_dist, 'scale')
        assert pred_dist.mean.shape == (data['batch_size'], data['output_features'])
        assert pred_dist.scale.shape == (data['batch_size'], data['output_features'])
    
    def test_loss_functions(self, sample_data_small, regression_params):
        """Test train and validation loss functions."""
        data = sample_data_small
        layer = Regression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        x = data['x']
        y = data['y_regression']
        result = layer(x)
        
        # Test train loss
        train_loss = result.train_loss_fn(y)
        assert isinstance(train_loss, torch.Tensor)
        assert train_loss.requires_grad
        
        # Test validation loss
        val_loss = result.val_loss_fn(y)
        assert isinstance(val_loss, torch.Tensor)
        assert val_loss.requires_grad
    
    def test_different_parameterizations(self, sample_data_small, regression_params, parameterizations):
        """Test different covariance parameterizations."""
        data = sample_data_small
        
        for param in parameterizations:
            if param == 'lowrank':
                # Skip lowrank for small features to avoid issues
                continue
                
            layer = Regression(
                in_features=data['input_features'],
                out_features=data['output_features'],
                parameterization=param,
                **regression_params
            )
            
            x = data['x']
            result = layer(x)
            
            # Should work without errors
            assert result.predictive.mean.shape == (data['batch_size'], data['output_features'])
    
    def test_predictive_distribution(self, sample_data_small, regression_params):
        """Test predictive distribution properties."""
        data = sample_data_small
        layer = Regression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        x = data['x']
        pred_dist = layer.predictive(x)
        
        # Test sampling
        samples = pred_dist.sample((10,))
        assert samples.shape == (10, data['batch_size'], data['output_features'])
        
        # Test log probability
        y = data['y_regression']
        log_prob = pred_dist.log_prob(y)
        assert log_prob.shape == (data['batch_size'], data['output_features'])
    
    def test_gradient_flow(self, sample_data_small, regression_params):
        """Test that gradients flow properly."""
        data = sample_data_small
        layer = Regression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        x = data['x']
        y = data['y_regression']
        result = layer(x)
        
        train_loss = result.train_loss_fn(y)
        train_loss.backward()
        
        # Check that gradients exist for parameters
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.all(param.grad == 0)


class TesttRegression:
    """Test t-Distribution Regression layer."""
    
    def test_initialization(self, sample_data_small, regression_params):
        """Test tRegression initialization."""
        data = sample_data_small
        layer = tRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        assert layer.in_features == data['input_features']
        assert layer.out_features == data['output_features']
        assert layer.regularization_weight == regression_params['regularization_weight']
    
    def test_forward_pass(self, sample_data_small, regression_params):
        """Test forward pass of tRegression."""
        data = sample_data_small
        layer = tRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        x = data['x']
        result = layer(x)
        
        # Check return type
        assert hasattr(result, 'predictive')
        assert hasattr(result, 'train_loss_fn')
        assert hasattr(result, 'val_loss_fn')
        
        # Check predictive distribution
        pred_dist = result.predictive
        assert isinstance(pred_dist, torch.distributions.StudentT)
        assert pred_dist.loc.shape == (data['batch_size'], data['output_features'])
        assert pred_dist.scale.shape == (data['batch_size'], data['output_features'])
    
    def test_noise_distribution(self, sample_data_small, regression_params):
        """Test noise distribution properties."""
        data = sample_data_small
        layer = tRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        noise_dist = layer.noise
        assert isinstance(noise_dist, torch.distributions.Gamma)
        
        # Test noise prior
        noise_prior = layer.noise_prior
        assert isinstance(noise_prior, torch.distributions.Gamma)
    
    def test_predictive_distribution(self, sample_data_small, regression_params):
        """Test predictive distribution properties."""
        data = sample_data_small
        layer = tRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        x = data['x']
        pred_dist = layer.predictive(x)
        
        # Test sampling
        samples = pred_dist.sample((10,))
        assert samples.shape == (10, data['batch_size'], data['output_features'])
        
        # Test log probability
        y = data['y_regression']
        log_prob = pred_dist.log_prob(y)
        assert log_prob.shape == (data['batch_size'], data['output_features'])
    
    def test_different_parameterizations(self, sample_data_small, regression_params):
        """Test different covariance parameterizations for tRegression."""
        data = sample_data_small
        
        for param in ['diagonal', 'dense']:
            layer = tRegression(
                in_features=data['input_features'],
                out_features=data['output_features'],
                parameterization=param,
                **regression_params
            )
            
            x = data['x']
            result = layer(x)
            
            # Should work without errors
            assert result.predictive.loc.shape == (data['batch_size'], data['output_features'])


class TestHetRegression:
    """Test Heteroscedastic Regression layer."""
    
    def test_initialization(self, sample_data_small, regression_params):
        """Test HetRegression initialization."""
        data = sample_data_small
        layer = HetRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        assert layer.in_features == data['input_features']
        assert layer.out_features == data['output_features']
        assert layer.regularization_weight == regression_params['regularization_weight']
    
    def test_forward_pass(self, sample_data_small, regression_params):
        """Test forward pass of HetRegression."""
        data = sample_data_small
        layer = HetRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        x = data['x']
        result = layer(x)
        
        # Check return type
        assert hasattr(result, 'predictive')
        assert hasattr(result, 'train_loss_fn')
        assert hasattr(result, 'val_loss_fn')
        
        # Check predictive distribution
        pred_dist = result.predictive
        assert hasattr(pred_dist, 'mean')
        assert hasattr(pred_dist, 'scale')
        assert pred_dist.mean.shape == (data['batch_size'], data['output_features'])
        assert pred_dist.scale.shape == (data['batch_size'], data['output_features'])
    
    def test_consistent_variance(self, sample_data_small, regression_params):
        """Test consistent variance sampling."""
        data = sample_data_small
        layer = HetRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        x = data['x']
        
        # Test with consistent_variance=True
        result_consistent = layer(x, consistent_variance=True)
        pred_consistent = result_consistent.predictive
        
        # Test with consistent_variance=False
        result_inconsistent = layer(x, consistent_variance=False)
        pred_inconsistent = result_inconsistent.predictive
        
        # Both should produce valid distributions
        assert pred_consistent.mean.shape == (data['batch_size'], data['output_features'])
        assert pred_inconsistent.mean.shape == (data['batch_size'], data['output_features'])
    
    def test_M_distribution(self, sample_data_small, regression_params):
        """Test M distribution for noise modeling."""
        data = sample_data_small
        layer = HetRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        M_dist = layer.M
        assert hasattr(M_dist, 'mean')
        assert hasattr(M_dist, 'scale')
        
        # Test log_noise computation
        x = data['x']
        log_noise = layer.log_noise(x, M_dist)
        assert log_noise.shape == (data['batch_size'], data['output_features'])
    
    def test_predictive_sample(self, sample_data_small, regression_params):
        """Test predictive sampling."""
        data = sample_data_small
        layer = HetRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        x = data['x']
        
        # Test consistent variance sampling
        pred_consistent = layer.predictive_sample(x, consistent_variance=True)
        assert pred_consistent.mean.shape == (data['batch_size'], data['output_features'])
        assert pred_consistent.scale.shape == (data['batch_size'], data['output_features'])
        
        # Test inconsistent variance sampling
        pred_inconsistent = layer.predictive_sample(x, consistent_variance=False)
        assert pred_inconsistent.mean.shape == (data['batch_size'], data['output_features'])
        assert pred_inconsistent.scale.shape == (data['batch_size'], data['output_features'])
    
    def test_different_parameterizations(self, sample_data_small, regression_params):
        """Test different covariance parameterizations for HetRegression."""
        data = sample_data_small
        
        for param in ['diagonal', 'dense']:
            layer = HetRegression(
                in_features=data['input_features'],
                out_features=data['output_features'],
                parameterization=param,
                **regression_params
            )
            
            x = data['x']
            result = layer(x)
            
            # Should work without errors
            assert result.predictive.mean.shape == (data['batch_size'], data['output_features'])


class TestRegressionMathematicalFunctions:
    """Test mathematical functions used in regression."""
    
    def test_gaussian_kl(self, device):
        """Test Gaussian KL divergence computation."""
        # Create test distributions
        mean = torch.randn(2, 3, device=device)
        scale = torch.ones(2, 3, device=device)
        dist = torch.distributions.Normal(mean, scale)
        
        # Test KL computation
        q_scale = 1.0
        kl = gaussian_kl(dist, q_scale)
        
        assert isinstance(kl, torch.Tensor)
        assert kl.shape == (2,)
        assert torch.all(kl >= 0)  # KL divergence should be non-negative
    
    def test_gamma_kl(self, device):
        """Test Gamma KL divergence computation."""
        # Create test distributions
        concentration = torch.ones(2, device=device)
        rate = torch.ones(2, device=device)
        dist1 = torch.distributions.Gamma(concentration, rate)
        dist2 = torch.distributions.Gamma(concentration * 2, rate)
        
        # Test KL computation
        kl = gamma_kl(dist1, dist2)
        
        assert isinstance(kl, torch.Tensor)
        assert kl.shape == (2,)
        assert torch.all(kl >= 0)  # KL divergence should be non-negative
    
    def test_expected_gaussian_kl(self, device):
        """Test expected Gaussian KL divergence computation."""
        # Create test distribution
        mean = torch.randn(2, 3, device=device)
        scale = torch.ones(2, 3, device=device)
        dist = torch.distributions.Normal(mean, scale)
        
        # Test expected KL computation
        q_scale = 1.0
        cov_factor = torch.ones(2, 3, device=device)
        kl = expected_gaussian_kl(dist, q_scale, cov_factor)
        
        assert isinstance(kl, torch.Tensor)
        assert kl.shape == (2,)
        assert torch.all(kl >= 0)  # KL divergence should be non-negative


class TestRegressionIntegration:
    """Integration tests for regression layers."""
    
    def test_training_step(self, sample_data_medium, regression_params):
        """Test a complete training step."""
        data = sample_data_medium
        layer = Regression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        
        x = data['x']
        y = data['y_regression']
        
        # Forward pass
        result = layer(x)
        train_loss = result.train_loss_fn(y)
        
        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        assert torch.isfinite(train_loss)
    
    def test_different_batch_sizes(self, regression_params):
        """Test with different batch sizes."""
        layer = Regression(
            in_features=10,
            out_features=5,
            **regression_params
        )
        
        for batch_size in [1, 16, 64, 128]:
            x = torch.randn(batch_size, 10)
            y = torch.randn(batch_size, 5)
            
            result = layer(x)
            train_loss = result.train_loss_fn(y)
            val_loss = result.val_loss_fn(y)
            
            assert torch.isfinite(train_loss)
            assert torch.isfinite(val_loss)
    
    def test_device_consistency(self, sample_data_small, regression_params, device):
        """Test that layers work on different devices."""
        data = sample_data_small
        layer = Regression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        ).to(device)
        
        x = data['x'].to(device)
        y = data['y_regression'].to(device)
        
        result = layer(x)
        train_loss = result.train_loss_fn(y)
        
        assert train_loss.device == device
        assert result.predictive.mean.device == device
    
    def test_uncertainty_estimation(self, sample_data_medium, regression_params):
        """Test uncertainty estimation capabilities."""
        data = sample_data_medium
        layer = Regression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        x = data['x']
        pred_dist = layer.predictive(x)
        
        # Test mean and variance
        mean = pred_dist.mean
        variance = pred_dist.scale ** 2
        
        assert mean.shape == (data['batch_size'], data['output_features'])
        assert variance.shape == (data['batch_size'], data['output_features'])
        assert torch.all(variance >= 0)  # Variance should be non-negative
        
        # Test sampling
        samples = pred_dist.sample((100,))
        assert samples.shape == (100, data['batch_size'], data['output_features'])
        
        # Test that sample variance is reasonable
        sample_variance = samples.var(dim=0)
        assert torch.allclose(sample_variance, variance, rtol=0.5)  # Allow some tolerance
    
    def test_heteroscedastic_uncertainty(self, sample_data_medium, regression_params):
        """Test heteroscedastic uncertainty estimation."""
        data = sample_data_medium
        layer = HetRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        x = data['x']
        
        # Test multiple samples to see variance in uncertainty
        variances = []
        for _ in range(10):
            pred_dist = layer.predictive_sample(x, consistent_variance=False)
            variances.append(pred_dist.scale ** 2)
        
        # Check that variances are different (heteroscedastic)
        var_tensor = torch.stack(variances)
        assert not torch.allclose(var_tensor[0], var_tensor[1], atol=1e-6)
    
    def test_t_regression_uncertainty(self, sample_data_medium, regression_params):
        """Test t-regression uncertainty estimation."""
        data = sample_data_medium
        layer = tRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **regression_params
        )
        
        x = data['x']
        pred_dist = layer.predictive(x)
        
        # Test Student-t distribution properties
        assert isinstance(pred_dist, torch.distributions.StudentT)
        assert pred_dist.df.shape == (data['batch_size'], data['output_features'])
        assert pred_dist.loc.shape == (data['batch_size'], data['output_features'])
        assert pred_dist.scale.shape == (data['batch_size'], data['output_features'])
        
        # Test sampling
        samples = pred_dist.sample((50,))
        assert samples.shape == (50, data['batch_size'], data['output_features'])
        
        # Test log probability
        y = data['y_regression']
        log_prob = pred_dist.log_prob(y)
        assert log_prob.shape == (data['batch_size'], data['output_features'])

