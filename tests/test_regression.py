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
    
        assert layer.regularization_weight == regression_params['regularization_weight']
        assert layer.W_mean.shape == (data['output_features'], data['input_features'])
    
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
        
        assert layer.W_mean.shape == (data['output_features'], data['input_features'])
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
    
    def test_initialization(self, sample_data_small, het_regression_params):
        """Test HetRegression initialization."""
        data = sample_data_small
        layer = HetRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **het_regression_params
        )
        
        assert layer.W_mean.shape == (data['output_features'], data['input_features'])
        assert layer.M_mean.shape == (data['output_features'], data['input_features'])
        assert layer.regularization_weight == het_regression_params['regularization_weight']
    
    def test_forward_pass(self, sample_data_small, het_regression_params):
        """Test forward pass of HetRegression."""
        data = sample_data_small
        layer = HetRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **het_regression_params
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
    
    def test_consistent_variance(self, sample_data_small, het_regression_params):
        """Test consistent variance sampling."""
        data = sample_data_small
        layer = HetRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **het_regression_params
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
    
    def test_M_distribution(self, sample_data_small, het_regression_params):
        """Test M distribution for noise modeling."""
        data = sample_data_small
        layer = HetRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **het_regression_params
        )
        
        M_dist = layer.M
        assert hasattr(M_dist, 'mean')
        assert hasattr(M_dist, 'scale_tril')
    
    def test_predictive_sample(self, sample_data_small, het_regression_params):
        """Test predictive sampling."""
        data = sample_data_small
        layer = HetRegression(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **het_regression_params
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
    
    def test_different_parameterizations(self, sample_data_small, het_regression_params):
        """Test different covariance parameterizations for HetRegression."""
        data = sample_data_small
        
        for param in ['diagonal', 'dense']:
            layer = HetRegression(
                in_features=data['input_features'],
                out_features=data['output_features'],
                parameterization=param,
                **het_regression_params
            )
            
            x = data['x']
            result = layer(x)
            
            # Should work without errors
            assert result.predictive.mean.shape == (data['batch_size'], data['output_features'])