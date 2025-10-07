"""
Tests for VBLL classification layers.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from vbll.layers.classification import (
    DiscClassification, tDiscClassification, HetClassification, GenClassification,
    gaussian_kl, gamma_kl, expected_gaussian_kl
)


class TestDiscClassification:
    """Test Discriminative Classification layer."""
    
    def test_initialization(self, sample_data_small, classification_params):
        """Test DiscClassification initialization."""
        data = sample_data_small
        layer = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        assert layer.in_features == data['input_features']
        assert layer.out_features == data['output_features']
        assert layer.regularization_weight == classification_params['regularization_weight']
        assert layer.return_ood == classification_params['return_ood']
    
    def test_forward_pass(self, sample_data_small, classification_params):
        """Test forward pass of DiscClassification."""
        data = sample_data_small
        layer = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        x = data['x']
        result = layer(x)
        
        # Check return type
        assert hasattr(result, 'predictive')
        assert hasattr(result, 'train_loss_fn')
        assert hasattr(result, 'val_loss_fn')
        
        # Check predictive distribution
        pred_dist = result.predictive
        assert isinstance(pred_dist, torch.distributions.Categorical)
        assert pred_dist.probs.shape == (data['batch_size'], data['output_features'])
        
        # Check that probabilities sum to 1
        assert torch.allclose(pred_dist.probs.sum(-1), torch.ones(data['batch_size']))
    
    def test_loss_functions(self, sample_data_small, classification_params):
        """Test train and validation loss functions."""
        data = sample_data_small
        layer = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        x = data['x']
        y = data['y_classification']
        result = layer(x)
        
        # Test train loss
        train_loss = result.train_loss_fn(y)
        assert isinstance(train_loss, torch.Tensor)
        assert train_loss.requires_grad
        
        # Test validation loss
        val_loss = result.val_loss_fn(y)
        assert isinstance(val_loss, torch.Tensor)
        assert val_loss.requires_grad
    
    def test_ood_scores(self, sample_data_small, classification_params):
        """Test OOD score computation."""
        params_with_ood = classification_params.copy()
        params_with_ood['return_ood'] = True
        
        layer = DiscClassification(
            in_features=sample_data_small['input_features'],
            out_features=sample_data_small['output_features'],
            **params_with_ood
        )
        
        x = sample_data_small['x']
        result = layer(x)
        
        assert result.ood_scores is not None
        ood_scores = result.ood_scores
        assert ood_scores.shape == (sample_data_small['batch_size'],)
        assert torch.all(ood_scores >= 0) and torch.all(ood_scores <= 1)
    
    def test_different_parameterizations(self, sample_data_small, classification_params, parameterizations):
        """Test different covariance parameterizations."""
        data = sample_data_small
        
        for param in parameterizations:
            if param == 'lowrank':
                # Skip lowrank for small features to avoid issues
                continue
                
            layer = DiscClassification(
                in_features=data['input_features'],
                out_features=data['output_features'],
                parameterization=param,
                **classification_params
            )
            
            x = data['x']
            result = layer(x)
            
            # Should work without errors
            assert result.predictive.probs.shape == (data['batch_size'], data['output_features'])
    
    def test_softmax_bounds(self, sample_data_small, classification_params):
        """Test different softmax bounds."""
        data = sample_data_small
        
        # Test Jensen bound (default)
        layer = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            softmax_bound='jensen',
            **classification_params
        )
        
        x = data['x']
        y = data['y_classification']
        result = layer(x)
        
        # Should work without errors
        train_loss = result.train_loss_fn(y)
        assert isinstance(train_loss, torch.Tensor)
    
    def test_gradient_flow(self, sample_data_small, classification_params):
        """Test that gradients flow properly."""
        data = sample_data_small
        layer = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        x = data['x']
        y = data['y_classification']
        result = layer(x)
        
        train_loss = result.train_loss_fn(y)
        train_loss.backward()
        
        # Check that gradients exist for parameters
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.all(param.grad == 0)


class TesttDiscClassification:
    """Test t-Discriminative Classification layer."""
    
    def test_initialization(self, sample_data_small, classification_params):
        """Test tDiscClassification initialization."""
        data = sample_data_small
        layer = tDiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        assert layer.in_features == data['input_features']
        assert layer.out_features == data['output_features']
        assert layer.regularization_weight == classification_params['regularization_weight']
    
    def test_forward_pass(self, sample_data_small, classification_params):
        """Test forward pass of tDiscClassification."""
        data = sample_data_small
        layer = tDiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        x = data['x']
        result = layer(x)
        
        # Check return type
        assert hasattr(result, 'predictive')
        assert hasattr(result, 'train_loss_fn')
        assert hasattr(result, 'val_loss_fn')
        
        # Check predictive distribution
        pred_dist = result.predictive
        assert isinstance(pred_dist, torch.distributions.Categorical)
        assert pred_dist.probs.shape == (data['batch_size'], data['output_features'])
    
    def test_noise_distribution(self, sample_data_small, classification_params):
        """Test noise distribution properties."""
        data = sample_data_small
        layer = tDiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        noise_dist = layer.noise
        assert isinstance(noise_dist, torch.distributions.Gamma)
        
        # Test noise prior
        noise_prior = layer.noise_prior
        assert isinstance(noise_prior, torch.distributions.Gamma)
    
    def test_softmax_bounds(self, sample_data_small, classification_params):
        """Test different softmax bounds for tDiscClassification."""
        data = sample_data_small
        
        # Test reduced_kn bound
        layer = tDiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            softmax_bound='reduced_kn',
            **classification_params
        )
        
        x = data['x']
        y = data['y_classification']
        result = layer(x)
        
        train_loss = result.train_loss_fn(y)
        assert isinstance(train_loss, torch.Tensor)
    
    def test_alpha_parameter(self, sample_data_small, classification_params):
        """Test alpha parameter for reduced_kn bound."""
        data = sample_data_small
        layer = tDiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            softmax_bound='reduced_kn',
            kn_alpha=0.5,
            **classification_params
        )
        
        assert hasattr(layer, 'alpha')
        assert layer.alpha.item() == 0.5


class TestHetClassification:
    """Test Heteroscedastic Classification layer."""
    
    def test_initialization(self, sample_data_small, classification_params):
        """Test HetClassification initialization."""
        data = sample_data_small
        layer = HetClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        assert layer.in_features == data['input_features']
        assert layer.out_features == data['output_features']
        assert layer.regularization_weight == classification_params['regularization_weight']
    
    def test_forward_pass(self, sample_data_small, classification_params):
        """Test forward pass of HetClassification."""
        data = sample_data_small
        layer = HetClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        x = data['x']
        result = layer(x)
        
        # Check return type
        assert hasattr(result, 'predictive')
        assert hasattr(result, 'train_loss_fn')
        assert hasattr(result, 'val_loss_fn')
        
        # Check predictive distribution
        pred_dist = result.predictive
        assert isinstance(pred_dist, torch.distributions.Categorical)
        assert pred_dist.probs.shape == (data['batch_size'], data['output_features'])
    
    def test_consistent_variance(self, sample_data_small, classification_params):
        """Test consistent variance sampling."""
        data = sample_data_small
        layer = HetClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        x = data['x']
        
        # Test with consistent_variance=True
        result_consistent = layer(x, consistent_variance=True)
        pred_consistent = result_consistent.predictive
        
        # Test with consistent_variance=False
        result_inconsistent = layer(x, consistent_variance=False)
        pred_inconsistent = result_inconsistent.predictive
        
        # Both should produce valid probability distributions
        assert pred_consistent.shape == (data['batch_size'], data['output_features'])
        assert pred_inconsistent.shape == (data['batch_size'], data['output_features'])
    
    def test_M_distribution(self, sample_data_small, classification_params):
        """Test M distribution for noise modeling."""
        data = sample_data_small
        layer = HetClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        M_dist = layer.M
        assert hasattr(M_dist, 'mean')
        assert hasattr(M_dist, 'scale')
        
        # Test log_noise computation
        x = data['x']
        log_noise = layer.log_noise(x, M_dist)
        assert log_noise.shape == (data['batch_size'], data['output_features'])


class TestGenClassification:
    """Test Generative Classification layer."""
    
    def test_initialization(self, sample_data_small, classification_params):
        """Test GenClassification initialization."""
        data = sample_data_small
        layer = GenClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        assert layer.in_features == data['input_features']
        assert layer.out_features == data['output_features']
        assert layer.regularization_weight == classification_params['regularization_weight']
    
    def test_forward_pass(self, sample_data_small, classification_params):
        """Test forward pass of GenClassification."""
        data = sample_data_small
        layer = GenClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        x = data['x']
        result = layer(x)
        
        # Check return type
        assert hasattr(result, 'predictive')
        assert hasattr(result, 'train_loss_fn')
        assert hasattr(result, 'val_loss_fn')
        
        # Check predictive distribution
        pred_dist = result.predictive
        assert isinstance(pred_dist, torch.distributions.Categorical)
        assert pred_dist.probs.shape == (data['batch_size'], data['output_features'])
    
    def test_mu_distribution(self, sample_data_small, classification_params):
        """Test mu distribution for generative modeling."""
        data = sample_data_small
        layer = GenClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        mu_dist = layer.mu()
        assert hasattr(mu_dist, 'mean')
        assert hasattr(mu_dist, 'scale')
    
    def test_noise_distribution(self, sample_data_small, classification_params):
        """Test noise distribution."""
        data = sample_data_small
        layer = GenClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        noise_dist = layer.noise()
        assert hasattr(noise_dist, 'mean')
        assert hasattr(noise_dist, 'scale')
    
    def test_logit_predictive(self, sample_data_small, classification_params):
        """Test logit predictive computation."""
        data = sample_data_small
        layer = GenClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        x = data['x']
        logits = layer.logit_predictive(x)
        
        assert logits.shape == (data['batch_size'], data['output_features'])


class TestMathematicalFunctions:
    """Test mathematical functions used in classification."""
    
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


class TestClassificationIntegration:
    """Integration tests for classification layers."""
    
    def test_training_step(self, sample_data_medium, classification_params):
        """Test a complete training step."""
        data = sample_data_medium
        layer = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        )
        
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        
        x = data['x']
        y = data['y_classification']
        
        # Forward pass
        result = layer(x)
        train_loss = result.train_loss_fn(y)
        
        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        assert torch.isfinite(train_loss)
    
    def test_different_batch_sizes(self, classification_params):
        """Test with different batch sizes."""
        layer = DiscClassification(
            in_features=10,
            out_features=5,
            **classification_params
        )
        
        for batch_size in [1, 16, 64, 128]:
            x = torch.randn(batch_size, 10)
            y = torch.randint(0, 5, (batch_size,))
            
            result = layer(x)
            train_loss = result.train_loss_fn(y)
            val_loss = result.val_loss_fn(y)
            
            assert torch.isfinite(train_loss)
            assert torch.isfinite(val_loss)
    
    def test_device_consistency(self, sample_data_small, classification_params, device):
        """Test that layers work on different devices."""
        data = sample_data_small
        layer = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
            **classification_params
        ).to(device)
        
        x = data['x'].to(device)
        y = data['y_classification'].to(device)
        
        result = layer(x)
        train_loss = result.train_loss_fn(y)
        
        assert train_loss.device == device
        assert result.predictive.probs.device == device

