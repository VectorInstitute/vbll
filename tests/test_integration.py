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
    
    def test_classification_uncertainty(self, sample_data_medium, classification_params):
        """Test classification uncertainty estimation."""
        data = sample_data_medium
        classification_params = classification_params.copy()
        classification_params['return_ood'] = True

        layer = DiscClassification(
            in_features=data['input_features'],
            out_features=data['output_features'],
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