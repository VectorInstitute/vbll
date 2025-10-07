"""
Pytest configuration and shared fixtures for VBLL tests.
"""
import pytest
import torch
import numpy as np
from typing import Tuple, Dict, Any


@pytest.fixture
def device():
    """Get the device for testing (CPU or GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def small_batch():
    """Small batch size for quick tests."""
    return 32


@pytest.fixture
def medium_batch():
    """Medium batch size for more comprehensive tests."""
    return 128


@pytest.fixture
def small_features():
    """Small feature dimension for quick tests."""
    return 10


@pytest.fixture
def medium_features():
    """Medium feature dimension for more comprehensive tests."""
    return 50


@pytest.fixture
def small_outputs():
    """Small output dimension for quick tests."""
    return 5


@pytest.fixture
def medium_outputs():
    """Medium output dimension for more comprehensive tests."""
    return 20


@pytest.fixture
def sample_data_small(small_batch, small_features, small_outputs, device):
    """Generate small sample data for testing."""
    x = torch.randn(small_batch, small_features, device=device)
    y_class = torch.randint(0, small_outputs, (small_batch,), device=device)
    y_reg = torch.randn(small_batch, small_outputs, device=device)
    return {
        'x': x,
        'y_classification': y_class,
        'y_regression': y_reg,
        'batch_size': small_batch,
        'input_features': small_features,
        'output_features': small_outputs
    }


@pytest.fixture
def sample_data_medium(medium_batch, medium_features, medium_outputs, device):
    """Generate medium sample data for testing."""
    x = torch.randn(medium_batch, medium_features, device=device)
    y_class = torch.randint(0, medium_outputs, (medium_batch,), device=device)
    y_reg = torch.randn(medium_batch, medium_outputs, device=device)
    return {
        'x': x,
        'y_classification': y_class,
        'y_regression': y_reg,
        'batch_size': medium_batch,
        'input_features': medium_features,
        'output_features': medium_outputs
    }


@pytest.fixture
def classification_params():
    """Common parameters for classification layers."""
    return {
        'regularization_weight': 0.1,
        'prior_scale': 1.0,
        'wishart_scale': 1.0,
        'dof': 2.0,
        'return_ood': False
    }

@pytest.fixture
def het_classification_params():
    """Common parameters for hetclassification layers."""
    return {
        'regularization_weight': 0.1,
        'prior_scale': 1.0,
        'noise_prior_scale': 2.0,
        'return_ood': False
    }


@pytest.fixture
def regression_params():
    """Common parameters for regression layers."""
    return {
        'regularization_weight': 0.1,
        'prior_scale': 1.0,
        'wishart_scale': 1e-2,
        'dof': 1.0
    }

@pytest.fixture
def het_regression_params():
    """Common parameters for heteroscedastic regression layers."""
    return {
        'regularization_weight': 0.1,
        'prior_scale': 1.0,
        'noise_prior_scale': 1e-2,
    }

@pytest.fixture
def parameterizations():
    """Available covariance parameterizations."""
    return ['diagonal', 'dense', 'lowrank']


@pytest.fixture
def softmax_bounds():
    """Available softmax bounds for classification."""
    return ['jensen', 'semimontecarlo', 'reduced_kn']
