"""
Benchmark tests for VBLL performance.
"""
import pytest
import torch
import time
import numpy as np
from vbll.layers.classification import DiscClassification
from vbll.layers.regression import Regression, HetRegression
from vbll.utils.distributions import Normal, DenseNormal


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_classification_speed(self, device):
        """Benchmark classification layer speed."""
        batch_size = 256
        input_features = 100
        output_features = 10
        
        x = torch.randn(batch_size, input_features, device=device)
        y = torch.randint(0, output_features, (batch_size,), device=device)
        
        layer = DiscClassification(
            in_features=input_features,
            out_features=output_features,
            regularization_weight=0.1
        ).to(device)
        
        # Warm up
        for _ in range(10):
            result = layer(x)
            loss = result.train_loss_fn(y)
            loss.backward()
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(100):
            result = layer(x)
            loss = result.train_loss_fn(y)
            loss.backward()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        print(f"Classification forward+backward time: {avg_time:.4f}s")
        
        # Should be reasonably fast
        assert avg_time < 0.1  # Less than 100ms per iteration
    
    def test_regression_speed(self, device):
        """Benchmark regression layer speed."""
        batch_size = 256
        input_features = 100
        output_features = 5
        
        x = torch.randn(batch_size, input_features, device=device)
        y = torch.randn(batch_size, output_features, device=device)
        
        layer = Regression(
            in_features=input_features,
            out_features=output_features,
            regularization_weight=0.1
        ).to(device)
        
        # Warm up
        for _ in range(10):
            result = layer(x)
            loss = result.train_loss_fn(y)
            loss.backward()
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(100):
            result = layer(x)
            loss = result.train_loss_fn(y)
            loss.backward()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        print(f"Regression forward+backward time: {avg_time:.4f}s")
        
        # Should be reasonably fast
        assert avg_time < 0.1  # Less than 100ms per iteration
    
    def test_heteroscedastic_speed(self, device):
        """Benchmark heteroscedastic regression speed."""
        batch_size = 256
        input_features = 100
        output_features = 5
        
        x = torch.randn(batch_size, input_features, device=device)
        y = torch.randn(batch_size, output_features, device=device)
        
        layer = HetRegression(
            in_features=input_features,
            out_features=output_features,
            regularization_weight=0.1
        ).to(device)
        
        # Warm up
        for _ in range(10):
            result = layer(x)
            loss = result.train_loss_fn(y)
            loss.backward()
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(100):
            result = layer(x)
            loss = result.train_loss_fn(y)
            loss.backward()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        print(f"Heteroscedastic regression forward+backward time: {avg_time:.4f}s")
        
        # Should be reasonably fast
        assert avg_time < 0.2  # Allow more time for heteroscedastic
    
    def test_distribution_operations_speed(self, device):
        """Benchmark distribution operations speed."""
        batch_size = 1000
        dim = 50
        
        mean = torch.randn(batch_size, dim, device=device)
        scale = torch.ones(batch_size, dim, device=device)
        dist = Normal(mean, scale)
        
        matrix = torch.randn(dim, 1, device=device)
        
        # Warm up
        for _ in range(10):
            result = dist @ matrix
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(1000):
            result = dist @ matrix
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 1000
        print(f"Distribution matrix multiplication time: {avg_time:.6f}s")
        
        # Should be very fast
        assert avg_time < 0.001  # Less than 1ms per operation
    
    def test_memory_usage(self, device):
        """Test memory usage with large tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory test")
        
        batch_size = 1000
        input_features = 200
        output_features = 50
        
        x = torch.randn(batch_size, input_features, device=device)
        y = torch.randint(0, output_features, (batch_size,), device=device)
        
        layer = DiscClassification(
            in_features=input_features,
            out_features=output_features,
            parameterization='dense',  # Most memory intensive
            regularization_weight=0.1
        ).to(device)
        
        # Check memory before
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated()
        
        # Forward pass
        result = layer(x)
        loss = result.train_loss_fn(y)
        loss.backward()
        
        # Check memory after
        memory_after = torch.cuda.memory_allocated()
        memory_used = memory_after - memory_before
        
        print(f"Memory used: {memory_used / 1024**2:.2f} MB")
        
        # Should not use excessive memory
        assert memory_used < 500 * 1024**2  # Less than 500MB


class TestScalabilityBenchmarks:
    """Scalability benchmark tests."""
    
    def test_batch_size_scalability(self, device):
        """Test scalability with different batch sizes."""
        input_features = 50
        output_features = 10
        
        batch_sizes = [32, 64, 128, 256, 512]
        times = []
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, input_features, device=device)
            y = torch.randint(0, output_features, (batch_size,), device=device)
            
            layer = DiscClassification(
                in_features=input_features,
                out_features=output_features,
                regularization_weight=0.1
            ).to(device)
            
            # Warm up
            for _ in range(5):
                result = layer(x)
                loss = result.train_loss_fn(y)
                loss.backward()
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(20):
                result = layer(x)
                loss = result.train_loss_fn(y)
                loss.backward()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 20
            times.append(avg_time)
            print(f"Batch size {batch_size}: {avg_time:.4f}s")
        
        # Check that time scales reasonably with batch size
        for i in range(1, len(times)):
            # Time should not increase more than linearly
            assert times[i] / times[i-1] < 2.0
    
    def test_feature_dimension_scalability(self, device):
        """Test scalability with different feature dimensions."""
        batch_size = 128
        output_features = 10
        
        feature_dims = [10, 25, 50, 100, 200]
        times = []
        
        for input_features in feature_dims:
            x = torch.randn(batch_size, input_features, device=device)
            y = torch.randint(0, output_features, (batch_size,), device=device)
            
            layer = DiscClassification(
                in_features=input_features,
                out_features=output_features,
                regularization_weight=0.1
            ).to(device)
            
            # Warm up
            for _ in range(5):
                result = layer(x)
                loss = result.train_loss_fn(y)
                loss.backward()
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(20):
                result = layer(x)
                loss = result.train_loss_fn(y)
                loss.backward()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 20
            times.append(avg_time)
            print(f"Feature dim {input_features}: {avg_time:.4f}s")
        
        # Check that time scales reasonably with feature dimension
        for i in range(1, len(times)):
            # Time should not increase more than quadratically
            assert times[i] / times[i-1] < 4.0
    
    def test_parameterization_comparison(self, device):
        """Compare performance of different parameterizations."""
        batch_size = 128
        input_features = 50
        output_features = 10
        
        parameterizations = ['diagonal', 'dense']
        times = {}
        
        for param in parameterizations:
            x = torch.randn(batch_size, input_features, device=device)
            y = torch.randint(0, output_features, (batch_size,), device=device)
            
            layer = DiscClassification(
                in_features=input_features,
                out_features=output_features,
                parameterization=param,
                regularization_weight=0.1
            ).to(device)
            
            # Warm up
            for _ in range(5):
                result = layer(x)
                loss = result.train_loss_fn(y)
                loss.backward()
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(20):
                result = layer(x)
                loss = result.train_loss_fn(y)
                loss.backward()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 20
            times[param] = avg_time
            print(f"Parameterization {param}: {avg_time:.4f}s")
        
        # Dense should be slower than diagonal
        assert times['dense'] > times['diagonal']


class TestNumericalStabilityBenchmarks:
    """Numerical stability benchmark tests."""
    
    def test_extreme_values_stability(self, device):
        """Test stability with extreme values."""
        batch_size = 100
        input_features = 20
        output_features = 5
        
        # Test with very small values
        x_small = torch.randn(batch_size, input_features, device=device) * 1e-10
        y_small = torch.randint(0, output_features, (batch_size,), device=device)
        
        layer = DiscClassification(
            in_features=input_features,
            out_features=output_features,
            regularization_weight=1e-10
        ).to(device)
        
        result = layer(x_small)
        loss = result.train_loss_fn(y_small)
        
        assert torch.isfinite(loss)
        assert torch.all(torch.isfinite(result.predictive.probs))
        
        # Test with very large values
        x_large = torch.randn(batch_size, input_features, device=device) * 1e10
        y_large = torch.randint(0, output_features, (batch_size,), device=device)
        
        layer = DiscClassification(
            in_features=input_features,
            out_features=output_features,
            regularization_weight=1e10
        ).to(device)
        
        result = layer(x_large)
        loss = result.train_loss_fn(y_large)
        
        assert torch.isfinite(loss)
        assert torch.all(torch.isfinite(result.predictive.probs))
    
    def test_gradient_stability(self, device):
        """Test gradient stability during training."""
        batch_size = 64
        input_features = 30
        output_features = 5
        
        x = torch.randn(batch_size, input_features, device=device)
        y = torch.randint(0, output_features, (batch_size,), device=device)
        
        layer = DiscClassification(
            in_features=input_features,
            out_features=output_features,
            regularization_weight=0.1
        ).to(device)
        
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        
        # Train for many iterations
        for epoch in range(100):
            optimizer.zero_grad()
            
            result = layer(x)
            loss = result.train_loss_fn(y)
            loss.backward()
            
            # Check gradient stability
            max_grad = max(p.grad.abs().max().item() for p in layer.parameters() if p.grad is not None)
            assert max_grad < 100.0  # Gradients should not explode
            
            optimizer.step()
            
            # Check loss stability
            assert torch.isfinite(loss)
    
    def test_precision_stability(self, device):
        """Test stability with different precisions."""
        batch_size = 32
        input_features = 20
        output_features = 5
        
        x = torch.randn(batch_size, input_features, device=device)
        y = torch.randint(0, output_features, (batch_size,), device=device)
        
        # Test with float32
        layer_f32 = DiscClassification(
            in_features=input_features,
            out_features=output_features,
            regularization_weight=0.1
        ).to(device)
        
        result_f32 = layer_f32(x.float())
        loss_f32 = result_f32.train_loss_fn(y)
        
        assert torch.isfinite(loss_f32)
        
        # Test with float64
        layer_f64 = DiscClassification(
            in_features=input_features,
            out_features=output_features,
            regularization_weight=0.1
        ).double().to(device)
        
        result_f64 = layer_f64(x.double())
        loss_f64 = result_f64.train_loss_fn(y)
        
        assert torch.isfinite(loss_f64)
        
        # Results should be similar
        assert abs(loss_f32.item() - loss_f64.item()) < 1.0

