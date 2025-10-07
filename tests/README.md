# VBLL Test Suite

This directory contains comprehensive tests for the VBLL (Variational Bayesian Last Layers) library.

## Test Structure

### Core Test Files

- **`test_distributions.py`** - Tests for distribution classes (`Normal`, `DenseNormal`, `DenseNormalPrec`, `LowRankNormal`)
- **`test_classification.py`** - Tests for classification layers (`DiscClassification`, `tDiscClassification`, `HetClassification`, `GenClassification`)
- **`test_regression.py`** - Tests for regression layers (`Regression`, `tRegression`, `HetRegression`)
- **`test_jax.py`** - Tests for JAX implementation
- **`test_integration.py`** - End-to-end integration tests
- **`test_utils.py`** - Tests for utility functions and edge cases
- **`test_benchmarks.py`** - Performance and scalability benchmarks

### Configuration Files

- **`conftest.py`** - Pytest fixtures and configuration
- **`__init__.py`** - Test package initialization

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_distributions.py

# Run specific test class
pytest tests/test_classification.py::TestDiscClassification

# Run specific test function
pytest tests/test_distributions.py::TestNormal::test_initialization
```

### Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only benchmark tests
pytest -m benchmark

# Skip slow tests
pytest -m "not slow"

# Skip GPU tests (if no GPU available)
pytest -m "not gpu"

# Skip JAX tests (if JAX not installed)
pytest -m "not jax"
```

### Performance Testing

```bash
# Run benchmark tests with timing
pytest tests/test_benchmarks.py -v --durations=0

# Run with performance profiling
pytest tests/test_benchmarks.py --profile
```

## Test Coverage

The test suite covers:

### Mathematical Components
- Distribution classes and their operations
- KL divergence computations
- Covariance matrix operations
- Matrix multiplication with distributions

### Classification Layers
- Discriminative classification (`DiscClassification`)
- t-distribution classification (`tDiscClassification`)
- Heteroscedastic classification (`HetClassification`)
- Generative classification (`GenClassification`)

### Regression Layers
- Standard regression (`Regression`)
- t-distribution regression (`tRegression`)
- Heteroscedastic regression (`HetRegression`)

### JAX Implementation
- JAX regression layers
- JAX distribution classes
- JIT compilation
- Vectorized operations

### Integration Tests
- End-to-end training loops
- Uncertainty estimation
- Cross-platform compatibility
- Memory efficiency

### Edge Cases
- Numerical stability
- Extreme parameter values
- Error handling
- Performance benchmarks

## Fixtures

The test suite provides several useful fixtures:

- **`device`** - PyTorch device (CPU or GPU)
- **`seed`** - Random seed for reproducibility
- **`sample_data_small/medium`** - Sample data for testing
- **`classification_params`** - Common classification parameters
- **`regression_params`** - Common regression parameters
- **`jax_key`** - JAX random key
- **`jax_sample_data`** - JAX sample data

## Performance Benchmarks

The benchmark tests measure:

- **Speed** - Forward and backward pass timing
- **Memory** - Memory usage with large tensors
- **Scalability** - Performance with different batch sizes and feature dimensions
- **Numerical Stability** - Behavior with extreme values
- **Cross-Platform** - CPU vs GPU performance

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:

```bash
# Run tests in CI environment
pytest --tb=short --disable-warnings

# Run with coverage
pytest --cov=vbll --cov-report=html

# Run with parallel execution
pytest -n auto
```

## Adding New Tests

When adding new tests:

1. **Follow naming conventions** - Use `test_` prefix for functions
2. **Use appropriate markers** - Mark tests as `@pytest.mark.slow`, `@pytest.mark.gpu`, etc.
3. **Add docstrings** - Explain what the test validates
4. **Use fixtures** - Leverage existing fixtures for common setup
5. **Test edge cases** - Include boundary conditions and error cases
6. **Verify numerical stability** - Check for NaN, inf, and extreme values

## Troubleshooting

### Common Issues

1. **CUDA out of memory** - Use smaller batch sizes or skip GPU tests
2. **JAX not installed** - Skip JAX tests with `-m "not jax"`
3. **Slow tests** - Skip with `-m "not slow"`
4. **Import errors** - Ensure VBLL is installed in development mode

### Debug Mode

```bash
# Run with debug output
pytest -s --tb=long

# Run single test with debug
pytest tests/test_distributions.py::TestNormal::test_initialization -s -v
```

## Test Data

The test suite uses synthetic data to avoid external dependencies:

- **Small datasets** - For quick unit tests
- **Medium datasets** - For integration tests
- **Large datasets** - For performance benchmarks
- **Edge cases** - Extreme values, empty tensors, etc.

All test data is generated deterministically using fixed random seeds for reproducibility.

