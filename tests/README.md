# VBLL Test Suite

This directory contains comprehensive tests for the VBLL (Variational Bayesian Last Layers) library.

## Test Structure

### Core Test Files

- **`test_distributions.py`** - Tests for distribution classes (`Normal`, `DenseNormal`, `DenseNormalPrec`, `LowRankNormal`)
- **`test_classification.py`** - Tests for classification layers (`DiscClassification`, `tDiscClassification`, `HetClassification`)
- **`test_regression.py`** - Tests for regression layers (`Regression`, `tRegression`, `HetRegression`)
- **`test_integration.py`** - End-to-end integration tests

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

## Adding New Tests

When adding new tests:

1. **Follow naming conventions** - Use `test_` prefix for functions
2. **Use appropriate markers** - Mark tests as `@pytest.mark.slow`, `@pytest.mark.gpu`, etc.
3. **Add docstrings** - Explain what the test validates
4. **Use fixtures** - Leverage existing fixtures for common setup
5. **Test edge cases** - Include boundary conditions and error cases
6. **Verify numerical stability** - Check for NaN, inf, and extreme values