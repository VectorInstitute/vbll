#!/usr/bin/env python3
"""
Test runner script for VBLL test suite.
"""
import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"❌ {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ {description} completed successfully")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run VBLL test suite")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--benchmarks", action="store_true", help="Run benchmark tests only")
    parser.add_argument("--jax", action="store_true", help="Include JAX tests")
    parser.add_argument("--gpu", action="store_true", help="Include GPU tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Add coverage
    if args.coverage:
        cmd.extend(["--cov=vbll", "--cov-report=html", "--cov-report=term"])
    
    # Determine test selection
    if args.quick:
        cmd.extend(["-m", "not slow"])
        print("🚀 Running quick tests (excluding slow tests)")
    elif args.unit:
        cmd.extend(["-m", "unit"])
        print("🧪 Running unit tests only")
    elif args.integration:
        cmd.extend(["-m", "integration"])
        print("🔗 Running integration tests only")
    elif args.benchmarks:
        cmd.extend(["-m", "benchmark"])
        print("⚡ Running benchmark tests only")
    else:
        print("🧪 Running all tests")
    
    # Add markers for optional dependencies
    markers = []
    if not args.jax:
        markers.append("not jax")
    if not args.gpu:
        markers.append("not gpu")
    
    if markers:
        cmd.extend(["-m", " and ".join(markers)])
    
    # Add test directory
    cmd.append("tests/")
    
    # Run the tests
    success = run_command(cmd, "VBLL Test Suite")
    
    if success:
        print("\n🎉 All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

