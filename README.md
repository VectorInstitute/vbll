# Variational Bayesian Last Layers (VBLL)

## Introduction
VBLL introduces a deterministic variational formulation for training Bayesian last layers in neural networks. This method offers a computationally efficient approach to improving uncertainty estimation in deep learning models. By leveraging this technique, VBLL can be trained and evaluated with quadratic complexity in last layer width, making it nearly computationally free to add to standard architectures. Our work focuses on enhancing predictive accuracy, calibration, and out-of-distribution detection over baselines in both regression and classification.

## Installation
```bash
# Clone the repository
git clone https://github.com/VectorInstitute/vbll.git

# Navigate into the repository directory
cd vbll

# Install required dependencies
pip install -e .
```

## Usage
The repository includes Jupyter Notebooks demonstrating the application of VBLL for regression and classification tasks. For detailed usage examples, please refer to the provided notebooks.

## Contributing
Contributions to the VBLL project are welcome. If you're interested in contributing, please read the contribution guidelines in the repository.

## Citation
If you find VBLL useful in your research, please consider citing our paper:

```bibtex
@inproceedings{harrison2024vbll,
  title={Variational Bayesian Last Layers},
  author={Harrison, James and Willes, John and Snoek, Jasper},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```


