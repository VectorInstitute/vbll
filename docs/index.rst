.. vbll documentation master file, created by
   sphinx-quickstart on Wed Apr  3 23:19:48 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Variational Bayesian Last Layers
================================

Overview
--------

VBLL introduces a deterministic variational formulation for training Bayesian last layers in neural networks. This method offers a computationally efficient approach to improving uncertainty estimation in deep learning models. By leveraging this technique, VBLL can be trained and evaluated with quadratic complexity in last layer width, making it nearly computationally free to add to standard architectures. Our work focuses on enhancing predictive accuracy, calibration, and out-of-distribution detection over baselines in both regression and classification.


Citation
--------

If you use VBLL in your research, please cite the following paper::

   @inproceedings{harrison2024vbll,
   title={Variational Bayesian Last Layers},
   author={Harrison, James and Willes, John and Snoek, Jasper},
   booktitle={International Conference on Learning Representations (ICLR)},
   year={2024}
   }



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   tutorials
   api_documentation






.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
