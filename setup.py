from setuptools import setup, find_packages
import os, sys

setup(
    name="vbll",
    version="0.4.8",
    packages=find_packages(),
    install_requires=["torch"],
)
