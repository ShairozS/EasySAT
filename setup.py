from setuptools import setup
import os

setup(
    name='easysat',
    version='0.1.0',
    description='A library for working with Boolean Satisfiability problems. Contains utilities for instance generation, resolution, analysis, and benchmarking.',
    url='https://github.com/ShairozS/EasySAT',
    author='Shairoz Sohail',
    author_email='shairozsohail@gmail.com',
    packages=['easysat'],
    install_requires=['python-sat']
)