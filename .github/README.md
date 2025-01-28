# discopula

> Discrete checkerboard copula modeling and implementation of new scoring methods pertaining to ordinal and categorical discrete data.

[![PyPI version](https://badge.fury.io/py/discopula.png)](https://badge.fury.io/py/discopula)
[![build](https://github.com/dmavani25/discopula/actions/workflows/test.yaml/badge.svg)](https://github.com/dmavani25/discopula/actions/workflows/test.yaml)
[![Documentation Status](https://readthedocs.org/projects/discopula/badge/?version=latest)](https://discopula.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/dmavani25/discopula/badge.png?branch=master)](https://coveralls.io/github/dmavani25/discopula?branch=master)
[![Built with PyPi Template](https://img.shields.io/badge/PyPi_Template-v0.6.1-blue.svg)](https://github.com/christophevg/pypi-template)

## Installation

This package (discopula) is hosted on PyPi, so for installation follow the following workflow ...

```console
$ pip install discopula
```

Now, you should be all set to use it in a Jupyter Notebook!

Alternatively, if you would like to use it in a project, we recommend you to have a virtual environment for your use of this package, then follow the following workflow. For best practices, it's recommended to use a virtual environment:

1. First, create and activate a virtual environment (Python 3.8+ recommended):

```bash
# Create virtual environment
$ python -m venv discopula-env

# Activate virtual environment (Mac/Linux)
$ source discopula-env/bin/activate

# Verify you're in the virtual environment
$ which python
```

2. Install package

```bash
$ pip install discopula
```

3. To deactivate the virtual environment, when done:

```bash
$ deactivate
```

## Documentation

Visit [Read the Docs](https://discopula.readthedocs.org) for the full documentation, including overviews and several examples.

## Examples

For detailed examples in Jupyter Notebooks and beyond (organized by functionality) please refer to our [GitHub repository's examples folder](https://github.com/dmavani25/discopula/tree/master/examples).

## Features

- Construction of checkerboard copulas from contingency tables and/or list of cases
- Calculation of marginal distributions and CDFs
- Computation of Checkerboard Copula Regression (CCR) and Prediction based on CCR
- Implementation of Checkerboard Copula Regression Association Measure (CCRAM) and the Scaled CCRAM (SCCRAM)
- Bootstrap functionality for CCR-based prediction, CCRAM and SCCRAM
- Permutation testing functionality for CCRAM & SCCRAM
- Vectorized implementations for improved performance
- Rigorous Edge-case Handling & Unit Testing with Pytest 

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.