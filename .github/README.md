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

Alternatively, if you would like to have a virtual environment for your use of this package, then follow the following workflow. For best practices, it's recommended to use a virtual environment:

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

For a quick overview, refer to the quick-start example below. More detailed examples organized by functionality can be found in our [GitHub repository's examples folder](https://github.com/dmavani25/discopula/tree/master/examples).

### Quick-Start Example

```python
import numpy as np
from discopula import CheckerboardCopula

# Create a sample contingency table
contingency_table = np.array([
    [0, 0, 20],
    [0, 10, 0],
    [20, 0, 0],
    [0, 10, 0],
    [0, 0, 20]
])

# Initialize copula from contingency table
copula = CheckerboardCopula.from_contingency_table(contingency_table)

# Basic properties
print(f"Shape of probability matrix P: {copula.P.shape}")
print(f"Marginal CDF X1: {copula.marginal_cdf_X1}")
print(f"Marginal CDF X2: {copula.marginal_cdf_X2}")

# Regression calculations
u1, u2 = 0.5, 0.5
print(f"E[U2|U1={u1}] = {copula.calculate_regression_U2_on_U1(u1):.6f}")
print(f"E[U1|U2={u2}] = {copula.calculate_regression_U1_on_U2(u2):.6f}")

# Association measures
print(f"CCRAM X1->X2: {copula.calculate_CCRAM_X1_X2():.6f}")
print(f"SCCRAM X1->X2: {copula.calculate_SCCRAM_X1_X2():.6f}")
```

### Quick-Start Example Output 

```text
Shape of probability matrix P: (5, 3)
Marginal CDF X1: [0.    0.25  0.375 0.625 0.75  1.   ]
Marginal CDF X2: [0.   0.25 0.5  1.  ]
E[U2|U1=0.5] = 0.125000
E[U1|U2=0.5] = 0.500000
CCRAM X1->X2: 0.843750
SCCRAM X1->X2: 1.000000
```

## Features

- Construction of checkerboard copulas from contingency tables
- Calculation of marginal distributions and CDFs
- Computation of conditional expectations (regression)
- Implementation of Checkerboard Copula Regression Association Measure (CCRAM)
- Standardized CCRAM (SCCRAM) calculations
- Vectorized implementations for improved performance
- Bootstrap functionality for CCRAM, SCCRAM, Checkerboard Copula Regression (CCR), and Prediction based on CCR
- Permutation testing functionality for CCRAM & SCCRAM
- Rigorous Edge-case Handling & Unit Testing with Pytest 

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.