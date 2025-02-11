## Getting Started

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