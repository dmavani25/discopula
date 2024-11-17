# discopula

> Discrete checkerboard copula modeling and implementation of new scoring methods pertaining to ordinal and categorical discrete data.

[![build](https://github.com/dmavani25/discopula/actions/workflows/test.yaml/badge.svg)](https://github.com/dmavani25/discopula/actions/workflows/test.yaml)
[![Documentation Status](https://readthedocs.org/projects/discopula/badge/?version=latest)](https://discopula.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/dmavani25/discopula/badge.svg?branch=master)](https://coveralls.io/github/dmavani25/discopula?branch=master)
[![Built with PyPi Template](https://img.shields.io/badge/PyPi_Template-v0.6.1-blue.svg)](https://github.com/christophevg/pypi-template)

## Overview

discopula is a Python package that implements checkerboard copulas for discrete ordinal data. It provides tools for:

- Creating checkerboard copulas from contingency tables
- Calculating copula regression scores
- Computing association measures (CCRAM and SCCRAM)
- Analyzing relationships between ordinal variables

## Features

- Easy-to-use API for working with discrete copulas
- Support for both standard and vectorized calculations
- Comprehensive association measures
- Built-in visualization capabilities
- Efficient numerical computations using NumPy

## Installation

```console
$ pip install discopula
```

## Contents

```{toctree}
:maxdepth: 1

whats-in-the-box.md
getting-started.md
contributing.md
code.md
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use discopula in your research, please cite:

```bibtex
@software{discopula2024,
  author = {Author Name},
  title = {discopula: A Python Package for Discrete Checkerboard Copulas},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/dmavani25/discopula}
}
```