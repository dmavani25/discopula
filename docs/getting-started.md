# Getting Started

## Installation

discopula is hosted on PyPI, making it easy to install using pip:

```console
$ pip install discopula
```

## Basic Usage

Here's a quick example to get you started:

```python
import numpy as np
from discopula import CheckerboardCopula

# Create a contingency table
contingency_table = np.array([
    [0, 0, 2],
    [0, 1, 0],
    [2, 0, 0],
    [0, 1, 0],
    [0, 0, 2]
])

# Create a CheckerboardCopula instance
copula = CheckerboardCopula.from_contingency_table(contingency_table)

# Calculate association measures
ccram_x1_x2 = copula.calculate_CCRAM_X1_X2()
sccram_x1_x2 = copula.calculate_SCCRAM_X1_X2()

print(f"CCRAM (X1->X2): {ccram_x1_x2}")
print(f"SCCRAM (X1->X2): {sccram_x1_x2}")
```

## Advanced Features

### Vectorized Operations

For improved performance with large datasets, use the vectorized methods:

```python
# Vectorized calculations
ccram_vectorized = copula.calculate_CCRAM_X1_X2_vectorized()
sccram_vectorized = copula.calculate_SCCRAM_X1_X2_vectorized()
```

### Working with Probability Matrices

You can also create a copula directly from a probability matrix:

```python
P = np.array([
    [0.0, 0.0, 0.25],
    [0.0, 0.125, 0.0],
    [0.25, 0.0, 0.0],
    [0.0, 0.125, 0.0],
    [0.0, 0.0, 0.25]
])

copula = CheckerboardCopula(P)
```