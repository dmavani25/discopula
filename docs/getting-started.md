# Getting Started

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

## Quick Start Example

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

### Example Output: 

```console
Shape of probability matrix P: (5, 3)
Marginal CDF X1: [0.    0.25  0.375 0.625 0.75  1.   ]
Marginal CDF X2: [0.   0.25 0.5  1.  ]
E[U2|U1=0.5] = 0.125000
E[U1|U2=0.5] = 0.500000
CCRAM X1->X2: 0.843750
SCCRAM X1->X2: 1.000000
```