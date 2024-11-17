# What's in The Box?

## CheckerboardCopula Class

The main class for working with discrete checkerboard copulas. It offers:

### Core Functionality

#### Copula Creation and Management
- Creation of copulas from contingency tables
- Automatic probability matrix normalization
- Marginal distribution calculations
- Conditional probability computations

#### Statistical Measures
- CCRAM (Checkerboard Copula Regression Association Measure)
  - Directional measure X1→X2
  - Directional measure X2→X1
- SCCRAM (Standardized CCRAM)
  - Normalized association measures
  - Scale-invariant results

### Key Features

#### Performance Optimization
- Vectorized implementations for large datasets
- Efficient memory usage
- Optimized numerical calculations

#### Robust Error Handling
- Input validation
- Numerical stability checks
- Comprehensive error messages

## Mathematical Framework

### Theoretical Foundation
The package implements the checkerboard copula theory for discrete ordinal data, providing:

- Copula density calculations
- Score-based regression methods
- Association measures for ordinal variables

### Implementation Details

#### Probability Matrix
- Joint probability matrix handling
- Marginal distribution calculations
- Conditional probability computations

#### Regression Functions
- Conditional expectation calculations
- Score-based regression methods
- Vectorized operations for efficiency

## Usage Examples

### Basic Usage
```python
import numpy as np
from discopula import CheckerboardCopula

# Create from contingency table
table = np.array([[1, 0], [0, 1]])
copula = CheckerboardCopula.from_contingency_table(table)

# Calculate association measure
ccram = copula.calculate_CCRAM_X1_X2()
```

### Advanced Features
```python
# Vectorized operations
sccram = copula.calculate_SCCRAM_X1_X2_vectorized()

# Access properties
prob_matrix = copula.P
cont_table = copula.contingency_table
```