import numpy as np
from discopula.checkerboard import CheckerboardCopula

# Unit Test for CheckerboardCopula
def test_checkerboard_copula():
    """
    Tests the CheckerboardCopula class for correctness in calculating copula density.
    """
    P = np.array([
        [0, 1/8, 0, 2/8],
        [0, 0, 2/8, 0],
        [0, 1/8, 0, 2/8]
    ])
    
    # Initialize the copula with joint probability matrix P
    copula = CheckerboardCopula(P)
    
    # Test cases
    test_cases = [
        ((0.3, 0.7), 0.18),   # Example test case with expected output
        ((0.1, 0.5), 0.03),    # Test point with low values
        ((0.5, 0.5), 0.25),  # Central point test
        ((0.8, 0.1), 0.07)     # Upper boundary test
    ]
    
    for u_values, expected in test_cases:
        result = copula.copula_density(u_values)
        assert abs(result - expected) < 0.01, f"Failed for {u_values}: Expected {expected}, got {result}"
    
    print("All tests passed for CheckerboardCopula.")

# Run the test
test_checkerboard_copula()