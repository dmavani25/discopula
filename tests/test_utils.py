import numpy as np
import pytest
from discopula.checkerboard.utils import (
    contingency_to_case_form,
    case_form_to_contingency,
    gen_contingency_to_case_form,
    gen_case_form_to_contingency
)

@pytest.fixture
def contingency_table():
    """
    Fixture to create a sample contingency table.
    """
    return np.array([
        [0, 0, 20],
        [0, 10, 0],
        [20, 0, 0],
        [0, 10, 0],
        [0, 0, 20]
    ])
    
@pytest.fixture
def case_form_data():
    """
    Fixture to create a sample case-form data array.
    """
    return np.array([
        [0, 2], [0, 2], [0, 2], [0, 2], [0, 2],
        [0, 2], [0, 2], [0, 2], [0, 2], [0, 2],
        [0, 2], [0, 2], [0, 2], [0, 2], [0, 2],
        [0, 2], [0, 2], [0, 2], [0, 2], [0, 2],
        [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
        [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
        [2, 0], [2, 0], [2, 0], [2, 0], [2, 0],
        [2, 0], [2, 0], [2, 0], [2, 0], [2, 0],
        [2, 0], [2, 0], [2, 0], [2, 0], [2, 0],
        [2, 0], [2, 0], [2, 0], [2, 0], [2, 0],
        [3, 1], [3, 1], [3, 1], [3, 1], [3, 1],
        [3, 1], [3, 1], [3, 1], [3, 1], [3, 1],
        [4, 2], [4, 2], [4, 2], [4, 2], [4, 2],
        [4, 2], [4, 2], [4, 2], [4, 2], [4, 2],
        [4, 2], [4, 2], [4, 2], [4, 2], [4, 2],
        [4, 2], [4, 2], [4, 2], [4, 2], [4, 2]
    ])

def test_contingency_to_case_form(contingency_table, case_form_data):
    """
    Test converting a contingency table to case-form data.
    """
    cases = contingency_to_case_form(contingency_table)
    np.testing.assert_array_equal(cases, case_form_data)

def test_case_form_to_contingency(contingency_table, case_form_data):
    """
    Test converting case-form data back to a contingency table.
    """
    n_rows, n_cols = contingency_table.shape
    reconstructed_table = case_form_to_contingency(case_form_data, n_rows, n_cols)
    np.testing.assert_array_equal(reconstructed_table, contingency_table)
    
def test_gen_contingency_to_case_form(contingency_table, case_form_data):
    """
    Test gen_contingency_to_case_form conversion.
    """
    cases = gen_contingency_to_case_form(contingency_table)
    # Sort both arrays to ensure consistent comparison
    np.testing.assert_array_equal(
        cases[np.lexsort(cases.T)],
        case_form_data[np.lexsort(case_form_data.T)]
    )

def test_gen_case_form_to_contingency(contingency_table, case_form_data):
    """
    Test gen_case_form_to_contingency conversion.
    """
    reconstructed = gen_case_form_to_contingency(case_form_data, contingency_table.shape)
    np.testing.assert_array_equal(reconstructed, contingency_table)

@pytest.fixture
def gen_contingency_table():
    """Fixture for a simple 2D contingency table."""
    return np.array([
        [2, 1],
        [0, 3]
    ])

@pytest.fixture
def gen_case_form_data():
    """Fixture for corresponding case-form data."""
    return np.array([
        [0, 0], [0, 0],  # 2 cases
        [0, 1],          # 1 case
        [1, 1], [1, 1], [1, 1]  # 3 cases
    ])

@pytest.fixture
def gen_3d_cases():
    """Fixture for 3D batched case data."""
    return np.array([
        [[0, 0], [0, 1]],
        [[1, 0], [1, 1]]
    ])

def test_gen_contingency_to_case_form_2d(gen_contingency_table, gen_case_form_data):
    """Test gen_contingency_to_case_form conversion."""
    cases = gen_contingency_to_case_form(gen_contingency_table)
    # Sort both arrays to ensure consistent comparison
    np.testing.assert_array_equal(
        cases[np.lexsort(cases.T)],
        gen_case_form_data[np.lexsort(gen_case_form_data.T)]
    )

def test_gen_case_form_to_contingency_2d(gen_contingency_table, gen_case_form_data):
    """Test gen_case_form_to_contingency with 2D data."""
    reconstructed = gen_case_form_to_contingency(gen_case_form_data, gen_contingency_table.shape)
    np.testing.assert_array_equal(reconstructed, gen_contingency_table)

def test_gen_case_form_to_contingency_3d(gen_3d_cases):
    """Test gen_case_form_to_contingency with 3D data."""
    shape = (2, 2)  # Expected shape of output table
    reconstructed = gen_case_form_to_contingency(gen_3d_cases, shape)
    expected = np.array([[1, 1], [1, 1]])  # Each position should have count 1
    np.testing.assert_array_equal(reconstructed, expected)

def test_gen_case_form_to_contingency_axis_order():
    """Test gen_case_form_to_contingency with custom axis ordering."""
    cases = np.array([[0, 1], [1, 0]])  # Two cases with swapped coordinates
    shape = (2, 2)
    
    # Normal order
    normal = gen_case_form_to_contingency(cases, shape)
    # Swapped axes
    swapped = gen_case_form_to_contingency(cases, shape, axis_order=[1, 0])
    
    np.testing.assert_array_equal(normal.T, swapped)