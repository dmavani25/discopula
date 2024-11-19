import numpy as np
import pytest
from discopula.n_dim_checkerboard import NDimensionalCheckerboardCopula

# @TODO under development

@pytest.fixture
def sample_2d():
    P = np.array([[0.2, 0.3], [0.1, 0.4]])
    return P

@pytest.fixture
def sample_3d():
    P = np.array([
        [[0.1, 0.1], [0.1, 0.1]],
        [[0.1, 0.2], [0.1, 0.2]]
    ])
    return P

@pytest.fixture
def sample_4d():
    P = np.random.rand(2, 3, 2, 2)
    return P / P.sum()

def test_initialization(sample_2d, sample_3d, sample_4d):
    copula_2d = NDimensionalCheckerboardCopula(sample_2d)
    assert copula_2d.n_dimensions == 2
    
    copula_3d = NDimensionalCheckerboardCopula(sample_3d)
    assert copula_3d.n_dimensions == 3
    
    copula_4d = NDimensionalCheckerboardCopula(sample_4d)
    assert copula_4d.n_dimensions == 4

@pytest.mark.parametrize("invalid_input,expected_error", [
    (np.array([0.5, 0.5]), ValueError),  # 1D array
    (np.array([[0.5, -0.5], [0.5, 0.5]]), ValueError),  # Negative values
    (np.array([[0.5, 0.5], [0.5, 0.5]]), ValueError),  # Non-normalized
])
def test_invalid_inputs(invalid_input, expected_error):
    with pytest.raises(expected_error):
        NDimensionalCheckerboardCopula(invalid_input)

@pytest.mark.parametrize("input_table,expected_dims", [
    ([[1, 2], [3, 4]], 2),  # Basic 2D case
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 3),  # 3D case
    (np.array([[1, 2], [3, 4]]), 2),  # Already numpy array
])
def test_from_contingency_table_valid(input_table, expected_dims):
    copula = NDimensionalCheckerboardCopula.from_contingency_table(input_table)
    assert copula.n_dimensions == expected_dims
    assert isinstance(copula.P, np.ndarray)
    assert np.allclose(copula.P.sum(), 1.0)

@pytest.mark.parametrize("invalid_input,expected_error,error_msg", [
    ([0.5], ValueError, "must be at least 2-dimensional"),  # 1D array
    ([[0, 0], [0, 0]], ValueError, "cannot be all zeros"),  # All zeros
    ([[-1, 2], [3, 4]], ValueError, "cannot contain negative values"),  # Negative values
])
def test_from_contingency_table_invalid(invalid_input, expected_error, error_msg):
    with pytest.raises(expected_error, match=error_msg):
        NDimensionalCheckerboardCopula.from_contingency_table(invalid_input)

def test_from_contingency_table_list_conversion():
    input_list = [[1, 2], [3, 4]]
    copula = NDimensionalCheckerboardCopula.from_contingency_table(input_list)
    assert isinstance(copula.P, np.ndarray)

def test_from_contingency_table_normalization():
    table = np.array([[10, 20], [30, 40]])  # Sum = 100
    copula = NDimensionalCheckerboardCopula.from_contingency_table(table)
    assert np.allclose(copula.P, table/100)

def test_marginal_calculations(sample_2d):
    copula = NDimensionalCheckerboardCopula(sample_2d)
    
    assert len(copula.marginal_pdfs) == 2
    assert np.allclose(copula.marginal_pdfs[0].sum(), 1.0)
    assert np.allclose(copula.marginal_pdfs[1].sum(), 1.0)
    
    assert len(copula.marginal_cdfs) == 2
    assert np.allclose(copula.marginal_cdfs[0][-1], 1.0)
    assert np.allclose(copula.marginal_cdfs[1][-1], 1.0)

def test_conditional_pmf(sample_3d):
    copula = NDimensionalCheckerboardCopula(sample_3d)
    cond_pmf = copula.get_conditional_pmf(0, [1,2], [0])
    assert np.allclose(cond_pmf.sum(), 1.0)

def test_regression(sample_2d):
    copula = NDimensionalCheckerboardCopula(sample_2d)
    reg_val = copula.calculate_regression(0, 1, 0.5)
    assert 0 <= reg_val <= 1

def test_CCRAM(sample_2d):
    copula = NDimensionalCheckerboardCopula(sample_2d)
    ccram = copula.calculate_CCRAM(0, 1)
    assert ccram >= 0

def test_SCCRAM(sample_2d):
    copula = NDimensionalCheckerboardCopula(sample_2d)
    sccram = copula.calculate_SCCRAM(0, 1)
    assert isinstance(sccram, float)

def test_from_contingency_table():
    contingency_table = np.array([[1, 2], [3, 4]])
    copula = NDimensionalCheckerboardCopula.from_contingency_table(contingency_table)
    assert copula.n_dimensions == 2