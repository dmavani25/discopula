import numpy as np
from discopula.checkerboard import CheckerboardCopula
import pytest

@pytest.fixture
def checkerboard_copula_dummy():
    """
    Fixture to create an instance of CheckerboardCopula with a specific probability matrix.
    """
    P = np.array([
        [0, 1/8, 0, 2/8],
        [0, 0, 2/8, 0],
        [0, 1/8, 0, 2/8]
    ])
    return CheckerboardCopula(P)

@pytest.mark.parametrize("u_values, expected", [
    ((0.3, 0.7), 0.18),   # Example test case with expected output
    ((0.1, 0.5), 0.03),   # Test point with low values
    ((0.5, 0.5), 0.25),   # Central point test
    ((0.8, 0.1), 0.07)    # Upper boundary test
])
def test_copula_density_dummy(checkerboard_copula_dummy, u_values, expected):
    """
    Tests the copula_density method of CheckerboardCopula for different points.
    """
    result = checkerboard_copula_dummy.copula_density(u_values)
    assert abs(result - expected) < 0.01, f"Failed for {u_values}: Expected {expected}, got {result}"


@pytest.fixture
def checkerboard_copula():
    """
    Fixture to create an instance of CheckerboardCopula with a specific probability matrix
    based on Example 1's Table 1.
    """
    P = np.array([
        [0, 0, 2/8],
        [0, 1/8, 0],
        [2/8, 0, 0],
        [0, 1/8, 0],
        [0, 0, 2/8]
    ])
    return CheckerboardCopula(P)

@pytest.mark.parametrize("expected_cdf_X1, expected_cdf_X2", [
     ([0, 2/8, 3/8, 5/8, 6/8, 1], [0, 2/8, 4/8, 1])
])
def test_marginal_cdfs(checkerboard_copula, expected_cdf_X1, expected_cdf_X2):
    """
    Tests the marginal CDFs of X1 and X2.
    """
    np.testing.assert_almost_equal(checkerboard_copula.marginal_cdf_X1, expected_cdf_X1, decimal=5,
                                   err_msg="Marginal CDF for X1 does not match expected values")
    np.testing.assert_almost_equal(checkerboard_copula.marginal_cdf_X2, expected_cdf_X2, decimal=5,
                                   err_msg="Marginal CDF for X2 does not match expected values")

@pytest.mark.parametrize("expected_scores_X1, expected_scores_X2", [
     ([2/16, 5/16, 8/16, 11/16, 14/16], [2/16, 6/16, 12/16])
])
def test_checkerboard_scores(checkerboard_copula, expected_scores_X1, expected_scores_X2):
    """
    Tests the checkerboard copula scores for X1 and X2.
    """
    np.testing.assert_almost_equal(checkerboard_copula.scores_X1, expected_scores_X1, decimal=5,
                                   err_msg="Checkerboard scores for X1 do not match expected values")
    np.testing.assert_almost_equal(checkerboard_copula.scores_X2, expected_scores_X2, decimal=5,
                                   err_msg="Checkerboard scores for X2 do not match expected values")

@pytest.mark.parametrize("expected_mean_S1, expected_variance_S1, expected_mean_S2, expected_variance_S2", [
     (0.5, 81/1024, 0.4167, 9/128)
])
def test_means_and_variances(checkerboard_copula, expected_mean_S1, expected_variance_S1, expected_mean_S2, expected_variance_S2):
    """
    Tests the means and variances for S1 and S2 based on the checkerboard copula scores.
    """
    mean_S1 = np.mean(checkerboard_copula.scores_X1)
    variance_S1 = np.var(checkerboard_copula.scores_X1)
    mean_S2 = np.mean(checkerboard_copula.scores_X2)
    variance_S2 = np.var(checkerboard_copula.scores_X2)
    
    assert abs(mean_S1 - expected_mean_S1) < 0.01, f"Mean for S1 does not match: Expected {expected_mean_S1}, got {mean_S1}"
    assert abs(variance_S1 - expected_variance_S1) < 0.01, f"Variance for S1 does not match: Expected {expected_variance_S1}, got {variance_S1}"
    assert abs(mean_S2 - expected_mean_S2) < 0.01, f"Mean for S2 does not match: Expected {expected_mean_S2}, got {mean_S2}"
    assert abs(variance_S2 - expected_variance_S2) < 0.01, f"Variance for S2 does not match: Expected {expected_variance_S2}, got {variance_S2}"

@pytest.mark.parametrize("u_values, expected", [
    ((0.3, 0.7), 0.15),   # Example test case with expected output
    ((0.1, 0.5), 0.0),   # Test point with low values
    ((0.5, 0.5), 0.25),   # Central point test
    ((0.8, 0.1), 0.1)    # Upper boundary test
])
def test_copula_density(checkerboard_copula, u_values, expected):
    """
    Tests the copula_density method of CheckerboardCopula for different points.
    """
    result = checkerboard_copula.copula_density(u_values)
    assert abs(result - expected) < 0.01, f"Failed for {u_values}: Expected {expected}, got {result}"
