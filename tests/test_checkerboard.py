import numpy as np
from discopula.checkerboard import CheckerboardCopula
import pytest

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

@pytest.mark.parametrize("expected_conditional_pmf_X2_given_X1", [
    np.array([
        [0, 0, 1],     # P(X2|X1=1) = [0, 0, 1]
        [0, 1, 0],     # P(X2|X1=2) = [0, 1, 0]
        [1, 0, 0],     # P(X2|X1=3) = [1, 0, 0]
        [0, 1, 0],     # P(X2|X1=4) = [0, 1, 0]
        [0, 0, 1]      # P(X2|X1=5) = [0, 0, 1]
    ])  
])
def test_conditional_pmf_X2_given_X1(checkerboard_copula, expected_conditional_pmf_X2_given_X1):
    """
    Tests the conditional PMF of X2 given X1.
    The expected values are based on the normalized rows of the joint probability matrix.
    """
    
    calculated_conditional_pmf_X2_given_X1 = checkerboard_copula.calculate_conditional_pmf_X2_given_X1()
    np.testing.assert_almost_equal(
        calculated_conditional_pmf_X2_given_X1,
        expected_conditional_pmf_X2_given_X1,
        decimal=5,
        err_msg="Conditional PMF of X2 given X1 does not match expected values"
    )

@pytest.mark.parametrize("expected_conditional_pmf_X1_given_X2", [
    np.array([
        [0., 0., 0.5],       # First row: P(X1=1|X2)
        [0., 0.5, 0.],       # Second row: P(X1=2|X2)
        [1., 0., 0.],        # Third row: P(X1=3|X2)
        [0., 0.5, 0.],       # Fourth row: P(X1=4|X2)
        [0., 0., 0.5]        # Fifth row: P(X1=5|X2)
    ]) 
])
def test_conditional_pmf_X1_given_X2(checkerboard_copula, expected_conditional_pmf_X1_given_X2):
    """
    Tests the conditional PMF of X1 given X2.
    The expected values are based on the normalized columns of the joint probability matrix.
    """
    
    calculated_conditional_pmf_X1_given_X2 = checkerboard_copula.calculate_conditional_pmf_X1_given_X2()
    np.testing.assert_almost_equal(
        calculated_conditional_pmf_X1_given_X2,
        expected_conditional_pmf_X1_given_X2,
        decimal=5,
        err_msg="Conditional PMF of X1 given X2 does not match expected values"
    )

def test_conditional_pmf_sums(checkerboard_copula):
    """
    Tests that the conditional PMFs sum to 1 along the appropriate axis
    (where there are non-zero marginals).
    """
    # Test X2|X1
    cond_pmf_X2_given_X1 = checkerboard_copula.calculate_conditional_pmf_X2_given_X1()
    row_sums = np.sum(cond_pmf_X2_given_X1, axis=1)
    np.testing.assert_array_almost_equal(
        row_sums,
        np.ones(5),
        decimal=5,
        err_msg="Conditional PMF of X2 given X1 rows do not sum to 1"
    )
    
    # Test X1|X2
    cond_pmf_X1_given_X2 = checkerboard_copula.calculate_conditional_pmf_X1_given_X2()
    col_sums = np.sum(cond_pmf_X1_given_X2, axis=0)  # Sum along columns for X1|X2
    np.testing.assert_array_almost_equal(
        col_sums,
        np.ones(3),
        decimal=5,
        err_msg="Conditional PMF of X1 given X2 columns do not sum to 1"
    )
    
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
     (0.5, 81/1024, 0.416, 9/128)
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

@pytest.mark.parametrize("u1, expected_regression_value", [
    (0, 12/16), # Inside [0, 1/8]
    (1/16, 12/16),  # Inside [0, 2/8]
    (3/8, 6/16),  # Inside (2/8, 3/8]
    (4/8, 2/16),    # Inside (3/8, 5/8]
    (5.5/8, 6/16),  # Inside (5/8, 6/8]
    (7/8, 12/16),   # Inside (6/8, 1]
    (1, 12/16)    # Inside (6/8, 1]
])
def test_regression_U2_on_U1(checkerboard_copula, u1, expected_regression_value):
    """
    Tests the regression function for specific u1 values that should match
    the given example values.
    """
    calculated_regression_value = checkerboard_copula.calculate_regression_U2_on_U1(u1)
    np.testing.assert_almost_equal(
        calculated_regression_value, 
        expected_regression_value,
        decimal=5,
        err_msg=f"Regression value for u1={u1} does not match expected value"
    )

@pytest.mark.parametrize("u1_values, expected_regression_values", [
    ([0, 1/16, 3/8, 4/8, 5.5/8, 7/8, 1], [12/16, 12/16, 6/16, 2/16, 6/16, 12/16, 12/16])
])
def test_regression_U2_on_U1_vectorized(checkerboard_copula, u1_values, expected_regression_values):
    """
    Tests the vectorized regression function with multiple u1 values.
    """
    calculated_regression_values = checkerboard_copula.calculate_regression_U2_on_U1_vectorized(u1_values)
    
    np.testing.assert_array_almost_equal(
        calculated_regression_values,
        expected_regression_values,
        decimal=5,
        err_msg="Vectorized regression values do not match expected values"
    )
    
@pytest.mark.parametrize("expected_ccram", [
    0.84375  # Based on manual calculation for the given P matrix (= 27/32)
])
def test_CCRAM_X1_X2(checkerboard_copula, expected_ccram):
    """
    Tests the CCRAM calculation for X1 and X2.
    """
    calculated_ccram = checkerboard_copula.calculate_CCRAM_X1_X2()
    np.testing.assert_almost_equal(
        calculated_ccram,
        expected_ccram,
        decimal=5,
        err_msg=f"CCRAM for X1 and X2 does not match the expected value {expected_ccram}"
    )

@pytest.mark.parametrize("expected_ccram_vectorized", [
    0.84375  # Based on manual calculation for the given P matrix (= 27/32)
])
def test_CCRAM_X1_X2_vectorized(checkerboard_copula, expected_ccram_vectorized):
    """
    Tests the vectorized CCRAM calculation for X1 and X2.
    """
    calculated_ccram_vectorized = checkerboard_copula.calculate_CCRAM_X1_X2_vectorized()
    np.testing.assert_almost_equal(
        calculated_ccram_vectorized,
        expected_ccram_vectorized,
        decimal=5,
        err_msg=f"Vectorized CCRAM for X1 and X2 does not match the expected value {expected_ccram_vectorized}"
    )
    
@pytest.mark.parametrize("expected_sigma_sq_S_times_12", [
    0.0703125 * 12  # Based on manual calculation for the given P matrix (= 0.84375)
])
def test_sigma_sq_S_times_12(checkerboard_copula, expected_sigma_sq_S_times_12):
    """
    Tests the calculation of sigma_sq_S.
    """
    calculated_sigma_sq_S = checkerboard_copula.calculate_sigma_sq_S()
    np.testing.assert_almost_equal(
        calculated_sigma_sq_S * 12,
        expected_sigma_sq_S_times_12,
        decimal=5,
        err_msg=f"Sigma squared S times 12 does not match the expected value {expected_sigma_sq_S_times_12}"
    )

@pytest.mark.parametrize("expected_sigma_sq_S_vectorized_times_12", [
    0.0703125 * 12  # Based on manual calculation for the given P matrix (= 0.84375)
])
def test_sigma_sq_S_vectorized_times_12(checkerboard_copula, expected_sigma_sq_S_vectorized_times_12):
    """
    Tests the vectorized calculation of sigma_sq_S.
    """
    calculated_sigma_sq_S_vectorized = checkerboard_copula.calculate_sigma_sq_S_vectorized()
    np.testing.assert_almost_equal(
        calculated_sigma_sq_S_vectorized * 12,
        expected_sigma_sq_S_vectorized_times_12,
        decimal=5,
        err_msg=f"Vectorized sigma squared S times 12 does not match the expected value {expected_sigma_sq_S_vectorized_times_12}"
    )
