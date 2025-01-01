import numpy as np
import pytest
from discopula.checkerboard.copula import CheckerboardCopula

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

def test_from_contingency_table_valid(contingency_table):
    """
    Test creation of CheckerboardCopula from valid contingency table.
    """
    cop = CheckerboardCopula.from_contingency_table(contingency_table)
    expected_P = contingency_table / contingency_table.sum()
    np.testing.assert_array_almost_equal(cop.P, expected_P)

def test_from_contingency_table_list():
    """
    Test creation of CheckerboardCopula from list instead of numpy array.
    """
    table_list = [
        [0, 0, 20],
        [0, 10, 0],
        [20, 0, 0],
        [0, 10, 0],
        [0, 0, 20]
    ]
    cop = CheckerboardCopula.from_contingency_table(table_list)
    assert isinstance(cop.P, np.ndarray)
    assert cop.P.shape == (5, 3)

@pytest.mark.parametrize("invalid_table,error_msg", [
    (np.array([[1, 2], [3, -1]]), "Contingency table cannot contain negative values"),
    (np.array([[0, 0], [0, 0]]), "Contingency table cannot be all zeros"),
    (np.array([1, 2, 3]), "Contingency table must be 2-dimensional"),
    (np.array([[[1, 2], [3, 4]]]), "Contingency table must be 2-dimensional")
])
def test_from_contingency_table_invalid(invalid_table, error_msg):
    """
    Test creation of CheckerboardCopula with invalid contingency tables.
    """
    with pytest.raises(ValueError, match=error_msg):
        CheckerboardCopula.from_contingency_table(invalid_table)

def test_contingency_table_property(contingency_table):
    """
    Test the contingency_table property returns approximately the original scale.
    """
    cop = CheckerboardCopula.from_contingency_table(contingency_table)
    recovered_table = cop.contingency_table
    # Check that the ratios between non-zero elements are preserved
    non_zero_orig = contingency_table[contingency_table > 0]
    non_zero_recovered = recovered_table[recovered_table > 0]
    ratio_orig = non_zero_orig / non_zero_orig.min()
    ratio_recovered = non_zero_recovered / non_zero_recovered.min()
    np.testing.assert_array_almost_equal(ratio_orig, ratio_recovered)

@pytest.mark.parametrize("invalid_P,error_msg", [
    (np.array([[0.5, 0.6], [0.2, 0.2]]), "Probability matrix P must sum to 1"),
    (np.array([[0.5, -0.1], [0.3, 0.3]]), "Probability matrix P must contain values between 0 and 1"),
    (np.array([[1.5, 0.1], [0.3, 0.3]]), "Probability matrix P must contain values between 0 and 1"),
    (np.array([0.2, 0.8]), "Probability matrix P must be 2-dimensional")
])
def test_invalid_probability_matrix(invalid_P, error_msg):
    """
    Test constructor with invalid probability matrices.
    """
    with pytest.raises(ValueError, match=error_msg):
        CheckerboardCopula(invalid_P)

def test_zero_probability_handling(contingency_table):
    """
    Test that zero probabilities are handled correctly in conditional PMFs.
    """
    cop = CheckerboardCopula.from_contingency_table(contingency_table)
    
    # Check that conditional PMFs sum to 1 for non-zero rows/columns
    non_zero_rows = np.where(contingency_table.sum(axis=1) > 0)[0]
    for row in non_zero_rows:
        np.testing.assert_almost_equal(
            cop.conditional_pmf_X2_given_X1[row].sum(),
            1.0
        )
    
    non_zero_cols = np.where(contingency_table.sum(axis=0) > 0)[0]
    for col in non_zero_cols:
        np.testing.assert_almost_equal(
            cop.conditional_pmf_X1_given_X2[:,col].sum(),
            1.0
        )

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
    
@pytest.mark.parametrize("u, ul, uj, expected_lambda_value", [
    (0, 0.1, 2/8, 0),  # u <= ul
    (1/16, 0, 2/8, 1/4),  # ul < u < uj
    (2/8, 2/8, 3/8, 0),  # u = ul
    (5/8, 3/8, 5/8, 1),  # u = uj
    (1, 0.5, 0.7, 1)  # u >= uj
])
def test_lambda_value(checkerboard_copula, u, ul, uj, expected_lambda_value):
    """
    Tests the lambda value for the checkerboard copula.
    """
    lambda_value = checkerboard_copula.lambda_function(u, ul, uj)
    np.testing.assert_almost_equal(
        lambda_value,
        expected_lambda_value,
        decimal=5,
        err_msg=f"Lambda value for u={u}, ul={ul}, uj={uj} does not match expected value"
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
    (0, 12/16), # Inside [0, 2/8]
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
def test_regression_U2_on_U1_batched(checkerboard_copula, u1_values, expected_regression_values):
    """
    Tests the batched regression function with multiple u1 values.
    """
    calculated_regression_values = checkerboard_copula.calculate_regression_U2_on_U1_batched(u1_values)
    
    np.testing.assert_array_almost_equal(
        calculated_regression_values,
        expected_regression_values,
        decimal=5,
        err_msg="Batched regression values do not match expected values"
    )
    
@pytest.mark.parametrize("u2, expected_regression_value", [
    (0, 1/2), # Inside [0, 2/8]
    (2/8, 1/2),  # Inside [0, 2/8]
    (3/8, 1/2),  # Inside (2/8, 3/8]
    (4/8, 1/2),    # Inside (3/8, 5/8]
    (6/8, 1/2),  # Inside (5/8, 6/8]
    (1, 1/2)    # Inside (6/8, 1]
])
def test_regression_U1_on_U2(checkerboard_copula, u2, expected_regression_value):
    """
    Tests the regression function for specific u2 values that should match
    the given example values.
    """
    calculated_regression_value = checkerboard_copula.calculate_regression_U1_on_U2(u2)
    np.testing.assert_almost_equal(
        calculated_regression_value, 
        expected_regression_value,
        decimal=5,
        err_msg=f"Regression value for u2={u2} does not match expected value"
    )

@pytest.mark.parametrize("u2_values, expected_regression_values", [
    ([0, 1/16, 3/8, 4/8, 5/8, 6/8, 1], [1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2])
])
def test_regression_U1_on_U2_batched(checkerboard_copula, u2_values, expected_regression_values):
    """
    Tests the batched regression function with multiple u2 values.
    """
    calculated_regression_values = checkerboard_copula.calculate_regression_U1_on_U2_batched(u2_values)
    
    np.testing.assert_array_almost_equal(
        calculated_regression_values,
        expected_regression_values,
        decimal=5,
        err_msg="Batched regression values do not match expected values"
    )
    
@pytest.mark.parametrize("expected_ccram", [
    0.84375  # Based on manual calculation for the given P matrix (= 27/32)
])
def test_CCRAM_X1_X2(checkerboard_copula, expected_ccram):
    """
    Tests the CCRAM calculation for X1 --> X2.
    """
    calculated_ccram = checkerboard_copula.calculate_CCRAM_X1_X2()
    np.testing.assert_almost_equal(
        calculated_ccram,
        expected_ccram,
        decimal=5,
        err_msg=f"CCRAM for X1 --> X2 does not match the expected value {expected_ccram}"
    )

@pytest.mark.parametrize("expected_ccram_vectorized", [
    0.84375  # Based on manual calculation for the given P matrix (= 27/32)
])
def test_CCRAM_X1_X2_vectorized(checkerboard_copula, expected_ccram_vectorized):
    """
    Tests the vectorized CCRAM calculation for X1 --> X2.
    """
    calculated_ccram_vectorized = checkerboard_copula.calculate_CCRAM_X1_X2_vectorized()
    np.testing.assert_almost_equal(
        calculated_ccram_vectorized,
        expected_ccram_vectorized,
        decimal=5,
        err_msg=f"Vectorized CCRAM for X1 --> X2 does not match the expected value {expected_ccram_vectorized}"
    )

@pytest.mark.parametrize("expected_ccram", [
    0.0  # Based on manual calculation for the given P matrix
])
def test_CCRAM_X2_X1(checkerboard_copula, expected_ccram):
    """
    Tests the CCRAM calculation for X2 --> X1.
    """
    calculated_ccram = checkerboard_copula.calculate_CCRAM_X2_X1()
    np.testing.assert_almost_equal(
        calculated_ccram,
        expected_ccram,
        decimal=5,
        err_msg=f"CCRAM for X2 --> X1 does not match the expected value {expected_ccram}"
    )

@pytest.mark.parametrize("expected_ccram_vectorized", [
    0.0  # Based on manual calculation for the given P matrix
])
def test_CCRAM_X2_X1_vectorized(checkerboard_copula, expected_ccram_vectorized):
    """
    Tests the vectorized CCRAM calculation for X2 --> X1.
    """
    calculated_ccram_vectorized = checkerboard_copula.calculate_CCRAM_X2_X1_vectorized()
    np.testing.assert_almost_equal(
        calculated_ccram_vectorized,
        expected_ccram_vectorized,
        decimal=5,
        err_msg=f"Vectorized CCRAM for X2 --> X1 does not match the expected value {expected_ccram_vectorized}"
    )

@pytest.mark.parametrize("expected_sigma_sq_S_times_12", [
    0.0703125 * 12  # Based on manual calculation for the given P matrix
])
def test_sigma_sq_S_X2_times_12(checkerboard_copula, expected_sigma_sq_S_times_12):
    """
    Tests the calculation of sigma_sq_S for X2.
    """
    calculated_sigma_sq_S = checkerboard_copula.calculate_sigma_sq_S_X2()
    np.testing.assert_almost_equal(
        calculated_sigma_sq_S * 12,
        expected_sigma_sq_S_times_12,
        decimal=5,
        err_msg=f"Sigma squared S times 12 for X2 does not match the expected value {expected_sigma_sq_S_times_12}"
    )

@pytest.mark.parametrize("expected_sigma_sq_S_vectorized_times_12", [
    0.0703125 * 12  # Based on manual calculation for the given P matrix
])
def test_sigma_sq_S_X2_vectorized_times_12(checkerboard_copula, expected_sigma_sq_S_vectorized_times_12):
    """
    Tests the vectorized calculation of sigma_sq_S for X2.
    """
    calculated_sigma_sq_S_vectorized = checkerboard_copula.calculate_sigma_sq_S_X2_vectorized()
    np.testing.assert_almost_equal(
        calculated_sigma_sq_S_vectorized * 12,
        expected_sigma_sq_S_vectorized_times_12,
        decimal=5,
        err_msg=f"Vectorized sigma squared S times 12 for X2 does not match the expected value {expected_sigma_sq_S_vectorized_times_12}"
    )
    
@pytest.mark.parametrize("expected_sigma_sq_S_times_12", [
    0.0791015625 * 12  # Based on manual calculation for the given P matrix
])
def test_sigma_sq_S_X1_times_12(checkerboard_copula, expected_sigma_sq_S_times_12):
    """
    Tests the calculation of sigma_sq_S for X1.
    """
    calculated_sigma_sq_S = checkerboard_copula.calculate_sigma_sq_S_X1()
    np.testing.assert_almost_equal(
        calculated_sigma_sq_S * 12,
        expected_sigma_sq_S_times_12,
        decimal=5,
        err_msg=f"Sigma squared S for X1 times 12 does not match the expected value {expected_sigma_sq_S_times_12}"
    )
    
@pytest.mark.parametrize("expected_sigma_sq_S_vectorized_times_12", [
    0.0791015625 * 12  # Based on manual calculation for the given P matrix
])
def test_sigma_sq_S_X1_vectorized_times_12(checkerboard_copula, expected_sigma_sq_S_vectorized_times_12):
    """
    Tests the vectorized calculation of sigma_sq_S for X1.
    """
    calculated_sigma_sq_S_vectorized = checkerboard_copula.calculate_sigma_sq_S_X1_vectorized()
    np.testing.assert_almost_equal(
        calculated_sigma_sq_S_vectorized * 12,
        expected_sigma_sq_S_vectorized_times_12,
        decimal=5,
        err_msg=f"Vectorized sigma squared S for X1 does not match the expected value {expected_sigma_sq_S_vectorized_times_12}"
    )
    
@pytest.mark.parametrize("expected_SCCRAM", [
    0.84375 / (12 * 0.0703125)  # Based on manual calculation for the given P matrix
])
def test_SCCRAM_X1_X2(checkerboard_copula, expected_SCCRAM):
    """
    Tests the calculation of the standardized Checkerboard Copula Regression Association Measure (SCCRAM).
    """
    calculated_SCCRAM = checkerboard_copula.calculate_SCCRAM_X1_X2()
    np.testing.assert_almost_equal(
        calculated_SCCRAM,
        expected_SCCRAM,
        decimal=5,
        err_msg=f"SCCRAM for X1 and X2 does not match the expected value {expected_SCCRAM}"
    )
    
@pytest.mark.parametrize("expected_SCCRAM_vectorized", [
    0.84375 / (12 * 0.0703125)  # Based on manual calculation for the given P matrix
])
def test_SCCRAM_X1_X2_vectorized(checkerboard_copula, expected_SCCRAM_vectorized):
    """
    Tests the vectorized calculation of the standardized Checkerboard Copula Regression Association Measure (SCCRAM).
    """
    calculated_SCCRAM_vectorized = checkerboard_copula.calculate_SCCRAM_X1_X2_vectorized()
    np.testing.assert_almost_equal(
        calculated_SCCRAM_vectorized,
        expected_SCCRAM_vectorized,
        decimal=5,
        err_msg=f"Vectorized SCCRAM for X1 and X2 does not match the expected value {expected_SCCRAM_vectorized}"
    )

@pytest.mark.parametrize("expected_SCCRAM", [
    0.0 / (0.0791015625 * 12)  # Based on manual calculation for the given P matrix
])
def test_SCCRAM_X2_X1(checkerboard_copula, expected_SCCRAM):
    """
    Tests the calculation of the standardized Checkerboard Copula Regression Association Measure (SCCRAM) for X2 --> X1.
    """
    calculated_SCCRAM = checkerboard_copula.calculate_SCCRAM_X2_X1()
    np.testing.assert_almost_equal(
        calculated_SCCRAM,
        expected_SCCRAM,
        decimal=5,
        err_msg=f"SCCRAM for X2 --> X1 does not match the expected value {expected_SCCRAM}"
    )
    
@pytest.mark.parametrize("expected_SCCRAM_vectorized", [
    0.0 / (0.0791015625 * 12)  # Based on manual calculation for the given P matrix
])
def test_SCCRAM_X2_X1_vectorized(checkerboard_copula, expected_SCCRAM_vectorized):
    """
    Tests the vectorized calculation of the standardized Checkerboard Copula Regression Association Measure (SCCRAM) for X2 --> X1.
    """
    calculated_SCCRAM_vectorized = checkerboard_copula.calculate_SCCRAM_X2_X1_vectorized()
    np.testing.assert_almost_equal(
        calculated_SCCRAM_vectorized,
        expected_SCCRAM_vectorized,
        decimal=5,
        err_msg=f"Vectorized SCCRAM for X2 --> X1 does not match the expected value {expected_SCCRAM_vectorized}"
    )
    
def test_copula_calculations_equivalence(contingency_table):
    """
    Test that calculations give same results whether initialized with 
    probability matrix or contingency table.
    """
    # Create copulas from probability matrix and contingency table
    P = contingency_table / contingency_table.sum()
    cop_from_P = CheckerboardCopula(P)
    cop_from_table = CheckerboardCopula.from_contingency_table(contingency_table)
    
    # Test various calculations
    np.testing.assert_array_almost_equal(
        cop_from_P.calculate_CCRAM_X1_X2(),
        cop_from_table.calculate_CCRAM_X1_X2()
    )
    np.testing.assert_array_almost_equal(
        cop_from_P.calculate_SCCRAM_X1_X2(),
        cop_from_table.calculate_SCCRAM_X1_X2()
    )
    np.testing.assert_array_almost_equal(
        cop_from_P.calculate_CCRAM_X2_X1(),
        cop_from_table.calculate_CCRAM_X2_X1()
    )
    np.testing.assert_array_almost_equal(
        cop_from_P.calculate_SCCRAM_X2_X1(),
        cop_from_table.calculate_SCCRAM_X2_X1()
    )
    
def test_get_predicted_category(checkerboard_copula):
    """Test get_predicted_category with various regression values."""
    test_cases = [
        # regression_value, marginal_cdf, expected_category
        (0.0, np.array([0, 0.25, 0.5, 1.0]), 0),  # At lower bound
        (0.25, np.array([0, 0.25, 0.5, 1.0]), 0),  # Below middle
        (0.5, np.array([0, 0.25, 0.5, 1.0]), 1),  # At boundary
        (0.75, np.array([0, 0.25, 0.5, 1.0]), 2),  # Above middle
        (1.0, np.array([0, 0.25, 0.5, 1.0]), 2),  # At upper bound
    ]
    
    for regression_value, cdf, expected in test_cases:
        predicted = checkerboard_copula.get_predicted_category(regression_value, cdf)
        assert predicted == expected, f"Failed for regression_value={regression_value}"

def test_get_predicted_category_batched(checkerboard_copula):
    """Test batched category prediction."""
    regression_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    marginal_cdf = np.array([0, 0.25, 0.5, 1.0])
    expected_categories = np.array([0, 0, 1, 2, 2])
    
    predicted = checkerboard_copula.get_predicted_category_batched(regression_values, marginal_cdf)
    np.testing.assert_array_equal(predicted, expected_categories)

@pytest.mark.parametrize("x1_category, expected_x2_category", [
    (0, 2),  # First category of X1 maps to third category of X2
    (1, 1),  # Second category of X1 maps to second category of X2  
    (2, 0),  # Third category of X1 maps to first category of X2
    (3, 1),  # Fourth category of X1 maps to second category of X2
    (4, 2),  # Fifth category of X1 maps to third category of X2
])
def test_predict_X2_from_X1(checkerboard_copula, x1_category, expected_x2_category):
    """Test prediction of X2 category from X1 category."""
    predicted = checkerboard_copula.predict_X2_from_X1(x1_category)
    assert predicted == expected_x2_category, f"Failed for X1 category {x1_category}"

def test_predict_X2_from_X1_batched(checkerboard_copula):
    """Test batched prediction of X2 from X1."""
    x1_categories = np.array([0, 1, 2, 3, 4])
    expected_x2_categories = np.array([2, 1, 0, 1, 2])
    
    predicted = checkerboard_copula.predict_X2_from_X1_batched(x1_categories)
    np.testing.assert_array_equal(predicted, expected_x2_categories)

def test_predict_X2_from_X1_invalid_category(checkerboard_copula):
    """Test prediction with invalid X1 category."""
    with pytest.raises(IndexError):
        checkerboard_copula.predict_X2_from_X1(5)  # Category 5 doesn't exist

def test_predict_X2_from_X1_batched_invalid(checkerboard_copula):
    """Test batched prediction with invalid X1 categories."""
    invalid_categories = np.array([0, 5, 2])  # Category 5 doesn't exist
    with pytest.raises(IndexError):
        checkerboard_copula.predict_X2_from_X1_batched(invalid_categories)

@pytest.mark.parametrize("x2_category, expected_x1_category", [
    (0, 2),  # First category of X2 maps to third category of X1
    (1, 2),  # Second category of X2 maps to mix of categories 2,4 of X1 -> median 2
    (2, 2),  # Third category of X2 maps to mix of categories 1,5 of X1 -> median 2
])
def test_predict_X1_from_X2(checkerboard_copula, x2_category, expected_x1_category):
    """Test prediction of X1 category from X2 category."""
    predicted = checkerboard_copula.predict_X1_from_X2(x2_category)
    assert predicted == expected_x1_category, f"Failed for X2 category {x2_category}"

def test_predict_X1_from_X2_batched(checkerboard_copula):
    """Test batched prediction of X1 from X2."""
    x2_categories = np.array([0, 1, 2])
    expected_x1_categories = np.array([2, 2, 2])
    
    predicted = checkerboard_copula.predict_X1_from_X2_batched(x2_categories)
    np.testing.assert_array_equal(predicted, expected_x1_categories)

def test_predict_X1_from_X2_invalid_category(checkerboard_copula):
    """Test prediction with invalid X2 category."""
    with pytest.raises(IndexError):
        checkerboard_copula.predict_X1_from_X2(3)  # Category 3 doesn't exist

def test_predict_X1_from_X2_batched_invalid(checkerboard_copula):
    """Test batched prediction with invalid X2 categories."""
    invalid_categories = np.array([0, 3, 1])  # Category 3 doesn't exist
    with pytest.raises(IndexError):
        checkerboard_copula.predict_X1_from_X2_batched(invalid_categories)

def test_prediction_consistency(checkerboard_copula):
    """Test that batched and individual predictions give same results."""
    x1_categories = np.array([0, 1, 2, 3, 4])
    x2_categories = np.array([0, 1, 2])
    
    # Test X2 from X1 predictions
    individual_x2_predictions = np.array([
        checkerboard_copula.predict_X2_from_X1(cat) 
        for cat in x1_categories
    ])
    batched_x2_predictions = checkerboard_copula.predict_X2_from_X1_batched(x1_categories)
    np.testing.assert_array_equal(individual_x2_predictions, batched_x2_predictions)
    
    # Test X1 from X2 predictions
    individual_x1_predictions = np.array([
        checkerboard_copula.predict_X1_from_X2(cat) 
        for cat in x2_categories
    ])
    batched_x1_predictions = checkerboard_copula.predict_X1_from_X2_batched(x2_categories)
    np.testing.assert_array_equal(individual_x1_predictions, batched_x1_predictions)

def test_prediction_special_cases(checkerboard_copula):
    """Test predictions for special cases like empty arrays."""
    
    # Single element arrays should work
    single_x1 = checkerboard_copula.predict_X2_from_X1_batched(np.array([0]))
    assert len(single_x1) == 1
    assert single_x1[0] == checkerboard_copula.predict_X2_from_X1(0)
    
    single_x2 = checkerboard_copula.predict_X1_from_X2_batched(np.array([0]))
    assert len(single_x2) == 1 
    assert single_x2[0] == checkerboard_copula.predict_X1_from_X2(0)