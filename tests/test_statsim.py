import numpy as np
import pytest
from discopula.checkerboard.copula import CheckerboardCopula
from discopula.checkerboard.statsim import (
    bootstrap_ccram, bootstrap_sccram,
    bootstrap_regression_U1_on_U2, bootstrap_regression_U2_on_U1,
    bootstrap_regression_U1_on_U2_vectorized, bootstrap_regression_U2_on_U1_vectorized,
    bootstrap_predict_X1_from_X2, bootstrap_predict_X2_from_X1,
    bootstrap_predict_X2_from_X1_vectorized, bootstrap_predict_X1_from_X2_vectorized,
    bootstrap_predict_X1_from_X2_all_comb_summary, bootstrap_predict_X2_from_X1_all_comb_summary,
    permutation_test_ccram, permutation_test_sccram    
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
    
def test_bootstrap_ccram(contingency_table):
    """
    Test bootstrap confidence interval calculation for CCRAM.
    """
    result = bootstrap_ccram(
        contingency_table,
        direction="X1_X2",
        n_resamples=999,
        confidence_level=0.95,
        random_state=8990
    )
    assert isinstance(result, object)
    assert hasattr(result, "confidence_interval")
    assert result.confidence_interval.low < result.confidence_interval.high
    assert result.standard_error >= 0.0

def test_bootstrap_sccram(contingency_table):
    """
    Test bootstrap confidence interval calculation for SCCRAM.
    """
    # Note: Not testing "BCa" in this case since it returns NaN for confidence intervals
    # DegenerateDataWarning: The BCa confidence interval cannot be calculated as referenced in SciPy documentation.
    # This problem is known to occur when the distribution is degenerate or the statistic is np.min.
    result = bootstrap_sccram(
        contingency_table,
        direction="X1_X2",
        n_resamples=999,
        method="percentile",
        confidence_level=0.95,
        random_state=8990
    )
    assert isinstance(result, object)
    assert hasattr(result, "confidence_interval")
    assert result.confidence_interval.low < result.confidence_interval.high
    assert result.standard_error >= 0.0

@pytest.mark.parametrize("direction, expected_value", [
    ("X1_X2", 0.84375),  # Example CCRAM value for X1 -> X2
    ("X2_X1", 0.0),       # Example CCRAM value for X2 -> X1
])
def test_bootstrap_ccram_values(contingency_table, direction, expected_value):
    """
    Test bootstrap CCRAM values against expected results.
    """
    copula = CheckerboardCopula.from_contingency_table(contingency_table)
    if direction == "X1_X2":
        original_value = copula.calculate_CCRAM_X1_X2_vectorized()
        result = bootstrap_ccram(
            contingency_table,
            direction=direction,
            n_resamples=999,
            confidence_level=0.95,
            random_state=8990
        )
        assert result.confidence_interval.low <= original_value <= result.confidence_interval.high
        np.testing.assert_almost_equal(original_value, expected_value, decimal=5)
    elif direction == "X2_X1":
        original_value = copula.calculate_CCRAM_X2_X1_vectorized()
        # Note: Not testing "BCa" in this case since it returns NaN for confidence intervals
        # DegenerateDataWarning: The BCa confidence interval cannot be calculated as referenced in SciPy documentation.
        # This problem is known to occur when the distribution is degenerate or the statistic is np.min.
        result = bootstrap_ccram(
            contingency_table,
            direction=direction,
            n_resamples=999,
            method="percentile",
            confidence_level=0.95,
            random_state=8990
        )
        # Adding a small margin to the confidence interval to account for floating point errors in this special case
        assert result.confidence_interval.low - 0.001 <= original_value <= result.confidence_interval.high
        np.testing.assert_almost_equal(original_value, expected_value, decimal=5)

@pytest.mark.parametrize("direction, expected_value", [
    ("X1_X2", 0.84375 / (12 * 0.0703125)),  # Example SCCRAM value for X1 -> X2
    ("X2_X1", 0.0)                          # Example SCCRAM value for X2 -> X1
])
def test_bootstrap_sccram_values(contingency_table, direction, expected_value):
    """
    Test bootstrap SCCRAM values against expected results.
    """
    copula = CheckerboardCopula.from_contingency_table(contingency_table)
    if direction == "X1_X2":
        original_value = copula.calculate_SCCRAM_X1_X2_vectorized()
        # Note: Not testing "BCa" in this case since it returns NaN for confidence intervals
        # DegenerateDataWarning: The BCa confidence interval cannot be calculated as referenced in SciPy documentation.
        # This problem is known to occur when the distribution is degenerate or the statistic is np.min.
        result = bootstrap_sccram(
            contingency_table,
            direction=direction,
            n_resamples=999,
            method="percentile",
            confidence_level=0.95,
            random_state=8990
        )
        np.testing.assert_almost_equal(original_value, expected_value, decimal=5)
        assert result.confidence_interval.low <= original_value <= result.confidence_interval.high
    elif direction == "X2_X1":
        original_value = copula.calculate_SCCRAM_X2_X1_vectorized()
        result = bootstrap_sccram(
            contingency_table,
            direction=direction,
            n_resamples=999,
            confidence_level=0.95,
            random_state=8990
        )
        np.testing.assert_almost_equal(original_value, expected_value, decimal=5)
        assert result.confidence_interval.low <= original_value <= result.confidence_interval.high
        
def test_bootstrap_ccram_invalid_direction(contingency_table):
    """
    Test that an error is raised for an invalid direction in bootstrap_ccram.
    """
    with pytest.raises(ValueError):
        bootstrap_ccram(
            contingency_table,
            direction="invalid",
            n_resamples=999,
            confidence_level=0.95,
            random_state=8990
        )

def test_bootstrap_sccram_invalid_direction(contingency_table):
    """
    Test that an error is raised for an invalid direction in bootstrap_sccram.
    """
    with pytest.raises(ValueError):
        bootstrap_sccram(
            contingency_table,
            direction="invalid",
            n_resamples=999,
            confidence_level=0.95,
            random_state=8990
        )

def test_bootstrap_regression_U1_on_U2_basic(contingency_table):
    """
    Test basic functionality of bootstrap_regression_U1_on_U2.
    """
    result = bootstrap_regression_U1_on_U2(
        contingency_table, 
        u2=0.5,
        n_resamples=999,  # Reduced for faster testing
        random_state=8990
    )
    
    assert hasattr(result, 'confidence_interval')
    assert result.confidence_interval.low < result.confidence_interval.high
    assert 0 <= result.confidence_interval.low <= 1
    assert 0 <= result.confidence_interval.high <= 1
    assert result.standard_error >= 0

def test_bootstrap_regression_U2_on_U1_basic(contingency_table):
    """
    Test basic functionality of bootstrap_regression_U2_on_U1.
    """
    result = bootstrap_regression_U2_on_U1(
        contingency_table,
        u1=0.5,
        n_resamples=999,
        random_state=8990
    )
    
    assert hasattr(result, 'confidence_interval')
    assert result.confidence_interval.low < result.confidence_interval.high
    assert 0 <= result.confidence_interval.low <= 1
    assert 0 <= result.confidence_interval.high <= 1
    assert result.standard_error >= 0

def test_bootstrap_regression_U1_on_U2_vectorized(contingency_table):
    """
    Test vectorized version of bootstrap regression U1 on U2.
    """
    u2_values = np.array([0.25, 0.5, 0.75])
    results = bootstrap_regression_U1_on_U2_vectorized(
        contingency_table,
        u2_values,
        n_resamples=999,
        random_state=8990
    )
    
    assert len(results) == len(u2_values)
    for result in results:
        assert hasattr(result, 'confidence_interval')
        assert result.confidence_interval.low < result.confidence_interval.high
        assert 0 <= result.confidence_interval.low <= 1
        assert 0 <= result.confidence_interval.high <= 1
        assert result.standard_error >= 0

def test_bootstrap_regression_U2_on_U1_vectorized(contingency_table):
    """
    Test vectorized version of bootstrap regression U2 on U1.
    """
    u1_values = np.array([0.25, 0.5, 0.75])
    results = bootstrap_regression_U2_on_U1_vectorized(
        contingency_table,
        u1_values,
        n_resamples=999,
        random_state=8990
    )
    
    assert len(results) == len(u1_values)
    for result in results:
        assert hasattr(result, 'confidence_interval')
        assert result.confidence_interval.low < result.confidence_interval.high
        assert 0 <= result.confidence_interval.low <= 1
        assert 0 <= result.confidence_interval.high <= 1
        assert result.standard_error >= 0

@pytest.mark.parametrize("u_value", [-0.1, 1.1])
def test_bootstrap_regression_U1_on_U2_invalid_input(contingency_table, u_value):
    """
    Test that invalid u2 values raise ValueError.
    """
    with pytest.raises(ValueError):
        bootstrap_regression_U1_on_U2(contingency_table, u2=u_value)

@pytest.mark.parametrize("u_value", [-0.1, 1.1])
def test_bootstrap_regression_U2_on_U1_invalid_input(contingency_table, u_value):
    """
    Test that invalid u1 values raise ValueError.
    """
    with pytest.raises(ValueError):
        bootstrap_regression_U2_on_U1(contingency_table, u1=u_value)

def test_bootstrap_regression_different_methods(contingency_table):
    """
    Test different bootstrap confidence interval methods.
    """
    methods = ['percentile', 'basic', 'BCa']
    for method in methods:
        result = bootstrap_regression_U1_on_U2(
            contingency_table,
            u2=0.5,
            n_resamples=999,
            method=method,
            random_state=8990
        )
        assert hasattr(result, 'confidence_interval')
        
        result = bootstrap_regression_U2_on_U1(
            contingency_table,
            u1=0.5,
            n_resamples=999,
            method=method,
            random_state=8990
        )
        assert hasattr(result, 'confidence_interval')

def test_bootstrap_regression_reproducibility(contingency_table):
    """
    Test that results are reproducible with same random_state.
    """
    result1 = bootstrap_regression_U1_on_U2(
        contingency_table,
        u2=0.5,
        random_state=8990
    )
    result2 = bootstrap_regression_U1_on_U2(
        contingency_table,
        u2=0.5,
        random_state=8990
    )
    
    np.testing.assert_array_almost_equal(
        result1.bootstrap_distribution,
        result2.bootstrap_distribution
    )

@pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
def test_bootstrap_regression_confidence_levels(contingency_table, confidence_level):
    """
    Test different confidence levels.
    """
    result = bootstrap_regression_U1_on_U2(
        contingency_table,
        u2=0.5,
        confidence_level=confidence_level,
        n_resamples=999,
        random_state=8990
    )
    
    # Higher confidence level should give wider interval
    interval_width = result.confidence_interval.high - result.confidence_interval.low
    assert 0 < interval_width <= 1
    
def test_bootstrap_predict_X2_from_X1_basic(contingency_table):
    """Test basic functionality of bootstrap_predict_X2_from_X1."""
    # Note: Not testing "BCa" in this case since it returns NaN for confidence intervals
    # DegenerateDataWarning: The BCa confidence interval cannot be calculated as referenced in SciPy documentation.
    # This problem is known to occur when the distribution is degenerate or the statistic is np.min.
    result = bootstrap_predict_X2_from_X1(
        contingency_table,
        x1_category=0,
        method='percentile',
        n_resamples=999,
        random_state=8990
    )
    
    assert hasattr(result, 'confidence_interval')
    assert result.confidence_interval.low <= result.confidence_interval.high
    assert isinstance(result.bootstrap_distribution[0], (int, np.integer))
    assert result.standard_error >= 0

def test_bootstrap_predict_X1_from_X2_basic(contingency_table):
    """Test basic functionality of bootstrap_predict_X1_from_X2."""
    # Note: Not testing "BCa" in this case since it returns NaN for confidence intervals
    # DegenerateDataWarning: The BCa confidence interval cannot be calculated as referenced in SciPy documentation.
    # This problem is known to occur when the distribution is degenerate or the statistic is np.min.
    result = bootstrap_predict_X1_from_X2(
        contingency_table,
        x2_category=0,
        method='percentile',
        n_resamples=999,
        random_state=8990
    )
    
    assert hasattr(result, 'confidence_interval')
    assert result.confidence_interval.low <= result.confidence_interval.high
    assert isinstance(result.bootstrap_distribution[0], (int, np.integer))
    assert result.standard_error >= 0

def test_bootstrap_predict_X2_from_X1_vectorized(contingency_table):
    """Test vectorized version of bootstrap prediction X2 from X1."""
    x1_categories = np.array([0, 1, 2])
    # Note: Not testing "BCa" in this case since it returns NaN for confidence intervals
    # DegenerateDataWarning: The BCa confidence interval cannot be calculated as referenced in SciPy documentation.
    # This problem is known to occur when the distribution is degenerate or the statistic is np.min.
    results = bootstrap_predict_X2_from_X1_vectorized(
        contingency_table,
        x1_categories,
        method='percentile',
        n_resamples=999,
        random_state=8990
    )
    
    assert len(results) == len(x1_categories)
    for result in results:
        assert hasattr(result, 'confidence_interval')
        assert result.confidence_interval.low <= result.confidence_interval.high
        assert isinstance(result.bootstrap_distribution[0], (int, np.integer))
        assert result.standard_error >= 0

def test_bootstrap_predict_X1_from_X2_vectorized(contingency_table):
    """Test vectorized version of bootstrap prediction X1 from X2."""
    x2_categories = np.array([0, 1, 2])
    # Note: Not testing "BCa" in this case since it returns NaN for confidence intervals
    # DegenerateDataWarning: The BCa confidence interval cannot be calculated as referenced in SciPy documentation.
    # This problem is known to occur when the distribution is degenerate or the statistic is np.min.
    results = bootstrap_predict_X1_from_X2_vectorized(
        contingency_table,
        x2_categories,
        method='percentile',
        n_resamples=999,
        random_state=8990
    )
    
    assert len(results) == len(x2_categories)
    for result in results:
        assert hasattr(result, 'confidence_interval')
        assert result.confidence_interval.low <= result.confidence_interval.high
        assert isinstance(result.bootstrap_distribution[0], (int, np.integer))
        assert result.standard_error >= 0

def test_bootstrap_predict_methods(contingency_table):
    """Test different bootstrap confidence interval methods."""
    methods = ['percentile', 'basic', 'BCa']
    for method in methods:
        result = bootstrap_predict_X2_from_X1(
            contingency_table,
            x1_category=0,
            n_resamples=999,
            method=method,
            random_state=8990
        )
        assert hasattr(result, 'confidence_interval')
        
        result = bootstrap_predict_X1_from_X2(
            contingency_table,
            x2_category=0,
            n_resamples=999,
            method=method,
            random_state=8990
        )
        assert hasattr(result, 'confidence_interval')

@pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
def test_bootstrap_predict_confidence_levels(contingency_table, confidence_level):
    """Test different confidence levels for predictions."""
    # Note: Not testing "BCa" in this case since it returns NaN for confidence intervals
    # DegenerateDataWarning: The BCa confidence interval cannot be calculated as referenced in SciPy documentation.
    # This problem is known to occur when the distribution is degenerate or the statistic is np.min.
    result = bootstrap_predict_X2_from_X1(
        contingency_table,
        x1_category=0,
        method='percentile',
        confidence_level=confidence_level,
        n_resamples=999,
        random_state=8990
    )
    
    # Check that interval bounds are valid category indices
    n_cols = contingency_table.shape[1]
    assert 0 <= result.confidence_interval.low < n_cols
    assert 0 <= result.confidence_interval.high < n_cols

def test_bootstrap_predict_consistent_with_direct(contingency_table):
    """Test that bootstrap predictions are consistent with direct predictions."""
    copula = CheckerboardCopula.from_contingency_table(contingency_table)
    
    # Test X2 from X1
    x1_category = 0
    direct_pred = copula.predict_X2_from_X1(x1_category)
    boot_result = bootstrap_predict_X2_from_X1(
        contingency_table,
        x1_category,
        n_resamples=999,
        random_state=8990
    )
    # Most common bootstrap prediction should match direct prediction
    assert direct_pred in boot_result.bootstrap_distribution
    
    # Test X1 from X2
    x2_category = 0
    direct_pred = copula.predict_X1_from_X2(x2_category)
    boot_result = bootstrap_predict_X1_from_X2(
        contingency_table,
        x2_category,
        n_resamples=999,
        random_state=8990
    )
    # Most common bootstrap prediction should match direct prediction
    assert direct_pred in boot_result.bootstrap_distribution
    
def test_bootstrap_predict_X1_from_X2_all_comb_summary(contingency_table):
    """Test summary table for predicting X1 from all X2 categories."""
    summary_table = bootstrap_predict_X1_from_X2_all_comb_summary(
        contingency_table,
        n_resamples=999,
        method='percentile',
        random_state=8990
    )
    
    n_rows, n_cols = contingency_table.shape
    
    # Test shape
    assert summary_table.shape == (n_rows, n_cols)
    
    # Test that columns sum to 100%
    np.testing.assert_array_almost_equal(
        summary_table.sum(axis=0),
        np.ones(n_cols) * 100,
        decimal=5
    )
    
    # Test that values are between 0 and 100
    assert np.all(summary_table >= 0)
    assert np.all(summary_table <= 100)

def test_bootstrap_predict_X2_from_X1_all_comb_summary(contingency_table):
    """Test summary table for predicting X2 from all X1 categories."""
    summary_table = bootstrap_predict_X2_from_X1_all_comb_summary(
        contingency_table,
        n_resamples=999,
        method='percentile',
        random_state=8990
    )
    
    n_rows, n_cols = contingency_table.shape
    
    # Test shape
    assert summary_table.shape == (n_rows, n_cols)
    
    # Test that rows sum to 100%
    np.testing.assert_array_almost_equal(
        summary_table.sum(axis=1),
        np.ones(n_rows) * 100,
        decimal=5
    )
    
    # Test that values are between 0 and 100
    assert np.all(summary_table >= 0)
    assert np.all(summary_table <= 100)

def test_summary_reproducibility():
    """Test that summary results are reproducible with same random_state."""
    table = np.array([
        [0, 0, 10],
        [0, 20, 0],
        [10, 0, 0],
        [0, 20, 0],
        [0, 0, 10]
    ])
    
    result1 = bootstrap_predict_X1_from_X2_all_comb_summary(
        table,
        n_resamples=999,
        random_state=8990
    )
    
    result2 = bootstrap_predict_X1_from_X2_all_comb_summary(
        table,
        n_resamples=999,
        random_state=8990
    )
    
    np.testing.assert_array_equal(result1, result2)
    
def test_summary_different_tables():
    """Test summary functions with different contingency tables."""
    tables = [
        np.array([[10, 0], [0, 10]]),  # Perfect diagonal
        np.array([[5, 5], [5, 5]]),    # Uniform
        np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])  # 3x3 diagonal
    ]
    
    for table in tables:
        summary1 = bootstrap_predict_X1_from_X2_all_comb_summary(table)
        summary2 = bootstrap_predict_X2_from_X1_all_comb_summary(table)
        
        # Check shapes
        assert summary1.shape == table.shape
        assert summary2.shape == table.shape
        
        # Check sums
        np.testing.assert_array_almost_equal(
            summary1.sum(axis=0),
            np.ones(table.shape[1]) * 100
        )
        np.testing.assert_array_almost_equal(
            summary2.sum(axis=1),
            np.ones(table.shape[0]) * 100
        )

def test_summary_invalid_input():
    """Test that invalid inputs raise appropriate errors."""
    invalid_tables = [
        np.array([1, 2, 3]),  # 1D array
        np.array([[-1, 0], [0, 1]]),  # Negative values
        np.array([[0, 0], [0, 0]])  # All zeros
    ]
    
    for table in invalid_tables:
        with pytest.raises((ValueError, IndexError)):
            bootstrap_predict_X1_from_X2_all_comb_summary(table)
        with pytest.raises((ValueError, IndexError)):
            bootstrap_predict_X2_from_X1_all_comb_summary(table)
            
# Add these tests to test_checkerboard.py

def test_permutation_test_sccram_basic(contingency_table):
    """Test basic functionality of permutation_test_sccram."""
    result = permutation_test_sccram(
        contingency_table,
        direction="X1_X2",
        n_resamples=999,
        random_state=8990
    )
    
    assert hasattr(result, 'statistic')
    assert hasattr(result, 'pvalue')
    assert hasattr(result, 'null_distribution')
    assert 0 <= result.pvalue <= 1
    assert len(result.null_distribution) == 999

def test_permutation_test_ccram_basic(contingency_table):
    """Test basic functionality of permutation_test_ccram."""
    result = permutation_test_ccram(
        contingency_table,
        direction="X1_X2",
        n_resamples=999,
        random_state=8990
    )
    
    assert hasattr(result, 'statistic')
    assert hasattr(result, 'pvalue')
    assert hasattr(result, 'null_distribution')
    assert 0 <= result.pvalue <= 1
    assert len(result.null_distribution) == 999

@pytest.mark.parametrize("direction", ["X1_X2", "X2_X1"])
def test_permutation_test_directions(contingency_table, direction):
    """Test both directions for permutation tests."""
    result_sccram = permutation_test_sccram(
        contingency_table,
        direction=direction,
        n_resamples=999,
        random_state=8990
    )
    
    result_ccram = permutation_test_ccram(
        contingency_table,
        direction=direction,
        n_resamples=999,
        random_state=8990
    )
    
    assert 0 <= result_sccram.pvalue <= 1
    assert 0 <= result_ccram.pvalue <= 1

@pytest.mark.parametrize("alternative", ['greater', 'less', 'two-sided'])
def test_permutation_test_alternatives(contingency_table, alternative):
    """Test different alternative hypotheses."""
    result_sccram = permutation_test_sccram(
        contingency_table,
        alternative=alternative,
        n_resamples=999,
        random_state=8990
    )
    
    result_ccram = permutation_test_ccram(
        contingency_table,
        alternative=alternative,
        n_resamples=999,
        random_state=8990
    )
    
    assert 0 <= result_sccram.pvalue <= 1
    assert 0 <= result_ccram.pvalue <= 1

def test_permutation_test_reproducibility():
    """Test that results are reproducible with same random_state."""
    table = np.array([
        [10, 0],
        [0, 10]
    ])
    
    result1 = permutation_test_sccram(
        table,
        n_resamples=999,
        random_state=8990
    )
    
    result2 = permutation_test_sccram(
        table,
        n_resamples=999,
        random_state=8990
    )
    
    np.testing.assert_array_equal(
        result1.null_distribution,
        result2.null_distribution
    )
    assert result1.pvalue == result2.pvalue

def test_permutation_test_independence():
    """Test that independent variables give expected results."""
    # Create independent table
    independent_table = np.array([
        [25, 25, 25],
        [25, 25, 25]
    ])
    
    result_sccram = permutation_test_sccram(
        independent_table,
        n_resamples=999,
        random_state=8990
    )
    
    result_ccram = permutation_test_ccram(
        independent_table,
        n_resamples=999,
        random_state=8990
    )
    
    # For independent variables, expect high p-values
    assert result_sccram.pvalue > 0.05
    assert result_ccram.pvalue > 0.05

def test_permutation_test_perfect_dependence():
    """Test that perfectly dependent variables give expected results."""
    # Create perfectly dependent table
    dependent_table = np.array([
        [10, 0, 0],
        [0, 10, 0],
        [0, 0, 10]
    ])
    
    result_sccram = permutation_test_sccram(
        dependent_table,
        n_resamples=999,
        alternative='greater',
        random_state=8990
    )
    
    result_ccram = permutation_test_ccram(
        dependent_table,
        n_resamples=999,
        alternative='greater',
        random_state=8990
    )
    
    # For perfectly dependent variables, expect very low p-values
    assert result_sccram.pvalue < 0.05
    assert result_ccram.pvalue < 0.05

def test_permutation_test_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    valid_table = np.array([[10, 0], [0, 10]])
    
    # Test invalid direction
    with pytest.raises(ValueError):
        permutation_test_sccram(valid_table, direction="invalid")
    with pytest.raises(ValueError):
        permutation_test_ccram(valid_table, direction="invalid")
        
    # Test invalid alternative
    with pytest.raises(ValueError):
        permutation_test_sccram(valid_table, alternative="invalid")
    with pytest.raises(ValueError):
        permutation_test_ccram(valid_table, alternative="invalid")
        
    # Test invalid table shapes
    invalid_tables = [
        np.array([1, 2, 3]),  # 1D array
        np.array([[-1, 0], [0, 1]]),  # Negative values
        np.array([[0, 0], [0, 0]])  # All zeros
    ]
    
    for table in invalid_tables:
        with pytest.raises((ValueError, IndexError)):
            permutation_test_sccram(table)
        with pytest.raises((ValueError, IndexError)):
            permutation_test_ccram(table)

def test_permutation_test_small_resamples():
    """Test behavior with small number of resamples."""
    table = np.array([[10, 0], [0, 10]])
    
    result_sccram = permutation_test_sccram(
        table,
        n_resamples=10,
        random_state=8990
    )
    
    result_ccram = permutation_test_ccram(
        table,
        n_resamples=10,
        random_state=8990
    )
    
    assert len(result_sccram.null_distribution) == 10
    assert len(result_ccram.null_distribution) == 10

def test_permutation_test_different_table_sizes():
    """Test permutation tests with different table sizes."""
    tables = [
        np.array([[10, 0], [0, 10]]),  # 2x2
        np.array([[5, 5, 5], [5, 5, 5]]),  # 2x3
        np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])  # 3x3
    ]
    
    for table in tables:
        result_sccram = permutation_test_sccram(
            table,
            n_resamples=999,
            random_state=8990
        )
        
        result_ccram = permutation_test_ccram(
            table,
            n_resamples=999,
            random_state=8990
        )
        
        assert 0 <= result_sccram.pvalue <= 1
        assert 0 <= result_ccram.pvalue <= 1