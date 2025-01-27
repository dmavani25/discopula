import numpy as np
import pandas as pd
import pytest
from discopula import GenericCheckerboardCopula

@pytest.fixture
def generic_copula():
    """Fixture providing a GenericCheckerboardCopula instance with test data."""
    P = np.array([
        [0, 0, 2/8],
        [0, 1/8, 0],
        [2/8, 0, 0],
        [0, 1/8, 0],
        [0, 0, 2/8]
    ])
    return GenericCheckerboardCopula(P)

@pytest.fixture
def contingency_table():
    """Fixture providing a test contingency table."""
    return np.array([
        [0, 0, 20],
        [0, 10, 0],
        [20, 0, 0],
        [0, 10, 0],
        [0, 0, 20]
    ])
    
@pytest.fixture
def table_4d():
    """Fixture for 4D contingency table."""
    table = np.zeros((2,3,2,6), dtype=int)
    
    # RDA Row 1 [0,2,0,*]
    table[0,2,0,1] = 1
    table[0,2,0,4] = 2
    table[0,2,0,5] = 4
    
    # RDA Row 2 [0,2,1,*]
    table[0,2,1,3] = 1
    table[0,2,1,4] = 3
    
    # RDA Row 3 [0,1,0,*]
    table[0,1,0,1] = 2
    table[0,1,0,2] = 3
    table[0,1,0,4] = 6
    table[0,1,0,5] = 6
    
    # RDA Row 4 [0,1,1,*]
    table[0,1,1,1] = 1
    table[0,1,1,3] = 2
    table[0,1,1,5] = 1
    
    # RDA Row 5 [0,0,0,*]
    table[0,0,0,4] = 2 
    table[0,0,0,5] = 2
    
    # RDA Row 6 [0,0,1,*]
    table[0,0,1,2] = 1
    table[0,0,1,3] = 1
    table[0,0,1,4] = 3
    
    # RDA Row 7 [1,2,0,*]
    table[1,2,0,2] = 3
    table[1,2,0,4] = 1
    table[1,2,0,5] = 2
    
    # RDA Row 8 [1,2,1,*]
    table[1,2,1,1] = 1
    table[1,2,1,4] = 3
    
    # RDA Row 9 [1,1,0,*]
    table[1,1,0,1] = 3
    table[1,1,0,2] = 4
    table[1,1,0,3] = 5
    table[1,1,0,4] = 6
    table[1,1,0,5] = 2
    
    # RDA Row 10 [1,1,1,*]
    table[1,1,1,0] = 1
    table[1,1,1,1] = 4
    table[1,1,1,2] = 4
    table[1,1,1,3] = 3
    table[1,1,1,5] = 1
    
    # RDA Row 11 [1,0,0,*]
    table[1,0,0,0] = 2
    table[1,0,0,1] = 2
    table[1,0,0,2] = 1
    table[1,0,0,3] = 5
    table[1,0,0,4] = 2
    
    # RDA Row 12 [1,0,1,*]
    table[1,0,1,0] = 2
    table[1,0,1,2] = 2
    table[1,0,1,3] = 3
    
    return table

@pytest.fixture
def cases_4d():
    """Fixture for 4D case-form data."""
    return np.array([
        # RDA Row 1
        [0,2,0,1],[0,2,0,4],[0,2,0,4],
        [0,2,0,5], [0,2,0,5],[0,2,0,5],[0,2,0,5],
        # RDA Row 2
        [0,2,1,3],[0,2,1,4],[0,2,1,4],[0,2,1,4],
        # RDA Row 3
        [0,1,0,1],[0,1,0,1],[0,1,0,2],[0,1,0,2],[0,1,0,2],
        [0,1,0,4],[0,1,0,4],[0,1,0,4],[0,1,0,4],[0,1,0,4],[0,1,0,4],
        [0,1,0,5],[0,1,0,5],[0,1,0,5],[0,1,0,5],[0,1,0,5],[0,1,0,5],
        # RDA Row 4
        [0,1,1,1],[0,1,1,3],[0,1,1,3],[0,1,1,5],
        # RDA Row 5
        [0,0,0,4],[0,0,0,4],[0,0,0,5],[0,0,0,5],
        # RDA Row 6
        [0,0,1,2],[0,0,1,3],[0,0,1,4],[0,0,1,4],[0,0,1,4],
        # RDA Row 7
        [1,2,0,2],[1,2,0,2],[1,2,0,2],[1,2,0,4],[1,2,0,5],[1,2,0,5],
        # RDA Row 8
        [1,2,1,1],[1,2,1,4],[1,2,1,4],[1,2,1,4],
        # RDA Row 9
        [1,1,0,1],[1,1,0,1],[1,1,0,1],[1,1,0,2],[1,1,0,2],[1,1,0,2],[1,1,0,2],
        [1,1,0,3],[1,1,0,3],[1,1,0,3],[1,1,0,3],[1,1,0,3],
        [1,1,0,4],[1,1,0,4],[1,1,0,4],[1,1,0,4],[1,1,0,4],[1,1,0,4],
        [1,1,0,5],[1,1,0,5],
        # RDA Row 10
        [1,1,1,0],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],
        [1,1,1,2],[1,1,1,2],[1,1,1,2],[1,1,1,2],
        [1,1,1,3],[1,1,1,3],[1,1,1,3],[1,1,1,5],
        # RDA Row 11
        [1,0,0,0],[1,0,0,0],[1,0,0,1],[1,0,0,1],[1,0,0,2],
        [1,0,0,3],[1,0,0,3],[1,0,0,3],[1,0,0,3],[1,0,0,3],
        [1,0,0,4],[1,0,0,4],
        # RDA Row 12
        [1,0,1,0],[1,0,1,0],[1,0,1,2],[1,0,1,2],
        [1,0,1,3],[1,0,1,3],[1,0,1,3]
    ])
    
@pytest.fixture
def expected_shape():
    """Fixture providing expected shape for the copula."""
    return (2, 3, 2, 6)

# Basic Creation Tests
def test_from_contingency_table_valid(contingency_table):
    """Test valid contingency table initialization."""
    copula = GenericCheckerboardCopula.from_contingency_table(contingency_table)
    expected_P = contingency_table / contingency_table.sum()
    np.testing.assert_array_almost_equal(copula.P, expected_P)

@pytest.mark.parametrize("invalid_table,error_msg", [
    (np.array([[1, 2], [3, -1]]), "Contingency table cannot contain negative values"),
    (np.array([[0, 0], [0, 0]]), "Contingency table cannot be all zeros"),
])
def test_invalid_contingency_tables(invalid_table, error_msg):
    """Test error handling for invalid contingency tables."""
    with pytest.raises(ValueError, match=error_msg):
        GenericCheckerboardCopula.from_contingency_table(invalid_table)

# Marginal Distribution Tests
@pytest.mark.parametrize("expected_cdf_0, expected_cdf_1", [
    ([0, 2/8, 3/8, 5/8, 6/8, 1], [0, 2/8, 4/8, 1])
])
def test_marginal_cdfs(generic_copula, expected_cdf_0, expected_cdf_1):
    """Test marginal CDF calculations."""
    np.testing.assert_almost_equal(generic_copula.marginal_cdfs[0], expected_cdf_0)
    np.testing.assert_almost_equal(generic_copula.marginal_cdfs[1], expected_cdf_1)

# Conditional PMF Tests
def test_conditional_pmfs(generic_copula):
    """Test conditional PMF calculations."""
    expected_1_given_0 = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    pmf = generic_copula._calculate_conditional_pmf(1, [0])
    np.testing.assert_array_almost_equal(pmf, expected_1_given_0)

# Regression Tests
@pytest.mark.parametrize("given_values, given_axes, target_axis, expected_value", [
    # Single conditioning
    ([0.0], [0], 1, 12/16),
    ([3/8], [0], 1, 6/16),
    ([1.0], [0], 1, 12/16),
])
def test_calculate_regression(generic_copula, given_values, given_axes, target_axis, expected_value):
    """Test regression calculation with multiple conditioning axes."""
    calculated = generic_copula._calculate_regression_batched(
        target_axis=target_axis,
        given_axes=given_axes,
        given_values=given_values
    )
    np.testing.assert_almost_equal(calculated, expected_value)

# CCRAM Tests
@pytest.mark.parametrize("from_axes, to_axis, expected_ccram", [
    ([0], 1, 0.84375),          # Single axis X1->X2
    ([1], 0, 0.0),              # Single axis X2->X1
])
def test_calculate_CCRAM(generic_copula, from_axes, to_axis, expected_ccram):
    """Test CCRAM calculations with multiple conditioning axes."""
    calculated = generic_copula.calculate_CCRAM(from_axes, to_axis, scaled=False)
    np.testing.assert_almost_equal(calculated, expected_ccram)
    
# CCRAM Vectorized Tests
@pytest.mark.parametrize("from_axes, to_axis, expected_ccram", [
    ([0], 1, 0.84375),          # Single axis X1->X2
    ([1], 0, 0.0),              # Single axis X2->X1
])
def test_calculate_CCRAM_vectorized(generic_copula, from_axes, to_axis, expected_ccram):
    """Test vectorized CCRAM calculations with multiple conditioning axes."""
    calculated = generic_copula.calculate_CCRAM_vectorized(from_axes, to_axis, scaled=False)
    np.testing.assert_almost_equal(calculated, expected_ccram)

# SCCRAM Tests
@pytest.mark.parametrize("from_axes, to_axis, expected_sccram", [
    ([0], 1, 0.84375/(12*0.0703125)),          # Single axis X1->X2
    ([1], 0, 0.0),                             # Single axis X2->X1
])
def test_calculate_SCCRAM(generic_copula, from_axes, to_axis, expected_sccram):
    """Test SCCRAM calculations with multiple conditioning axes."""
    calculated = generic_copula.calculate_CCRAM(from_axes, to_axis, scaled=True)
    np.testing.assert_almost_equal(calculated, expected_sccram)
    
# SCCRAM Vectorized Tests
@pytest.mark.parametrize("from_axes, to_axis, expected_sccram", [
    ([0], 1, 0.84375/(12*0.0703125)),          # Single axis X1->X2
    ([1], 0, 0.0),                             # Single axis X2->X1
])
def test_calculate_SCCRAM_vectorized(generic_copula, from_axes, to_axis, expected_sccram):
    """Test vectorized SCCRAM calculations with multiple conditioning axes."""
    calculated = generic_copula.calculate_CCRAM_vectorized(from_axes, to_axis, scaled=True)
    np.testing.assert_almost_equal(calculated, expected_sccram)

# Category Prediction Tests
@pytest.mark.parametrize("source_categories, from_axes, to_axis, expected_categories", [
    # Single axis prediction
    ([0], [0], 1, [2]),
    ([1], [0], 1, [1]),
    ([2], [0], 1, [0]),
    ([3], [0], 1, [1]),
    ([4], [0], 1, [2]),     
])
def test_predict_category_multi(generic_copula, source_categories, from_axes, to_axis, expected_categories):
    """Test category prediction with multiple conditioning axes."""
    predicted = generic_copula._predict_category_batched_multi(
        source_categories=source_categories,
        from_axes=from_axes,
        to_axis=to_axis
    )
    np.testing.assert_array_equal(predicted, expected_categories)

# Add Multi-axis Category Predictions Test
def test_get_category_predictions_multi(generic_copula):
    """Test category predictions with multiple conditioning axes."""
    df = generic_copula.get_category_predictions_multi(
        from_axes=[1],
        to_axis=0,
        axis_names={0: "Income", 1: "Education"}
    )
    
    assert isinstance(df, pd.DataFrame)
    assert "Predicted Income Category" in df.columns
    assert "Education Category" in df.columns

# Add Consistency Tests for Multi-axis
def test_multi_axis_consistency(generic_copula):
    """Test consistency between single and multiple axis calculations."""
    single_axis = generic_copula.calculate_CCRAM([0], 1)
    multi_axis = generic_copula.calculate_CCRAM_vectorized([0], 1)
    np.testing.assert_almost_equal(single_axis, multi_axis)

# Invalid Cases Tests
def test_invalid_predictions(generic_copula):
    """Test invalid prediction handling."""
    with pytest.raises(IndexError):
        generic_copula._predict_category(5, 0, 1)

# Special Cases Tests
def test_prediction_special_cases(generic_copula):
    """Test edge cases in predictions."""
    single_pred = generic_copula._predict_category_batched_multi(np.array([0]), 0, 1)
    assert len(single_pred) == 1
    assert single_pred[0] == generic_copula._predict_category(0, 0, 1)

# Consistency Tests
def test_calculation_consistency(contingency_table):
    """Test consistency across different initialization methods."""
    P = contingency_table / contingency_table.sum()
    cop1 = GenericCheckerboardCopula(P)
    cop2 = GenericCheckerboardCopula.from_contingency_table(contingency_table)
    
    np.testing.assert_array_almost_equal(
        cop1.calculate_CCRAM(0, 1),
        cop2.calculate_CCRAM(0, 1)
    )

def test_vectorized_consistency(generic_copula):
    """Test consistency between vectorized and non-vectorized methods."""
    regular = generic_copula.calculate_CCRAM(0, 1)
    vectorized = generic_copula.calculate_CCRAM_vectorized(0, 1)
    np.testing.assert_almost_equal(regular, vectorized)
        
def test_calculate_scores_valid(generic_copula):
    """Test valid calculation of scores."""
    scores_0 = generic_copula.calculate_scores(0)
    scores_1 = generic_copula.calculate_scores(1)

    # Check exact expected values
    expected_scores_0 = np.array([0.125, 0.3125, 0.5, 0.6875, 0.875], dtype=np.float64)
    expected_scores_1 = np.array([0.125, 0.375, 0.75], dtype=np.float64)
    
    np.testing.assert_array_almost_equal(scores_0, expected_scores_0)
    np.testing.assert_array_almost_equal(scores_1, expected_scores_1)
    
def test_calculate_scores_invalid_axis(generic_copula):
    """Test invalid axis handling for score calculation."""
    with pytest.raises(KeyError):
        generic_copula.calculate_scores(2)  # Invalid axis index

def test_calculate_variance_S_valid(generic_copula):
    """Test valid calculation of score variance."""
    var_0 = generic_copula.calculate_variance_S(0)
    var_1 = generic_copula.calculate_variance_S(1)
    
    # Check return type
    assert isinstance(var_0, (float, np.float64))
    assert isinstance(var_1, (float, np.float64))
    
    # Variance should be non-negative
    assert var_0 >= 0
    assert var_1 >= 0
    
    # Check exact expected values
    expected_var_0, expected_var_1 = 0.0791015625, 0.0703125
    np.testing.assert_almost_equal(var_0, expected_var_0)
    np.testing.assert_almost_equal(var_1, expected_var_1)
    

def test_calculate_variance_S_invalid_axis(generic_copula):
    """Test invalid axis handling for variance calculation."""
    with pytest.raises(KeyError):
        generic_copula.calculate_variance_S(2)  # Invalid axis index

def test_from_cases_creation(cases_4d, table_4d, expected_shape):
    """Test creation of copula from cases data."""
    cop = GenericCheckerboardCopula.from_cases(cases_4d, expected_shape)
    assert cop.ndim == 4
    assert cop.P.shape == expected_shape
    assert np.all(cop.P >= 0)
    assert np.isclose(cop.P.sum(), 1.0)
    assert np.all(cop.contingency_table == table_4d)

def test_from_cases_marginal_pdfs(cases_4d, expected_shape):
    """Test marginal PDFs calculation from cases."""
    cop = GenericCheckerboardCopula.from_cases(cases_4d, expected_shape)
    
    # Test marginal PDFs exist for each dimension
    assert len(cop.marginal_pdfs) == 4
    
    # Test each marginal PDF sums to 1
    for axis in range(4):
        pdf = cop.marginal_pdfs[axis]
        assert np.isclose(np.sum(pdf), 1.0)

def test_from_cases_marginal_cdfs(cases_4d, expected_shape):
    """Test marginal CDFs calculation from cases."""
    cop = GenericCheckerboardCopula.from_cases(cases_4d, expected_shape)
    
    # Test CDFs exist for each dimension
    assert len(cop.marginal_cdfs) == 4
    
    # Test CDF properties
    for axis in range(4):
        cdf = cop.marginal_cdfs[axis]
        assert cdf[0] == 0  # CDF starts at 0
        assert np.isclose(cdf[-1], 1.0)  # CDF ends at 1
        assert np.all(np.diff(cdf) >= 0)  # CDF is monotonically increasing

def test_from_cases_scores(cases_4d, expected_shape):
    """Test scores calculation from cases."""
    cop = GenericCheckerboardCopula.from_cases(cases_4d, expected_shape)
    
    # Test scores exist for each dimension
    for axis in range(4):
        scores = cop.calculate_scores(axis)
        assert len(scores) == expected_shape[axis]

def test_from_cases_variance(cases_4d, expected_shape):
    """Test variance calculation from cases."""
    cop = GenericCheckerboardCopula.from_cases(cases_4d, expected_shape)
    
    # Test variance for the last dimension
    variance = cop.calculate_variance_S(3)
    assert isinstance(variance, (float, np.float64))
    assert variance >= 0

def test_from_cases_invalid_input():
    """Test error handling for invalid inputs."""
    invalid_cases = np.array([[0,1], [1,2]])  # Wrong number of dimensions
    invalid_shape = (2, 2)  # Wrong shape specification
    
    with pytest.raises(ValueError):
        GenericCheckerboardCopula.from_cases(invalid_cases, (2,2,2,2))
    
    with pytest.raises(ValueError):
        GenericCheckerboardCopula.from_cases(cases_4d, invalid_shape)

def test_from_cases_contingency_table(cases_4d, expected_shape):
    """Test contingency table properties from cases."""
    cop = GenericCheckerboardCopula.from_cases(cases_4d, expected_shape)
    
    # Test contingency table properties
    table = cop.contingency_table
    assert table.shape == expected_shape
    assert np.all(table >= 0)  # Non-negative counts
    assert np.sum(table) == len(cases_4d)  # Sum equals number of cases
    
def test_4d_ccram_calculations(cases_4d, expected_shape):
    """Test CCRAM calculations for 4D case with multiple conditioning axes."""
    cop = GenericCheckerboardCopula.from_cases(cases_4d, expected_shape)
    
    # Test various axis combinations
    test_cases = [
        ([0], 1, 0.015560174514475982),
        ([0, 1], 2, 0.03214055500887074),
        ([0, 1, 2], 3, 0.2663199625336037),
        ([1, 2, 3], 0, 0.7325390169591219)
    ]
    
    for from_axes, to_axis, expected in test_cases:
        # Regular CCRAM
        ccram = cop.calculate_CCRAM(from_axes, to_axis, scaled=False)
        assert 0 <= ccram <= 1
        print(from_axes, to_axis, ccram)
        assert np.isclose(ccram, expected)
        
        # Vectorized CCRAM
        ccram_vec = cop.calculate_CCRAM_vectorized(from_axes, to_axis, scaled=False)
        np.testing.assert_almost_equal(ccram, ccram_vec)

def test_4d_prediction_multi(cases_4d, expected_shape):
    """Test multi-axis prediction for 4D case."""
    cop = GenericCheckerboardCopula.from_cases(cases_4d, expected_shape)
    
    # Test various prediction scenarios
    test_cases = [
        # source_categories, from_axes, to_axis
        ([0], [0], 1),
        ([0, 1], [0, 1], 2),
        ([0, 1, 0], [0, 1, 2], 3)
    ]
    
    for source_cats, from_axes, to_axis in test_cases:
        predicted = cop._predict_category_batched_multi(
            source_categories=source_cats,
            from_axes=from_axes,
            to_axis=to_axis
        )
        assert isinstance(predicted, np.ndarray)
        assert predicted.shape == (1,)
        assert 0 <= predicted[0] < expected_shape[to_axis]

def test_4d_conditional_pmf(cases_4d, expected_shape):
    """Test conditional PMF calculations for 4D case."""
    cop = GenericCheckerboardCopula.from_cases(cases_4d, expected_shape)
    
    # Test various conditioning combinations
    test_cases = [
        (0, [1]),             # Single conditioning
        (0, [1, 2])      # Triple conditioning
    ]
    
    for target, given_axes in test_cases:
        pmf = cop._calculate_conditional_pmf(target, given_axes)
        assert isinstance(pmf, np.ndarray)
        print(pmf.sum(axis=target))
        assert np.allclose(pmf.sum(axis=target), 1.0)  # Proper conditional distribution

def test_4d_scores_expected_values(cases_4d, expected_shape):
    """Test score calculations for 4D case."""
    cop = GenericCheckerboardCopula.from_cases(cases_4d, expected_shape)
    
    for axis in range(4):
        scores = cop.calculate_scores(axis)
        assert len(scores) == expected_shape[axis]
        assert np.all(0 <= np.array(scores)) and np.all(np.array(scores) <= 1)
        # Scores should be monotonically increasing
        assert np.all(np.diff(scores) >= 0)

def test_4d_category_predictions_dataframe(cases_4d, expected_shape):
    """Test category predictions output format for 4D case."""
    cop = GenericCheckerboardCopula.from_cases(cases_4d, expected_shape)
    
    axis_names = {
        0: "First",
        1: "Second", 
        2: "Third",
        3: "Fourth"
    }
    
    from_axes = [0,1,2]
    to_axis = 3
    
    df = cop.get_category_predictions_multi(
        from_axes=from_axes,
        to_axis=to_axis,
        axis_names=axis_names
    )
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    for axis in from_axes:
        assert f"{axis_names[axis]} Category" in df.columns
    assert f"Predicted {axis_names[to_axis]} Category" in df.columns