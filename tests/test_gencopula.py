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

# Basic Creation Tests
def test_from_contingency_table_valid(contingency_table):
    """Test valid contingency table initialization."""
    copula = GenericCheckerboardCopula.from_contingency_table(contingency_table)
    expected_P = contingency_table / contingency_table.sum()
    np.testing.assert_array_almost_equal(copula.P, expected_P)

@pytest.mark.parametrize("invalid_table,error_msg", [
    (np.array([[1, 2], [3, -1]]), "Contingency table cannot contain negative values"),
    (np.array([[0, 0], [0, 0]]), "Contingency table cannot be all zeros"),
    (np.array([1, 2, 3]), "Contingency table must be 2-dimensional")
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
@pytest.mark.parametrize("u, target_axis, given_axis, expected_value", [
    (0, 1, 0, 12/16),
    (3/8, 1, 0, 6/16),
    (1, 1, 0, 12/16)
])
def test_calculate_regression(generic_copula, u, target_axis, given_axis, expected_value):
    """Test regression calculation."""
    calculated = generic_copula._calculate_regression(
        target_axis=target_axis,
        given_axis=given_axis,
        given_value=u
    )
    np.testing.assert_almost_equal(calculated, expected_value)

# CCRAM Tests
@pytest.mark.parametrize("from_axis, to_axis, expected_ccram", [
    (0, 1, 0.84375),  # X1->X2
    (1, 0, 0.0)       # X2->X1
])
def test_calculate_CCRAM(generic_copula, from_axis, to_axis, expected_ccram):
    """Test CCRAM calculations."""
    calculated = generic_copula.calculate_CCRAM(from_axis, to_axis, is_scaled=False)
    np.testing.assert_almost_equal(calculated, expected_ccram)
    
# CCRAM Vectorized Tests
@pytest.mark.parametrize("from_axis, to_axis, expected_ccram", [
    (0, 1, 0.84375),  # X1->X2
    (1, 0, 0.0)       # X2->X1
])
def test_calculate_CCRAM_vectorized(generic_copula, from_axis, to_axis, expected_ccram):
    """Test vectorized CCRAM calculations."""
    calculated = generic_copula.calculate_CCRAM_vectorized(from_axis, to_axis, is_scaled=False)
    np.testing.assert_almost_equal(calculated, expected_ccram)

# SCCRAM Tests
@pytest.mark.parametrize("from_axis, to_axis, expected_sccram", [
    (0, 1, 0.84375/(12*0.0703125)),  # X1->X2
    (1, 0, 0.0)                      # X2->X1
])
def test_calculate_SCCRAM(generic_copula, from_axis, to_axis, expected_sccram):
    """Test SCCRAM calculations."""
    calculated = generic_copula.calculate_CCRAM(from_axis, to_axis, is_scaled=True)
    np.testing.assert_almost_equal(calculated, expected_sccram)
    
# SCCRAM Vectorized Tests
@pytest.mark.parametrize("from_axis, to_axis, expected_sccram", [
    (0, 1, 0.84375/(12*0.0703125)),  # X1->X2
    (1, 0, 0.0)                      # X2->X1
])
def test_calculate_SCCRAM_vectorized(generic_copula, from_axis, to_axis, expected_sccram):
    """Test vectorized SCCRAM calculations."""
    calculated = generic_copula.calculate_CCRAM_vectorized(from_axis, to_axis, is_scaled=True)
    np.testing.assert_almost_equal(calculated, expected_sccram)

# Category Prediction Tests
@pytest.mark.parametrize("source_category, from_axis, to_axis, expected_category", [
    (0, 0, 1, 2),  # First category of axis 0 maps to third category of axis 1
    (1, 0, 1, 1),  # Second category of axis 0 maps to second category of axis 1
    (2, 0, 1, 0),  # Third category of axis 0 maps to first category of axis 1
])
def test_predict_category(generic_copula, source_category, from_axis, to_axis, expected_category):
    """Test category prediction."""
    predicted = generic_copula._predict_category(source_category, from_axis, to_axis)
    assert predicted == expected_category

def test_predict_category_batched(generic_copula):
    """Test batched category prediction."""
    source_categories = np.array([0, 1, 2, 3, 4])
    expected = np.array([2, 1, 0, 1, 2])
    predicted = generic_copula._predict_category_batched(source_categories, 0, 1)
    np.testing.assert_array_equal(predicted, expected)

# Invalid Cases Tests
def test_invalid_predictions(generic_copula):
    """Test invalid prediction handling."""
    with pytest.raises(IndexError):
        generic_copula._predict_category(5, 0, 1)
    with pytest.raises(IndexError):
        generic_copula._predict_category_batched(np.array([0, 5, 2]), 0, 1)

# Special Cases Tests
def test_prediction_special_cases(generic_copula):
    """Test edge cases in predictions."""
    single_pred = generic_copula._predict_category_batched(np.array([0]), 0, 1)
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
    
def test_get_category_predictions_basic(generic_copula):
    """Test basic functionality of get_category_predictions."""
    df = generic_copula.get_category_predictions(0, 1)
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == generic_copula.P.shape[0]
    assert list(df.columns) == ['X Category', 'Predicted Y Category']
    
    # Check predictions match individual predictions
    for idx, row in df.iterrows():
        expected = generic_copula._predict_category(idx, 0, 1)
        assert row['Predicted Y Category'] == expected

def test_get_category_predictions_custom_names(generic_copula):
    """Test get_category_predictions with custom axis names."""
    df = generic_copula.get_category_predictions(
        0, 1, 
        from_axis_name="Income", 
        to_axis_name="Education"
    )
    assert list(df.columns) == ['Income Category', 'Predicted Education Category']

def test_get_category_predictions_known_values(generic_copula):
    """Test get_category_predictions against known mappings."""
    expected_predictions = {
        0: 2,
        1: 1,
        2: 0, 
        3: 1, 
        4: 2 
    }
    
    df = generic_copula.get_category_predictions(0, 1)
    for source_cat, predicted_cat in expected_predictions.items():
        assert df.iloc[source_cat]['Predicted Y Category'] == predicted_cat

def test_get_category_predictions_invalid_axes(generic_copula):
    """Test get_category_predictions with invalid axes."""
    with pytest.raises(IndexError):
        generic_copula.get_category_predictions(2, 1)  # Invalid from_axis
    with pytest.raises(IndexError):
        generic_copula.get_category_predictions(0, 2)  # Invalid to_axis