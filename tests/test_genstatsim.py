import numpy as np
import pytest
from io import StringIO
import sys
from discopula import bootstrap_ccram, permutation_test_ccram, bootstrap_predict_category_summary, display_prediction_summary
from discopula.checkerboard.genstatsim import (
    _bootstrap_predict_category,
    _bootstrap_predict_category_vectorized,
)

@pytest.fixture
def contingency_table():
    """Fixture to create a sample contingency table."""
    return np.array([
        [0, 0, 20],
        [0, 10, 0],
        [20, 0, 0],
        [0, 10, 0],
        [0, 0, 20]
    ])

def test_bootstrap_ccram_basic(contingency_table):
    """Test basic functionality of bootstrap_ccram."""
    result = bootstrap_ccram(
        contingency_table,
        from_axis=0,
        to_axis=1,
        n_resamples=999,
        random_state=8990
    )
    
    assert hasattr(result, "confidence_interval")
    assert hasattr(result, "bootstrap_distribution")
    assert hasattr(result, "standard_error")
    assert hasattr(result, "histogram_fig")
    assert result.confidence_interval[0] < result.confidence_interval[1]
    assert result.standard_error >= 0

def test_bootstrap_ccram_scaled(contingency_table):
    """Test bootstrap_ccram with scaling."""
    result = bootstrap_ccram(
        contingency_table,
        from_axis=0,
        to_axis=1,
        is_scaled=True,
        n_resamples=9999,
        random_state=8990
    )
    
    assert "SCCRAM" in result.metric_name
    assert hasattr(result, "confidence_interval")
    assert result.confidence_interval[0] < result.confidence_interval[1]

@pytest.mark.parametrize("from_axis,to_axis,expected_metric", [
    (0, 1, "CCRAM 0->1"),
    (1, 0, "CCRAM 1->0")
])
def test_bootstrap_ccram_directions(contingency_table, from_axis, to_axis, expected_metric):
    """Test different directional calculations."""
    result = bootstrap_ccram(
        contingency_table,
        from_axis=from_axis,
        to_axis=to_axis,
        n_resamples=999,
        random_state=8990
    )
    
    assert result.metric_name == expected_metric
    assert hasattr(result, "confidence_interval")
    assert result.confidence_interval[0] < result.confidence_interval[1]

@pytest.fixture
def simple_table():
    """Simple 2x2 contingency table."""
    return np.array([[10, 0], [0, 10]])

@pytest.fixture
def complex_table():
    """More complex 3x3 contingency table."""
    return np.array([
        [10, 0, 0],
        [0, 10, 0],
        [0, 0, 10]
    ])

def test_bootstrap_predict_category_basic(simple_table):
    """Test basic functionality of bootstrap category prediction."""
    result = _bootstrap_predict_category(
        simple_table,
        source_category=0,
        from_axis=0,
        to_axis=1,
        n_resamples=999,
        random_state=42
    )
    
    assert hasattr(result, 'confidence_interval')
    assert hasattr(result, 'bootstrap_distribution')
    assert len(result.bootstrap_distribution) == 999
    assert all(x in [0, 1] for x in result.bootstrap_distribution)

def test_bootstrap_predict_category_vectorized(complex_table):
    """Test vectorized predictions for multiple categories."""
    results = _bootstrap_predict_category_vectorized(
        complex_table,
        source_categories=np.array([0, 1]),
        from_axis=0,
        to_axis=1,
        n_resamples=999,
        random_state=42
    )
    
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(len(r.bootstrap_distribution) == 999 for r in results)

def test_bootstrap_predict_category_summary(complex_table):
    """Test summary table generation."""
    summary = bootstrap_predict_category_summary(
        complex_table,
        from_axis=0,
        to_axis=1,
        n_resamples=999,
        random_state=42
    )
    
    assert isinstance(summary, np.ndarray)
    assert summary.shape == (3, 3)
    assert np.all(summary >= 0)
    assert np.all(summary <= 100)
    assert np.allclose(np.sum(summary, axis=0), [100, 100, 100], atol=1e-10)

def test_display_prediction_summary(complex_table):
    """Test summary display formatting."""
    summary = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 100]])
    
    # Capture stdout
    stdout = StringIO()
    sys.stdout = stdout
    
    display_prediction_summary(summary, "A", "B")
    
    sys.stdout = sys.__stdout__
    output = stdout.getvalue()
    
    assert "Prediction Summary" in output
    assert "From A to B:" in output
    assert "A=0" in output
    assert "B=0" in output
    assert "100" in output

def test_reproducibility_for_pred(simple_table):
    """Test result reproducibility with same random state."""
    result1 = _bootstrap_predict_category(
        simple_table,
        source_category=0,
        from_axis=0,
        to_axis=1,
        random_state=42
    )
    
    result2 = _bootstrap_predict_category(
        simple_table,
        source_category=0,
        from_axis=0,
        to_axis=1,
        random_state=42
    )
    
    np.testing.assert_array_equal(
        result1.bootstrap_distribution,
        result2.bootstrap_distribution
    )

def test_invalid_inputs_for_pred():
    """Test handling of invalid inputs."""
    valid_table = np.array([[10, 0], [0, 10]])
    
    # Test invalid axes
    with pytest.raises((ValueError, IndexError, KeyError)):
        _bootstrap_predict_category(valid_table, 0, 2, 1)
    
    # Test invalid category
    with pytest.raises((ValueError, IndexError)):
        _bootstrap_predict_category(valid_table, 5, 0, 1)
    
    # Test invalid table shapes
    invalid_tables = [
        np.array([1, 2]),  # 1D
        np.array([[-1, 0], [0, 1]]),  # Negative values
        np.array([[0, 0], [0, 0]])  # All zeros
    ]
    
    for table in invalid_tables:
        with pytest.raises((ValueError, IndexError)):
            _bootstrap_predict_category(table, 0, 0, 1)

def test_confidence_levels(simple_table):
    """Test different confidence levels."""
    levels = [0.90, 0.95, 0.99]
    
    for level in levels:
        result = _bootstrap_predict_category(
            simple_table,
            source_category=0,
            from_axis=0,
            to_axis=1,
            confidence_level=level,
            random_state=42
        )
        
        ci = result.confidence_interval
        assert ci.low <= ci.high
        assert 0 <= ci.low <= 1
        assert 0 <= ci.high <= 1

def test_different_methods(simple_table):
    """Test different bootstrap methods."""
    methods = ['percentile', 'basic']
    
    for method in methods:
        result = _bootstrap_predict_category(
            simple_table,
            source_category=0,
            from_axis=0,
            to_axis=1,
            method=method,
            random_state=42
        )
        
        assert hasattr(result, 'confidence_interval')

def test_permutation_test_ccram_basic(contingency_table):
    """Test basic functionality of permutation_test_ccram."""
    result = permutation_test_ccram(
        contingency_table,
        from_axis=0,
        to_axis=1,
        n_resamples=999,
        random_state=8990
    )
    
    assert hasattr(result, "observed_value")
    assert hasattr(result, "p_value")
    assert hasattr(result, "null_distribution")
    assert hasattr(result, "histogram_fig")
    assert 0 <= result.p_value <= 1
    assert len(result.null_distribution) == 999

@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_permutation_test_alternatives(contingency_table, alternative):
    """Test different alternative hypotheses."""
    result = permutation_test_ccram(
        contingency_table,
        from_axis=0,
        to_axis=1,
        alternative=alternative,
        n_resamples=999,
        random_state=8990
    )
    
    assert 0 <= result.p_value <= 1

def test_reproducibility():
    """Test that results are reproducible with same random_state."""
    table = np.array([[10, 0], [0, 10]])
    
    # Bootstrap reproducibility
    boot_result1 = bootstrap_ccram(
        table,
        from_axis=0,
        to_axis=1,
        random_state=8990
    )
    
    boot_result2 = bootstrap_ccram(
        table,
        from_axis=0,
        to_axis=1,
        random_state=8990
    )
    
    np.testing.assert_array_almost_equal(
        boot_result1.bootstrap_distribution,
        boot_result2.bootstrap_distribution
    )
    
    # Permutation test reproducibility
    perm_result1 = permutation_test_ccram(
        table,
        from_axis=0,
        to_axis=1,
        random_state=8990
    )
    
    perm_result2 = permutation_test_ccram(
        table,
        from_axis=0,
        to_axis=1,
        random_state=8990
    )
    
    np.testing.assert_array_equal(
        perm_result1.null_distribution,
        perm_result2.null_distribution
    )
    assert perm_result1.p_value == perm_result2.p_value

def test_invalid_inputs():
    """Test invalid inputs raise appropriate errors."""
    valid_table = np.array([[10, 0], [0, 10]])
    
    # Test invalid axes
    with pytest.raises(KeyError):
        bootstrap_ccram(valid_table, from_axis=2, to_axis=1)
    with pytest.raises((KeyError, IndexError)):
        permutation_test_ccram(valid_table, from_axis=0, to_axis=2)
        
    # Test invalid alternative
    with pytest.raises(ValueError):
        permutation_test_ccram(valid_table, alternative="invalid")
        
    # Test invalid tables
    invalid_tables = [
        np.array([1, 2, 3]),  # 1D array
        np.array([[-1, 0], [0, 1]]),  # Negative values
        np.array([[0, 0], [0, 0]])  # All zeros
    ]
    
    for table in invalid_tables:
        with pytest.raises((ValueError, IndexError)):
            bootstrap_ccram(table, from_axis=0, to_axis=1)
        with pytest.raises((ValueError, IndexError)):
            permutation_test_ccram(table, from_axis=0, to_axis=1)