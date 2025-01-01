import numpy as np
import pytest
from discopula import bootstrap_ccram, permutation_test_ccram

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
        n_resamples=999,
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