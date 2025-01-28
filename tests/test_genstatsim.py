import numpy as np
import pandas as pd
import pytest
from io import StringIO
import sys
from discopula import (
    GenericCheckerboardCopula,
    bootstrap_ccram, 
    permutation_test_ccram, 
    bootstrap_predict_category_summary, 
)
from discopula.checkerboard.genstatsim import _bootstrap_predict_category_multi

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

def test_bootstrap_ccram_basic(contingency_table):
    """Test basic functionality of bootstrap_ccram."""
    gen_copula = GenericCheckerboardCopula.from_contingency_table(contingency_table)
    result = bootstrap_ccram(
        gen_copula,
        from_axes=[0],
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

def test_bootstrap_ccram_multiple_axes(table_4d):
    """Test bootstrap_ccram with multiple conditioning axes."""
    gen_copula = GenericCheckerboardCopula.from_contingency_table(table_4d)
    result = bootstrap_ccram(
        gen_copula,
        from_axes=[0, 1, 2],
        to_axis=3,
        n_resamples=999,
        random_state=8990
    )
    print(result)
    assert "(0,1,2)->3" in result.metric_name
    assert hasattr(result, "confidence_interval")
    assert result.confidence_interval[0] < result.confidence_interval[1]

def test_bootstrap_predict_category_multi_basic(contingency_table):
    """Test basic multi-axis category prediction."""
    result = _bootstrap_predict_category_multi(
        contingency_table,
        source_categories=[0],
        from_axes=[0],
        to_axis=1,
        n_resamples=999,
        random_state=8990
    )
    
    assert hasattr(result, 'confidence_interval')
    assert hasattr(result, 'bootstrap_distribution')
    assert len(result.bootstrap_distribution) == 999

def test_bootstrap_predict_category_multi_axes(table_4d):
    """Test category prediction with multiple conditioning axes."""
    result = _bootstrap_predict_category_multi(
        table_4d,
        source_categories=[0, 1],
        from_axes=[0, 1],
        to_axis=2,
        n_resamples=999,
        random_state=8990
    )
    
    assert hasattr(result, 'bootstrap_distribution')
    assert len(result.bootstrap_distribution) == 999

def test_prediction_summary_multi(table_4d):
    """Test multi-dimensional prediction summary."""
    gen_copula = GenericCheckerboardCopula.from_contingency_table(table_4d)
    summary_df = bootstrap_predict_category_summary(
        gen_copula,
        from_axes=[0, 1],
        from_axes_names=["X0","X1"],
        to_axis=2,
        n_resamples=999,
        random_state=8990
    )
    
    assert isinstance(summary_df, pd.DataFrame)
    assert np.all(summary_df >= 0)
    assert np.all(summary_df <= 100)

def test_display_prediction_summary_multi(table_4d):
    """Test display of multi-dimensional prediction summary."""
    gen_copula = GenericCheckerboardCopula.from_contingency_table(table_4d)
    
    # Capture stdout
    stdout = StringIO()
    sys.stdout = stdout
    
    summary_df = bootstrap_predict_category_summary(
        gen_copula,
        from_axes=[0, 1],
        from_axes_names=["First", "Second"],
        to_axis=2,
        to_axis_name="Third",
        n_resamples=999,
        random_state=8990
    )
    
    sys.stdout = sys.__stdout__
    output = stdout.getvalue()
    
    assert isinstance(summary_df, pd.DataFrame)
    assert "Prediction Summary" in output
    assert "First" in output
    assert "Second" in output
    assert "Third" in output

def test_permutation_test_multiple_axes(table_4d):
    """Test permutation test with multiple conditioning axes."""
    gen_copula = GenericCheckerboardCopula.from_contingency_table(table_4d)
    result = permutation_test_ccram(
        gen_copula,
        from_axes=[0, 1, 2],
        to_axis=3,
        n_resamples=999,
        random_state=8990
    )
    print(result.p_value)
    assert "(0,1,2)->3" in result.metric_name
    assert hasattr(result, "p_value")
    assert 0 <= result.p_value <= 1
    assert len(result.null_distribution) == 999

def test_invalid_inputs_multi():
    """Test invalid inputs for multi-axis functionality."""
    valid_table = np.array([[10, 0], [0, 10]])
    gen_copula = GenericCheckerboardCopula.from_contingency_table(valid_table)
    # Test invalid axes combinations
    with pytest.raises((IndexError, KeyError)):
        bootstrap_ccram(gen_copula, from_axes=[2, 3], to_axis=1)
    
    # Test duplicate axes
    with pytest.raises(IndexError):
        bootstrap_ccram(gen_copula, from_axes=[0, 0], to_axis=1)

def test_reproducibility_multi(table_4d):
    """Test reproducibility with multiple axes."""
    gen_copula = GenericCheckerboardCopula.from_contingency_table(table_4d)
    result1 = bootstrap_ccram(
        gen_copula,
        from_axes=[0, 1],
        to_axis=2,
        random_state=8990
    )
    
    result2 = bootstrap_ccram(
        gen_copula,
        from_axes=[0, 1],
        to_axis=2,
        random_state=8990
    )
    
    np.testing.assert_array_almost_equal(
        result1.bootstrap_distribution,
        result2.bootstrap_distribution
    )