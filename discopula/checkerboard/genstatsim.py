import numpy as np
from scipy.stats import bootstrap, permutation_test
from dataclasses import dataclass
from typing import Tuple, List
import matplotlib.pyplot as plt
import pandas as pd
from .gencopula import GenericCheckerboardCopula
from .utils import gen_contingency_to_case_form, gen_case_form_to_contingency

@dataclass
class CustomBootstrapResult:
    """Container for bootstrap simulation results with visualization capabilities.
    
    Parameters
    ----------
    metric_name : str
        Name of the metric being bootstrapped
    observed_value : float
        Original observed value of the metric
    confidence_interval : Tuple[float, float]
        Lower and upper confidence interval bounds
    bootstrap_distribution : np.ndarray
        Array of bootstrapped values
    standard_error : float 
        Standard error of the bootstrap distribution
    histogram_fig : plt.Figure, optional
        Matplotlib figure of distribution plot
    """
    metric_name: str
    observed_value: float
    confidence_interval: Tuple[float, float]
    bootstrap_distribution: np.ndarray
    standard_error: float
    histogram_fig: plt.Figure = None

    def plot_distribution(self, title=None):
        """Plot bootstrap distribution with observed value."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate data range
            data_range = np.ptp(self.bootstrap_distribution)
            if data_range == 0:
                # If all values are identical, use a single bin
                bins = 1
            else:
                # Try to use 50 bins, fall back to fewer if needed
                bins = min(50, max(1, int(np.sqrt(len(self.bootstrap_distribution)))))
            
            ax.hist(self.bootstrap_distribution, bins=bins, density=True, alpha=0.7)
            ax.axvline(self.observed_value, color='red', linestyle='--', 
                    label=f'Observed {self.metric_name}')
            ax.set_xlabel(f'{self.metric_name} Value')
            ax.set_ylabel('Density')
            ax.set_title(title or 'Bootstrap Distribution')
            ax.legend()
            self.histogram_fig = fig
            return fig
        except Exception as e:
            print(f"Warning: Could not create bootstrap distribution plot: {str(e)}")
            self.histogram_fig = None
            return None

def bootstrap_ccram(contingency_table: np.ndarray,
                   from_axis: int,
                   to_axis: int, 
                   is_scaled: bool = False,
                   n_resamples: int = 9999,
                   confidence_level: float = 0.95,
                   method: str = 'percentile',
                   random_state = None) -> CustomBootstrapResult:
    """Perform bootstrap simulation for (S)CCRAM measure.
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        Input contingency table
    from_axis : int
        Source axis index
    to_axis : int  
        Target axis index
    is_scaled : bool, default=False
        Whether to use scaled CCRAM (SCCRAM)
    n_resamples : int, default=9999
        Number of bootstrap resamples
    confidence_level : float, default=0.95
        Confidence level for intervals
    method : str, default='percentile'
        Bootstrap CI method
    random_state : optional
        Random state for reproducibility
        
    Returns
    -------
    CustomBootstrapResult
        Bootstrap results including CIs and distribution
    """
    # Name the metric based on whether it's scaled
    metric_name = f"{'SCCRAM' if is_scaled else 'CCRAM'} {from_axis}->{to_axis}"
    
    # Calculate observed value first
    gen_copula = GenericCheckerboardCopula.from_contingency_table(contingency_table)
    observed_ccram = gen_copula.calculate_CCRAM_vectorized(from_axis, to_axis, is_scaled)
    
    # Convert contingency table to case form
    cases = gen_contingency_to_case_form(contingency_table)
    
    # Define axis ordering explicitly 
    resampling_axes = [from_axis, to_axis]
    
    # Split into source and target variables
    x_source, x_target = cases[:, from_axis], cases[:, to_axis]
    data = (x_source, x_target)

    def ccram_stat(x_source, x_target, axis=0):
        if x_source.ndim > 1:
            batch_size = x_source.shape[0]
            cases = np.stack([
                np.column_stack((
                    x_source[i].reshape(-1, 1), 
                    x_target[i].reshape(-1, 1)
                )) for i in range(batch_size)
            ])
        else:
            cases = np.column_stack((
                x_source.reshape(-1, 1), 
                x_target.reshape(-1, 1)
            ))

        # Reconstruct table preserving original axis order
        if cases.ndim == 3:
            results = []
            for batch_cases in cases:
                table = gen_case_form_to_contingency(
                    batch_cases, 
                    shape=contingency_table.shape,
                    axis_order=resampling_axes
                )
                copula = GenericCheckerboardCopula.from_contingency_table(table)
                value = copula.calculate_CCRAM_vectorized(from_axis, to_axis, is_scaled)
                results.append(value)
            return np.array(results)
        else:
            table = gen_case_form_to_contingency(
                cases, 
                shape=contingency_table.shape,
                axis_order=resampling_axes
            )
            copula = GenericCheckerboardCopula.from_contingency_table(table)
            value = copula.calculate_CCRAM_vectorized(from_axis, to_axis, is_scaled)
            return value

    # Perform bootstrap
    res = bootstrap(
        data,
        ccram_stat,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        paired=True,
        vectorized=True
    )
    
    # Create custom result with correct observed value
    cust_boot_res = CustomBootstrapResult(
        metric_name=metric_name,
        observed_value=observed_ccram,
        confidence_interval=res.confidence_interval,
        bootstrap_distribution=res.bootstrap_distribution,
        standard_error=res.standard_error
    )
    
    boot_dist_fig = cust_boot_res.plot_distribution(f'Bootstrap Distribution: {metric_name} {from_axis}->{to_axis}')
    cust_boot_res.histogram_fig = boot_dist_fig
    
    return cust_boot_res

def _bootstrap_predict_category(
    contingency_table: np.ndarray,
    source_category: int,
    from_axis: int,
    to_axis: int,
    n_resamples: int = 9999,
    confidence_level: float = 0.95,
    method: str = 'percentile',
    random_state = None
):
    """Helper Function: Bootstrap confidence intervals for generic category prediction."""
    # Convert table to case form
    cases = gen_contingency_to_case_form(contingency_table)
    
    # Split variables
    x_source, x_target = cases[:, from_axis], cases[:, to_axis]
    data = (x_source, x_target)

    def prediction_stat(x_source, x_target, axis=0):
        if x_source.ndim > 1:
            batch_size = x_source.shape[0]
            cases = np.stack([np.column_stack((x_source[i], x_target[i])) 
                            for i in range(batch_size)])
        else:
            cases = np.column_stack((x_source, x_target))
            
        # Reconstruct table
        if cases.ndim == 3:
            results = []
            for batch_cases in cases:
                table = gen_case_form_to_contingency(
                    batch_cases,
                    shape=contingency_table.shape,
                    axis_order=[from_axis, to_axis]
                )
                copula = GenericCheckerboardCopula.from_contingency_table(table)
                pred = copula._predict_category(source_category, from_axis, to_axis)
                results.append(pred)
            return np.array(results)
        else:
            table = gen_case_form_to_contingency(
                cases,
                shape=contingency_table.shape,
                axis_order=[from_axis, to_axis]
            )
            copula = GenericCheckerboardCopula.from_contingency_table(table)
            return copula._predict_category(source_category, from_axis, to_axis)

    # Perform bootstrap
    return bootstrap(
        data,
        prediction_stat,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        paired=True,
        vectorized=True
    )

def _bootstrap_predict_category_vectorized(
    contingency_table: np.ndarray,
    source_categories: np.ndarray,
    from_axis: int,
    to_axis: int,
    n_resamples: int = 9999,
    confidence_level: float = 0.95,
    method: str = 'percentile',
    random_state = None
) -> List:
    """Helper Function: Vectorized bootstrapping for multiple category predictions."""
    source_categories = np.asarray(source_categories)
    results = []
    
    for source_cat in source_categories:
        res = _bootstrap_predict_category(
            contingency_table,
            source_cat,
            from_axis,
            to_axis,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method=method,
            random_state=random_state
        )
        results.append(res)
    
    return results

def bootstrap_predict_category_summary(
    contingency_table: np.ndarray,
    from_axis: int,
    to_axis: int,
    n_resamples: int = 9999,
    confidence_level: float = 0.95,
    method: str = 'percentile',
    random_state = None
) -> np.ndarray:
    """Generate bootstrap summary table for all category combinations.
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        Contingency table
    from_axis : int
        Source axis index
    to_axis : int
        Target axis index
    n_resamples : int, default=9999
        Number of resamples
    confidence_level : float, default=0.95
        Confidence level
    method : str, default='percentile'
        Bootstrap CI method
    random_state : optional
        Random state

    Returns
    -------
    numpy.ndarray
        Summary table of prediction proportions
    """
    source_dim = contingency_table.shape[from_axis]
    target_dim = contingency_table.shape[to_axis]
    
    # Get predictions for all source categories
    results = _bootstrap_predict_category_vectorized(
        contingency_table,
        np.arange(source_dim),
        from_axis,
        to_axis,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state
    )
    
    # Initialize summary table
    summary = np.zeros((target_dim, source_dim))
    
    # Fill summary table with prediction proportions
    for source_cat in range(source_dim):
        bootstrap_preds = results[source_cat].bootstrap_distribution
        unique_preds, counts = np.unique(bootstrap_preds, return_counts=True)
        total = len(bootstrap_preds)
        
        for val, count in zip(unique_preds, counts):
            summary[int(val), source_cat] = (count / total) * 100
            
    return summary

def display_prediction_summary(summary_matrix: np.ndarray, 
                             from_axis_name: str = "X",
                             to_axis_name: str = "Y") -> None:
    """Display prediction summary matrix in a nicely formatted table.
    
    Parameters
    ----------
    summary_matrix : np.ndarray
        Matrix of prediction percentages
    from_axis_name : str, default="X"
        Name of source variable
    to_axis_name : str, default="Y"  
        Name of target variable
    """
    # Round percentages to 1 decimal place
    summary_rounded = np.round(summary_matrix, 1)
    
    # Create row and column labels
    n_rows, n_cols = summary_matrix.shape
    row_labels = [f"{to_axis_name}={i}" for i in range(n_rows)]
    col_labels = [f"{from_axis_name}={i}" for i in range(n_cols)]
    
    # Create pandas DataFrame
    df = pd.DataFrame(
        summary_rounded,
        index=row_labels,
        columns=col_labels
    )
    
    # Display with styling
    print("\nPrediction Summary (% of bootstrap samples)")
    print(f"From {from_axis_name} to {to_axis_name}:")
    print("-" * 50)
    print(df.to_string(float_format=lambda x: f"{x:5.1f}%"))
    print("-" * 50)

@dataclass 
class CustomPermutationResult:
    """Container for permutation test results with visualization capabilities.
    
    Parameters
    ----------
    metric_name : str
        Name of the metric being tested
    observed_value : float
        Original observed value
    p_value : float
        Permutation test p-value
    null_distribution : np.ndarray
        Array of values under null hypothesis
    histogram_fig : plt.Figure, optional
        Matplotlib figure of distribution plot
    """
    metric_name: str
    observed_value: float
    p_value: float
    null_distribution: np.ndarray
    histogram_fig: plt.Figure = None

    def plot_distribution(self, title=None):
        """Plot null distribution with observed value."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate data range
            data_range = np.ptp(self.null_distribution)
            if data_range == 0:
                # If all values are identical, use a single bin
                bins = 1
            else:
                # Try to use 50 bins, fall back to fewer if needed
                bins = min(50, max(1, int(np.sqrt(len(self.null_distribution)))))
                
            ax.hist(self.null_distribution, bins=bins, density=True, alpha=0.7)
            ax.axvline(self.observed_value, color='red', linestyle='--', 
                    label=f'Observed {self.metric_name}')
            ax.set_xlabel(f'{self.metric_name} Value')
            ax.set_ylabel('Density')
            ax.set_title(title or 'Null Distribution')
            ax.legend()
            self.histogram_fig = fig
            return fig
        except Exception as e:
            print(f"Warning: Could not create null distribution plot: {str(e)}")
            self.histogram_fig = None
            return None

def permutation_test_ccram(contingency_table: np.ndarray,
                          from_axis: int = 0,
                          to_axis: int = 1,
                          is_scaled: bool = False,
                          alternative: str ='greater',
                          n_resamples: int = 9999,
                          random_state: int = None) -> CustomPermutationResult:
    """Perform permutation test for (S)CCRAM measure.
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        Input contingency table
    from_axis : int, default=0
        Source axis index
    to_axis : int, default=1
        Target axis index
    is_scaled : bool, default=False
        Whether to use scaled CCRAM (SCCRAM)
    alternative : str, default='greater'
        Alternative hypothesis ('greater', 'less', 'two-sided')
    n_resamples : int, default=9999
        Number of permutations
    random_state : int, optional
        Random state for reproducibility
        
    Returns
    -------
    CustomPermutationResult
        Test results including p-value and null distribution
    """
    # Name the metric based on whether it's scaled
    metric_name = f"{'SCCRAM' if is_scaled else 'CCRAM'} {from_axis}->{to_axis}"
    
    # Convert contingency table to case form
    cases = gen_contingency_to_case_form(contingency_table)
    
    # Define axis ordering explicitly 
    resampling_axes = [from_axis, to_axis]
    
    # Split into source and target variables
    x_source, x_target = cases[:, from_axis], cases[:, to_axis]
    data = (x_source, x_target)

    def ccram_stat(x_source, x_target, axis=0):
        if x_source.ndim > 1:
            batch_size = x_source.shape[0]
            cases = np.stack([
                np.column_stack((
                    x_source[i].reshape(-1, 1), 
                    x_target[i].reshape(-1, 1)
                )) for i in range(batch_size)
            ])
        else:
            cases = np.column_stack((
                x_source.reshape(-1, 1), 
                x_target.reshape(-1, 1)
            ))

        # Reconstruct table preserving original axis order
        if cases.ndim == 3:
            results = []
            for batch_cases in cases:
                table = gen_case_form_to_contingency(
                    batch_cases, 
                    shape=contingency_table.shape,
                    axis_order=resampling_axes
                )
                copula = GenericCheckerboardCopula.from_contingency_table(table)
                value = copula.calculate_CCRAM_vectorized(from_axis, to_axis, is_scaled)
                results.append(value)
            return np.array(results)
        else:
            table = gen_case_form_to_contingency(
                cases, 
                shape=contingency_table.shape,
                axis_order=resampling_axes
            )
            copula = GenericCheckerboardCopula.from_contingency_table(table)
            value = copula.calculate_CCRAM_vectorized(from_axis, to_axis, is_scaled)
            return value

    # Perform permutation test
    perm = permutation_test(
        data, 
        ccram_stat,
        permutation_type='pairings',
        alternative=alternative,
        n_resamples=n_resamples,
        random_state=random_state,
        vectorized=True
    )
    
    # Create result
    result = CustomPermutationResult(
        metric_name=metric_name,
        observed_value=perm.statistic,
        p_value=perm.pvalue,
        null_distribution=perm.null_distribution
    )
    
    null_dist_fig = result.plot_distribution(f'Null Distribution: {metric_name}')
    result.histogram_fig = null_dist_fig
    
    return result