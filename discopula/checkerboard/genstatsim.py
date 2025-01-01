import numpy as np
from scipy.stats import bootstrap, permutation_test
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
from .gencopula import GenericCheckerboardCopula
from .utils import gen_contingency_to_case_form, gen_case_form_to_contingency

@dataclass
class CustomBootstrapResult:
    """Custom container for bootstrap results."""
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

@dataclass 
class CustomPermutationResult:
    """Custom container for permutation test results."""
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
    """Perform permutation test for (S)CCRAM."""
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

# Example usage
if __name__ == '__main__':
    table = np.array([
        [0, 0, 20],
        [0, 10, 0],
        [20, 0, 0],
        [0, 10, 0],
        [0, 0, 20]
    ])
    
    # Bootstrap analysis
    boot_result = bootstrap_ccram(
        table, from_axis=0, to_axis=1, n_resamples=9999
    )
    print(f"Observed CCRAM: {boot_result.observed_value:.4f}")
    print(f"95% CI: ({boot_result.confidence_interval[0]:.4f}, "
          f"{boot_result.confidence_interval[1]:.4f})")
    print(f"Standard Error: {boot_result.standard_error:.4f}")
    boot_result.histogram_fig.show()
    
    # Permutation test
    perm_result = permutation_test_ccram(
        table, from_axis=0, to_axis=1, n_resamples=9999
    )
    print(f"Observed CCRAM: {perm_result.observed_value:.4f}")
    print(f"P-value: {perm_result.p_value:.4f}")
    perm_result.histogram_fig.show()