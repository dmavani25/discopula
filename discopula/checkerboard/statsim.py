import numpy as np
from scipy.stats import bootstrap, permutation_test
from .copula import CheckerboardCopula
from .utils import case_form_to_contingency, contingency_to_case_form

def bootstrap_ccram(contingency_table, direction="X1_X2", n_resamples=9999, confidence_level=0.95, method='BCa', random_state=None):
    """Calculate bootstrap confidence intervals for CCRAM measure.

    Performs bootstrap resampling to estimate confidence intervals for the
    Checkerboard Copula Regression Association Measure (CCRAM) in specified direction (default X1 --> X2).
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        2D array representing contingency table of observed frequencies.
    direction : {'X1_X2', 'X2_X1'}, default='X1_X2'
    n_resamples : int, default=9999
        Number of bootstrap resamples to generate.
    confidence_level : float, default=0.95
        Confidence level for interval calculation (between 0 and 1).
    method : {'percentile', 'basic', 'BCa'}, default='BCa'
        Method to calculate bootstrap confidence intervals:
        - 'percentile': Standard percentile method
        - 'basic': Basic bootstrap interval
        - 'BCa': Bias-corrected and accelerated bootstrap
    random_state : {None, int, numpy.random.Generator,
                   numpy.random.RandomState}, optional
        Random state for reproducibility.

    Returns
    -------
    scipy.stats.BootstrapResult
        Object containing:
        - confidence_interval: namedtuple with low and high bounds
        - bootstrap_distribution: array of CCRAM values for resamples
        - standard_error: bootstrap estimate of standard error

    Examples
    --------
    >>> table = np.array([[10, 0], [0, 10]])
    >>> result = bootstrap_ccram(table, n_resamples=999)
    >>> print(f"95% CI: ({result.confidence_interval.low:.4f}, "
    ...       f"{result.confidence_interval.high:.4f})")

    Notes
    -----
    Implementation process:
    1. Converts contingency table to case form
    2. Performs paired bootstrap resampling
    3. Calculates CCRAM for each resample
    4. Computes confidence intervals using specified method
    
    The BCa method is recommended as it corrects for both bias and skewness
    in the bootstrap distribution.

    See Also
    --------
    contingency_to_case_form : Converts table to resampling format
    case_form_to_contingency : Converts resampled cases back to table
    scipy.stats.bootstrap : Underlying bootstrap implementation

    References
    ----------
    .. [1] Efron, B. (1987). "Better Bootstrap Confidence Intervals"
    """
    # Convert contingency table to case form
    cases = contingency_to_case_form(contingency_table)
    
    # Split into X1 and X2 variables
    x1, x2 = cases[:, 0], cases[:, 1]
    data = (x1, x2)
    
    def ccram_stat(x1, x2, axis=0):
        # Handle batched data
        if x1.ndim > 1:
            batch_size = x1.shape[0]
            cases = np.stack([np.column_stack((x1[i], x2[i])) 
                            for i in range(batch_size)])
        else:
            cases = np.column_stack((x1, x2))
            
        n_rows, n_cols = contingency_table.shape
        resampled_table = case_form_to_contingency(cases, n_rows, n_cols)
        
        # Handle single vs batched tables
        if resampled_table.ndim == 3:
            results = []
            for table in resampled_table:
                copula = CheckerboardCopula.from_contingency_table(table)
                if direction == "X1_X2":
                    results.append(copula.calculate_CCRAM_X1_X2_vectorized())
                elif direction == "X2_X1":
                    results.append(copula.calculate_CCRAM_X2_X1_vectorized())
                else:
                    raise ValueError("Invalid direction. Use 'X1_X2' or 'X2_X1'")
            return np.array(results)
        else:
            copula = CheckerboardCopula.from_contingency_table(resampled_table)
            if direction == "X1_X2":
                return copula.calculate_CCRAM_X1_X2_vectorized()
            else:
                return copula.calculate_CCRAM_X2_X1_vectorized()

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
    
    return res

def bootstrap_sccram(contingency_table, direction="X1_X2", n_resamples=9999, confidence_level=0.95, method='BCa', random_state=None):
    """Calculate bootstrap confidence intervals for SCCRAM measure.

    Performs bootstrap resampling to estimate confidence intervals for the
    Scaled Checkerboard Copula Regression Association Measure (SCCRAM) in specified direction (default X1 --> X2).
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        2D array representing contingency table of observed frequencies.
    direction : {'X1_X2', 'X2_X1'}, default='X1_X2'
    n_resamples : int, default=9999
        Number of bootstrap resamples to generate.
    confidence_level : float, default=0.95
        Confidence level for interval calculation (between 0 and 1).
    method : {'percentile', 'basic', 'BCa'}, default='BCa'
        Method to calculate bootstrap confidence intervals:
        - 'percentile': Standard percentile method
        - 'basic': Basic bootstrap interval
        - 'BCa': Bias-corrected and accelerated bootstrap
    random_state : {None, int, numpy.random.Generator,
                   numpy.random.RandomState}, optional
        Random state for reproducibility.

    Returns
    -------
    scipy.stats.BootstrapResult
        Object containing:
        - confidence_interval: namedtuple with low and high bounds
        - bootstrap_distribution: array of SCCRAM values for resamples
        - standard_error: bootstrap estimate of standard error

    Examples
    --------
    >>> table = np.array([[10, 0], [0, 10]])
    >>> result = bootstrap_sccram(table, n_resamples=999)
    >>> print(f"95% CI: ({result.confidence_interval.low:.4f}, "
    ...       f"{result.confidence_interval.high:.4f})")

    Notes
    -----
    Implementation process:
    1. Converts contingency table to case form
    2. Performs paired bootstrap resampling
    3. Calculates SCCRAM for each resample
    4. Computes confidence intervals using specified method
    
    The BCa method is recommended as it corrects for both bias and skewness
    in the bootstrap distribution.

    See Also
    --------
    contingency_to_case_form : Converts table to resampling format
    case_form_to_contingency : Converts resampled cases back to table
    scipy.stats.bootstrap : Underlying bootstrap implementation

    References
    ----------
    .. [1] Efron, B. (1987). "Better Bootstrap Confidence Intervals"
    """
    # Convert contingency table to case form
    cases = contingency_to_case_form(contingency_table)
    
    # Split into X1 and X2 variables
    x1, x2 = cases[:, 0], cases[:, 1]
    data = (x1, x2)
    
    def sccram_stat(x1, x2, axis=0):
        # Handle batched data
        if x1.ndim > 1:
            batch_size = x1.shape[0]
            cases = np.stack([np.column_stack((x1[i], x2[i])) 
                            for i in range(batch_size)])
        else:
            cases = np.column_stack((x1, x2))
            
        n_rows, n_cols = contingency_table.shape
        resampled_table = case_form_to_contingency(cases, n_rows, n_cols)
        
        # Handle single vs batched tables
        if resampled_table.ndim == 3:
            results = []
            for table in resampled_table:
                copula = CheckerboardCopula.from_contingency_table(table)
                if direction == "X1_X2":
                    results.append(copula.calculate_SCCRAM_X1_X2_vectorized())
                elif direction == "X2_X1":
                    results.append(copula.calculate_SCCRAM_X2_X1_vectorized())
                else:
                    raise ValueError("Invalid direction. Use 'X1_X2' or 'X2_X1'")
            return np.array(results)
        else:
            copula = CheckerboardCopula.from_contingency_table(resampled_table)
            if direction == "X1_X2":
                return copula.calculate_SCCRAM_X1_X2_vectorized()
            else:
                return copula.calculate_SCCRAM_X2_X1_vectorized()

    # Perform bootstrap
    res = bootstrap(
        data,
        sccram_stat, 
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        paired=True,
        vectorized=True
    )
    
    return res

def bootstrap_regression_U1_on_U2(contingency_table, u2, n_resamples=9999, confidence_level=0.95, method='BCa', random_state=None):
    """Calculate bootstrap confidence intervals for regression E[U1|U2=u2].
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        2D array representing contingency table of observed frequencies.
    u2 : float
        The U2 value at which to evaluate the regression (between 0 and 1).
    n_resamples : int, default=9999
        Number of bootstrap resamples to generate.
    confidence_level : float, default=0.95
        Confidence level for interval calculation (between 0 and 1).
    method : {'percentile', 'basic', 'BCa'}, default='BCa'
        Method to calculate bootstrap confidence intervals.
    random_state : {None, int, numpy.random.Generator,
                   numpy.random.RandomState}, optional
        Random state for reproducibility.

    Returns
    -------
    scipy.stats.BootstrapResult
        Object containing:
        - confidence_interval: namedtuple with low and high bounds
        - bootstrap_distribution: array of regression values for resamples
        - standard_error: bootstrap estimate of standard error

    Examples
    --------
    >>> table = np.array([[10, 0], [0, 10]])
    >>> result = bootstrap_regression_U1_on_U2(table, u2=0.5)
    >>> print(f"95% CI: ({result.confidence_interval.low:.4f}, "
    ...       f"{result.confidence_interval.high:.4f})")
    """
    # Input validation
    if not 0 <= u2 <= 1:
        raise ValueError("u2 must be between 0 and 1")
        
    # Convert contingency table to case form
    cases = contingency_to_case_form(contingency_table)
    
    # Split into X1 and X2 variables
    x1, x2 = cases[:, 0], cases[:, 1]
    data = (x1, x2)
    
    def regression_stat(x1, x2, axis=0):
        # Handle batched data
        if x1.ndim > 1:
            batch_size = x1.shape[0]
            cases = np.stack([np.column_stack((x1[i], x2[i])) 
                            for i in range(batch_size)])
        else:
            cases = np.column_stack((x1, x2))
            
        n_rows, n_cols = contingency_table.shape
        resampled_table = case_form_to_contingency(cases, n_rows, n_cols)
        
        # Handle single vs batched tables
        if resampled_table.ndim == 3:
            results = []
            for table in resampled_table:
                copula = CheckerboardCopula.from_contingency_table(table)
                results.append(copula.calculate_regression_U1_on_U2(u2))
            return np.array(results)
        else:
            copula = CheckerboardCopula.from_contingency_table(resampled_table)
            return copula.calculate_regression_U1_on_U2(u2)

    # Perform bootstrap
    res = bootstrap(
        data,
        regression_stat, 
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        paired=True,
        vectorized=True
    )
    
    return res


def bootstrap_regression_U1_on_U2_vectorized(contingency_table, u2_values, n_resamples=9999, confidence_level=0.95, method='BCa', random_state=None):
    """Calculate bootstrap confidence intervals for regression E[U1|U2=u2] for multiple u2 values.
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        2D array representing contingency table of observed frequencies.
    u2_values : array-like
        Array of U2 values at which to evaluate the regression (between 0 and 1).
    n_resamples : int, default=9999
        Number of bootstrap resamples to generate.
    confidence_level : float, default=0.95
        Confidence level for interval calculation (between 0 and 1).
    method : {'percentile', 'basic', 'BCa'}, default='BCa'
        Method to calculate bootstrap confidence intervals.
    random_state : {None, int, numpy.random.Generator,
                   numpy.random.RandomState}, optional
        Random state for reproducibility.

    Returns
    -------
    list of scipy.stats.BootstrapResult
        List of bootstrap results for each u2 value, each containing:
        - confidence_interval: namedtuple with low and high bounds
        - bootstrap_distribution: array of regression values for resamples
        - standard_error: bootstrap estimate of standard error

    Examples
    --------
    >>> table = np.array([[10, 0], [0, 10]])
    >>> u2s = np.array([0.25, 0.5, 0.75])
    >>> results = bootstrap_regression_U1_on_U2_vectorized(table, u2s)
    >>> for u2, res in zip(u2s, results):
    ...     print(f"u2={u2:.2f}: CI=({res.confidence_interval.low:.4f}, "
    ...           f"{res.confidence_interval.high:.4f})")
    """
    # Input validation
    u2_values = np.asarray(u2_values)
    
    # Perform bootstrap for each u2 value
    results = []
    for u2 in u2_values:
        res = bootstrap_regression_U1_on_U2(
            contingency_table, 
            u2,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method=method,
            random_state=random_state
        )
        results.append(res)
        
    return results

def bootstrap_regression_U2_on_U1(contingency_table, u1, n_resamples=9999, confidence_level=0.95, method='BCa', random_state=None):
    """Calculate bootstrap confidence intervals for regression E[U2|U1=u1].
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        2D array representing contingency table of observed frequencies.
    u1 : float
        The U1 value at which to evaluate the regression (between 0 and 1).
    n_resamples : int, default=9999
        Number of bootstrap resamples to generate.
    confidence_level : float, default=0.95
        Confidence level for interval calculation (between 0 and 1).
    method : {'percentile', 'basic', 'BCa'}, default='BCa'
        Method to calculate bootstrap confidence intervals.
    random_state : {None, int, numpy.random.Generator,
                   numpy.random.RandomState}, optional
        Random state for reproducibility.

    Returns
    -------
    scipy.stats.BootstrapResult
        Object containing:
        - confidence_interval: namedtuple with low and high bounds
        - bootstrap_distribution: array of regression values for resamples
        - standard_error: bootstrap estimate of standard error

    Examples
    --------
    >>> table = np.array([[10, 0], [0, 10]])
    >>> result = bootstrap_regression_U2_on_U1(table, u1=0.5)
    >>> print(f"95% CI: ({result.confidence_interval.low:.4f}, "
    ...       f"{result.confidence_interval.high:.4f})")
    """
    # Input validation
    if not 0 <= u1 <= 1:
        raise ValueError("u1 must be between 0 and 1")
        
    # Convert contingency table to case form
    cases = contingency_to_case_form(contingency_table)
    
    # Split into X1 and X2 variables
    x1, x2 = cases[:, 0], cases[:, 1]
    data = (x1, x2)
    
    def regression_stat(x1, x2, axis=0):
        # Handle batched data
        if x1.ndim > 1:
            batch_size = x1.shape[0]
            cases = np.stack([np.column_stack((x1[i], x2[i])) 
                            for i in range(batch_size)])
        else:
            cases = np.column_stack((x1, x2))
            
        n_rows, n_cols = contingency_table.shape
        resampled_table = case_form_to_contingency(cases, n_rows, n_cols)
        
        # Handle single vs batched tables
        if resampled_table.ndim == 3:
            results = []
            for table in resampled_table:
                copula = CheckerboardCopula.from_contingency_table(table)
                results.append(copula.calculate_regression_U2_on_U1(u1))
            return np.array(results)
        else:
            copula = CheckerboardCopula.from_contingency_table(resampled_table)
            return copula.calculate_regression_U2_on_U1(u1)

    # Perform bootstrap
    res = bootstrap(
        data,
        regression_stat, 
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        paired=True,
        vectorized=True
    )
    
    return res

def bootstrap_regression_U2_on_U1_vectorized(contingency_table, u1_values, n_resamples=9999, confidence_level=0.95, method='BCa', random_state=None):
    """Calculate bootstrap confidence intervals for regression E[U2|U1=u1] for multiple u1 values.
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        2D array representing contingency table of observed frequencies.
    u1_values : array-like
        Array of U1 values at which to evaluate the regression (between 0 and 1).
    n_resamples : int, default=9999
        Number of bootstrap resamples to generate.
    confidence_level : float, default=0.95
        Confidence level for interval calculation (between 0 and 1).
    method : {'percentile', 'basic', 'BCa'}, default='BCa'
        Method to calculate bootstrap confidence intervals.
    random_state : {None, int, numpy.random.Generator,
                   numpy.random.RandomState}, optional
        Random state for reproducibility.

    Returns
    -------
    list of scipy.stats.BootstrapResult
        List of bootstrap results for each u1 value, each containing:
        - confidence_interval: namedtuple with low and high bounds
        - bootstrap_distribution: array of regression values for resamples
        - standard_error: bootstrap estimate of standard error

    Examples
    --------
    >>> table = np.array([[10, 0], [0, 10]])
    >>> u1s = np.array([0.25, 0.5, 0.75])
    >>> results = bootstrap_regression_U2_on_U1_vectorized(table, u1s)
    >>> for u1, res in zip(u1s, results):
    ...     print(f"u1={u1:.2f}: CI=({res.confidence_interval.low:.4f}, "
    ...           f"{res.confidence_interval.high:.4f})")
    """
    # Input validation
    u1_values = np.asarray(u1_values)
    
    # Perform bootstrap for each u1 value
    results = []
    for u1 in u1_values:
        res = bootstrap_regression_U2_on_U1(
            contingency_table, 
            u1,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method=method,
            random_state=random_state
        )
        results.append(res)
        
    return results

def bootstrap_predict_X2_from_X1(contingency_table, x1_category, n_resamples=9999, confidence_level=0.95, method='BCa', random_state=None):
    """Calculate bootstrap confidence intervals for predicting X2 category from X1.
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        2D array representing contingency table of observed frequencies.
    x1_category : int
        The X1 category index for which to predict X2 (0-based).
    n_resamples : int, default=9999
        Number of bootstrap resamples to generate.
    confidence_level : float, default=0.95
        Confidence level for interval calculation (between 0 and 1).
    method : {'percentile', 'basic', 'BCa'}, default='BCa'
        Method to calculate bootstrap confidence intervals.
    random_state : {None, int, numpy.random.Generator,
                   numpy.random.RandomState}, optional
        Random state for reproducibility.

    Returns
    -------
    scipy.stats.BootstrapResult
        Object containing:
        - confidence_interval: namedtuple with low and high bounds
        - bootstrap_distribution: array of predicted X2 categories for resamples
        - standard_error: bootstrap estimate of standard error

    Examples
    --------
    >>> table = np.array([[10, 0], [0, 10]])
    >>> result = bootstrap_predict_X2_from_X1(table, x1_category=0)
    >>> print(f"95% CI: ({result.confidence_interval.low:.4f}, "
    ...       f"{result.confidence_interval.high:.4f})")
    """
    # Convert contingency table to case form
    cases = contingency_to_case_form(contingency_table)
    
    # Split into X1 and X2 variables
    x1, x2 = cases[:, 0], cases[:, 1]
    data = (x1, x2)
    
    def prediction_stat(x1, x2, axis=0):
        # Handle batched data
        if x1.ndim > 1:
            batch_size = x1.shape[0]
            cases = np.stack([np.column_stack((x1[i], x2[i])) 
                            for i in range(batch_size)])
        else:
            cases = np.column_stack((x1, x2))
            
        n_rows, n_cols = contingency_table.shape
        resampled_table = case_form_to_contingency(cases, n_rows, n_cols)
        
        # Handle single vs batched tables
        if resampled_table.ndim == 3:
            results = []
            for table in resampled_table:
                copula = CheckerboardCopula.from_contingency_table(table)
                results.append(copula.predict_X2_from_X1(x1_category))
            return np.array(results)
        else:
            copula = CheckerboardCopula.from_contingency_table(resampled_table)
            return copula.predict_X2_from_X1(x1_category)

    # Perform bootstrap
    res = bootstrap(
        data,
        prediction_stat, 
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        paired=True,
        vectorized=True
    )
    
    return res

def bootstrap_predict_X2_from_X1_vectorized(contingency_table, x1_categories, n_resamples=9999, confidence_level=0.95, method='BCa', random_state=None):
    """Calculate bootstrap confidence intervals for predicting X2 from multiple X1 categories.
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        2D array representing contingency table of observed frequencies.
    x1_categories : array-like
        Array of X1 category indices for which to predict X2 (0-based).
    n_resamples : int, default=9999
        Number of bootstrap resamples to generate.
    confidence_level : float, default=0.95
        Confidence level for interval calculation (between 0 and 1).
    method : {'percentile', 'basic', 'BCa'}, default='BCa'
        Method to calculate bootstrap confidence intervals.
    random_state : {None, int, numpy.random.Generator,
                   numpy.random.RandomState}, optional
        Random state for reproducibility.

    Returns
    -------
    list of scipy.stats.BootstrapResult
        List of bootstrap results for each X1 category, each containing:
        - confidence_interval: namedtuple with low and high bounds
        - bootstrap_distribution: array of predicted X2 categories for resamples
        - standard_error: bootstrap estimate of standard error

    Examples
    --------
    >>> table = np.array([[10, 0], [0, 10]])
    >>> x1s = np.array([0, 1])
    >>> results = bootstrap_predict_X2_from_X1_vectorized(table, x1s)
    >>> for x1, res in zip(x1s, results):
    ...     print(f"X1={x1}: CI=({res.confidence_interval.low:.4f}, "
    ...           f"{res.confidence_interval.high:.4f})")
    """
    # Input validation
    x1_categories = np.asarray(x1_categories)
    
    # Perform bootstrap for each x1 category
    results = []
    for x1_cat in x1_categories:
        res = bootstrap_predict_X2_from_X1(
            contingency_table, 
            x1_cat,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method=method,
            random_state=random_state
        )
        results.append(res)
        
    return results

def bootstrap_predict_X1_from_X2(contingency_table, x2_category, n_resamples=9999, confidence_level=0.95, method='BCa', random_state=None):
    """Calculate bootstrap confidence intervals for predicting X1 category from X2.
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        2D array representing contingency table of observed frequencies.
    x2_category : int
        The X2 category index for which to predict X1 (0-based).
    n_resamples : int, default=9999
        Number of bootstrap resamples to generate.
    confidence_level : float, default=0.95
        Confidence level for interval calculation (between 0 and 1).
    method : {'percentile', 'basic', 'BCa'}, default='BCa'
        Method to calculate bootstrap confidence intervals.
    random_state : {None, int, numpy.random.Generator,
                   numpy.random.RandomState}, optional
        Random state for reproducibility.

    Returns
    -------
    scipy.stats.BootstrapResult
        Object containing:
        - confidence_interval: namedtuple with low and high bounds
        - bootstrap_distribution: array of predicted X1 categories for resamples
        - standard_error: bootstrap estimate of standard error

    Examples
    --------
    >>> table = np.array([[10, 0], [0, 10]])
    >>> result = bootstrap_predict_X1_from_X2(table, x2_category=0)
    >>> print(f"95% CI: ({result.confidence_interval.low:.4f}, "
    ...       f"{result.confidence_interval.high:.4f})")
    """
    # Convert contingency table to case form
    cases = contingency_to_case_form(contingency_table)
    
    # Split into X1 and X2 variables
    x1, x2 = cases[:, 0], cases[:, 1]
    data = (x1, x2)
    
    def prediction_stat(x1, x2, axis=0):
        # Handle batched data
        if x1.ndim > 1:
            batch_size = x1.shape[0]
            cases = np.stack([np.column_stack((x1[i], x2[i])) 
                            for i in range(batch_size)])
        else:
            cases = np.column_stack((x1, x2))
            
        n_rows, n_cols = contingency_table.shape
        resampled_table = case_form_to_contingency(cases, n_rows, n_cols)
        
        # Handle single vs batched tables
        if resampled_table.ndim == 3:
            results = []
            for table in resampled_table:
                copula = CheckerboardCopula.from_contingency_table(table)
                results.append(copula.predict_X1_from_X2(x2_category))
            return np.array(results)
        else:
            copula = CheckerboardCopula.from_contingency_table(resampled_table)
            return copula.predict_X1_from_X2(x2_category)

    # Perform bootstrap
    res = bootstrap(
        data,
        prediction_stat, 
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        paired=True,
        vectorized=True
    )
    
    return res

def bootstrap_predict_X1_from_X2_vectorized(contingency_table, x2_categories, n_resamples=9999, confidence_level=0.95, method='BCa', random_state=None):
    """Calculate bootstrap confidence intervals for predicting X1 from multiple X2 categories.
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        2D array representing contingency table of observed frequencies.
    x2_categories : array-like
        Array of X2 category indices for which to predict X1 (0-based).
    n_resamples : int, default=9999
        Number of bootstrap resamples to generate.
    confidence_level : float, default=0.95
        Confidence level for interval calculation (between 0 and 1).
    method : {'percentile', 'basic', 'BCa'}, default='BCa'
        Method to calculate bootstrap confidence intervals.
    random_state : {None, int, numpy.random.Generator,
                   numpy.random.RandomState}, optional
        Random state for reproducibility.

    Returns
    -------
    list of scipy.stats.BootstrapResult
        List of bootstrap results for each X2 category, each containing:
        - confidence_interval: namedtuple with low and high bounds
        - bootstrap_distribution: array of predicted X1 categories for resamples
        - standard_error: bootstrap estimate of standard error

    Examples
    --------
    >>> table = np.array([[10, 0], [0, 10]])
    >>> x2s = np.array([0, 1])
    >>> results = bootstrap_predict_X1_from_X2_vectorized(table, x2s)
    >>> for x2, res in zip(x2s, results):
    ...     print(f"X2={x2}: CI=({res.confidence_interval.low:.4f}, "
    ...           f"{res.confidence_interval.high:.4f})")
    """
    # Input validation
    x2_categories = np.asarray(x2_categories)
    
    # Perform bootstrap for each x2 category
    results = []
    for x2_cat in x2_categories:
        res = bootstrap_predict_X1_from_X2(
            contingency_table, 
            x2_cat,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method=method,
            random_state=random_state
        )
        results.append(res)
        
    return results

def bootstrap_predict_X1_from_X2_all_comb_summary(contingency_table, n_resamples=9999, confidence_level=0.95, method='BCa', random_state=None):
    """Calculate bootstrap summary classification table for predicting X1 from all X2 categories.
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        2D array representing contingency table of observed frequencies.
    n_resamples : int, default=9999
        Number of bootstrap resamples to generate.
    confidence_level : float, default=0.95
        Confidence level for interval calculation (between 0 and 1).
    method : {'percentile', 'basic', 'BCa'}, default='BCa'
        Method to calculate bootstrap confidence intervals.
    random_state : {None, int, numpy.random.Generator,
                   numpy.random.RandomState}, optional
        Random state for reproducibility.

    Returns
    -------
    summary : numpy.ndarray
        2D array showing the proportion of X1=j from CCR at each combination of categories of X2
    """
    n_rows, n_cols = contingency_table.shape
    
    # Get bootstrap predictions for each X2 category
    results = bootstrap_predict_X1_from_X2_vectorized(
        contingency_table, 
        np.arange(n_cols),
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state
    )
    
    # Initialize summary table
    summary_table = np.zeros((n_rows, n_cols))
    
    # For each X2 category
    for x2_cat in range(n_cols):
        bootstrap_preds = results[x2_cat].bootstrap_distribution
        
        # Count occurrences of each possible X1 category
        unique_vals, counts = np.unique(bootstrap_preds, return_counts=True)
        total_samples = len(bootstrap_preds)
        
        # Calculate percentage for each prediction category
        for val, count in zip(unique_vals, counts):
            summary_table[int(val), x2_cat] = (count / total_samples) * 100
    
    return summary_table

def bootstrap_predict_X2_from_X1_all_comb_summary(contingency_table, n_resamples=9999, confidence_level=0.95, method='BCa', random_state=None):
    """Calculate bootstrap summary classification table for predicting X2 from all X1 categories.
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        2D array representing contingency table of observed frequencies.
    n_resamples : int, default=9999
        Number of bootstrap resamples to generate.
    confidence_level : float, default=0.95
        Confidence level for interval calculation (between 0 and 1).
    method : {'percentile', 'basic', 'BCa'}, default='BCa'
        Method to calculate bootstrap confidence intervals.
    random_state : {None, int, numpy.random.Generator,
                   numpy.random.RandomState}, optional
        Random state for reproducibility.

    Returns
    -------
    summary : numpy.ndarray
        2D array showing the proportion of X2=j from CCR at each combination of categories of X1
    """
    n_rows, n_cols = contingency_table.shape
    
    # Get bootstrap predictions for each X1 category
    results = bootstrap_predict_X2_from_X1_vectorized(
        contingency_table, 
        np.arange(n_rows),
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state
    )
    
    # Initialize summary table
    summary_table = np.zeros((n_rows, n_cols))
    
    # For each X1 category
    for x1_cat in range(n_rows):
        bootstrap_preds = results[x1_cat].bootstrap_distribution
        
        # Count occurrences of each possible X2 category
        unique_vals, counts = np.unique(bootstrap_preds, return_counts=True)
        total_samples = len(bootstrap_preds)
        
        # Calculate percentage for each prediction category
        for val, count in zip(unique_vals, counts):
            summary_table[x1_cat, int(val)] = (count / total_samples) * 100
    
    return summary_table

def permutation_test_sccram(contingency_table, direction="X1_X2", alternative='greater', n_resamples=9999, random_state=None):
    """Performs permutation test for H0: SCCRAM = 0 (independence between variables).
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        2D array representing contingency table of observed frequencies
    direction : {'X1_X2', 'X2_X1'}, default='X1_X2'
        Direction of dependency to test
    alternative : {'greater', 'less', 'two-sided'}, default='greater'
        H1: SCCRAM > 0 (Default)
    n_resamples : int, default=9999
        Number of permutation resamples
    random_state : {None, int, numpy.random.Generator}, optional
        Random state for reproducibility
        
    Returns
    -------
    PermutationTestResult
        Object containing:
        - statistic: Observed SCCRAM value
        - pvalue: p-value from permutation test
        - null_distribution: SCCRAM values under null hypothesis
    
    Notes
    -----
    Tests independence between variables by permuting one variable while keeping
    the other fixed. Uses case-form data representation for permutations.
    """
    # Convert table to case form
    cases = contingency_to_case_form(contingency_table)
    
    # Split into X1 and X2 variables
    x1, x2 = cases[:, 0], cases[:, 1]
    data = (x1, x2)  # Keep X1 fixed, permute X2
    
    def sccram_stat(x1, x2, axis=0):
        # Handle batched data 
        if x1.ndim > 1:
            batch_size = x1.shape[0]
            cases = np.stack([np.column_stack((x1[i], x2[i])) 
                            for i in range(batch_size)])
        else:
            cases = np.column_stack((x1, x2))
            
        n_rows, n_cols = contingency_table.shape
        resampled_table = case_form_to_contingency(cases, n_rows, n_cols)
        
        # Handle single vs batched tables
        if resampled_table.ndim == 3:
            results = []
            for table in resampled_table:
                copula = CheckerboardCopula.from_contingency_table(table)
                if direction == "X1_X2":
                    results.append(copula.calculate_SCCRAM_X1_X2_vectorized())
                elif direction == "X2_X1":
                    results.append(copula.calculate_SCCRAM_X2_X1_vectorized())
                else:
                    raise ValueError("Invalid direction. Use 'X1_X2' or 'X2_X1'")
            return np.array(results)
        else:
            copula = CheckerboardCopula.from_contingency_table(resampled_table)
            if direction == "X1_X2":
                return copula.calculate_SCCRAM_X1_X2_vectorized()
            else:
                return copula.calculate_SCCRAM_X2_X1_vectorized()

    # Perform permutation test
    res = permutation_test(
        data,
        sccram_stat,
        permutation_type='pairings', # Permute pairings between variables
        n_resamples=n_resamples,
        alternative=alternative,
        random_state=random_state,
        vectorized=True
    )
    
    return res

def permutation_test_ccram(contingency_table, direction="X1_X2", alternative='greater', n_resamples=9999, random_state=None):
    """Performs permutation test for H0: CCRAM = 0 (independence between variables).
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        2D array representing contingency table of observed frequencies
    direction : {'X1_X2', 'X2_X1'}, default='X1_X2'
        Direction of dependency to test
    alternative : {'greater', 'less', 'two-sided'}, default='greater'
        H1: CCRAM > 0 (Default)
    n_resamples : int, default=9999
        Number of permutation resamples
    random_state : {None, int, numpy.random.Generator}, optional
        Random state for reproducibility
        
    Returns
    -------
    PermutationTestResult
        Object containing:
        - statistic: Observed CCRAM value
        - pvalue: p-value from permutation test
        - null_distribution: CCRAM values under null hypothesis
    
    Notes
    -----
    Tests independence between variables by permuting one variable while keeping
    the other fixed. Uses case-form data representation for permutations.
    """
    # Convert table to case form
    cases = contingency_to_case_form(contingency_table)
    
    # Split into X1 and X2 variables
    x1, x2 = cases[:, 0], cases[:, 1]
    data = (x1, x2)  # Keep X1 fixed, permute X2
    
    def ccram_stat(x1, x2, axis=0):
        # Handle batched data 
        if x1.ndim > 1:
            batch_size = x1.shape[0]
            cases = np.stack([np.column_stack((x1[i], x2[i])) 
                            for i in range(batch_size)])
        else:
            cases = np.column_stack((x1, x2))
            
        n_rows, n_cols = contingency_table.shape
        resampled_table = case_form_to_contingency(cases, n_rows, n_cols)
        
        # Handle single vs batched tables
        if resampled_table.ndim == 3:
            results = []
            for table in resampled_table:
                copula = CheckerboardCopula.from_contingency_table(table)
                if direction == "X1_X2":
                    results.append(copula.calculate_CCRAM_X1_X2_vectorized())
                elif direction == "X2_X1":
                    results.append(copula.calculate_CCRAM_X2_X1_vectorized())
                else:
                    raise ValueError("Invalid direction. Use 'X1_X2' or 'X2_X1'")
            return np.array(results)
        else:
            copula = CheckerboardCopula.from_contingency_table(resampled_table)
            if direction == "X1_X2":
                return copula.calculate_CCRAM_X1_X2_vectorized()
            else:
                return copula.calculate_CCRAM_X2_X1_vectorized()

    # Perform permutation test
    res = permutation_test(
        data,
        ccram_stat,
        permutation_type='pairings', # Permute pairings between variables
        n_resamples=n_resamples,
        alternative=alternative,
        random_state=random_state,
        vectorized=True
    )
    
    return res

if __name__ == "__main__":
    table = np.array([
        [0, 0, 20],
        [0, 10, 0],
        [20, 0, 0],
        [0, 10, 0],
        [0, 0, 20]
    ])
    res = bootstrap_ccram(table,"X2_X1", method="percentile")
    print(res.confidence_interval)
    print(res.standard_error)