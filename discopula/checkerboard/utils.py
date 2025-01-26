import numpy as np
        
def contingency_to_case_form(contingency_table):
    """Convert a contingency table to case-form data representation.
    
    Transforms a contingency table of counts into a sequence of individual cases,
    where each case represents one observation with its row and column indices.
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        A 2D array where entry (i,j) represents the count/frequency of 
        observations in category i of variable X1 and category j of variable X2.
        Must contain non-negative integers.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (N, 2) where N is the total count of observations.
        Each row contains [i, j] indices representing one observation in 
        category i of X1 and category j of X2.

    Examples
    --------
    >>> table = np.array([[1, 0], [0, 2]])
    >>> cases = contingency_to_case_form(table)
    >>> print(cases)
    [[0 0]
     [1 1]
     [1 1]]

    Notes
    -----
    - Zero entries in the contingency table are skipped
    - The output array length equals the sum of all entries in the table
    - Each non-zero entry (i,j) with count k generates k copies of [i,j]
    - Used for converting data to format needed by bootstrap resampling

    See Also
    --------
    case_form_to_contingency : Inverse operation converting cases to table
    get_bootstrap_ci_for_ccram : Uses this function for bootstrap preparation
    """
    rows, cols = contingency_table.shape
    cases = []
    for i in range(rows):
        for j in range(cols):
            count = contingency_table[i, j]
            if count > 0:
                cases.extend([[i, j]] * count)
    return np.array(cases)

def case_form_to_contingency(cases, n_rows, n_cols):
    """Convert case-form data to contingency table format.
    
    Transforms sequence of cases (individual observations) back into a contingency
    table. Handles both single sample and batched (multiple resampled) data.
    
    Parameters
    ----------
    cases : numpy.ndarray
        Either 2D array of shape (N, 2) for single sample or 3D array of shape
        (n_samples, N, 2) for batched data. Each case is represented by [i,j]
        indices for categories of X1 and X2.
    n_rows : int
        Number of rows (categories of X1) in output contingency table.
    n_cols : int
        Number of columns (categories of X2) in output contingency table.

    Returns
    -------
    numpy.ndarray
        If input is 2D: contingency table of shape (n_rows, n_cols)
        If input is 3D: batch of tables of shape (n_samples, n_rows, n_cols)
        Each entry (i,j) contains count of cases with indices [i,j].

    Examples
    --------
    >>> cases = np.array([[0, 0], [1, 1], [1, 1]])
    >>> table = case_form_to_contingency(cases, 2, 2)
    >>> print(table)
    [[1 0]
     [0 2]]

    Notes
    -----
    - Handles both single sample and batched data automatically
    - For batched data, processes each sample independently
    - Output dimensions determined by n_rows and n_cols parameters
    - Used primarily in bootstrap resampling calculations

    See Also
    --------
    contingency_to_case_form : Inverse operation converting table to cases
    get_bootstrap_ci_for_ccram : Uses this function in bootstrap process
    """
    if cases.ndim == 3:  # Handling batched data
        n_samples = cases.shape[0]
        tables = np.zeros((n_samples, n_rows, n_cols), dtype=int)
        for k in range(n_samples):
            for case in cases[k]:
                i, j = case
                tables[k, i, j] += 1
        return tables
    else:  # Single sample
        table = np.zeros((n_rows, n_cols), dtype=int)
        for case in cases:
            i, j = case
            table[i, j] += 1
        return table

def gen_contingency_to_case_form(contingency_table: np.ndarray) -> np.ndarray:
    """Convert N-dimensional contingency table to case form.
    
    Parameters
    ----------
    contingency_table : np.ndarray
        N-dimensional contingency table
        
    Returns
    -------
    np.ndarray
        Array of cases where each row represents coordinates
    """
    # Get indices of non-zero elements
    indices = np.nonzero(contingency_table)
    counts = contingency_table[indices]
    
    # Create cases list
    cases = []
    for idx, count in zip(zip(*indices), counts):
        cases.extend([list(idx)] * int(count))
    
    return np.array(cases)

def gen_case_form_to_contingency(cases: np.ndarray, 
                                shape: tuple,
                                axis_order: list = None) -> np.ndarray:
    """Convert cases to contingency table with specified axis ordering.
    
    Parameters
    ----------
    cases : np.ndarray
        Array of cases where each row is a sample
    shape : tuple
        Shape of output contingency table
    axis_order : list, optional
        Order of axes for reconstruction
    
    Returns
    -------
    np.ndarray
        Reconstructed contingency table
    """
    if axis_order is None:
        axis_order = list(range(cases.shape[1]))
        
    table = np.zeros(shape, dtype=int)
    
    # Handle both 2D and 3D cases
    if cases.ndim == 3:
        # For batched data
        for batch in cases:
            for case in batch:
                idx = tuple(int(x) for x in case[axis_order])
                table[idx] += 1
    else:
        # For single batch
        for case in cases:
            idx = tuple(int(x) for x in case[axis_order])
            table[idx] += 1
            
    return table