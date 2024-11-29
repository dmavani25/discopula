import numpy as np

class CheckerboardCopula:
    """
    A class to calculate the checkerboard copula density and scores for ordinal random vectors.

    This class implements methods for computing copula densities, regression scores,
    and various association measures (CCRAM/SCCRAM) for ordinal data analysis.

    Attributes
    ----------
    P : numpy.ndarray
        Joint probability matrix.
    marginal_pdf_X1 : numpy.ndarray
        Marginal probability density function for X1.
    marginal_pdf_X2 : numpy.ndarray
        Marginal probability density function for X2.
    marginal_cdf_X1 : numpy.ndarray
        Marginal cumulative distribution function for X1.
    marginal_cdf_X2 : numpy.ndarray
        Marginal cumulative distribution function for X2.
    conditional_pmf_X2_given_X1 : numpy.ndarray
        Conditional probability mass function of X2 given X1.
    conditional_pmf_X1_given_X2 : numpy.ndarray
        Conditional probability mass function of X1 given X2.
    scores_X1 : numpy.ndarray
        Checkerboard copula scores for X1.
    scores_X2 : numpy.ndarray
        Checkerboard copula scores for X2.
            
        
    Example
    -------
    >>> import numpy as np
    >>> P = np.array([
            [0, 0, 2/8],
            [0, 1/8, 0],
            [2/8, 0, 0],
            [0, 1/8, 0],
            [0, 0, 2/8]
        ])  
    >>> copula = CheckerboardCopula(P)
    >>> sccram = copula.calculate_SCCRAM_X1_X2()

    References
    ----------
    - [1] Genest, C. and J. Nešlehová (2007). A primer on copulas for count data.
        Astin Bulletin 37(2), 475–515.
    - [2] Wei, Zheng & Kim, Daeyoung, 2021. "On exploratory analytic method for 
            multi-way contingency tables with an ordinal response variable and 
            categorical explanatory variables," Journal of Multivariate Analysis.
    """
    
    @classmethod
    def from_contingency_table(cls, contingency_table):
        """
        Create a CheckerboardCopula instance from a contingency table.

        Parameters
        ----------
        contingency_table : numpy.ndarray
            A 2D contingency table of counts/frequencies.

        Returns
        -------
        CheckerboardCopula
            A new instance initialized with the probability matrix.

        Raises
        ------
        ValueError
            If the input table contains negative values or all zeros.
            If the input table is not 2-dimensional.

        Examples
        --------
        >>> table = np.array([
            [0, 0, 20],
            [0, 10, 0],
            [20, 0, 0],
            [0, 10, 0],
            [0, 0, 20]
        ])
        >>> copula = CheckerboardCopula.from_contingency_table(table)
        """
        if not isinstance(contingency_table, np.ndarray):
            contingency_table = np.array(contingency_table)
            
        if contingency_table.ndim != 2:
            raise ValueError("Contingency table must be 2-dimensional")
            
        if np.any(contingency_table < 0):
            raise ValueError("Contingency table cannot contain negative values")
            
        total_count = contingency_table.sum()
        if total_count == 0:
            raise ValueError("Contingency table cannot be all zeros")
            
        P = contingency_table / total_count
        return cls(P)
    
    def __init__(self, P):
        """
        Initialize CheckerboardCopula with a joint probability matrix.

        Parameters
        ----------
        P : numpy.ndarray
            Joint probability matrix of shape (m, n). Values must be in [0, 1] 
            and sum to 1.

        Raises
        ------
        ValueError
            If P is not 2D.
            If P contains values outside [0, 1].
            If P does not sum to 1.

        Notes
        -----
        Calculates and stores:
        - Marginal PDFs and CDFs
        - Conditional PMFs
        - Checkerboard scores
        """
        if not isinstance(P, np.ndarray):
            P = np.array(P)
            
        if P.ndim != 2:
            raise ValueError("Probability matrix P must be 2-dimensional")
            
        if np.any(P < 0) or np.any(P > 1):
            raise ValueError("Probability matrix P must contain values between 0 and 1")
        
        if not np.allclose(P.sum(), 1.0, rtol=1e-10, atol=1e-10):
            raise ValueError("Probability matrix P must sum to 1")
        
        self.P = P
        
        # Calculate marginal distributions
        self.marginal_pdf_X1 = P.sum(axis=1) / P.sum()
        self.marginal_pdf_X2 = P.sum(axis=0) / P.sum()
        self.marginal_cdf_X1 = np.insert(np.cumsum(P.sum(axis=1)) / P.sum(), 0, 0)
        self.marginal_cdf_X2 = np.insert(np.cumsum(P.sum(axis=0)) / P.sum(), 0, 0)
        
        # Calculate conditional distributions
        self.conditional_pmf_X2_given_X1 = self.calculate_conditional_pmf_X2_given_X1()
        self.conditional_pmf_X1_given_X2 = self.calculate_conditional_pmf_X1_given_X2()
        
        # Calculate scores
        self.scores_X1 = self.calculate_checkerboard_scores(self.marginal_cdf_X1)
        self.scores_X2 = self.calculate_checkerboard_scores(self.marginal_cdf_X2)

    @property
    def contingency_table(self):
        """Get the contingency table by rescaling the probability matrix.
        
        This property converts the internal probability matrix (P) back to an 
        approximate contingency table of counts. Since the exact original counts
        cannot be recovered, it scales the probabilities by finding the smallest 
        non-zero probability and using its reciprocal as a multiplier.
        
        Returns
        -------
        numpy.ndarray
            A matrix of integer counts representing the contingency table.
            The values are rounded to the nearest integer after scaling.
        
        Notes
        -----
        The scaling process works by:
        1. Finding the smallest non-zero probability in the matrix
        2. Using its reciprocal as the scaling factor
        3. Multiplying all probabilities by this scale
        4. Rounding to nearest integers
        
        Warning
        -------
        This is an approximation of the original contingency table since the
        exact counts cannot be recovered from probabilities alone.
        """
        # Multiply by the smallest number that makes all entries close to integers
        scale = 1 / np.min(self.P[self.P > 0]) if np.any(self.P > 0) else 1
        return np.round(self.P * scale).astype(int)

    def calculate_conditional_pmf_X2_given_X1(self):
        """Calculate the conditional probability mass function of X2 given X1.
        
        Computes P(X2|X1) by dividing each element in the joint probability matrix
        by its corresponding row sum. This gives the conditional distribution of X2
        for each value of X1.

        Returns
        -------
        numpy.ndarray
            A matrix of shape (n_rows, n_cols) containing the conditional 
            probabilities P(X2|X1). Each row represents a value of X1, and each
            column represents the conditional probability of X2 given that X1 value.
            Rows that sum to zero in the original matrix will result in rows of 
            zeros in the output to avoid division by zero.

        Notes
        -----
        The conditional PMF is calculated as:
            P(X2|X1) = P(X1,X2) / P(X1)
        where P(X1) is obtained by summing P(X1,X2) over all values of X2 (row sums).
        
        See Also
        --------
        calculate_conditional_pmf_X1_given_X2 : Calculates P(X1|X2)
        """
        # Calculate row-wise probabilities
        row_sums = self.P.sum(axis=1, keepdims=True)
        # Avoid division by zero by using np.divide with setting zero where row_sums is zero
        conditional_pmf = np.divide(self.P, row_sums, out=np.zeros_like(self.P), where=row_sums!=0)
        return conditional_pmf

    def calculate_conditional_pmf_X1_given_X2(self):
        """Calculate the conditional probability mass function of X1 given X2.
        
        Computes P(X1|X2) by dividing each element in the joint probability matrix
        by its corresponding column sum. This gives the conditional distribution of X1
        for each value of X2.

        Returns
        -------
        numpy.ndarray
            A matrix of shape (n_rows, n_cols) containing the conditional 
            probabilities P(X1|X2). Each column represents a value of X2, and each
            row represents the conditional probability of X1 given that X2 value.
            Columns that sum to zero in the original matrix will result in columns 
            of zeros in the output to avoid division by zero.

        Notes
        -----
        The conditional PMF is calculated as:
            P(X1|X2) = P(X1, X2) / P(X2)
        where P(X2) is obtained by summing P(X1,X2) over all values of X1 (column sums).
        
        See Also
        --------
        calculate_conditional_pmf_X2_given_X1 : Calculates P(X2|X1)
        """
        # Calculate column-wise probabilities
        col_sums = self.P.sum(axis=0, keepdims=True)
        # Avoid division by zero by using np.divide with setting zero where col_sums is zero
        conditional_pmf = np.divide(self.P, col_sums, out=np.zeros_like(self.P), where=col_sums!=0)
        return conditional_pmf  # Return the matrix directly without transposing

    def lambda_function(self, u, ul, uj):
        """Calculate lambda function for checkerboard copula.
        
        Computes a piecewise linear function that maps input values to [0,1] based on
        lower and upper bounds. This lambda function is used in constructing the
        checkerboard copula.

        Parameters
        ----------
        u : float
            Input value to be transformed
        ul : float 
            Lower bound of the interval
        uj : float
            Upper bound of the interval

        Returns
        -------
        float
            Lambda value in [0,1], calculated as:
            - 0.0 if u <= ul
            - 1.0 if u >= uj
            - (u - ul)/(uj - ul) otherwise

        Notes
        -----
        This is a piecewise linear function that:
        1. Returns 0 for inputs below or at the lower bound
        2. Returns 1 for inputs above or at the upper bound
        3. Linearly interpolates between bounds for inputs in between

        See Also
        --------
        calculate_checkerboard_copula : Uses this lambda function for copula construction
        """
        if u <= ul:
            return 0.0
        elif u >= uj:
            return 1.0
        else:
            return (u - ul) / (uj - ul)
    
    def calculate_checkerboard_scores(self, marginal_cdf):
        """Calculate checkerboard copula scores for ordinal variable.

        Computes scores for ordinal variables by taking midpoints between consecutive
        values in the marginal cumulative distribution function (CDF). These scores
        are used in constructing the checkerboard copula.

        Parameters
        ----------
        marginal_cdf : numpy.ndarray
            Marginal CDF values in ascending order, representing the cumulative
            probabilities for each category of the ordinal variable.

        Returns 
        -------
        numpy.ndarray
            Array of scores computed as midpoints between consecutive CDF values.
            The length will be len(marginal_cdf) - 1.

        Notes
        -----
        Implements Definition 2 from reference paper. For each position j,
        the score is calculated as:
            score[j] = (CDF[j-1] + CDF[j]) / 2
        
        Examples
        --------
        >>> cdf = np.array([0.2, 0.5, 1.0])
        >>> scores = calculate_checkerboard_scores(cdf)

        See Also
        --------
        calculate_checkerboard_copula : Uses these scores in copula construction
        """
        scores = [(marginal_cdf[j - 1] + marginal_cdf[j]) / 2 for j in range(1, len(marginal_cdf))]
        return scores
    
    def calculate_regression_U2_on_U1(self, u1):
        """Calculate checkerboard copula regression of U2 on U1.

        Computes E[U2|U1=u1] for a given u1 value using the conditional 
        probability mass function.

        Parameters
        ----------
        u1 : float
            Value between 0 and 1 representing the U1 coordinate. This value
            determines which interval of the checkerboard copula to use for
            calculating the conditional expectation.

        Returns
        -------
        float
            Conditional expectation E[U2|U1=u1], representing the expected value
            of U2 given the specific U1 value.

        Raises
        ------
        ValueError
            If u1 is not between 0 and 1

        Notes
        -----
        The calculation follows these steps:
        1. Determines which interval u1 falls into using breakpoints
        2. Retrieves conditional PMF for that interval
        3. Computes expectation using scores and conditional probabilities

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> reg_value = copula.calculate_regression_U2_on_U1(0.5)

        See Also
        --------
        calculate_regression_U1_on_U2 : Computes the reverse regression E[U1|U2=u2]
        conditional_pmf_X2_given_X1 : Used to compute conditional probabilities
        """
        # Define the breakpoints from the marginal CDF
        breakpoints = self.marginal_cdf_X1[1:-1]  # Remove 0 and 1
        
        # Find which interval u1 falls into
        # searchsorted with side='left' gives us the index where u1 would be inserted
        interval_idx = np.searchsorted(breakpoints, u1, side='left')
        
        # Get the conditional PMF for the determined interval
        conditional_pmf = self.conditional_pmf_X2_given_X1[interval_idx]
        
        # Calculate regression value using the conditional PMF and scores
        regression_value = np.sum(conditional_pmf * self.scores_X2)
        
        return regression_value

    def calculate_regression_U2_on_U1_batched(self, u1_values):
        """Vectorized calculation of U2 on U1 regression for multiple u1 values.
        
        A batch-optimized version that computes conditional expectations E[U2|U1=u1]
        for multiple u1 values simultaneously using vectorized operations.

        Parameters
        ----------
        u1_values : numpy.ndarray
            Array of values between 0 and 1 representing U1 coordinates for which
            to compute the regression values.

        Returns
        -------
        numpy.ndarray
            Array of regression values E[U2|U1=u1] for each input u1, with the
            same shape as the input array.

        Notes
        -----
        Implementation steps:
        1. Converts input to numpy array
        2. Identifies intervals for all u1 values using vectorized searchsorted
        3. Computes regression values for unique intervals
        4. Maps results back to original input shape
        
        This vectorized implementation is significantly faster than calling
        calculate_regression_U2_on_U1() repeatedly for large input arrays.

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> u1s = np.array([0.2, 0.5, 0.8])
        >>> reg_values = copula.calculate_regression_U2_on_U1_batched(u1s)
        >>> # Returns array of regression values for each input

        See Also
        --------
        calculate_regression_U2_on_U1 : Single-value version of this method
        calculate_regression_U1_on_U2_batched : Batch computation of reverse regression
        """
        # Convert input to numpy array if it isn't already
        u1_values = np.asarray(u1_values)
        
        # Initialize output array
        results = np.zeros_like(u1_values, dtype=float)
        
        # Find intervals for all u1 values at once
        # Use searchsorted with side='left' to handle edge cases correctly
        intervals = np.searchsorted(self.marginal_cdf_X1[1:-1], u1_values, side='left')
        
        # Calculate regression values for each unique interval
        for interval_idx in np.unique(intervals):
            mask = (intervals == interval_idx)
            conditional_pmf = self.conditional_pmf_X2_given_X1[interval_idx]
            regression_value = np.sum(conditional_pmf * self.scores_X2)
            results[mask] = regression_value
                
        return results
    
    def calculate_regression_U1_on_U2(self, u2):
        """Calculate checkerboard copula regression of U1 on U2.

        Computes E[U1|U2=u2] for a given u2 value using the conditional 
        probability mass function.

        Parameters
        ----------
        u2 : float
            Value between 0 and 1 representing the U2 coordinate. This value
            determines which interval of the checkerboard copula to use for
            calculating the conditional expectation.

        Returns
        -------
        float
            Conditional expectation E[U1|U2=u2], representing the expected value
            of U1 given the specific U2 value.

        Raises
        ------
        ValueError
            If u2 is not between 0 and 1

        Notes
        -----
        The calculation follows these steps:
        1. Determines which interval u2 falls into using breakpoints
        2. Retrieves conditional PMF for that interval
        3. Computes expectation using scores and conditional probabilities

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> reg_value = copula.calculate_regression_U1_on_U2(0.5)
        >>> # Returns expected value of U1 given U2 = 0.5

        See Also
        --------
        calculate_regression_U2_on_U1 : Computes the reverse regression E[U2|U1=u1]
        conditional_pmf_X1_given_X2 : Used to compute conditional probabilities
        """
        # Define the breakpoints from the marginal CDF
        breakpoints = self.marginal_cdf_X2[1:-1]  # Remove 0 and 1
        # Find which interval u2 falls into
        interval_idx = np.searchsorted(breakpoints, u2, side='left')
        # Get the conditional PMF for the determined interval
        conditional_pmf = self.conditional_pmf_X1_given_X2[:,interval_idx]
        
        # Calculate regression value using the conditional PMF and scores
        regression_value = np.sum(conditional_pmf * self.scores_X1)
        return regression_value
    
    def calculate_regression_U1_on_U2_batched(self, u2_values):
        """Vectorized calculation of U1 on U2 regression for multiple u2 values.
        
        A batch-optimized version that computes conditional expectations E[U1|U2=u2]
        for multiple u2 values simultaneously using vectorized operations.

        Parameters
        ----------
        u2_values : numpy.ndarray or array-like
            Array of values between 0 and 1 representing U2 coordinates for which
            to compute the regression values.

        Returns
        -------
        numpy.ndarray
            Array of regression values E[U1|U2=u2] for each input u2, with the
            same shape as the input array.

        Notes
        -----
        Implementation steps:
        1. Converts input to numpy array
        2. Identifies intervals for all u2 values using vectorized searchsorted
        3. Computes regression values for unique intervals
        4. Maps results back to original input shape
        
        This vectorized implementation is significantly faster than calling
        calculate_regression_U1_on_U2() repeatedly for large input arrays.

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> u2s = np.array([0.2, 0.5, 0.8])
        >>> reg_values = copula.calculate_regression_U1_on_U2_batched(u2s)
        >>> # Returns array of regression values for each input

        See Also
        --------
        calculate_regression_U1_on_U2 : Single-value version of this method
        calculate_regression_U2_on_U1_batched : Batch computation of reverse regression
        """
        # Convert input to numpy array if it isn't already
        u2_values = np.asarray(u2_values)
        
        # Initialize output array
        results = np.zeros_like(u2_values, dtype=float)
        
        # Find intervals for all u2 values at once
        # Use searchsorted with side='left' to handle edge cases correctly
        intervals = np.searchsorted(self.marginal_cdf_X2[1:-1], u2_values, side='left')
        
        # Calculate regression values for each unique interval
        for interval_idx in np.unique(intervals):
            mask = (intervals == interval_idx)
            conditional_pmf = self.conditional_pmf_X1_given_X2[:,interval_idx]
            regression_value = np.sum(conditional_pmf * self.scores_X1)
            results[mask] = regression_value
            
        return results
    
    def calculate_CCRAM_X1_X2(self):
        """Calculate the Checkerboard Copula Regression Association Measure (CCRAM) for X1 --> X2.

        Computes the CCRAM measure that quantifies the directional association from
        X1 to X2 based on the nonlinear regression function of the checkerboard copula.

        Returns
        -------
        float
            CCRAM value in [0,1] where:
            - 0 indicates no association from X1 to X2
            - 1 indicates perfect association from X1 to X2

        Notes
        -----
        The calculation follows these steps:
        1. For each X1 category:
            - Gets probability P(X1)
            - Computes regression E[U2|U1] at corresponding U1 value
            - Calculates weighted squared deviation from 0.5
        2. Sums all weighted deviations
        3. Multiplies by 12 to scale to [0,1] range

        The formula is:
            CCRAM = 12 * sum(P(X1) * (E[U2|U1] - 0.5)^2)

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> ccram = copula.calculate_CCRAM_X1_X2()
        >>> # Returns association measure from X1 to X2

        See Also
        --------
        calculate_CCRAM_X2_X1 : Computes the reverse association measure X2 --> X1
        calculate_regression_U2_on_U1 : Used to compute conditional expectations
        """
        weighted_expectation = 0.0
        for p_x1, u1 in zip(self.marginal_pdf_X1, self.marginal_cdf_X1[1:]):
            regression_value = self.calculate_regression_U2_on_U1(u1)
            weighted_expectation += p_x1 * (regression_value - 0.5) ** 2
        return 12 * weighted_expectation
    
    def calculate_CCRAM_X1_X2_vectorized(self) -> float:
        """Calculate vectorized CCRAM (X1 -> X2).
        
        Computes the Checkerboard Copula Regression Association Measure
        for the relationship X1 -> X2 using vectorized operations for
        improved performance.
        
        Returns
        -------
        float
            The CCRAM value between 0 and 1, where:
            - 0 indicates no association from X1 to X2
            - 1 indicates perfect association from X1 to X2
                
        Notes
        -----
        This is the vectorized version of calculate_CCRAM_X1_X2() and should
        be preferred for better performance.
        
        The calculation uses the formula:
            CCRAM = 12 * sum(P(X1) * (E[U2|U1] - 0.5)^2)
        
        Implementation steps:
        1. Compute regression values E[U2|U1] for all X1 categories at once
        2. Calculate weighted squared deviations from 0.5
        3. Sum and scale by 12
        
        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> ccram = copula.calculate_CCRAM_X1_X2_vectorized()
        
        See Also
        --------
        calculate_CCRAM_X1_X2 : Non-vectorized version
        calculate_CCRAM_X2_X1_vectorized : Vectorized X2 -> X1 measure
        
        References
        ----------
        See paper section 3.2 for CCRAM definition
        """
        regression_values = self.calculate_regression_U2_on_U1_batched(self.marginal_cdf_X1[1:])
        weighted_expectation = np.sum(self.marginal_pdf_X1 * (regression_values - 0.5) ** 2)
        return 12 * weighted_expectation
    
    def calculate_CCRAM_X2_X1(self):
        """Calculate the Checkerboard Copula Regression Association Measure (CCRAM) for X2 --> X1.

        Computes the CCRAM measure that quantifies the directional association from
        X2 to X1 based on the nonlinear regression function of the checkerboard copula.

        Returns
        -------
        float
            CCRAM value in [0,1] where:
            - 0 indicates no association from X2 to X1
            - 1 indicates perfect association from X2 to X1

        Notes
        -----
        The calculation follows these steps:
        1. For each X2 category:
            - Gets probability P(X2)
            - Computes regression E[U1|U2] at corresponding U2 value
            - Calculates weighted squared deviation from 0.5
        2. Sums all weighted deviations
        3. Multiplies by 12 to scale to [0,1] range

        The formula is:
            CCRAM = 12 * sum(P(X2) * (E[U1|U2] - 0.5)^2)

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> ccram = copula.calculate_CCRAM_X2_X1()
        >>> # Returns association measure from X2 to X1

        See Also
        --------
        calculate_CCRAM_X1_X2 : Computes the reverse association measure X1 --> X2
        calculate_regression_U1_on_U2 : Used to compute conditional expectations
        calculate_CCRAM_X2_X1_vectorized : Vectorized version of this method
        """
        weighted_expectation = 0.0
        for p_x2, u2 in zip(self.marginal_pdf_X2, self.marginal_cdf_X2[1:]):
            regression_value = self.calculate_regression_U1_on_U2(u2)
            weighted_expectation += p_x2 * (regression_value - 0.5) ** 2
        return 12 * weighted_expectation
    
    def calculate_CCRAM_X2_X1_vectorized(self):
        """Calculate vectorized CCRAM (X2 -> X1).
        
        Computes the Checkerboard Copula Regression Association Measure
        for the relationship X2 -> X1 using vectorized operations for
        improved performance.

        Returns
        -------
        float
            The CCRAM value between 0 and 1, where:
            - 0 indicates no association from X2 to X1
            - 1 indicates perfect association from X2 to X1
                
        Notes
        -----
        This is the vectorized version of calculate_CCRAM_X2_X1() and should
        be preferred for better performance.
        
        The calculation uses the formula:
            CCRAM = 12 * sum(P(X2) * (E[U1|U2] - 0.5)^2)
        
        Implementation steps:
        1. Compute regression values E[U1|U2] for all X2 categories at once
        2. Calculate weighted squared deviations from 0.5
        3. Sum and scale by 12

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> ccram = copula.calculate_CCRAM_X2_X1_vectorized()
        >>> # Returns vectorized association measure from X2 to X1

        See Also
        --------
        calculate_CCRAM_X2_X1 : Non-vectorized version
        calculate_CCRAM_X1_X2_vectorized : Vectorized X1 -> X2 measure
        calculate_regression_U1_on_U2_batched : Used for vectorized regression
        """
        regression_values = self.calculate_regression_U1_on_U2_batched(self.marginal_cdf_X2[1:])
        weighted_expectation = np.sum(self.marginal_pdf_X2 * (regression_values - 0.5) ** 2)
        return 12 * weighted_expectation
    
    def calculate_sigma_sq_S_X2(self):
        """Calculate variance of the checkerboard copula score S for X2.

        Implements the formula:
        σ²_Sⱼ = Σᵢⱼ₌₁^Iⱼ û_ᵢⱼ₋₁ û_ᵢⱼ p̂₊ᵢⱼ₊/4

        Returns
        -------
        float
            Variance of score S for X2 variable, used in SCCRAM calculation

        Notes
        -----
        Implementation steps:
        1. Gets consecutive CDF values (û_ᵢⱼ₋₁, û_ᵢⱼ)
        2. For each category i:
            - Multiplies previous CDF × next CDF × PDF
            - Sums these products
        3. Divides sum by 4 to get final variance

        Mathematical components:
        - û_ᵢⱼ₋₁: Previous CDF value
        - û_ᵢⱼ: Next CDF value
        - p̂₊ᵢⱼ₊: Marginal PDF value
        
        References
        ----------
        See paper section 3.3 for variance formula derivation

        See Also
        --------
        calculate_sigma_sq_S_X1 : Calculates variance for X1 scores
        calculate_SCCRAM : Uses this variance in standardization
        """
        # Get consecutive CDF values
        u_prev = self.marginal_cdf_X2[:-1]  # û_ᵢⱼ₋₁
        u_next = self.marginal_cdf_X2[1:]   # û_ᵢⱼ
        
        # Calculate each term in the sum
        terms = []
        for i in range(len(self.marginal_pdf_X2)):
            if i < len(u_prev) and i < len(u_next):
                term = u_prev[i] * u_next[i] * self.marginal_pdf_X2[i]
                terms.append(term)
        
        # Calculate sigma_sq_S
        sigma_sq_S = sum(terms) / 4.0
        
        return sigma_sq_S
    
    def calculate_sigma_sq_S_X2_vectorized(self):
        """Calculate variance of score S for X2 using vectorized operations.

        Implements the formula σ²_Sⱼ = Σᵢⱼ₌₁^Iⱼ û_ᵢⱼ₋₁ û_ᵢⱼ p̂₊ᵢⱼ₊/4 using
        numpy's vectorized operations for improved performance.

        Returns
        -------
        float
            Variance of score S for X2 variable, used in SCCRAM calculation

        Notes
        -----
        Implementation steps:
        1. Gets consecutive CDF values as vectors:
            - û_ᵢⱼ₋₁: Previous CDF values (u_prev)
            - û_ᵢⱼ: Next CDF values (u_next)
        2. Performs element-wise multiplication:
            terms = u_prev * u_next * marginal_pdf
        3. Sums products and divides by 4

        This vectorized version is more efficient than iterative calculation,
        especially for large datasets.

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> var_s = copula.calculate_sigma_sq_S_X2_vectorized()

        See Also
        --------
        calculate_sigma_sq_S_X2 : Non-vectorized version
        calculate_sigma_sq_S_X1_vectorized : Vectorized variance for X1
        calculate_SCCRAM : Uses this variance in standardization
        """
        # Get consecutive CDF values
        u_prev = self.marginal_cdf_X2[:-1]  # û_ᵢⱼ₋₁
        u_next = self.marginal_cdf_X2[1:]   # û_ᵢⱼ
        
        # Vectorized multiplication of all terms
        terms = u_prev * u_next * self.marginal_pdf_X2
        
        # Calculate sigma_sq_S
        sigma_sq_S = np.sum(terms) / 4.0
        
        return sigma_sq_S
    
    def calculate_sigma_sq_S_X1(self):
        """Calculate variance of the checkerboard copula score S for X1.

        Implements the formula:
        σ²_Sⱼ = Σᵢⱼ₌₁^Iⱼ û_ᵢⱼ₋₁ û_ᵢⱼ p̂₊ᵢⱼ₊/4

        Returns
        -------
        float
            Variance of score S for X1 variable, used in SCCRAM calculation

        Notes
        -----
        Implementation steps:
        1. Gets consecutive CDF values (û_ᵢⱼ₋₁, û_ᵢⱼ):
            - u_prev: Previous CDF values
            - u_next: Next CDF values
        2. For each category i:
            - Multiplies previous CDF × next CDF × PDF
            - Accumulates products in terms list
        3. Divides sum by 4 to get final variance

        Mathematical components:
        - û_ᵢⱼ₋₁: Previous CDF value
        - û_ᵢⱼ: Next CDF value
        - p̂₊ᵢⱼ₊: Marginal PDF value

        See Also
        --------
        calculate_sigma_sq_S_X1_vectorized : Vectorized version of this method
        calculate_sigma_sq_S_X2 : Calculates variance for X2 scores
        calculate_SCCRAM : Uses this variance in standardization

        References
        ----------
        See paper section 3.3 for variance formula derivation
        """
        # Get consecutive CDF values
        u_prev = self.marginal_cdf_X1[:-1]  # û_ᵢⱼ₋₁
        u_next = self.marginal_cdf_X1[1:]  # û_ᵢⱼ
        
        # Calculate each term in the sum
        terms = []
        for i in range(len(self.marginal_pdf_X1)):
            if i < len(u_prev) and i < len(u_next):
                term = u_prev[i] * u_next[i] * self.marginal_pdf_X1[i]
                terms.append(term)
        
        # Calculate sigma_sq_S
        sigma_sq_S = sum(terms) / 4.0
        
        return sigma_sq_S
    
    def calculate_sigma_sq_S_X1_vectorized(self):
        """Calculate variance of score S for X1 using vectorized operations.

        Implements the formula σ²_Sⱼ = Σᵢⱼ₌₁^Iⱼ û_ᵢⱼ₋₁ û_ᵢⱼ p̂₊ᵢⱼ₊/4 using
        numpy's vectorized operations for improved performance.

        Returns
        -------
        float
            Variance of score S for X1 variable, used in SCCRAM calculation

        Notes
        -----
        Implementation steps:
        1. Gets consecutive CDF values as vectors:
            - û_ᵢⱼ₋₁: Previous CDF values (u_prev)
            - û_ᵢⱼ: Next CDF values (u_next)
        2. Performs element-wise multiplication:
            terms = u_prev * u_next * marginal_pdf
        3. Sums products and divides by 4

        This vectorized version is more efficient than iterative calculation,
        especially for large datasets.

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> var_s = copula.calculate_sigma_sq_S_X1_vectorized()

        See Also
        --------
        calculate_sigma_sq_S_X1 : Non-vectorized version
        calculate_sigma_sq_S_X2_vectorized : Vectorized variance for X2
        calculate_SCCRAM : Uses this variance in standardization

        References
        ----------
        See paper section 3.3 for formula derivation
        """
        # Get consecutive CDF values
        u_prev = self.marginal_cdf_X1[:-1]  # û_ᵢⱼ₋₁
        u_next = self.marginal_cdf_X1[1:] # û_ᵢⱼ
        
        # Vectorized multiplication of all terms
        terms = u_prev * u_next * self.marginal_pdf_X1
        
        # Calculate sigma_sq_S
        sigma_sq_S = np.sum(terms) / 4.0
        
        return sigma_sq_S
    
    def calculate_SCCRAM_X1_X2(self):
        """Calculate standardized CCRAM for X1 -> X2 relationship.

        The standardized measure accounts for the variance in the scores
        and provides a normalized association measure.

        Returns
        -------
        float
            SCCRAM value between 0 and 1, where:
            - 0 indicates no standardized association from X1 to X2
            - 1 indicates perfect standardized association from X1 to X2

        Notes
        -----
        Implementation formula:
            SCCRAM = CCRAM / (12 * σ²_S)

        where:
        - CCRAM is the unstandardized measure
        - σ²_S is the variance of scores for X2
        - 12 is the scaling factor

        The standardization adjusts for the variance in the ordinal scores,
        making the measure more comparable across different variable pairs.

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> sccram = copula.calculate_SCCRAM_X1_X2()

        See Also
        --------
        calculate_CCRAM_X1_X2 : Computes unstandardized measure
        calculate_sigma_sq_S_X2 : Computes score variance
        calculate_SCCRAM_X2_X1 : Computes reverse standardized measure
        """
        ccram = self.calculate_CCRAM_X1_X2()
        sigma_sq_S = self.calculate_sigma_sq_S_X2()
        
        # Handle edge case where variance is zero
        if sigma_sq_S < 1e-10:  # Use small threshold instead of exact 0
            if ccram < 1e-10:
                return 0.0  # No association case
            else:
                return 1.0  # Perfect association case
        
        return ccram / (12 * sigma_sq_S)
    
    def calculate_SCCRAM_X1_X2_vectorized(self):
        """Calculate standardized CCRAM for X1 -> X2 using vectorized operations.

        Computes the Standardized Checkerboard Copula Regression Association 
        Measure (SCCRAM) for X1 --> X2 relationship using vectorized operations
        for improved performance.

        Returns
        -------
        float
            SCCRAM value between 0 and 1, where:
            - 0 indicates no standardized association from X1 to X2
            - 1 indicates perfect standardized association from X1 to X2

        Notes
        -----
        Implementation formula:
            SCCRAM = CCRAM / (12 * σ²_S)

        Uses vectorized operations for both:
        1. CCRAM calculation
        2. Score variance calculation (σ²_S)

        This vectorized version is more efficient than the non-vectorized
        version for large datasets.

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> sccram = copula.calculate_SCCRAM_X1_X2_vectorized()

        See Also
        --------
        calculate_SCCRAM_X1_X2 : Non-vectorized version
        calculate_CCRAM_X1_X2_vectorized : Vectorized CCRAM calculation
        calculate_sigma_sq_S_X2_vectorized : Vectorized variance calculation
        """
        ccram_vectorized = self.calculate_CCRAM_X1_X2_vectorized()
        sigma_sq_S_vectorized = self.calculate_sigma_sq_S_X2_vectorized()
        
        # Handle edge case where variance is zero
        if sigma_sq_S_vectorized < 1e-10:  # Use small threshold instead of exact 0
            if ccram_vectorized < 1e-10:
                return 0.0  # No association case
            else:
                return 1.0  # Perfect association case
                
        return ccram_vectorized / (12 * sigma_sq_S_vectorized)


    def calculate_SCCRAM_X2_X1(self):
        """Calculate standardized CCRAM for X2 -> X1 relationship.

        The standardized measure accounts for the variance in the scores
        and provides a normalized association measure.

        Returns
        -------
        float
            SCCRAM value between 0 and 1, where:
            - 0 indicates no standardized association from X2 to X1
            - 1 indicates perfect standardized association from X2 to X1

        Notes
        -----
        Implementation formula:
            SCCRAM = CCRAM / (12 * σ²_S)
        
        where:
        - CCRAM is the unstandardized measure
        - σ²_S is the variance of scores for X1
        - 12 is the scaling factor

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> sccram = copula.calculate_SCCRAM_X2_X1()

        See Also
        --------
        calculate_CCRAM_X2_X1 : Computes unstandardized measure
        calculate_sigma_sq_S_X1 : Computes score variance
        calculate_SCCRAM_X1_X2 : Computes reverse standardized measure
        """
        ccram = self.calculate_CCRAM_X2_X1()
        sigma_sq_S = self.calculate_sigma_sq_S_X1()
        
        # Handle edge case where variance is zero
        if sigma_sq_S < 1e-10:  # Use small threshold instead of exact 0
            if ccram < 1e-10:
                return 0.0  # No association case
            else:
                return 1.0  # Perfect association case
        
        return ccram / (12 * sigma_sq_S)
    
    def calculate_SCCRAM_X2_X1_vectorized(self):
        """Calculate standardized CCRAM for X2 -> X1 using vectorized operations.

        Computes the Standardized Checkerboard Copula Regression Association 
        Measure (SCCRAM) for X2 --> X1 relationship using vectorized operations
        for improved performance.

        Returns
        -------
        float
            SCCRAM value between 0 and 1, where:
            - 0 indicates no standardized association from X2 to X1
            - 1 indicates perfect standardized association from X2 to X1

        Notes
        -----
        Implementation formula:
            SCCRAM = CCRAM / (12 * σ²_S)

        Uses vectorized operations for both:
        1. CCRAM calculation
        2. Score variance calculation (σ²_S)

        This vectorized version is more efficient than the non-vectorized
        version for large datasets.

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> sccram = copula.calculate_SCCRAM_X2_X1_vectorized()

        See Also
        --------
        calculate_SCCRAM_X2_X1 : Non-vectorized version
        calculate_CCRAM_X2_X1_vectorized : Vectorized CCRAM calculation
        calculate_sigma_sq_S_X1_vectorized : Vectorized variance calculation
        """
        ccram_vectorized = self.calculate_CCRAM_X2_X1_vectorized()
        sigma_sq_S_vectorized = self.calculate_sigma_sq_S_X1_vectorized()
        
        # Handle edge case where variance is zero
        if sigma_sq_S_vectorized < 1e-10:  # Use small threshold instead of exact 0
            if ccram_vectorized < 1e-10:
                return 0.0  # No association case
            else:
                return 1.0  # Perfect association case
                
        return ccram_vectorized / (12 * sigma_sq_S_vectorized)
    
    def get_predicted_category(self, regression_value, marginal_cdf):
        """Get predicted category based on regression value.
        
        Parameters
        ----------
        regression_value : float
            Value from CCR regression function (between 0 and 1)
        marginal_cdf : array-like 
            Marginal CDF values defining category boundaries
            
        Returns
        -------
        int
            Index of predicted category (0-based)
            
        Notes
        -----
        Finds category i* where u_{i*-1} < regression_value ≤ u_{i*}
        """
        # searchsorted finds index where regression_value would be inserted 
        # to maintain order - this gives us the category index
        i_star = np.searchsorted(marginal_cdf[1:-1], regression_value, side='left')
        return i_star
    
    def get_predicted_category_batched(self, regression_values, marginal_cdf):
        """Get predicted categories for a batch of regression values.
        
        Parameters
        ----------
        regression_values : numpy.ndarray or array-like
            Array of regression values to predict categories for (between 0 and 1)
        marginal_cdf : array-like
            Marginal CDF values defining category boundaries
            
        Returns
        -------
        numpy.ndarray
        Array of predicted category indices (0-based)
        """
        # searchsorted finds index where regression_value would be inserted
        # to maintain order - this gives us the category index
        return np.searchsorted(marginal_cdf[1:-1], regression_values, side='left')

    def predict_X2_from_X1(self, x1_category):
        """Predict category of X2 given category of X1.
        
        Parameters
        ----------
        x1_category : int
            Category index of X1 (0-based)
            
        Returns
        -------
        int
            Predicted category index for X2 (0-based)
        """
        # Get corresponding u1 value for given X1 category
        u1 = self.marginal_cdf_X1[x1_category + 1]  # +1 since CDF includes 0
        
        # Get regression value 
        u2_star = self.calculate_regression_U2_on_U1(u1)
        
        # Get predicted category
        return self.get_predicted_category(u2_star, self.marginal_cdf_X2)
    
    def predict_X2_from_X1_batched(self, x1_categories):
        """Vectorized prediction of X2 given X1 categories.
        
        Predicts the category of X2 given categories of X1 for multiple
        X1 values simultaneously using vectorized operations.

        Parameters
        ----------
        x1_categories : numpy.ndarray or array-like
            Array of category indices for X1 (0-based) for which to predict X2.

        Returns
        -------
        numpy.ndarray
            Array of predicted category indices for X2 (0-based) with the same shape
            as the input array.

        Notes
        -----
        Implementation steps:
        1. Convert input to numpy array if it isn't already
        2. Get corresponding u1 values for all X1 categories
        3. Compute regression values E[U2|U1] for all u1 values
        4. Get predicted categories for all regression values

        This vectorized version is more efficient than calling predict_X2_from_X1()
        repeatedly for large input arrays.

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> x1s = np.array([0, 1, 1])
        >>> x2s = copula.predict_X2_from_X1_batched(x1s)

        See Also
        --------
        predict_X2_from_X1 : Single-value version of this method
        predict_X1_from_X2_batched : Vectorized reverse prediction
        """
        # Convert input to numpy array if it isn't already
        x1_categories = np.asarray(x1_categories)
        
        # Get corresponding u1 values for all X1 categories
        u1_values = self.marginal_cdf_X1[x1_categories + 1]
        
        # Compute regression values for all u1 values
        u2_star_values = self.calculate_regression_U2_on_U1_batched(u1_values)
        
        # Get predicted categories for all regression values
        return self.get_predicted_category_batched(u2_star_values, self.marginal_cdf_X2)
    
    def predict_X1_from_X2(self, x2_category):
        """Predict category of X1 given category of X2.
        
        Parameters
        ----------
        x2_category : int
            Category index of X2 (0-based)
            
        Returns
        -------
        int
            Predicted category index for X1 (0-based)
        """
        # Get corresponding u2 value for given X2 category
        u2 = self.marginal_cdf_X2[x2_category + 1]
        
        # Get regression value
        u1_star = self.calculate_regression_U1_on_U2(u2)
        
        # Get predicted category
        return self.get_predicted_category(u1_star, self.marginal_cdf_X1)
    
    def predict_X1_from_X2_batched(self, x2_categories):
        """Vectorized prediction of X1 given X2 categories.
        
        Predicts the category of X1 given categories of X2 for multiple
        X2 values simultaneously using vectorized operations.

        Parameters
        ----------
        x2_categories : numpy.ndarray or array-like
            Array of category indices for X2 (0-based) for which to predict X1.

        Returns
        -------
        numpy.ndarray
            Array of predicted category indices for X1 (0-based) with the same shape
            as the input array.

        Notes
        -----
        Implementation steps:
        1. Convert input to numpy array if it isn't already
        2. Get corresponding u2 values for all X2 categories
        3. Compute regression values E[U1|U2] for all u2 values
        4. Get predicted categories for all regression values

        This vectorized version is more efficient than calling predict_X1_from_X2()
        repeatedly for large input arrays.

        Examples
        --------
        >>> copula = CheckerboardCopula(P)
        >>> x1s = np.array([0, 1, 1])
        >>> x2s = copula.predict_X1_from_X2_batched(x1s)

        See Also
        --------
        predict_X1_from_X2 : Single-value version of this method
        predict_X2_from_X1_batched : Vectorized reverse prediction
        """
        # Convert input to numpy array if it isn't already
        x2_categories = np.asarray(x2_categories)
        
        # Get corresponding u2 values for all X2 categories
        u2_values = self.marginal_cdf_X2[x2_categories + 1]
        
        # Compute regression values for all u2 values
        u1_star_values = self.calculate_regression_U1_on_U2_batched(u2_values)
        
        # Get predicted categories for all regression values
        return self.get_predicted_category_batched(u1_star_values, self.marginal_cdf_X1)