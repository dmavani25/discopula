import numpy as np

class GenericCheckerboardCopula:
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
        """Initialize with joint probability matrix P."""
        if not isinstance(P, np.ndarray):
            P = np.array(P)
            
        if P.ndim != 2:
            raise ValueError("P must be 2-dimensional")
            
        if np.any(P < 0) or np.any(P > 1):
            raise ValueError("P must contain values in [0,1]")
            
        if not np.allclose(P.sum(), 1.0):
            raise ValueError("P must sum to 1")
            
        self.P = P
        self.ndim = P.ndim
        
        # Calculate and store marginals for each axis
        self.marginal_pdfs = {}
        self.marginal_cdfs = {}
        self.scores = {}
        
        for axis in range(self.ndim):
            # Calculate marginal PDF
            pdf = P.sum(axis=tuple(i for i in range(self.ndim) if i != axis))
            self.marginal_pdfs[axis] = pdf
            
            # Calculate marginal CDF
            cdf = np.insert(np.cumsum(pdf), 0, 0)
            self.marginal_cdfs[axis] = cdf
            
            # Calculate scores
            self.scores[axis] = self._calculate_scores(cdf)
            
        # Store conditional PMFs
        self.conditional_pmfs = {}
        
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
    
    def calculate_CCRAM(self, from_axis, to_axis, is_scaled=False):
        """Calculate (Standardized) Checkerboard Copula Regression Association Measure.
        
        Parameters
        ----------
        from_axis : int
            Source axis for directional association
        to_axis : int
            Target axis for directional association
        is_scaled : bool, optional
            Whether to return standardized measure (default: False)
            
        Returns
        -------
        float
            (S)CCRAM value in [0,1] indicating strength of association
        """
        weighted_expectation = 0.0
        for p_x, u in zip(self.marginal_pdfs[from_axis], 
                        self.marginal_cdfs[from_axis][1:]):
            regression_value = self._calculate_regression(
                target_axis=to_axis,
                given_axis=from_axis, 
                given_value=u
            )
            weighted_expectation += p_x * (regression_value - 0.5) ** 2
        
        ccram = 12 * weighted_expectation
        
        if not is_scaled:
            return ccram
            
        # Calculate scaled version
        sigma_sq_S = self._calculate_sigma_sq_S(to_axis)
        if sigma_sq_S < 1e-10:
            return 1.0 if ccram >= 1e-10 else 0.0
        return ccram / (12 * sigma_sq_S)

    def calculate_CCRAM_vectorized(self, from_axis, to_axis, is_scaled=False):
        """Calculate (Standardized) CCRAM using vectorized operations.
        
        Parameters
        ----------
        from_axis : int
            Source axis for directional association
        to_axis : int
            Target axis for directional association
        is_scaled : bool, optional
            Whether to return standardized measure (default: False)
            
        Returns
        -------
        float
            (S)CCRAM value in [0,1] indicating strength of association
        """
        regression_values = self._calculate_regression_batched(
            target_axis=to_axis,
            given_axis=from_axis,
            given_values=self.marginal_cdfs[from_axis][1:]
        )
        weighted_expectation = np.sum(
            self.marginal_pdfs[from_axis] * (regression_values - 0.5) ** 2
        )
        ccram = 12 * weighted_expectation
        
        if not is_scaled:
            return ccram
            
        # Calculate scaled version
        sigma_sq_S = self._calculate_sigma_sq_S_vectorized(to_axis)
        if sigma_sq_S < 1e-10:
            return 1.0 if ccram >= 1e-10 else 0.0
        return ccram / (12 * sigma_sq_S)

    def predict_category(self, source_category, from_axis, to_axis):
        """Predict category for target axis given source category.
        
        Parameters
        ----------
        source_category : int
            Category index of source axis (0-based)
        from_axis : int
            Source axis index
        to_axis : int
            Target axis index
                
        Returns
        -------
        int
            Predicted category index for target axis (0-based)
        """
        # Get corresponding u value for source category
        u_source = self.marginal_cdfs[from_axis][source_category + 1]
        
        # Get regression value
        u_target = self._calculate_regression(
            target_axis=to_axis,
            given_axis=from_axis,
            given_value=u_source
        )
        
        # Get predicted category
        return self._get_predicted_category(u_target, self.marginal_cdfs[to_axis])

    def predict_category_batched(self, source_categories, from_axis, to_axis):
        """Vectorized prediction of target categories.
        
        Parameters
        ----------
        source_categories : numpy.ndarray
            Array of source category indices (0-based)
        from_axis : int
            Source axis index
        to_axis : int
            Target axis index
            
        Returns
        -------
        numpy.ndarray
            Array of predicted category indices for target axis
        """
        # Convert input to numpy array
        source_categories = np.asarray(source_categories)
        
        # Get corresponding u values for all source categories
        u_source_values = self.marginal_cdfs[from_axis][source_categories + 1]
        
        # Compute regression values for all u values
        u_target_values = self._calculate_regression_batched(
            target_axis=to_axis,
            given_axis=from_axis,
            given_values=u_source_values
        )
        
        # Get predicted categories
        return self._get_predicted_category_batched(
            u_target_values, 
            self.marginal_cdfs[to_axis]
        )
    
    def _calculate_conditional_pmf(self, target_axis, given_axes):
        """Calculate conditional probability mass function.
        
        Parameters
        ----------
        target_axis : int
            The axis for which to calculate conditional probabilities
        given_axes : list of int
            The axes that are being conditioned on
            
        Returns
        -------
        numpy.ndarray
            Array containing conditional probabilities P(target|given)
            
        Notes
        -----
        Calculates P(target|given) = P(target,given) / P(given)
        """
        if not isinstance(given_axes, (list, tuple)):
            given_axes = [given_axes]
            
        # Key for storing in conditional_pmfs dict
        key = (target_axis, tuple(sorted(given_axes)))
        
        # Return cached result if available
        if key in self.conditional_pmfs:
            return self.conditional_pmfs[key]
        
        # Calculate axes to sum over (marginalize)
        all_axes = set(range(self.ndim))
        keep_axes = set([target_axis] + list(given_axes))
        sum_axes = tuple(all_axes - keep_axes)
        
        # Calculate joint probability P(target,given)
        if sum_axes:
            joint_prob = self.P.sum(axis=sum_axes)
        else:
            joint_prob = self.P
            
        # Calculate marginal probability P(given)
        marginal_prob = joint_prob.sum(axis=target_axis, keepdims=True)
        
        # Calculate conditional probability P(target|given)
        with np.errstate(divide='ignore', invalid='ignore'):
            conditional_prob = np.divide(
                joint_prob, 
                marginal_prob,
                out=np.zeros_like(joint_prob),
                where=marginal_prob!=0
            )
        
        # Cache and return result
        self.conditional_pmfs[key] = conditional_prob
        return conditional_prob
    
    def _calculate_regression(self, target_axis, given_axis, given_value):
        """Calculate regression E[target|given=value].
        
        Parameters
        ----------
        target_axis : int
            Axis for which to calculate expected value
        given_axis : int  
            Conditioning axis
        given_value : float
            Value between 0 and 1 for conditioning variable

        Returns
        -------
        float
            Conditional expectation E[target|given=value]
        """
        # Get breakpoints from marginal CDF
        breakpoints = self.marginal_cdfs[given_axis][1:-1]
        
        # Find interval index
        interval_idx = np.searchsorted(breakpoints, given_value, side='left')
        
        # Get conditional PMF
        conditional_pmf = self._calculate_conditional_pmf(
            target_axis=target_axis,
            given_axes=[given_axis]
        )
        
        # Select appropriate slice of conditional PMF
        if given_axis == 0:
            pmf_slice = conditional_pmf[interval_idx]
        else:
            pmf_slice = conditional_pmf[:,interval_idx]
        
        # Calculate regression using scores and PMF
        regression_value = np.sum(pmf_slice * self.scores[target_axis])
        
        return regression_value

    def _calculate_regression_batched(self, target_axis, given_axis, given_values):
        """Vectorized calculation of regression for multiple values.
        
        Parameters
        ----------
        target_axis : int
            Axis for which to calculate expected values
        given_axis : int
            Conditioning axis  
        given_values : numpy.ndarray
            Array of values between 0 and 1

        Returns
        -------
        numpy.ndarray
            Array of regression values with same shape as given_values
        """
        given_values = np.asarray(given_values)
        results = np.zeros_like(given_values, dtype=float)
        
        # Get breakpoints
        breakpoints = self.marginal_cdfs[given_axis][1:-1]
        
        # Find intervals for all values
        intervals = np.searchsorted(breakpoints, given_values, side='left')
        
        # Get conditional PMF
        conditional_pmf = self._calculate_conditional_pmf(
            target_axis=target_axis, 
            given_axes=[given_axis]
        )
        
        # Calculate for each unique interval
        for interval_idx in np.unique(intervals):
            mask = (intervals == interval_idx)
            if given_axis == 0:
                pmf_slice = conditional_pmf[interval_idx]
            else:
                pmf_slice = conditional_pmf[:,interval_idx]
                
            regression_value = np.sum(pmf_slice * self.scores[target_axis])
            results[mask] = regression_value
            
        return results
    
    def _calculate_scores(self, marginal_cdf):
        """Calculate checkerboard scores from marginal CDF."""
        return [(marginal_cdf[j-1] + marginal_cdf[j])/2 
                for j in range(1, len(marginal_cdf))]
    
    def _lambda_function(self, u, ul, uj):
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
        
    def _get_predicted_category(self, regression_value, marginal_cdf):
        """Get predicted category based on regression value.
        
        Parameters
        ----------
        regression_value : float
            Value from regression function (between 0 and 1)
        marginal_cdf : array-like 
            Marginal CDF values defining category boundaries
            
        Returns
        -------
        int
            Index of predicted category (0-based)
        """
        return np.searchsorted(marginal_cdf[1:-1], regression_value, side='left')

    def _get_predicted_category_batched(self, regression_values, marginal_cdf):
        """Get predicted categories for multiple regression values.
        
        Parameters
        ----------
        regression_values : numpy.ndarray
            Array of regression values to predict categories for
        marginal_cdf : array-like
            Marginal CDF values defining category boundaries
            
        Returns
        -------
        numpy.ndarray
            Array of predicted category indices (0-based)
        """
        return np.searchsorted(marginal_cdf[1:-1], regression_values, side='left')
    
    def _calculate_sigma_sq_S(self, axis):
        """Calculate variance of score S for given axis.
        
        Parameters
        ----------
        axis : int
            Axis for which to calculate score variance
            
        Returns
        -------
        float
            Variance of score S for given axis
        """
        # Get consecutive CDF values
        u_prev = self.marginal_cdfs[axis][:-1]
        u_next = self.marginal_cdfs[axis][1:]
        
        # Calculate each term in the sum
        terms = []
        for i in range(len(self.marginal_pdfs[axis])):
            if i < len(u_prev) and i < len(u_next):
                term = u_prev[i] * u_next[i] * self.marginal_pdfs[axis][i]
                terms.append(term)
        
        # Calculate sigma_sq_S
        sigma_sq_S = sum(terms) / 4.0
        return sigma_sq_S

    def _calculate_sigma_sq_S_vectorized(self, axis):
        """Calculate variance of score S using vectorized operations.
        
        Parameters
        ----------
        axis : int
            Axis for which to calculate score variance
            
        Returns
        -------
        float
            Variance of score S for given axis
        """
        # Get consecutive CDF values
        u_prev = self.marginal_cdfs[axis][:-1]
        u_next = self.marginal_cdfs[axis][1:]
        
        # Vectorized multiplication of all terms
        terms = u_prev * u_next * self.marginal_pdfs[axis]
        
        # Calculate sigma_sq_S
        sigma_sq_S = np.sum(terms) / 4.0
        return sigma_sq_S