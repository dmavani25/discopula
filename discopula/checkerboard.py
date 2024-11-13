import numpy as np

class CheckerboardCopula:
    """
    A class to calculate the checkerboard copula density and scores for ordinal random vectors,
    including regression functionality for conditional copula scores.
    """

    def __init__(self, P):
        """
        Initializes the CheckerboardCopula with the joint probability matrix P and
        calculates the marginal cumulative distribution functions (CDFs).
        
        Args:
            P (numpy.ndarray): The joint probability matrix.
        """
        self.P = P
        # Normalized cumulative sums for marginal CDFs to ensure they are proper CDFs ranging from 0 to 1.
        self.marginal_cdf_X1 = np.insert(np.cumsum(P.sum(axis=1)) / P.sum(), 0, 0)  # Marginal CDF for X1
        self.marginal_cdf_X2 = np.insert(np.cumsum(P.sum(axis=0)) / P.sum(), 0, 0)  # Marginal CDF for X2
        self.conditional_pmf_X2_given_X1 = self.calculate_conditional_pmf_X2_given_X1()
        self.conditional_pmf_X1_given_X2 = self.calculate_conditional_pmf_X1_given_X2()
        self.scores_X1 = self.calculate_checkerboard_scores(self.marginal_cdf_X1)
        self.scores_X2 = self.calculate_checkerboard_scores(self.marginal_cdf_X2)

    def calculate_conditional_pmf_X2_given_X1(self):
        """
        Calculates the conditional PMF of X2 given X1.
        
        Returns:
            numpy.ndarray: 5x3 matrix of conditional probabilities P(X2|X1)
        """
        # Calculate row-wise probabilities
        row_sums = self.P.sum(axis=1, keepdims=True)
        # Avoid division by zero by using np.divide with setting zero where row_sums is zero
        conditional_pmf = np.divide(self.P, row_sums, out=np.zeros_like(self.P), where=row_sums!=0)
        return conditional_pmf

    def calculate_conditional_pmf_X1_given_X2(self):
        """
        Calculates the conditional PMF of X1 given X2.
        
        Returns:
            numpy.ndarray: 5x3 matrix of conditional probabilities P(X1|X2)
        """
        # Calculate column-wise probabilities
        col_sums = self.P.sum(axis=0, keepdims=True)
        # Avoid division by zero by using np.divide with setting zero where col_sums is zero
        conditional_pmf = np.divide(self.P, col_sums, out=np.zeros_like(self.P), where=col_sums!=0)
        return conditional_pmf  # Return the matrix directly without transposing

    def lambda_function(self, u, ul, uj):
        """
        Calculates the lambda function for given u, ul, and uj values as per the checkerboard copula definition.
        """
        if u <= ul:
            return 0.0
        elif u >= uj:
            return 1.0
        else:
            return (u - ul) / (uj - ul)
    
    def calculate_checkerboard_scores(self, marginal_cdf):
        """
        Calculates the checkerboard copula scores for an ordinal variable as per Definition 2.
        """
        scores = [(marginal_cdf[j - 1] + marginal_cdf[j]) / 2 for j in range(1, len(marginal_cdf))]
        return scores
    
    def calculate_regression_U2_on_U1(self, u1):
        """
        Calculates the checkerboard copula regression of U2 on U1.
        For a given value u1, returns r(u1) which represents E[U2|U1=u1].
        
        Args:
            u1 (float): A value between 0 and 1 representing the U1 coordinate
            
        Returns:
            float: The conditional expectation E[U2|U1=u1]
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

    def calculate_regression_U2_on_U1_vectorized(self, u1_values):
        """
        Vectorized version of calculate_regression_U2_on_U1 that can handle arrays of u1 values.
        
        Args:
            u1_values (numpy.ndarray): Array of values between 0 and 1
            
        Returns:
            numpy.ndarray: Array of regression values
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
        
    def copula_density(self, u_values):
        """
        Calculates the checkerboard copula density for a given point (u1, u2).
        NOTE: This function might be brittle. Needs to be tested thoroughly & a second look.
        """
        d = len(u_values)  # dimension of copula (should be 2 for this case)
        result = 0.0
        
        # Iterate through all elements in P to calculate the copula density
        for i in range(len(self.P)):
            for j in range(len(self.P[i])):
                lambda_s = 1.0
                for k in range(d):
                    if k == 0:
                        ul = self.marginal_cdf_X1[i] if i > 0 else 0
                        uj = self.marginal_cdf_X1[i+1]
                        lambda_val = self.lambda_function(u_values[k], ul, uj)
                        lambda_s *= lambda_val
                    else:
                        ul = self.marginal_cdf_X2[j] if j > 0 else 0
                        uj = self.marginal_cdf_X2[j+1]
                        lambda_val = self.lambda_function(u_values[k], ul, uj)
                        lambda_s *= lambda_val
                result += lambda_s * self.P[i][j]
        
        return result