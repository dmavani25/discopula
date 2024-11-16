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

        # Marginal PDFs (densities are just the probabilities for discrete distributions)
        self.marginal_pdf_X1 = P.sum(axis=1) / P.sum()  # Marginal PDF for X1
        self.marginal_pdf_X2 = P.sum(axis=0) / P.sum()  # Marginal PDF for X2

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
    
    def calculate_CCRAM_X1_X2(self):
        """
        Calculates the Checkerboard Copula Regression Association Measure (CCRAM) for X1 and X2.
        """
        weighted_expectation = 0.0
        for p_x1, u1 in zip(self.marginal_pdf_X1, self.marginal_cdf_X1[1:]):
            regression_value = self.calculate_regression_U2_on_U1(u1)
            weighted_expectation += p_x1 * (regression_value - 0.5) ** 2
        return 12 * weighted_expectation
    
    def calculate_CCRAM_X1_X2_vectorized(self):
        """
        Vectorized version of calculate_CCRAM_X1_X2 that can handle arrays of u1
        values.
        """
        regression_values = self.calculate_regression_U2_on_U1_vectorized(self.marginal_cdf_X1[1:])
        weighted_expectation = np.sum(self.marginal_pdf_X1 * (regression_values - 0.5) ** 2)
        return 12 * weighted_expectation
    
    def calculate_sigma_sq_S(self):
        """
        Calculates the variance of the checkerboard copula score S.
        Formula: σ²_Sⱼ = Σᵢⱼ₌₁^Iⱼ û_ᵢⱼ₋₁ û_ᵢⱼ p̂₊ᵢⱼ₊/4
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
    
    def calculate_sigma_sq_S_vectorized(self):
        """
        Calculates the variance of the checkerboard copula score S using vectorized operations.
        Formula: σ²_Sⱼ = Σᵢⱼ₌₁^Iⱼ û_ᵢⱼ₋₁ û_ᵢⱼ p̂₊ᵢⱼ₊/4 (vectorized)
        """
        # Get consecutive CDF values
        u_prev = self.marginal_cdf_X2[:-1]  # û_ᵢⱼ₋₁
        u_next = self.marginal_cdf_X2[1:]   # û_ᵢⱼ
        
        # Vectorized multiplication of all terms
        terms = u_prev * u_next * self.marginal_pdf_X2
        
        # Calculate sigma_sq_S
        sigma_sq_S = np.sum(terms) / 4.0
        
        return sigma_sq_S

# For quick testing purposes
if __name__ == '__main__':
    P = np.array([
        [0, 0, 2/8],
        [0, 1/8, 0],
        [2/8, 0, 0],
        [0, 1/8, 0],
        [0, 0, 2/8]
    ])
    
    cop = CheckerboardCopula(P)