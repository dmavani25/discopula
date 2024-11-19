import numpy as np
# from itertools import combinations

# @TODO under development

class NDimensionalCheckerboardCopula:
    @classmethod
    def from_contingency_table(cls, contingency_table):
        if not isinstance(contingency_table, np.ndarray):
            contingency_table = np.array(contingency_table)
            
        if contingency_table.ndim < 2:
            raise ValueError("Contingency table must be at least 2-dimensional")
            
        if np.any(contingency_table < 0):
            raise ValueError("Contingency table cannot contain negative values")
            
        total_count = contingency_table.sum()
        if total_count == 0:
            raise ValueError("Contingency table cannot be all zeros")
            
        P = contingency_table / total_count
        return cls(P)
    
    def __init__(self, P):
        if not isinstance(P, np.ndarray):
            P = np.array(P)
            
        if P.ndim < 2:
            raise ValueError("Probability array P must be at least 2-dimensional")
            
        if np.any(P < 0) or np.any(P > 1):
            raise ValueError("Probability array P must contain values between 0 and 1")
        
        if not np.allclose(P.sum(), 1.0, rtol=1e-10, atol=1e-10):
            raise ValueError("Probability array P must sum to 1")
        
        self.P = P
        self.n_dimensions = P.ndim
        
        # Calculate marginals for each dimension
        self.marginal_pdfs = []
        self.marginal_cdfs = []
        
        for dim in range(self.n_dimensions):
            axes = tuple(i for i in range(self.n_dimensions) if i != dim)
            pdf = np.apply_over_axes(np.sum, P, axes).flatten()
            cdf = np.insert(np.cumsum(pdf), 0, 0)
            
            self.marginal_pdfs.append(pdf)
            self.marginal_cdfs.append(cdf)
            
        self.scores = [self.calculate_checkerboard_scores(cdf) for cdf in self.marginal_cdfs]
        
    def calculate_checkerboard_scores(self, marginal_cdf):
        return [(marginal_cdf[j - 1] + marginal_cdf[j]) / 2 for j in range(1, len(marginal_cdf))]
    
    def get_conditional_pmf(self, target_dim, given_dims, given_values):
        """Calculate conditional PMF for target dimension given other dimensions"""
        slices = [slice(None)] * self.n_dimensions
        for dim, val in zip(given_dims, given_values):
            slices[dim] = val
            
        conditional_slice = self.P[tuple(slices)]
        sum_axis = tuple(i for i in range(self.n_dimensions) if i != target_dim and i not in given_dims)
        
        if sum_axis:
            conditional_slice = np.sum(conditional_slice, axis=sum_axis)
            
        normalizer = np.sum(conditional_slice)
        return conditional_slice / normalizer if normalizer > 0 else np.zeros_like(conditional_slice)
    
    def calculate_regression(self, target_dim, given_dim, u_value):
        """Calculate regression E[U_target|U_given=u_value]"""
        breakpoints = self.marginal_cdfs[given_dim][1:-1]
        interval_idx = np.searchsorted(breakpoints, u_value, side='left')
        
        conditional_pmf = self.get_conditional_pmf(target_dim, [given_dim], [interval_idx])
        return np.sum(conditional_pmf * self.scores[target_dim])
    
    def calculate_CCRAM(self, target_dim, given_dim):
        """Calculate CCRAM for target_dim given given_dim"""
        weighted_expectation = 0.0
        pdf = self.marginal_pdfs[given_dim]
        cdf = self.marginal_cdfs[given_dim][1:]
        
        for p, u in zip(pdf, cdf):
            regression_value = self.calculate_regression(target_dim, given_dim, u)
            weighted_expectation += p * (regression_value - 0.5) ** 2
            
        return 12 * weighted_expectation
    
    def calculate_sigma_sq_S(self, dim):
        """Calculate score variance for dimension dim"""
        u_prev = self.marginal_cdfs[dim][:-1]
        u_next = self.marginal_cdfs[dim][1:]
        terms = u_prev * u_next * self.marginal_pdfs[dim]
        return np.sum(terms) / 4.0
    
    def calculate_SCCRAM(self, target_dim, given_dim):
        """Calculate standardized CCRAM"""
        ccram = self.calculate_CCRAM(target_dim, given_dim)
        sigma_sq_S = self.calculate_sigma_sq_S(target_dim)
        return ccram / (12 * sigma_sq_S)