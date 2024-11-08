import numpy as np

class CheckerboardCopula:
    """
    A class to calculate the checkerboard copula density and scores for ordinal random vectors.
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
        self.scores_X1 = self.calculate_checkerboard_scores(self.marginal_cdf_X1)
        self.scores_X2 = self.calculate_checkerboard_scores(self.marginal_cdf_X2)

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

    def copula_density(self, u_values):
        """
        Calculates the checkerboard copula density for a given point (u1, u2).
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
    
    def calculate_checkerboard_scores(self, marginal_cdf):
        """
        Calculates the checkerboard copula scores for an ordinal variable as per Definition 2.
        """
        scores = [(marginal_cdf[j - 1] + marginal_cdf[j]) / 2 for j in range(1, len(marginal_cdf))]
        return scores
