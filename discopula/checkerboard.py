import numpy as np

class CheckerboardCopula:
    """
    A class to calculate the checkerboard copula density for ordinal random vectors.
    """

    def __init__(self, P):
        """
        Initializes the CheckerboardCopula with the joint probability matrix P and
        calculates the marginal cumulative distribution functions (CDFs).
        
        Args:
            P (numpy.ndarray): The joint probability matrix.
        """
        self.P = P
        self.marginal_cdf_X1 = np.cumsum(P.sum(axis=1))
        self.marginal_cdf_X2 = np.cumsum(P.sum(axis=0))

    def lambda_function(self, u, ul, uj):
        """
        Calculates the lambda function for given u, ul, and uj values as per the checkerboard copula definition.
        
        Args:
            u (float): The target value for copula density calculation.
            ul (float): The lower bound for the interval.
            uj (float): The upper bound for the interval.
        
        Returns:
            float: Lambda function result.
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
        
        Args:
            u_values (tuple): A tuple of (u1, u2) values within the [0, 1] interval.
        
        Returns:
            float: Checkerboard copula density at the given point.
        """
        d = len(u_values)  # dimension of copula (should be 2 for this case)
        result = 0.0
        
        # Iterate through all elements in P to calculate the copula density
        for i in range(len(self.P)):
            for j in range(len(self.P[i])):
                lambda_s = 1.0
                for k in range(d):
                    if k == 0:
                        ul = self.marginal_cdf_X1[i-1] if i > 0 else 0
                        uj = self.marginal_cdf_X1[i]
                        lambda_val = self.lambda_function(u_values[k], ul, uj)
                        lambda_s *= lambda_val
                    else:
                        ul = self.marginal_cdf_X2[j-1] if j > 0 else 0
                        uj = self.marginal_cdf_X2[j]
                        lambda_val = self.lambda_function(u_values[k], ul, uj)
                        lambda_s *= lambda_val
                result += lambda_s * self.P[i][j]
        
        return result