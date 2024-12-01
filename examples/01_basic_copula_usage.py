import numpy as np
from discopula import CheckerboardCopula

def basic_copula_example():
    # Create a sample contingency table
    contingency_table = np.array([
        [0, 0, 20],
        [0, 10, 0],
        [20, 0, 0],
        [0, 10, 0],
        [0, 0, 20]
    ])

    # Initialize copula from contingency table
    copula = CheckerboardCopula.from_contingency_table(contingency_table)

    # Basic properties
    print("Basic Copula Properties:")
    print(f"    Shape of constructed probability matrix P: {copula.P.shape}")
    print(f"    Constructed Probability matrix P:\n{copula.P}")
    print(f"    Marginal PDF X1: {copula.marginal_pdf_X1}")
    print(f"    Marginal CDF X1: {copula.marginal_cdf_X1}")
    print(f"    Marginal PDF X2: {copula.marginal_pdf_X2}")
    print(f"    Marginal CDF X2: {copula.marginal_cdf_X2}")
    print(f"    Conditional PMF X2|X1:\n{copula.conditional_pmf_X2_given_X1}")
    print(f"    Conditional PMF X1|X2:\n{copula.conditional_pmf_X1_given_X2}")
    print(f"    Scores X1: {copula.scores_X1}")
    print(f"    Mean Scores X1: {np.mean(copula.scores_X1)}")
    print(f"    Variance of Scores X1: {np.var(copula.scores_X1)}")
    print(f"    Scores X2: {copula.scores_X2}")
    print(f"    Mean Scores X2: {np.mean(copula.scores_X2)}")
    print(f"    Variance of Scores X2: {np.var(copula.scores_X2)}")

    # Calculate regression U2 on U1 for specific points
    u1_points = [0, 1/16, 3/8, 4/8, 5.5/8, 7/8, 1]
    print("\nRegression Calculations for U2 on U1:")
    for u1 in u1_points:
        reg = copula.calculate_regression_U2_on_U1(u1)
        print(f"    E[U2|U1={u1}] = {reg:.4f}")
    
    # Calculate regression U2 on U1 in batched manner
    u1_points = np.array([0, 1/16, 3/8, 4/8, 5.5/8, 7/8, 1])
    print("\nBatched Regression Calculations for U2 on U1:")
    reg = copula.calculate_regression_U2_on_U1_batched(u1_points)
    for u1, r in zip(u1_points, reg):
        print(f"    E[U2|U1={u1}] = {r:.4f}")
        
    # Calculate regression U1 on U2 for specific points
    u2_points = [0, 1/16, 3/8, 4/8, 5/8, 6/8, 1]
    print("\nRegression Calculations for U1 on U2:")
    for u2 in u2_points:
        reg = copula.calculate_regression_U1_on_U2(u2)
        print(f"    E[U1|U2={u2}] = {reg:.4f}")
        
    # Calculate regression U1 on U2 in batched manner
    u2_points = np.array([0, 1/16, 3/8, 4/8, 5/8, 6/8, 1])
    print("\nBatched Regression Calculations for U1 on U2:")
    reg = copula.calculate_regression_U1_on_U2_batched(u2_points)
    for u2, r in zip(u2_points, reg):
        print(f"    E[U1|U2={u2}] = {r:.4f}")

    # Calculate association measures
    print("\nAssociation Measures:")
    print(f"    CCRAM X1->X2: {copula.calculate_CCRAM_X1_X2():.4f}")
    print(f"    CCRAM X1->X2 (vectorized): {copula.calculate_CCRAM_X1_X2_vectorized():.4f}")
    print(f"    SCCRAM X1->X2: {copula.calculate_SCCRAM_X1_X2():.4f}")
    print(f"    SCCRAM X1->X2 (vectorized): {copula.calculate_SCCRAM_X1_X2_vectorized():.4f}")
    print(f"    CCRAM X2->X1: {copula.calculate_CCRAM_X2_X1():.4f}")
    print(f"    CCRAM X2->X1 (vectorized): {copula.calculate_CCRAM_X2_X1_vectorized():.4f}")
    print(f"    SCCRAM X2->X1: {copula.calculate_SCCRAM_X2_X1():.4f}")
    print(f"    SCCRAM X2->X1 (vectorized): {copula.calculate_SCCRAM_X2_X1_vectorized():.4f}")
    
    # Calculate variance of checkerboard copula scores
    print("\nVariance of Checkerboard Copula Scores (σ²_Sⱼ):")
    print(f"    σ²_S1: {copula.calculate_sigma_sq_S_X1():.4f}")
    print(f"    σ²_S1 (vectorized): {copula.calculate_sigma_sq_S_X1_vectorized():.4f}")
    print(f"    σ²_S2: {copula.calculate_sigma_sq_S_X2():.4f}")
    print(f"    σ²_S2 (vectorized): {copula.calculate_sigma_sq_S_X2_vectorized():.4f}")
    
    # Get Predicted Category X2 from X1 after Regression
    print("\nPredicted Category X2 from X1 after Regression:")
    x1_categories = np.array([0, 1, 2, 3, 4])
    for x1 in x1_categories:
        pred_x2 = copula.predict_X2_from_X1(x1)
        print(f"    Predicted X2 given X1={x1}: {pred_x2}")
    
    # Get Predicted Category X1 from X2 after Regression
    print("\nPredicted Category X1 from X2 after Regression:")
    x2_categories = np.array([0, 1, 2])
    for x2 in x2_categories:
        pred_x1 = copula.predict_X1_from_X2(x2)
        print(f"    Predicted X1 given X2={x2}: {pred_x1}")
    
    # Get Predicted Category X2 from X1 after Regression in batched manner
    predictions_x2 = copula.predict_X2_from_X1_batched(x1_categories)
    print(f"\nPredicted X2 given X1={x1_categories}: {predictions_x2}")
        
    # Get Predicted Category X1 from X2 after Regression in batched manner
    predictions_x1 = copula.predict_X1_from_X2_batched(x2_categories)
    print(f"\nPredicted X1 given X2={x2_categories}: {predictions_x1}")

if __name__ == "__main__":
    basic_copula_example()
    """Expected Output:

        Basic Copula Properties:
            Shape of constructed probability matrix P: (5, 3)
            Constructed Probability matrix P:
        [[0.    0.    0.25 ]
        [0.    0.125 0.   ]
        [0.25  0.    0.   ]
        [0.    0.125 0.   ]
        [0.    0.    0.25 ]]
            Marginal PDF X1: [0.25  0.125 0.25  0.125 0.25 ]
            Marginal CDF X1: [0.    0.25  0.375 0.625 0.75  1.   ]
            Marginal PDF X2: [0.25 0.25 0.5 ]
            Marginal CDF X2: [0.   0.25 0.5  1.  ]
            Conditional PMF X2|X1:
        [[0. 0. 1.]
        [0. 1. 0.]
        [1. 0. 0.]
        [0. 1. 0.]
        [0. 0. 1.]]
            Conditional PMF X1|X2:
        [[0.  0.  0.5]
        [0.  0.5 0. ]
        [1.  0.  0. ]
        [0.  0.5 0. ]
        [0.  0.  0.5]]
            Scores X1: [np.float64(0.125), np.float64(0.3125), np.float64(0.5), np.float64(0.6875), np.float64(0.875)]
            Mean Scores X1: 0.5
            Variance of Scores X1: 0.0703125
            Scores X2: [np.float64(0.125), np.float64(0.375), np.float64(0.75)]
            Mean Scores X2: 0.4166666666666667
            Variance of Scores X2: 0.06597222222222222

        Regression Calculations for U2 on U1:
            E[U2|U1=0] = 0.7500
            E[U2|U1=0.0625] = 0.7500
            E[U2|U1=0.375] = 0.3750
            E[U2|U1=0.5] = 0.1250
            E[U2|U1=0.6875] = 0.3750
            E[U2|U1=0.875] = 0.7500
            E[U2|U1=1] = 0.7500

        Batched Regression Calculations for U2 on U1:
            E[U2|U1=0.0] = 0.7500
            E[U2|U1=0.0625] = 0.7500
            E[U2|U1=0.375] = 0.3750
            E[U2|U1=0.5] = 0.1250
            E[U2|U1=0.6875] = 0.3750
            E[U2|U1=0.875] = 0.7500
            E[U2|U1=1.0] = 0.7500

        Regression Calculations for U1 on U2:
            E[U1|U2=0] = 0.5000
            E[U1|U2=0.0625] = 0.5000
            E[U1|U2=0.375] = 0.5000
            E[U1|U2=0.5] = 0.5000
            E[U1|U2=0.625] = 0.5000
            E[U1|U2=0.75] = 0.5000
            E[U1|U2=1] = 0.5000

        Batched Regression Calculations for U1 on U2:
            E[U1|U2=0.0] = 0.5000
            E[U1|U2=0.0625] = 0.5000
            E[U1|U2=0.375] = 0.5000
            E[U1|U2=0.5] = 0.5000
            E[U1|U2=0.625] = 0.5000
            E[U1|U2=0.75] = 0.5000
            E[U1|U2=1.0] = 0.5000

        Association Measures:
            CCRAM X1->X2: 0.8438
            CCRAM X1->X2 (vectorized): 0.8438
            SCCRAM X1->X2: 1.0000
            SCCRAM X1->X2 (vectorized): 1.0000
            CCRAM X2->X1: 0.0000
            CCRAM X2->X1 (vectorized): 0.0000
            SCCRAM X2->X1: 0.0000
            SCCRAM X2->X1 (vectorized): 0.0000

        Variance of Checkerboard Copula Scores (σ²_Sⱼ):
            σ²_S1: 0.0791
            σ²_S1 (vectorized): 0.0791
            σ²_S2: 0.0703
            σ²_S2 (vectorized): 0.0703

        Predicted Category X2 from X1 after Regression:
            Predicted X2 given X1=0: 2
            Predicted X2 given X1=1: 1
            Predicted X2 given X1=2: 0
            Predicted X2 given X1=3: 1
            Predicted X2 given X1=4: 2

        Predicted Category X1 from X2 after Regression:
            Predicted X1 given X2=0: 2
            Predicted X1 given X2=1: 2
            Predicted X1 given X2=2: 2

        Predicted X2 given X1=[0 1 2 3 4]: [2 1 0 1 2]

        Predicted X1 given X2=[0 1 2]: [2 2 2]
    """