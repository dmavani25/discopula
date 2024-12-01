import numpy as np
from discopula import CheckerboardCopula
from discopula import (
    bootstrap_ccram, bootstrap_sccram,
    bootstrap_regression_U2_on_U1, bootstrap_regression_U1_on_U2,
    bootstrap_regression_U2_on_U1_vectorized, bootstrap_regression_U1_on_U2_vectorized,
    bootstrap_predict_X1_from_X2, 
    bootstrap_predict_X2_from_X1,
    bootstrap_predict_X1_from_X2_vectorized, 
    bootstrap_predict_X2_from_X1_vectorized,
    bootstrap_predict_X1_from_X2_all_comb_summary, 
    bootstrap_predict_X2_from_X1_all_comb_summary,
)

def bootstrap_example():
    # Example contingency table
    contingency_table = np.array([
        [0, 0, 10],
        [0, 20, 0],
        [10, 0, 0],
        [0, 20, 0],
        [0, 0, 10]
    ])
    
    # Bootstrap SCCRAM X1 -> X2
    # Note: Not using "BCa" in this case since it returns NaN for confidence intervals
    # DegenerateDataWarning: The BCa confidence interval cannot be calculated as referenced in SciPy documentation.
    # This problem is known to occur when the distribution is degenerate or the statistic is np.min.
    print("\nBootstrap SCCRAM X1 -> X2:")
    result_sccram_x1x2 = bootstrap_sccram(
        contingency_table,
        direction="X1_X2",
        n_resamples=9999,
        method="percentile",
        confidence_level=0.95,
        random_state=8990
    )
    print(f"    Confidence Interval: {result_sccram_x1x2.confidence_interval}")
    print(f"    Bootstrap Distribution Range: {result_sccram_x1x2.bootstrap_distribution.min()} - {result_sccram_x1x2.bootstrap_distribution.max()}")
    print(f"    Standard Error: {result_sccram_x1x2.standard_error:.4f}")
    
    # Bootstrap SCCRAM X2 -> X1
    print("\nBootstrap SCCRAM X2 -> X1:")
    result_sccram_x2x1 = bootstrap_sccram(
        contingency_table,
        direction="X2_X1",
        n_resamples=9999,
        method="BCa",
        confidence_level=0.95,
        random_state=8990
    )
    print(f"    Confidence Interval: {result_sccram_x2x1.confidence_interval}")
    print(f"    Bootstrap Distribution Range: {result_sccram_x2x1.bootstrap_distribution.min()} - {result_sccram_x2x1.bootstrap_distribution.max()}")
    print(f"    Standard Error: {result_sccram_x2x1.standard_error:.4f}")
    
    # Bootstrap CCRAM X1 -> X2
    print("\nBootstrap CCRAM X1 -> X2:")
    result_ccram_x1x2 = bootstrap_ccram(
        contingency_table,
        direction="X1_X2",
        n_resamples=9999,
        method="BCa",
        confidence_level=0.95,
        random_state=8990
    )
    print(f"    Confidence Interval: {result_ccram_x1x2.confidence_interval}")
    print(f"    Bootstrap Distribution Range: {result_ccram_x1x2.bootstrap_distribution.min()} - {result_ccram_x1x2.bootstrap_distribution.max()}")
    print(f"    Standard Error: {result_ccram_x1x2.standard_error:.4f}")
    
    # Bootstrap CCRAM X2 -> X1
    # Note: Not using "BCa" in this case since it returns NaN for confidence intervals
    # DegenerateDataWarning: The BCa confidence interval cannot be calculated as referenced in SciPy documentation.
    # This problem is known to occur when the distribution is degenerate or the statistic is np.min.
    print("\nBootstrap CCRAM X2 -> X1:")
    result_ccram_x2x1 = bootstrap_ccram(
        contingency_table,
        direction="X2_X1",
        n_resamples=9999,
        method="percentile",
        confidence_level=0.95,
        random_state=8990
    )
    print(f"    Confidence Interval: {result_ccram_x2x1.confidence_interval}")
    print(f"    Bootstrap Distribution Range: {result_ccram_x2x1.bootstrap_distribution.min()} - {result_ccram_x2x1.bootstrap_distribution.max()}")
    print(f"    Standard Error: {result_ccram_x2x1.standard_error:.4f}")
    
    # Bootstrap regression U1 on U2
    print("\nBootstrap Regression U1 on U2:")
    result_reg_u1u2 = bootstrap_regression_U1_on_U2(
        contingency_table,
        u2=0.5,
        n_resamples=9999,
        method="percentile",
        confidence_level=0.95,
        random_state=8990
    )
    print(f"    Confidence Interval: {result_reg_u1u2.confidence_interval}")
    print(f"    Bootstrap Distribution Range: {result_reg_u1u2.bootstrap_distribution.min()} - {result_reg_u1u2.bootstrap_distribution.max()}")
    print(f"    Standard Error: {result_reg_u1u2.standard_error:.4f}")
    
    # Bootstrap regression U2 on U1
    print("\nBootstrap Regression U2 on U1:")
    result_reg_u2u1 = bootstrap_regression_U2_on_U1(
        contingency_table,
        u1=0.5,
        n_resamples=9999,
        method="percentile",
        confidence_level=0.95,
        random_state=8990
    )
    print(f"    Confidence Interval: {result_reg_u2u1.confidence_interval}")
    print(f"    Bootstrap Distribution Range: {result_reg_u2u1.bootstrap_distribution.min()} - {result_reg_u2u1.bootstrap_distribution.max()}")
    print(f"    Standard Error: {result_reg_u2u1.standard_error:.4f}")
    
    # Bootstrap regression U1 on U2 in batched manner
    u2_points = np.array([0, 1/16, 3/8, 4/8, 5/8, 6/8, 1])
    print("\nBootstrap Batched Regression U1 on U2:")
    results_reg_u1u2_batch = bootstrap_regression_U1_on_U2_vectorized(
        contingency_table,
        u2_points,
        n_resamples=9999,
        method="percentile",
        confidence_level=0.95,
        random_state=8990
    )
    for u2, res in zip(u2_points, results_reg_u1u2_batch):
        print(f"    When U2 = {u2}:")
        print(f"        Confidence Interval: {res.confidence_interval}")
        print(f"        Bootstrap Distribution Range: {res.bootstrap_distribution.min()} - {res.bootstrap_distribution.max()}")
        print(f"        Standard Error: {res.standard_error:.4f}")
        
    # Bootstrap regression U2 on U1 in batched manner
    u1_points = np.array([0, 1/16, 3/8, 4/8, 5.5/8, 7/8, 1])
    print("\nBootstrap Batched Regression U2 on U1:")
    results_reg_u2u1_batch = bootstrap_regression_U2_on_U1_vectorized(
        contingency_table,
        u1_points,
        n_resamples=9999,
        method="percentile",
        confidence_level=0.95,
        random_state=8990
    )
    for u1, res in zip(u1_points, results_reg_u2u1_batch):
        print(f"    When U1 = {u1}:")
        print(f"        Confidence Interval: {res.confidence_interval}")
        print(f"        Bootstrap Distribution Range: {res.bootstrap_distribution.min()} - {res.bootstrap_distribution.max()}")
        print(f"        Standard Error: {res.standard_error:.4f}")

    # Single prediction with confidence interval
    print("\nBootstrap Prediction X1 from X2 = 0:")
    x2_category = 0
    result_x2 = bootstrap_predict_X1_from_X2(contingency_table, x2_category, method='percentile')
    print(f"    Bootstrap Distribution Range: {result_x2.bootstrap_distribution.min()} - {result_x2.bootstrap_distribution.max()}")

    # Multiple predictions with confidence intervals
    print("\nBootstrap Prediction X1 from X2 (Vectorized):")
    x2_categories = np.array([0, 1, 2])
    results_x2_vectorized = bootstrap_predict_X1_from_X2_vectorized(contingency_table, x2_categories, method='percentile')
    for x2, res in zip(x2_categories, results_x2_vectorized):
        print(f"    Bootstrap Distribution Range for X2={x2}: {res.bootstrap_distribution.min()} - {res.bootstrap_distribution.max()}")
        
    # Summary table for all X2 categories
    print("\nBootstrap Prediction X1 from X2 (All Combinations Summary):")
    summary_table_x1x2 = bootstrap_predict_X1_from_X2_all_comb_summary(contingency_table, method='percentile', n_resamples=1000)
    print(summary_table_x1x2)
    
    # Single prediction with confidence interval
    print("\nBootstrap Prediction X2 from X1 = 0:")
    x1_category = 0
    result_x1 = bootstrap_predict_X2_from_X1(contingency_table, x1_category, method='percentile')
    print(f"    Bootstrap Distribution Range: {result_x1.bootstrap_distribution.min()} - {result_x1.bootstrap_distribution.max()}")
    
    # Multiple predictions with confidence intervals
    print("\nBootstrap Prediction X2 from X1 (Vectorized):")
    x1_categories = np.array([0, 1, 2, 3, 4])
    results_x1_vectorized = bootstrap_predict_X2_from_X1_vectorized(contingency_table, x1_categories, method='percentile')
    for x1, res in zip(x1_categories, results_x1_vectorized):
        print(f"Bootstrap Distribution Range for X1={x1}: {res.bootstrap_distribution.min()} - {res.bootstrap_distribution.max()}")
        
    # Summary table for all X1 categories
    print("\nBootstrap Prediction X2 from X1 (All Combinations Summary):")
    summary_table_x2x1 = bootstrap_predict_X2_from_X1_all_comb_summary(contingency_table, method='percentile', n_resamples=1000)
    print(summary_table_x2x1)
    
if __name__ == "__main__":
    bootstrap_example()
    """Example Output:
    
        Bootstrap SCCRAM X1 -> X2:
            Confidence Interval: ConfidenceInterval(low=np.float64(0.9999999999999997), high=np.float64(1.0000000000000004))
            Bootstrap Distribution Range: 0.9999999999999993 - 1.0000000000000007
            Standard Error: 0.0000

        Bootstrap SCCRAM X2 -> X1:
            Confidence Interval: ConfidenceInterval(low=np.float64(0.0), high=np.float64(0.0))
            Bootstrap Distribution Range: 0.0 - 0.4039921722113503
            Standard Error: 0.0458

        Bootstrap CCRAM X1 -> X2:
            Confidence Interval: ConfidenceInterval(low=np.float64(0.6690437317784259), high=np.float64(0.8562857142857145))
            Bootstrap Distribution Range: 0.4289562682215745 - 0.8816326530612244
            Standard Error: 0.0492

        Bootstrap CCRAM X2 -> X1:
            Confidence Interval: ConfidenceInterval(low=np.float64(0.0005340739118911274), high=np.float64(0.15688375959524514))
            Bootstrap Distribution Range: 0.0 - 0.3611195335276968
            Standard Error: 0.0427

        Bootstrap Regression U1 on U2:
            Confidence Interval: ConfidenceInterval(low=np.float64(0.4328571428571429), high=np.float64(0.5661679536679537))
            Bootstrap Distribution Range: 0.3681467181467182 - 0.6176691729323307
            Standard Error: 0.0341

        Bootstrap Regression U2 on U1:
            Confidence Interval: ConfidenceInterval(low=np.float64(0.04285714285714286), high=np.float64(0.4571428571428572))
            Bootstrap Distribution Range: 0.02142857142857143 - 0.5428571428571429
            Standard Error: 0.1472

        Bootstrap Batched Regression U1 on U2:
            When U2 = 0.0:
                Confidence Interval: ConfidenceInterval(low=np.float64(0.39285714285714285), high=np.float64(0.6071428571428572))
                Bootstrap Distribution Range: 0.0 - 0.7000000000000001
                Standard Error: 0.0554
            When U2 = 0.0625:
                Confidence Interval: ConfidenceInterval(low=np.float64(0.39285714285714285), high=np.float64(0.6071428571428572))
                Bootstrap Distribution Range: 0.2857142857142857 - 0.7000000000000001
                Standard Error: 0.0548
            When U2 = 0.375:
                Confidence Interval: ConfidenceInterval(low=np.float64(0.4328571428571429), high=np.float64(0.5661679536679537))
                Bootstrap Distribution Range: 0.3681467181467182 - 0.6176691729323307
                Standard Error: 0.0341
            When U2 = 0.5:
                Confidence Interval: ConfidenceInterval(low=np.float64(0.4328571428571429), high=np.float64(0.5661679536679537))
                Bootstrap Distribution Range: 0.3681467181467182 - 0.6176691729323307
                Standard Error: 0.0341
            When U2 = 0.625:
                Confidence Interval: ConfidenceInterval(low=np.float64(0.4310805984555984), high=np.float64(0.5690544217687074))
                Bootstrap Distribution Range: 0.35 - 0.6706349206349207
                Standard Error: 0.0352
            When U2 = 0.75:
                Confidence Interval: ConfidenceInterval(low=np.float64(0.36309523809523814), high=np.float64(0.6342105263157894))
                Bootstrap Distribution Range: 0.25000000000000006 - 0.7888888888888889
                Standard Error: 0.0665
            When U2 = 1.0:
                Confidence Interval: ConfidenceInterval(low=np.float64(0.33492063492063495), high=np.float64(0.6650793650793652))
                Bootstrap Distribution Range: 0.1554945054945055 - 0.844805194805195
                Standard Error: 0.0826

        Bootstrap Batched Regression U2 on U1:
            When U1 = 0.0:
                Confidence Interval: ConfidenceInterval(low=np.float64(0.807142857142857), high=np.float64(0.9071428571428571))
                Bootstrap Distribution Range: 0.7571428571428571 - 0.95
                Standard Error: 0.0271
            When U1 = 0.0625:
                Confidence Interval: ConfidenceInterval(low=np.float64(0.7785714285714287), high=np.float64(0.9071428571428571))
                Bootstrap Distribution Range: 0.38571428571428573 - 0.95
                Standard Error: 0.0611
            When U1 = 0.375:
                Confidence Interval: ConfidenceInterval(low=np.float64(0.05714285714285714), high=np.float64(0.4928571428571429))
                Bootstrap Distribution Range: 0.02142857142857143 - 0.5642857142857144
                Standard Error: 0.1398
            When U1 = 0.5:
                Confidence Interval: ConfidenceInterval(low=np.float64(0.04285714285714286), high=np.float64(0.4571428571428572))
                Bootstrap Distribution Range: 0.02142857142857143 - 0.5428571428571429
                Standard Error: 0.1472
            When U1 = 0.6875:
                Confidence Interval: ConfidenceInterval(low=np.float64(0.3357142857142857), high=np.float64(0.5))
                Bootstrap Distribution Range: 0.03571428571428572 - 0.7642857142857142
                Standard Error: 0.0601
            When U1 = 0.875:
                Confidence Interval: ConfidenceInterval(low=np.float64(0.40678571428571575), high=np.float64(0.8857142857142859))
                Bootstrap Distribution Range: 0.33571428571428574 - 0.9214285714285717
                Standard Error: 0.1871
            When U1 = 1.0:
                Confidence Interval: ConfidenceInterval(low=np.float64(0.807142857142857), high=np.float64(0.9071428571428571))
                Bootstrap Distribution Range: 0.7571428571428571 - 0.95
                Standard Error: 0.0271

        Bootstrap Prediction X1 from X2 = 0:
            Bootstrap Distribution Range: 2 - 2

        Bootstrap Prediction X1 from X2 (Vectorized):
            Bootstrap Distribution Range for X2=0: 2 - 2
            Bootstrap Distribution Range for X2=1: 1 - 3
            Bootstrap Distribution Range for X2=2: 0 - 4

        Bootstrap Prediction X1 from X2 (All Combinations Summary):
        [[  0.    0.    0. ]
        [  0.   12.   25. ]
        [100.   75.1  44.8]
        [  0.   12.9  30. ]
        [  0.    0.    0.2]]

        Bootstrap Prediction X2 from X1 = 0:
            Bootstrap Distribution Range: 2 - 2

        Bootstrap Prediction X2 from X1 (Vectorized):
        Bootstrap Distribution Range for X1=0: 2 - 2
        Bootstrap Distribution Range for X1=1: 1 - 1
        Bootstrap Distribution Range for X1=2: 0 - 0
        Bootstrap Distribution Range for X1=3: 1 - 1
        Bootstrap Distribution Range for X1=4: 2 - 2

        Bootstrap Prediction X2 from X1 (All Combinations Summary):
        [[  0.   0. 100.]
        [  0. 100.   0.]
        [100.   0.   0.]
        [  0. 100.   0.]
        [  0.   0. 100.]]
    """