# Temporarily commented out for testing
if __name__ == "__main__":
    import numpy as np
    from discopula.checkerboard.copula import CheckerboardCopula
    from discopula.checkerboard.statsim import (
        bootstrap_predict_X1_from_X2, 
        bootstrap_predict_X2_from_X1,
        bootstrap_predict_X1_from_X2_vectorized, 
        bootstrap_predict_X2_from_X1_vectorized,
        bootstrap_predict_X1_from_X2_all_comb_summary, 
        bootstrap_predict_X2_from_X1_all_comb_summary,
        permutation_test_ccram, 
        permutation_test_sccram
    )
    
    # Example contingency table
    table = np.array([
        [0, 0, 10],
        [0, 20, 0],
        [10, 0, 0],
        [0, 20, 0],
        [0, 0, 10]
    ])

    # Copula object from contingency table
    copula = CheckerboardCopula.from_contingency_table(table)

    # Single prediction with confidence interval
    x2_category = 0
    result = bootstrap_predict_X1_from_X2(table, x2_category, method='percentile')
    print(f"Bootstrap distribution: {result.bootstrap_distribution.min()} - {result.bootstrap_distribution.max()}")

    # Multiple predictions with confidence intervals
    x2_categories = np.array([0, 1, 2])
    results = bootstrap_predict_X1_from_X2_vectorized(table, x2_categories, method='percentile')
    for x2, res in zip(x2_categories, results):
        print(f"Bootstrap distribution: {res.bootstrap_distribution.min()} - {res.bootstrap_distribution.max()}")
        
    # Summary table for all X2 categories
    summary_table = bootstrap_predict_X1_from_X2_all_comb_summary(table, method='percentile', n_resamples=1000)
    print(summary_table)
    
    # Single prediction with confidence interval
    x1_category = 0
    result = bootstrap_predict_X2_from_X1(table, x1_category, method='percentile')
    print(f"Bootstrap distribution: {result.bootstrap_distribution.min()} - {result.bootstrap_distribution.max()}")
    
    # Multiple predictions with confidence intervals
    x1_categories = np.array([0, 1, 2, 3, 4])
    results = bootstrap_predict_X2_from_X1_vectorized(table, x1_categories, method='percentile')
    for x1, res in zip(x1_categories, results):
        print(f"Bootstrap distribution: {res.bootstrap_distribution.min()} - {res.bootstrap_distribution.max()}")
        
    # Summary table for all X1 categories
    summary_table = bootstrap_predict_X2_from_X1_all_comb_summary(table, method='percentile', n_resamples=1000)
    print(summary_table)
    
    # Test independence between variables - SCCRAM X1_X2
    result_sccram_x1x2 = permutation_test_sccram(
        table,
        direction="X1_X2", 
        n_resamples=9999
    )
    
    print("\nSCCRAM Results (X1->X2):")
    print(f"Observed SCCRAM: {result_sccram_x1x2.statistic:.4f}")
    print(f"P-value: {result_sccram_x1x2.pvalue:.4f}")
    print(f"Null distribution: {result_sccram_x1x2.null_distribution.min():.4f} - {result_sccram_x1x2.null_distribution.max():.4f}")
    print(f"Relative Error: {1.0/np.sqrt(9999 * result_sccram_x1x2.pvalue):.4f}")

    # Test independence between variables - SCCRAM X2_X1
    result_sccram_x2x1 = permutation_test_sccram(
        table,
        direction="X2_X1", 
        n_resamples=9999
    )

    print("\nSCCRAM Results (X2->X1):")
    print(f"Observed SCCRAM: {result_sccram_x2x1.statistic:.4f}")
    print(f"P-value: {result_sccram_x2x1.pvalue:.4f}")
    print(f"Null distribution: {result_sccram_x2x1.null_distribution.min():.4f} - {result_sccram_x2x1.null_distribution.max():.4f}")
    print(f"Relative Error: {1.0/np.sqrt(9999 * result_sccram_x2x1.pvalue):.4f}")
    
    # Test independence between variables - CCRAM X1_X2
    result_ccram_x1x2 = permutation_test_ccram(
        table,
        direction="X1_X2",
        alternative="two-sided",
        n_resamples=1000000
    )

    print("\nCCRAM Results (X1->X2):")
    print(f"Observed CCRAM: {result_ccram_x1x2.statistic:.4f}")
    print(f"P-value: {result_ccram_x1x2.pvalue:.4f}")
    print(f"Null distribution: {result_ccram_x1x2.null_distribution.min():.4f} - {result_ccram_x1x2.null_distribution.max():.4f}")
    print(f"Relative Error: {1.0/np.sqrt(1000000 * result_ccram_x1x2.pvalue):.4f}")

    # Test independence between variables - CCRAM X2_X1
    result_ccram_x2x1 = permutation_test_ccram(
        table,
        direction="X2_X1",
        alternative="two-sided", 
        n_resamples=1000000
    )

    print("\nCCRAM Results (X2->X1):")
    print(f"Observed CCRAM: {result_ccram_x2x1.statistic:.4f}")
    print(f"P-value: {result_ccram_x2x1.pvalue:.4f}")
    print(f"Null distribution: {result_ccram_x2x1.null_distribution.min():.4f} - {result_ccram_x2x1.null_distribution.max():.4f}")
    print(f"Relative Error: {1.0/np.sqrt(1000000 * result_ccram_x2x1.pvalue):.4f}")