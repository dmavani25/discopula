import numpy as np
from discopula import permutation_test_ccram, permutation_test_sccram

def statistical_testing_example():
    # Example contingency table
    contingency_table = np.array([
        [0, 0, 10],
        [0, 20, 0],
        [10, 0, 0],
        [0, 20, 0],
        [0, 0, 10]
    ])

    # Test independence between variables - SCCRAM X1_X2
    result_sccram_x1x2 = permutation_test_sccram(
        contingency_table,
        direction="X1_X2", 
        alternative="greater",
        n_resamples=9999
    )
    print("\nSCCRAM Results (X1->X2) with 9999 Re-samples:")
    print(f"    Observed SCCRAM: {result_sccram_x1x2.statistic:.4f}")
    print(f"    P-value: {result_sccram_x1x2.pvalue:.4f}")
    print(f"    Null distribution: {result_sccram_x1x2.null_distribution.min():.4f} - {result_sccram_x1x2.null_distribution.max():.4f}")
    print(f"    Standard Deviation of Null Distribution: {result_sccram_x1x2.null_distribution.std():.4f}")
    print(f"    Relative Error: {1.0/np.sqrt(9999 * result_sccram_x1x2.pvalue):.4f}")

    # Test independence between variables - SCCRAM X2_X1
    result_sccram_x2x1 = permutation_test_sccram(
        contingency_table,
        direction="X2_X1", 
        alternative="greater",
        n_resamples=9999
    )
    print("\nSCCRAM Results (X2->X1) with 9999 Re-samples:")
    print(f"    Observed SCCRAM: {result_sccram_x2x1.statistic:.4f}")
    print(f"    P-value: {result_sccram_x2x1.pvalue:.4f}")
    print(f"    Null distribution: {result_sccram_x2x1.null_distribution.min():.4f} - {result_sccram_x2x1.null_distribution.max():.4f}")
    print(f"    Standard Deviation of Null Distribution: {result_sccram_x2x1.null_distribution.std():.4f}")
    print(f"    Relative Error: {1.0/np.sqrt(9999 * result_sccram_x2x1.pvalue):.4f}")

    # Test independence between variables - CCRAM X1_X2
    result_ccram_x1x2 = permutation_test_ccram(
        contingency_table,
        direction="X1_X2",
        alternative="greater",
        n_resamples=1000000
    )
    print("\nCCRAM Results (X1->X2) with 1000000 Re-samples:")
    print(f"    Observed CCRAM: {result_ccram_x1x2.statistic:.4f}")
    print(f"    P-value: {result_ccram_x1x2.pvalue:.4f}")
    print(f"    Null distribution: {result_ccram_x1x2.null_distribution.min():.4f} - {result_ccram_x1x2.null_distribution.max():.4f}")
    print(f"    Standard Deviation of Null Distribution: {result_ccram_x1x2.null_distribution.std():.4f}")
    print(f"    Relative Error: {1.0/np.sqrt(1000000 * result_ccram_x1x2.pvalue):.4f}")

    # Test independence between variables - CCRAM X2_X1
    result_ccram_x2x1 = permutation_test_ccram(
        contingency_table,
        direction="X2_X1",
        alternative="greater", 
        n_resamples=1000000
    )
    print("\nCCRAM Results (X2->X1) with 1000000 Re-samples:")
    print(f"    Observed CCRAM: {result_ccram_x2x1.statistic:.4f}")
    print(f"    P-value: {result_ccram_x2x1.pvalue:.4f}")
    print(f"    Null distribution: {result_ccram_x2x1.null_distribution.min():.4f} - {result_ccram_x2x1.null_distribution.max():.4f}")
    print(f"    Standard Deviation of Null Distribution: {result_ccram_x2x1.null_distribution.std():.4f}")
    print(f"    Relative Error: {1.0/np.sqrt(1000000 * result_ccram_x2x1.pvalue):.4f}")


if __name__ == "__main__":
    statistical_testing_example()
    """Example Output:
    
        SCCRAM Results (X1->X2) with 9999 Re-samples:
            Observed SCCRAM: 1.0000
            P-value: 0.0001
            Null distribution: 0.0011 - 0.3440
            Standard Deviation of Null Distribution: 0.0391
            Relative Error: 1.0001

        SCCRAM Results (X2->X1) with 9999 Re-samples:
            Observed SCCRAM: 0.0000
            P-value: 0.9984
            Null distribution: 0.0000 - 0.2354
            Standard Deviation of Null Distribution: 0.0285
            Relative Error: 0.0100

        CCRAM Results (X1->X2) with 1000000 Re-samples:
            Observed CCRAM: 0.7872
            P-value: 0.0000
            Null distribution: 0.0005 - 0.3390
            Standard Deviation of Null Distribution: 0.0310
            Relative Error: 1.0000

        CCRAM Results (X2->X1) with 1000000 Re-samples:
            Observed CCRAM: 0.0000
            P-value: 0.9985
            Null distribution: 0.0000 - 0.3125
            Standard Deviation of Null Distribution: 0.0267
            Relative Error: 0.0010
    
    """