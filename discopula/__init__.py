"""
  Discopula is a Python package for the estimation of copula-based models for discrete data. 
"""

from discopula.checkerboard.copula import CheckerboardCopula
from discopula.checkerboard.utils import contingency_to_case_form, case_form_to_contingency
from discopula.checkerboard.statsim import (
        bootstrap_ccram, bootstrap_sccram,
        bootstrap_regression_U1_on_U2, bootstrap_regression_U2_on_U1,
        bootstrap_regression_U1_on_U2_vectorized, bootstrap_regression_U2_on_U1_vectorized,
        bootstrap_predict_X1_from_X2, bootstrap_predict_X2_from_X1,
        bootstrap_predict_X1_from_X2_vectorized, bootstrap_predict_X2_from_X1_vectorized,
        bootstrap_predict_X1_from_X2_all_comb_summary, bootstrap_predict_X2_from_X1_all_comb_summary,
        permutation_test_ccram, permutation_test_sccram
    )

__version__ = "0.1.0"
__all__ = [
  "CheckerboardCopula", "contingency_to_case_form", "case_form_to_contingency",
  "bootstrap_ccram", "bootstrap_sccram",
  "bootstrap_regression_U1_on_U2", "bootstrap_regression_U2_on_U1",
  "bootstrap_regression_U1_on_U2_vectorized", "bootstrap_regression_U2_on_U1_vectorized",
  "bootstrap_predict_X2_from_X1", "bootstrap_predict_X1_from_X2",
  "bootstrap_predict_X2_from_X1_vectorized", "bootstrap_predict_X1_from_X2_vectorized",
  "bootstrap_predict_X1_from_X2_all_comb_summary", "bootstrap_predict_X2_from_X1_all_comb_summary",
  "permutation_test_ccram", "permutation_test_sccram"]
