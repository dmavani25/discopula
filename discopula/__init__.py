"""
  Discopula is a Python package for the estimation of copula-based models for discrete data. 
"""
from discopula.checkerboard.gencopula import GenericCheckerboardCopula
from discopula.checkerboard.utils import gen_contingency_to_case_form, gen_case_form_to_contingency
from discopula.checkerboard.genstatsim import (
        bootstrap_ccram,
        bootstrap_predict_category_summary,
        permutation_test_ccram,
    )

__version__ = "0.6.1"
__all__ = [
  "GenericCheckerboardCopula",
  "gen_contingency_to_case_form",
  "gen_case_form_to_contingency",
  "bootstrap_ccram",
  "bootstrap_predict_category_summary",
  "permutation_test_ccram",
]
