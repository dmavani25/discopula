"""
  Discopula is a Python package for the estimation of copula-based models for discrete data. 
"""
from discopula.checkerboard.utils import gen_contingency_to_case_form, gen_case_form_to_contingency
from discopula.checkerboard.genstatsim import (
        bootstrap_ccram,
        permutation_test_ccram,
    )
from discopula.checkerboard.gencopula import GenericCheckerboardCopula

__version__ = "0.2.0"
__all__ = [
  "GenericCheckerboardCopula"
  "gen_contingency_to_case_form",
  "gen_case_form_to_contingency",
  "bootstrap_ccram",
  "permutation_test_ccram",
]
