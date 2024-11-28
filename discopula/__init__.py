"""
  Discopula is a Python package for the estimation of copula-based models for discrete data. 
"""

from discopula.checkerboard.copula import CheckerboardCopula
from discopula.checkerboard.copula import contingency_to_case_form, case_form_to_contingency
from discopula.checkerboard.copula import bootstrap_ccram, bootstrap_sccram
__version__ = "0.0.3"
__all__ = ["CheckerboardCopula", "contingency_to_case_form", "case_form_to_contingency", "bootstrap_ccram", "bootstrap_sccram"]
