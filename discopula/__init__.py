"""
  Discopula is a Python package for the estimation of copula-based models for discrete data. 
"""

from discopula.checkerboard.copula import CheckerboardCopula
from discopula.checkerboard.copula import contingency_to_case_form, case_form_to_contingency
from discopula.checkerboard.copula import bootstrap_ccram, bootstrap_sccram
from discopula.checkerboard.copula import bootstrap_regression_U1_on_U2, bootstrap_regression_U2_on_U1
from discopula.checkerboard.copula import bootstrap_regression_U1_on_U2_vectorized, bootstrap_regression_U2_on_U1_vectorized
from discopula.checkerboard.copula import bootstrap_predict_X2_from_X1, bootstrap_predict_X1_from_X2
from discopula.checkerboard.copula import bootstrap_predict_X2_from_X1_vectorized, bootstrap_predict_X1_from_X2_vectorized
__version__ = "0.0.4"
__all__ = [
  "CheckerboardCopula", "contingency_to_case_form", "case_form_to_contingency",
  "bootstrap_ccram", "bootstrap_sccram",
  "bootstrap_regression_U1_on_U2", "bootstrap_regression_U2_on_U1",
  "bootstrap_regression_U1_on_U2_vectorized", "bootstrap_regression_U2_on_U1_vectorized",
  "bootstrap_predict_X2_from_X1", "bootstrap_predict_X1_from_X2",
  "bootstrap_predict_X2_from_X1_vectorized", "bootstrap_predict_X1_from_X2_vectorized"]
