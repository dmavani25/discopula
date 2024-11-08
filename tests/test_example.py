"""
  Basic boilerplate to illustrate setting up unit tests.
"""

from discopula.module import Template


def test_template():
    """
    Test the only functionality there is, saying hello ;-)
    """
    tpl = Template()
    assert tpl.hello().my_name_is("Christophe") == "hello Christophe"
