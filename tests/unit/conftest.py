from ruleskit import Activation, Rule, RegressionRule, ClassificationRule
import pytest


@pytest.fixture
def clean():
    yield
    Activation.clean_files()


@pytest.fixture
def prepare_fit():
    calc_attributes_r = Rule.calc_attributes
    calc_attributes_rr = RegressionRule.calc_attributes
    calc_attributes_cr = ClassificationRule.calc_attributes
    yield
    Rule.calc_attributes = calc_attributes_r
    RegressionRule.calc_attributes = calc_attributes_rr
    ClassificationRule.calc_attributes = calc_attributes_cr
