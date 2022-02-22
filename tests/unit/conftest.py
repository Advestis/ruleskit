from ruleskit import Activation, Rule, RegressionRule, ClassificationRule, RuleSet
import pytest


@pytest.fixture
def clean():
    yield
    Activation.clean_files()


@pytest.fixture
def clean_for_stacked_fit():
    RuleSet.STACKED_FIT = True
    yield
    Activation.clean_files()
    RuleSet.STACKED_FIT = False


@pytest.fixture
def prepare_fit():
    calc_attributes_r = Rule.calc_attributes
    calc_attributes_rr = RegressionRule.calc_attributes
    calc_attributes_cr = ClassificationRule.calc_attributes
    yield
    Rule.calc_attributes = calc_attributes_r
    RegressionRule.calc_attributes = calc_attributes_rr
    ClassificationRule.calc_attributes = calc_attributes_cr
