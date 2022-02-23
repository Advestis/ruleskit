from ruleskit import Activation, Rule, RegressionRule, ClassificationRule, RuleSet, Thresholds
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
def clean_for_stacked_fit_th():
    RuleSet.STACKED_FIT = True
    yield
    Activation.clean_files()
    RuleSet.STACKED_FIT = False
    ClassificationRule.SET_THRESHOLDS(None)
    RegressionRule.SET_THRESHOLDS(None)


@pytest.fixture
def prepare_fit():
    calc_attributes_r = Rule.calc_attributes
    calc_attributes_rr = RegressionRule.calc_attributes
    calc_attributes_cr = ClassificationRule.calc_attributes
    yield
    Rule.calc_attributes = calc_attributes_r
    RegressionRule.calc_attributes = calc_attributes_rr
    ClassificationRule.calc_attributes = calc_attributes_cr
