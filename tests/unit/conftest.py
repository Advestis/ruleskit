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
