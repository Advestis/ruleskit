import numpy as np
from ruleskit import HyperrectangleCondition
from ruleskit import Rule
from ruleskit import RuleSet
import pytest


@pytest.mark.parametrize(
    "rule1, rule2, rs",
    [
        (
            Rule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
            Rule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            RuleSet(
                [
                    Rule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                    Rule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
                ]
            ),
        ),
    ],
)
def test_add(rule1, rule2, rs):
    res = RuleSet([rule1])
    res += RuleSet([rule2])
    np.testing.assert_equal(res, rs)


@pytest.mark.parametrize(
    "xs, y, rule_list, coverage",
    [
        (
            np.array([[5, 3], [3, 4], [2, np.nan]]),
            np.array([1, 3, 2]),
            [
                Rule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                Rule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            2 / 3,
        ),
    ],
)
def test_coverage(clean, xs, y, rule_list, coverage):
    [r.fit(xs, y) for r in rule_list]
    res = RuleSet(rule_list, remember_activation=True, stack_activation=True)
    np.testing.assert_equal(res.ruleset_coverage, coverage)
    res.calc_activation(xs)
    np.testing.assert_equal(res.ruleset_coverage, coverage)


@pytest.mark.parametrize(
    "rs",
    [
        RuleSet(
            [
                Rule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                Rule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ]
        ),
    ],
)
def test_save(rs):
    rs.save("tests/unit/data/ruleset.csv")


@pytest.mark.parametrize(
    "rs",
    [
        RuleSet(
            [
                Rule(HyperrectangleCondition([0, 1], bmins=[1, 1], bmaxs=[2, 3])),
                Rule(HyperrectangleCondition([1, 2], bmins=[4, 4], bmaxs=[5, 4])),
            ]
        ),
    ],
)
def test_save(rs):
    rs.save("tests/unit/data/ruleset.csv")


def test_load():
    rs = RuleSet()
    rs.load("tests/unit/data/ruleset_test.csv")
    for rule in rs:
        print(rule)
