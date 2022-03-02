import numpy as np
import pandas as pd

from ruleskit import HyperrectangleCondition, RegressionRule, ClassificationRule
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
    "xs,"
    "y,"
    "xs_test,"
    "y_test,"
    "rule_list,"
    "weights,"
    "exp_coverage,"
    "exp_act,"
    "exp_stacked_act,"
    "exp_covs,"
    "exp_preds,"
    "exp_crits,"
    "exp_pred,"
    "exp_crit,"
    "ruleset_pred_crit_fails",
    [
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            None,
            np.array([1, 3, 2, 1]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            None,
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8, columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, 1], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, 2, 1.5, 1.75]),
            1.8125 / 3.0,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            None,
            np.array([1, 3, 2, 1]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            pd.Series([1, 0], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8, columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, 1], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 1.5, 1.5]),
            0.25,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            None,
            np.array([1, 3, 2, 1]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            pd.Series([1], index=["X_0 in [1, 2]"]),
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8, columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, 1], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 1.5, 1.5]),
            0.25,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            None,
            np.array([1, 3, 2, 1]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            pd.Series([0.75, 1], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8, columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, 1], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, 2, 1.5, 1.785714286]),
            0.6224489795918368,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            None,
            np.array([1, 3, 2, 1]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            "criterion",
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8, columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, 1], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, 2, 1.5, 1.9]),
            2.06 / 3,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            None,
            np.array([1, 3, 2, 2, 3]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            None,
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2 / 3, 2 / 3, 1], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, 3, 2, 2, np.nan]),
            1.0,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            None,
            np.array([1, 3, 2, 2, 3]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            pd.Series([1, 0, 1], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2 / 3, 2 / 3, 1], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 2, 2, 2]),
            2 / 3,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            None,
            np.array([1, 3, 2, 2, 3]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            pd.Series([0.5, 1], index=["X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2 / 3, 2 / 3, 1], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, 3, np.nan, 2, 3]),
            1.0,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            None,
            np.array([1, 3, 2, 2, 3]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            "criterion",
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2 / 3, 2 / 3, 1], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, 3, 2, 2, np.nan]),
            1.0,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array(["a", "c", "b", "b", "c"]),
            None,
            np.array(["a", "c", "b", "b", "c"]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            "criterion",
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series(["b", "c", "b"], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2 / 3, 2 / 3, 1], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, "c", "b", "b", np.nan]),
            1.0,
            False,
        ),

        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            None,
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8,
                         columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, np.nan], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, 2, 1.5, 1.75]),
            1.8125 / 3.0,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            pd.Series([1, 0], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8,
                         columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, np.nan], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 1.5, 1.5]),
            0.25,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            pd.Series([1], index=["X_0 in [1, 2]"]),
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8,
                         columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, np.nan], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 1.5, 1.5]),
            0.25,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            pd.Series([0.75, 1], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8,
                         columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, np.nan], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, 2, 1.5, 1.785714286]),
            0.6224489795918368,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            "criterion",
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8,
                         columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, np.nan], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 1.5, 1.5]),
            0.25,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            None,
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([0, np.nan, np.nan],
                      index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, 3, 2, 2, np.nan]),
            1.0,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            pd.Series([1, 0, 1], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([0, np.nan, np.nan],
                      index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 2, 2, 2]),
            2 / 3,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            pd.Series([0.5, 1], index=["X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([0, np.nan, np.nan],
                      index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, 3, np.nan, 2, 3]),
            1.0,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            "criterion",
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([0, np.nan, np.nan],
                      index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 2, 2, 2]),
            2. / 3.,
            True
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array(["a", "c", "b", "b", "c"]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array(["a", "b"]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            "criterion",
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series(["b", "c", "b"], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([0, np.nan, np.nan],
                      index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, "b", "b", "b"]),
            2. / 3.,
            True
        ),
    ],
)
def test_unstacked_fit(
    clean,
    xs,
    y,
    xs_test,
    y_test,
    rule_list,
    weights,
    exp_coverage,
    exp_act,
    exp_stacked_act,
    exp_covs,
    exp_preds,
    exp_crits,
    exp_pred,
    exp_crit,
    ruleset_pred_crit_fails
):
    res = RuleSet(rule_list, remember_activation=True, stack_activation=True)
    res.fit(y, xs)
    res.eval(y_test, xs_test)
    assert res.ruleset_coverage == exp_coverage
    np.testing.assert_equal(res.activation, exp_act)
    pd.testing.assert_frame_equal(res.stacked_activations, exp_stacked_act)
    for r in res:
        assert r.coverage == exp_covs[str(r.condition)]
        assert r.prediction == exp_preds[str(r.condition)]
        if np.isnan(r.criterion):
            assert np.isnan(exp_crits[str(r.condition)])
        else:
            assert r.criterion == exp_crits[str(r.condition)]
    if ruleset_pred_crit_fails:
        with pytest.raises(ValueError) as e:
            prediction = res.calc_prediction(y=y, weights=weights)
            pd.testing.assert_series_equal(prediction, exp_pred)
            assert "No rules had non-zero/non-NaN weights" in str(e)
    else:
        prediction = res.calc_prediction(y=y, weights=weights)
        pd.testing.assert_series_equal(prediction, exp_pred)
        assert round(res.calc_criterion(y=y, predictions_vector=prediction), 6) == round(exp_crit, 6)


@pytest.mark.parametrize(
    "xs,"
    "y,"
    "xs_test,"
    "y_test,"
    "rule_list,"
    "weights,"
    "exp_coverage,"
    "exp_act,"
    "exp_stacked_act,"
    "exp_covs,"
    "exp_preds,"
    "exp_crits,"
    "exp_pred,"
    "exp_crit,"
    "ruleset_pred_crit_fails",
    [
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            None,
            np.array([1, 3, 2, 1]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            None,
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8, columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, 1], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, 2, 1.5, 1.75]),
            1.8125 / 3.0,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            None,
            np.array([1, 3, 2, 1]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            pd.Series([1, 0], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8, columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, 1], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 1.5, 1.5]),
            0.25,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            None,
            np.array([1, 3, 2, 1]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            pd.Series([1], index=["X_0 in [1, 2]"]),
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8, columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, 1], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 1.5, 1.5]),
            0.25,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            None,
            np.array([1, 3, 2, 1]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            pd.Series([0.75, 1], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8, columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, 1], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, 2, 1.5, 1.785714286]),
            0.6224489795918368,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            None,
            np.array([1, 3, 2, 1]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            "criterion",
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8, columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, 1], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, 2, 1.5, 1.9]),
            2.06 / 3,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            None,
            np.array([1, 3, 2, 2, 3]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            None,
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2 / 3, 2 / 3, 1], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, 3, 2, 2, np.nan]),
            1.0,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            None,
            np.array([1, 3, 2, 2, 3]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            pd.Series([1, 0, 1], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2 / 3, 2 / 3, 1], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 2, 2, 2]),
            2 / 3,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            None,
            np.array([1, 3, 2, 2, 3]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            pd.Series([0.5, 1], index=["X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2 / 3, 2 / 3, 1], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, 3, np.nan, 2, 3]),
            1.0,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            None,
            np.array([1, 3, 2, 2, 3]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            "criterion",
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2 / 3, 2 / 3, 1], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, 3, 2, 2, np.nan]),
            1.0,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array(["a", "c", "b", "b", "c"]),
            None,
            np.array(["a", "c", "b", "b", "c"]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            "criterion",
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series(["b", "c", "b"], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2 / 3, 2 / 3, 1], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, "c", "b", "b", np.nan]),
            1.0,
            False,
        ),

        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            None,
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8,
                         columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, np.nan], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, 2, 1.5, 1.75]),
            1.8125 / 3.0,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            pd.Series([1, 0], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8,
                         columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, np.nan], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 1.5, 1.5]),
            0.25,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            pd.Series([1], index=["X_0 in [1, 2]"]),
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8,
                         columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, np.nan], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 1.5, 1.5]),
            0.25,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            pd.Series([0.75, 1], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8,
                         columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, np.nan], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, 2, 1.5, 1.785714286]),
            0.6224489795918368,
            False,
        ),
        (
            np.array([[5, 3], [3, 4], [2, np.nan], [2, 4]]),
            np.array([1, 3, 2, 1]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
            ],
            "criterion",
            0.75,
            np.array([0, 1, 1, 1]),
            pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8,
                         columns=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.5, 0.5], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([1.5, 2], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([0.25, np.nan], index=["X_0 in [1, 2]", "X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 1.5, 1.5]),
            0.25,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            None,
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([0, np.nan, np.nan],
                      index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, 3, 2, 2, np.nan]),
            1.0,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            pd.Series([1, 0, 1], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([0, np.nan, np.nan],
                      index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 2, 2, 2]),
            2 / 3,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            pd.Series([0.5, 1], index=["X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([0, np.nan, np.nan],
                      index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, 3, np.nan, 2, 3]),
            1.0,
            False,
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array([1, 3, 2, 2, 3]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array([1, 2]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            "criterion",
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([2, 3, 2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([0, np.nan, np.nan],
                      index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, 2, 2, 2]),
            2. / 3.,
            True
        ),
        (
            np.array([[5, 3], [5, 4], [2, np.nan], [2, 4], [2, 6]]),
            np.array(["a", "c", "b", "b", "c"]),
            np.array([[1, 2], [np.nan, 3]]),
            np.array(["a", "b"]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[6])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 4], bmaxs=[3, 5])),
            ],
            "criterion",
            0.8,
            np.array([0, 1, 1, 1, 1]),
            pd.DataFrame(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
                dtype=np.uint8,
                columns=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"],
            ),
            pd.Series([0.6, 0.6, 0.2], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series(["b", "c", "b"], index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([0, np.nan, np.nan],
                      index=["X_0 in [1, 2]", "X_1 in [4, 6]", "X_0 in [2, 3] AND X_1 in [4, 5]"]),
            pd.Series([np.nan, np.nan, "b", "b", "b"]),
            2. / 3.,
            True
        ),
    ],
)
def test_stacked_fit(
    clean_for_stacked_fit,
    xs,
    y,
    xs_test,
    y_test,
    rule_list,
    weights,
    exp_coverage,
    exp_act,
    exp_stacked_act,
    exp_covs,
    exp_preds,
    exp_crits,
    exp_pred,
    exp_crit,
    ruleset_pred_crit_fails
):
    res = RuleSet(rule_list, remember_activation=True, stack_activation=True)
    res.fit(y, xs)
    res.eval(y_test, xs_test)
    assert res.ruleset_coverage == exp_coverage
    np.testing.assert_equal(res.activation, exp_act)
    pd.testing.assert_frame_equal(res.stacked_activations, exp_stacked_act)
    for r in res:
        assert r.coverage == exp_covs[str(r.condition)]
        assert r.prediction == exp_preds[str(r.condition)]
        if np.isnan(r.criterion):
            assert np.isnan(exp_crits[str(r.condition)])
        else:
            assert r.criterion == exp_crits[str(r.condition)]
    if ruleset_pred_crit_fails:
        with pytest.raises(ValueError) as e:
            prediction = res.calc_prediction(y=y, weights=weights)
            pd.testing.assert_series_equal(prediction, exp_pred)
            assert "No rules had non-zero/non-NaN weights" in str(e)
    else:
        prediction = res.calc_prediction(y=y, weights=weights)
        pd.testing.assert_series_equal(prediction, exp_pred)
        assert round(res.calc_criterion(y=y, predictions_vector=prediction), 6) == round(exp_crit, 6)


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


def test_not_fitted():
    res = RuleSet(
        [
            Rule(HyperrectangleCondition([0, 1], bmins=[1, 1], bmaxs=[2, 3])),
            Rule(HyperrectangleCondition([1, 2], bmins=[4, 4], bmaxs=[5, 4])),
        ]
    )
    with pytest.raises(ValueError) as e:
        res.eval(y=np.array([0, 1, 2]))
        assert "Not all rules of the ruleset were fitted" in str(e)
