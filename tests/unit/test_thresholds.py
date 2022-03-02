import numpy as np
import pandas as pd
import pytest

from ruleskit import Thresholds, RegressionRule, HyperrectangleCondition, RuleSet, ClassificationRule


class Dummy:
    def __init__(self, **kwargs):
        self.crit_max = 2
        for key in kwargs:
            setattr(self, key, kwargs[key])


def test_thresholds():
    ts = Thresholds("tests/unit/data/thresholds.json")
    assert not ts("coverage", Dummy(coverage=0.03))
    assert ts("coverage", Dummy(coverage=0.06))
    assert not ts("criterion", Dummy(criterion=-1.5))
    assert ts("criterion", Dummy(criterion=1.5))


@pytest.mark.parametrize(
    "xs,"
    "y,"
    "rule_list,"
    "coverage,"
    "exp_act,"
    "exp_stacked_act,"
    "exp_covs,"
    "exp_preds,"
    "exp_crits,"
    "exp_stacked_act_2,"
    "exp_covs_2,"
    "exp_preds_2,"
    "exp_crits_2,"
    "stacked_fit",
    [
        (
            np.array(
                [
                    [5, 3],
                    [3, 4],
                    [2, np.nan],
                    [2, 4]
                ]
            ),
            np.array([-4, 3, 2, 1]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
                RegressionRule(HyperrectangleCondition([1], bmins=[0], bmaxs=[3])),
                RegressionRule(HyperrectangleCondition([0], bmins=[2], bmaxs=[5])),
                RegressionRule(HyperrectangleCondition([0, 1], bmins=[2, 0], bmaxs=[5, 3])),
            ],
            1.0,
            np.array([1, 1, 1, 1]),
            pd.DataFrame([
                [0, 0, 1, 1, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 1, 0, 1, 0]
            ],
             dtype=np.uint8,
             columns=[
                 "X_0 in [1, 2]",
                 "X_1 in [4, 5]",
                 "X_1 in [0, 3]",
                 "X_0 in [2, 5]",
                 "X_0 in [2, 5] AND X_1 in [0, 3]"
             ]
            ),
            pd.Series(
                [0.5, 0.5, 0.25, 1.0, 0.25],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                    "X_1 in [0, 3]",
                    "X_0 in [2, 5]",
                    "X_0 in [2, 5] AND X_1 in [0, 3]"
                ]
            ),
            pd.Series(
                [1.5, 2, -4, 0.5, -4],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                    "X_1 in [0, 3]",
                    "X_0 in [2, 5]",
                    "X_0 in [2, 5] AND X_1 in [0, 3]"
                ]
            ),
            pd.Series(
                [0.25, 1, 0, 7.25, 0],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                    "X_1 in [0, 3]",
                    "X_0 in [2, 5]",
                    "X_0 in [2, 5] AND X_1 in [0, 3]"
                ]
            ),

            pd.DataFrame([
                [0, 1],
                [1, 1],
                [0, 1],
                [1, 1]
            ],
             dtype=np.uint8,
             columns=[
                 "X_1 in [4, 5]",
                 "X_0 in [2, 5]",
             ]
            ),
            pd.Series(
                [0.5, 1.0],
                index=[
                    "X_1 in [4, 5]",
                    "X_0 in [2, 5]",
                ]
            ),
            pd.Series(
                [2, 0.5],
                index=[
                    "X_1 in [4, 5]",
                    "X_0 in [2, 5]",
                ]
            ),
            pd.Series(
                [1, 7.25],
                index=[
                    "X_1 in [4, 5]",
                    "X_0 in [2, 5]",
                ]
            ),

            True
        ),
        (
            np.array(
                [
                    [5, 3],
                    [3, 4],
                    [2, np.nan],
                    [2, 4]
                ]
            ),
            np.array([-4, 3, 2, 1]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[0], bmaxs=[3])),
                ClassificationRule(HyperrectangleCondition([0], bmins=[2], bmaxs=[5])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 0], bmaxs=[5, 3])),
            ],
            1.0,
            np.array([1, 1, 1, 1]),
            pd.DataFrame([
                [0, 0, 1, 1, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 1, 0, 1, 0]
            ],
             dtype=np.uint8,
             columns=[
                 "X_0 in [1, 2]",
                 "X_1 in [4, 5]",
                 "X_1 in [0, 3]",
                 "X_0 in [2, 5]",
                 "X_0 in [2, 5] AND X_1 in [0, 3]"
             ]
            ),
            pd.Series(
                [0.5, 0.5, 0.25, 1.0, 0.25],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                    "X_1 in [0, 3]",
                    "X_0 in [2, 5]",
                    "X_0 in [2, 5] AND X_1 in [0, 3]"
                ]
            ),
            pd.Series(
                [1, 1, -4, -4, -4],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                    "X_1 in [0, 3]",
                    "X_0 in [2, 5]",
                    "X_0 in [2, 5] AND X_1 in [0, 3]"
                ]
            ),
            pd.Series(
                [0.5, 0.5, 1, 0.25, 1],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                    "X_1 in [0, 3]",
                    "X_0 in [2, 5]",
                    "X_0 in [2, 5] AND X_1 in [0, 3]"
                ]
            ),

            pd.DataFrame([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ],
                dtype=np.uint8,
                columns=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                ]
            ),
            pd.Series(
                [0.5, 0.5],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                ]
            ),
            pd.Series(
                [1, 1],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                ]
            ),
            pd.Series(
                [0.5, 0.5],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                ]
            ),

            True
        ),
        (
            np.array(
                [
                    [5, 3],
                    [3, 4],
                    [2, np.nan],
                    [2, 4]
                ]
            ),
            np.array([-4, 3, 2, 1]),
            [
                RegressionRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                RegressionRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
                RegressionRule(HyperrectangleCondition([1], bmins=[0], bmaxs=[3])),
                RegressionRule(HyperrectangleCondition([0], bmins=[2], bmaxs=[5])),
                RegressionRule(HyperrectangleCondition([0, 1], bmins=[2, 0], bmaxs=[5, 3])),
            ],
            1.0,
            np.array([1, 1, 1, 1]),
            pd.DataFrame([
                [0, 0, 1, 1, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 1, 0, 1, 0]
            ],
             dtype=np.uint8,
             columns=[
                 "X_0 in [1, 2]",
                 "X_1 in [4, 5]",
                 "X_1 in [0, 3]",
                 "X_0 in [2, 5]",
                 "X_0 in [2, 5] AND X_1 in [0, 3]"
             ]
            ),
            pd.Series(
                [0.5, 0.5, 0.25, 1.0, 0.25],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                    "X_1 in [0, 3]",
                    "X_0 in [2, 5]",
                    "X_0 in [2, 5] AND X_1 in [0, 3]"
                ]
            ),
            pd.Series(
                [1.5, 2, -4, 0.5, -4],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                    "X_1 in [0, 3]",
                    "X_0 in [2, 5]",
                    "X_0 in [2, 5] AND X_1 in [0, 3]"
                ]
            ),
            pd.Series(
                [0.25, 1, 0, 7.25, 0],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                    "X_1 in [0, 3]",
                    "X_0 in [2, 5]",
                    "X_0 in [2, 5] AND X_1 in [0, 3]"
                ]
            ),

            pd.DataFrame([
                [0, 1],
                [1, 1],
                [0, 1],
                [1, 1]
            ],
             dtype=np.uint8,
             columns=[
                 "X_1 in [4, 5]",
                 "X_0 in [2, 5]",
             ]
            ),
            pd.Series(
                [0.5, 1.0],
                index=[
                    "X_1 in [4, 5]",
                    "X_0 in [2, 5]",
                ]
            ),
            pd.Series(
                [2, 0.5],
                index=[
                    "X_1 in [4, 5]",
                    "X_0 in [2, 5]",
                ]
            ),
            pd.Series(
                [1, 7.25],
                index=[
                    "X_1 in [4, 5]",
                    "X_0 in [2, 5]",
                ]
            ),

            False
        ),
        (
            np.array(
                [
                    [5, 3],
                    [3, 4],
                    [2, np.nan],
                    [2, 4]
                ]
            ),
            np.array([-4, 3, 2, 1]),
            [
                ClassificationRule(HyperrectangleCondition([0], bmins=[1], bmaxs=[2])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[4], bmaxs=[5])),
                ClassificationRule(HyperrectangleCondition([1], bmins=[0], bmaxs=[3])),
                ClassificationRule(HyperrectangleCondition([0], bmins=[2], bmaxs=[5])),
                ClassificationRule(HyperrectangleCondition([0, 1], bmins=[2, 0], bmaxs=[5, 3])),
            ],
            1.0,
            np.array([1, 1, 1, 1]),
            pd.DataFrame([
                [0, 0, 1, 1, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 1, 0, 1, 0]
            ],
             dtype=np.uint8,
             columns=[
                 "X_0 in [1, 2]",
                 "X_1 in [4, 5]",
                 "X_1 in [0, 3]",
                 "X_0 in [2, 5]",
                 "X_0 in [2, 5] AND X_1 in [0, 3]"
             ]
            ),
            pd.Series(
                [0.5, 0.5, 0.25, 1.0, 0.25],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                    "X_1 in [0, 3]",
                    "X_0 in [2, 5]",
                    "X_0 in [2, 5] AND X_1 in [0, 3]"
                ]
            ),
            pd.Series(
                [1, 1, -4, -4, -4],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                    "X_1 in [0, 3]",
                    "X_0 in [2, 5]",
                    "X_0 in [2, 5] AND X_1 in [0, 3]"
                ]
            ),
            pd.Series(
                [0.5, 0.5, 1, 0.25, 1],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                    "X_1 in [0, 3]",
                    "X_0 in [2, 5]",
                    "X_0 in [2, 5] AND X_1 in [0, 3]"
                ]
            ),

            pd.DataFrame([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ],
                dtype=np.uint8,
                columns=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                ]
            ),
            pd.Series(
                [0.5, 0.5],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                ]
            ),
            pd.Series(
                [1, 1],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                ]
            ),
            pd.Series(
                [0.5, 0.5],
                index=[
                    "X_0 in [1, 2]",
                    "X_1 in [4, 5]",
                ]
            ),

            False
        ),
    ],
)
def test_thresholds_fit(
        clean_for_stacked_fit_th,
        xs,
        y,
        rule_list,
        coverage,
        exp_act,
        exp_stacked_act,
        exp_covs,
        exp_preds,
        exp_crits,
        exp_stacked_act_2,
        exp_covs_2,
        exp_preds_2,
        exp_crits_2,
        stacked_fit
):
    res = RuleSet(rule_list, remember_activation=True, stack_activation=True)
    if not stacked_fit:
        RuleSet.STACKED_FIT = False
    res.fit(y, xs)
    res.eval(y, xs)
    assert res.ruleset_coverage == coverage
    np.testing.assert_equal(res.activation, exp_act)
    pd.testing.assert_frame_equal(res.stacked_activations, exp_stacked_act)
    for r in res:
        assert r.coverage == exp_covs[str(r.condition)]
        assert r.prediction == exp_preds[str(r.condition)]
        assert r.criterion == exp_crits[str(r.condition)]

    ClassificationRule.SET_THRESHOLDS("tests/unit/data/thresholds.json")
    RegressionRule.SET_THRESHOLDS("tests/unit/data/thresholds.json")

    res.fit(y, xs)
    res.eval(y, xs)
    pd.testing.assert_frame_equal(res.stacked_activations, exp_stacked_act_2)
    for r in res:
        assert r.coverage == exp_covs_2[str(r.condition)]
        assert r.prediction == exp_preds_2[str(r.condition)]
        assert r.criterion == exp_crits_2[str(r.condition)]
