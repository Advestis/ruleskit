from ruleskit import Rule, RegressionRule, ClassificationRule, HyperrectangleCondition, Activation
from bitarray import bitarray
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import itertools


# noinspection PyUnresolvedReferences,PyCallingNonCallable
@pytest.mark.parametrize(
    "condition_activation_error, theclass",
    itertools.product(
        [
            (
                HyperrectangleCondition(features_indexes=[0, 1], features_names=["A", "B"], bmins=[0, 1], bmaxs=[2, 3]),
                Activation(np.array([0, 1, 1, 0]), to_file=False),
                None,
            ),
            (None, None, None),
            (None, Activation(np.array([0, 1, 1, 0]), to_file=False), ValueError),
            (
                HyperrectangleCondition(features_indexes=[0, 1], features_names=["A", "B"], bmins=[0, 1], bmaxs=[2, 3]),
                None,
                None,
            ),
        ],
        [Rule, RegressionRule, ClassificationRule],
    ),
)
def test_init(clean, condition_activation_error, theclass):
    condition, activation, error = condition_activation_error
    if error is not None:
        with pytest.raises(error):
            _ = theclass(condition, activation)
    else:
        r = theclass(condition, activation)
        assert r.condition == condition
        if activation is not None:
            np.testing.assert_equal(r.activation, activation.raw)
        else:
            assert r.activation is None
        assert r._activation == activation


@pytest.mark.parametrize(
    "x_y_condition_activation_s_error, theclass",
    itertools.product(
        [
            (
                np.array([[1, 3], [3, 4], [2, np.nan]]),
                np.array([1, 3, 2]),
                HyperrectangleCondition([0], bmins=[1], bmaxs=[2]),
                np.array([1, 0, 1]),
                "101",
                None,
            ),
            (
                np.array([[1, 3], [3, 4], [2, np.nan]]),
                np.array([1, 3, 2]),
                HyperrectangleCondition([1], bmins=[3], bmaxs=[5]),
                np.array([1, 1, 0]),
                "110",
                None,
            ),
            (
                np.array([[1, 3], [3, 4], [2, np.nan]]),
                np.array([1, 3, 2]),
                HyperrectangleCondition([0, 1], bmins=[1, 3], bmaxs=[2, 5]),
                np.array([1, 0, 0]),
                "100",
                None,
            ),
            (
                pd.DataFrame(data=[[1, 3], [3, 4], [2, np.nan]], columns=["A", "B"]),
                np.array([1, 3, 2]),
                HyperrectangleCondition([0, 1], features_names=["A", "B"], bmins=[1, 3], bmaxs=[2, 5]),
                np.array([1, 0, 0]),
                "100",
                None,
            ),
        ],
        [Rule, RegressionRule, ClassificationRule],
    ),
)
def test_fit(clean, x_y_condition_activation_s_error, theclass):
    x, y, condition, activation, s, error = x_y_condition_activation_s_error
    # noinspection PyCallingNonCallable
    rule = theclass(condition=condition)
    rule.fit(xs=x, y=y)
    np.testing.assert_equal(rule.activation, activation)
    np.testing.assert_equal(rule._activation.as_bitarray, bitarray(s))
    assert rule._activation.as_integer == int(s, 2)


@pytest.mark.parametrize(
    "x, y, condition, activation",
    [
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            np.array([1, 3, 2]),
            HyperrectangleCondition([0, 1], bmins=[1, 3], bmaxs=[2, 5]),
            np.array([1, 0, 0]),
        ),
    ],
)
def test_local_activation(clean, x, y, condition, activation):
    Rule.LOCAL_ACTIVATION = False
    rule = RegressionRule(condition=condition)
    rule.fit(xs=x, y=y)
    assert not isinstance(rule._activation.data, Path)
    Rule.LOCAL_ACTIVATION = True
    rule = RegressionRule(condition=condition)
    rule.fit(xs=x, y=y)
    assert isinstance(rule._activation.data, Path) and rule._activation.data.is_file()


@pytest.mark.parametrize(
    "x, y, condition, cov, pred",
    [
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            np.array([1, 3, 2]),
            HyperrectangleCondition([0], bmins=[1], bmaxs=[2]),
            2 / 3,
            1.5,
        ),
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            np.array([1, 3, 2]),
            HyperrectangleCondition([1], bmins=[3], bmaxs=[5]),
            2 / 3,
            2,
        ),
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            np.array([1, 3, 2]),
            HyperrectangleCondition([0, 1], bmins=[1, 3], bmaxs=[2, 5]),
            1 / 3,
            1,
        ),
    ],
)
def test_regression_attributes(clean, x, y, condition, cov, pred):
    rule = RegressionRule(condition=condition)
    rule.fit(xs=x, y=y)
    np.testing.assert_equal(rule.coverage, cov)
    np.testing.assert_equal(rule.prediction, pred)


@pytest.mark.parametrize(
    "x, y, condition, proba, pred, crit",
    [
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            np.array(["a", "b", "a"]),
            HyperrectangleCondition([0], bmins=[1], bmaxs=[2]),
            [("a", 1.0)],
            "a",
            1.0,
        ),
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            np.array(["a", "b", "a"]),
            HyperrectangleCondition([1], bmins=[4], bmaxs=[5]),
            [("b", 1.0)],
            "b",
            1.0,
        ),
        (
            np.array([[1, 3], [2, 4], [5, np.nan]]),
            np.array(["a", "b", "a"]),
            HyperrectangleCondition([0], bmins=[1], bmaxs=[5]),
            [("a", 2 / 3), ("b", 1 / 3)],
            "a",
            2 / 3,
        ),
    ],
)
def test_classification_attributes(clean, x, y, condition, proba, pred, crit):
    rule = ClassificationRule(condition=condition)
    rule.fit(xs=x, y=y)
    rule.eval(y=y)
    np.testing.assert_equal(rule._prediction, proba)
    np.testing.assert_equal(rule.prediction, pred)
    np.testing.assert_equal(rule.criterion, crit)


@pytest.mark.parametrize(
    "x, y, condition1, condition2, activation1, activation2, activation_test",
    [
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            np.array([1, 3, 2]),
            HyperrectangleCondition([0], bmins=[1], bmaxs=[2]),
            HyperrectangleCondition([1], bmins=[3], bmaxs=[5]),
            np.array([1, 0, 1]),
            np.array([1, 1, 0]),
            np.array([1, 0, 0]),
        ),
    ],
)
def test_and(clean, x, y, condition1, condition2, activation1, activation2, activation_test):
    rule1 = RegressionRule(condition=condition1)
    rule1.fit(xs=x, y=y)
    rule2 = RegressionRule(condition=condition2)
    rule2.fit(xs=x, y=y)

    new_rule = rule1 & rule2
    np.testing.assert_equal(new_rule.activation, activation_test)
    new_rule.fit(xs=x, y=y)
    np.testing.assert_equal(new_rule.activation, activation_test)


@pytest.mark.parametrize(
    "condition_x_y_x2_expected, theclass",
    itertools.product(
        [
            (
                HyperrectangleCondition([0, 1], bmins=[1, 3], bmaxs=[2, 5]),
                np.array([[1, 3], [3, 4], [2, np.nan]]),
                np.array([1.5, 3, 2]),
                np.array([[1, 3], [0, 4], [0, 5], [np.nan, np.nan]]),
                np.array([1.5, np.nan, np.nan, np.nan]),
            ),
            (
                HyperrectangleCondition([0, 1], bmins=[1, 3], bmaxs=[2, 5]),
                np.array([[1, 3], [3, 4], [2, np.nan]]),
                np.array([1.5, 3, 2]),
                None,
                np.array([1.5, np.nan, np.nan]),
            ),
            (
                HyperrectangleCondition([0, 1], features_names=["A", "B"], bmins=[1, 3], bmaxs=[2, 5]),
                pd.DataFrame(data=[[1, 3], [3, 4], [2, np.nan]], columns=["A", "B"]),
                np.array([1.5, 3, 2]),
                np.array([[1, 3], [0, 4], [0, 5], [np.nan, np.nan]]),
                np.array([1.5, np.nan, np.nan, np.nan]),
            ),
            (
                HyperrectangleCondition([0, 1], features_names=["A", "B"], bmins=[1, 3], bmaxs=[2, 5]),
                pd.DataFrame(data=[[1, 3], [3, 4], [2, np.nan]], columns=["A", "B"]),
                np.array([1.5, 3, 2]),
                None,
                np.array([1.5, np.nan, np.nan]),
            ),
            (
                HyperrectangleCondition([0, 1], features_names=["A", "B"], bmins=[1, 3], bmaxs=[2, 5]),
                np.array([[1, 3], [3, 4], [2, np.nan]]),
                np.array([1.5, 3, 2]),
                pd.DataFrame([[1, 3], [0, 4], [0, 5], [np.nan, np.nan]], columns=["A", "B"]),
                np.array([1.5, np.nan, np.nan, np.nan]),
            ),
            (
                HyperrectangleCondition([0, 1], features_names=["A", "B"], bmins=[1, 3], bmaxs=[2, 5]),
                pd.DataFrame(data=[[1, 3], [3, 4], [2, np.nan]], columns=["A", "B"]),
                np.array([1.5, 3, 2]),
                pd.DataFrame([[1, 3], [0, 4], [0, 5], [np.nan, np.nan]], columns=["A", "B"]),
                np.array([1.5, np.nan, np.nan, np.nan]),
            ),
            (
                HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                np.array([[3, 1], [4, 3], [np.nan, 2]]),
                np.array([1.5, 3, 2]),
                np.array([[3, 1], [4, 0], [5, 0], [np.nan, np.nan]]),
                np.array([1.5, np.nan, np.nan, np.nan]),
            ),
            (
                HyperrectangleCondition([1, 0], features_names=["A", "B"], bmins=[1, 3], bmaxs=[2, 5]),
                np.array([[3, 1], [4, 3], [np.nan, 2]]),
                np.array([1.5, 3, 2]),
                np.array([[3, 1], [4, 0], [5, 0], [np.nan, np.nan]]),
                np.array([1.5, np.nan, np.nan, np.nan]),
            ),
            (
                HyperrectangleCondition([0, 1], features_names=["B", "A"], bmins=[1, 3], bmaxs=[2, 5]),
                pd.DataFrame([[3, 1], [4, 3], [np.nan, 2]], columns=["A", "B"]),
                np.array([1.5, 3, 2]),
                pd.DataFrame([[3, 1], [4, 0], [5, 0], [np.nan, np.nan]], columns=["A", "B"]),
                np.array([1.5, np.nan, np.nan, np.nan]),
            ),
            (
                HyperrectangleCondition([0, 1], features_names=["B", "A"], bmins=[1, 3], bmaxs=[2, 5]),
                pd.DataFrame([[3, 1], [4, 3], [np.nan, 2]], columns=["A", "B"]),
                np.array(["a", "b", "c"]),
                pd.DataFrame([[3, 1], [4, 0], [5, 0], [np.nan, np.nan]], columns=["A", "B"]),
                np.array(["a", "nan", "nan", "nan"]),
            ),
        ],
        [Rule, RegressionRule, ClassificationRule],
    ),
)
def test_predict(clean, condition_x_y_x2_expected, theclass):
    condition, x, y, x2, expected = condition_x_y_x2_expected
    # noinspection PyCallingNonCallable
    rule = theclass(condition=condition)
    if y.dtype == str and theclass != ClassificationRule:
        return

    if theclass == Rule:
        rule.fit(xs=x, y=y)
        assert rule.prediction is None
        return
    if y.dtype.type == np.str_ and theclass != ClassificationRule:
        return

    rule.fit(xs=x, y=y)
    pred = rule.predict(x2)
    if theclass == ClassificationRule:
        np.testing.assert_equal(pred, expected)
    else:
        np.testing.assert_almost_equal(pred, expected)


@pytest.mark.parametrize(
    "rule1, rule2, pred1, pred2, expected",
    [
        (
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 0, 1]), to_file=False)),
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 1, 0]), to_file=False)),
            0.2,
            0.3,
            1/3,
        ),
        (
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 0, 1]), to_file=False)),
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 1, 0]), to_file=False)),
            0.2,
            -0.3,
            -1/3,
        ),
        (
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 0, 1]), to_file=False)),
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 1, 0]), to_file=False)),
            -0.2,
            -0.3,
            1/3,
        ),
        (
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 0, 1]), to_file=False)),
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 0, 0]), to_file=False)),
            0.2,
            0.3,
            2/3,
        ),
        (
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 0, 1]), to_file=False)),
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 0, 0]), to_file=False)),
            -0.2,
            0.3,
            -2/3,
        ),
        (
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 0, 1]), to_file=False)),
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 0, 0]), to_file=False)),
            -0.2,
            -0.3,
            2/3,
        ),
        (
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 0, 1]), to_file=False)),
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 0, 1]), to_file=False)),
            0.2,
            0.3,
            1.0,
        ),
        (
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([1, 0, 1]), to_file=False)),
            Rule(condition=HyperrectangleCondition([1, 0], bmins=[1, 3], bmaxs=[2, 5]),
                 activation=Activation(np.array([0, 1, 0]), to_file=False)),
            0.2,
            0.3,
            0.0,
        ),
    ],
)
def test_correlation(clean, rule1, rule2, pred1, pred2, expected):
    rule1._prediction = pred1
    rule2._prediction = pred2
    assert rule1.get_correlation(rule2) == expected


@pytest.mark.parametrize(
    "condition_x_y_xtest_ytest_act_pred_crit, theclass",
    itertools.product(
        [
            (
                HyperrectangleCondition([0], bmins=[0], bmaxs=[1]),
                np.array([
                    [0, 1],
                    [2, 3],
                    [4, 5]
                ]),
                np.array([1, 0, 2]),
                np.array([
                    [0, 1],
                    [0, 1],
                    [0, 1]
                ]),
                np.array([1, 4, 5]),
                np.array([1, 0, 0]),
                {ClassificationRule: 1, RegressionRule: 1},
                {ClassificationRule: 1./3., RegressionRule: 25/3.}
            )
        ],
        [Rule, RegressionRule, ClassificationRule],
    )
)
def test_test(clean, condition_x_y_xtest_ytest_act_pred_crit, theclass):
    condition, x, y, xtest, ytest, act, pred, crit = condition_x_y_xtest_ytest_act_pred_crit
    # noinspection PyCallingNonCallable
    rule = theclass(condition=condition)
    if y.dtype == str and theclass != ClassificationRule:
        return

    if theclass == Rule:
        rule.fit(xs=x, y=y, xs_test=xtest, y_test=ytest)
        assert rule.prediction is None
        return
    if y.dtype.type == np.str_ and theclass != ClassificationRule:
        return

    rule.fit(xs=x, y=y)
    rule.eval(xs=xtest, y=ytest)
    np.testing.assert_equal(act, rule.activation)
    np.testing.assert_equal(pred[theclass], rule.prediction)
    np.testing.assert_equal(crit[theclass], rule.criterion)


def test_not_fitted():
    rule = Rule(HyperrectangleCondition([0, 1], bmins=[1, 1], bmaxs=[2, 3]))
    with pytest.raises(ValueError) as e:
        rule.eval(y=np.array([0, 1, 2]))
        assert "Must have fitted the rule before" in str(e)

