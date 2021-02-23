from rule.hyperrectanglecondition import HyperrectangleCondition
from rule.rule import Rule
from rule.rule import compress, decompress
import numpy as np
import pytest


@pytest.mark.parametrize(
    "x, y, condition, activation, compressed_activation, cov, pred",
    [
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            np.array([1, 3, 2]),
            HyperrectangleCondition([0], bmins=[1], bmaxs=[2]),
            np.array([1, 0, 1]),
            (5, "1,1,2,3"),
            2/3,
            1.5,
        ),
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            np.array([1, 3, 2]),
            HyperrectangleCondition([1], bmins=[3], bmaxs=[5]),
            np.array([1, 1, 0]),
            (6, "1,2,3"),
            2/3,
            2
        ),
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            np.array([1, 3, 2]),
            HyperrectangleCondition([0, 1], bmins=[1, 3], bmaxs=[2, 5]),
            np.array([1, 0, 0]),
            (4, "1,1,3"),
            1/3,
            1
        ),
    ],
)
def test_activation(x, y, condition, activation, compressed_activation, cov, pred):
    rule = Rule(condition=condition)
    rule.fit(xs=x, y=y)
    np.testing.assert_equal(rule.activation, activation)
    if isinstance(rule._activation, int):
        assert rule._activation == compressed_activation[0]
    else:
        assert rule._activation == compressed_activation[1]
    np.testing.assert_equal(rule.coverage, cov)
    np.testing.assert_equal(rule.prediction, pred)


@pytest.mark.parametrize(
    "act, exp",
    [
        (np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]), "0,1,2,4,7,10,15"),
        (np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]), "1,2,4,7,10,15"),
    ],
)
def test_compression(act, exp):
    print("activation:", act)
    comp = compress(act)
    print("Compressed:", comp)
    print("expected:", exp)
    np.testing.assert_equal(comp, exp)


@pytest.mark.parametrize(
    "comp, exp",
    [
        ("0,1,2,4,7,10,15", np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1])),
        ("1,2,4,7,10,15", np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1])),
    ],
)
def test_decompression(comp, exp):
    print("Compressed:", comp)
    act = decompress(comp)
    print("activation:", act)
    print("expected:", exp)
    np.testing.assert_equal(act, exp)
