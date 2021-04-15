import numpy as np
from ruleskit import HyperrectangleCondition
import pytest


@pytest.mark.parametrize(
    "x, condition, output",
    [
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            HyperrectangleCondition([0], bmins=[1], bmaxs=[2]),
            np.array([1, 0, 1]),
        ),
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            HyperrectangleCondition([1], bmins=[3], bmaxs=[5]),
            np.array([1, 1, 0]),
        ),
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            HyperrectangleCondition([0, 1], bmins=[1, 3], bmaxs=[2, 5]),
            np.array([1, 0, 0]),
        ),
    ],
)
def test_evaluate(x, condition, output):
    res = condition.evaluate(x)
    # noinspection PyUnresolvedReferences
    np.testing.assert_equal(res.raw, output)


@pytest.mark.parametrize(
    "condition1, condition2, output",
    [
        (
            HyperrectangleCondition([0], bmins=[1], bmaxs=[2]),
            HyperrectangleCondition([1], bmins=[1], bmaxs=[2]),
            HyperrectangleCondition([0, 1], bmins=[1, 1], bmaxs=[2, 2]),
        ),
    ],
)
def test_add(condition1, condition2, output):
    res = condition1 & condition2
    np.testing.assert_equal(res, output)

    res = condition1 + condition2
    np.testing.assert_equal(res, output)
