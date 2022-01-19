import numpy as np
import pandas as pd
from ruleskit import HyperrectangleCondition
import pytest

from ruleskit.condition import DuplicatedFeatures


@pytest.mark.parametrize(
    "fi, fn, bi, ba, impossible, err",
    [
        ([0, 1], None, [0, 1], [2, 3], False, None),
        ([0, 1], ["A", "B"], [0, 1], [2, 3], False, None),
        (None, ["A", "B"], [0, 1], [2, 3], False, None),
        (None, None, [0, 1], [2, 3], False, ValueError),
        ([0, 1], None, [0, 1, 2], [2, 3], False, ValueError),
        ([0, 1], None, [0, 1], [1, 2, 3], False, ValueError),
        ([0, 1], ["A", "B", "C"], [1, 2], [2, 3], False, ValueError),
        ([0.2, 1], ["A", "B"], [1, 2], [2, 3], False, TypeError),
        ([0, 1], [1, "B"], [0, 1], [2, 3], False, TypeError),
        ([0, 1], ["A", "B"], ["0", 1], [2, 3], False, TypeError),
        ([0, 1], ["A", "B"], [0, 1], [2, "3"], False, TypeError),
        ([0, 1], ["A", "B"], [2, 1], [0, 3], True, None),
        ([0, 0], ["A", "B"], [2, 1], [0, 3], True, DuplicatedFeatures),
        ([0, 1], ["A", "A"], [2, 1], [0, 3], True, DuplicatedFeatures),
    ],
)
def test_init(fi, fn, bi, ba, impossible, err):
    if err is not None:
        with pytest.raises(err):
            _ = HyperrectangleCondition(features_indexes=fi, features_names=fn, bmins=bi, bmaxs=ba)
    else:
        cond = HyperrectangleCondition(features_indexes=fi, features_names=fn, bmins=bi, bmaxs=ba)
        assert cond.features_indexes == fi if fi is not None else list(range(len(fn)))
        assert cond.features_names == fn if fn is not None else [f"X_{i}" for i in cond.features_indexes]
        assert cond.bmins == bi
        assert cond.bmaxs == ba
        assert cond.impossible is impossible


@pytest.mark.parametrize(
    "fi, fn, bi, ba, fie, fne, bie, bae, according_to",
    [
        ([0, 1], None, [0, 1], [2, 3], [0, 1], ["X_0", "X_1"], [0, 1], [2, 3], "index"),
        ([1, 0], None, [1, 0], [3, 2], [0, 1], ["X_0", "X_1"], [0, 1], [2, 3], "index"),
        ([0, 1], None, [0, 1], [2, 3], [0, 1], ["X_0", "X_1"], [0, 1], [2, 3], "name"),
        ([1, 0], None, [1, 0], [3, 2], [0, 1], ["X_0", "X_1"], [0, 1], [2, 3], "name"),
        ([1, 0], ["A", "B"], [1, 0], [3, 2], [1, 0], ["A", "B"], [1, 0], [3, 2], "name"),
        ([1, 0], ["A", "B"], [1, 0], [3, 2], [0, 1], ["B", "A"], [0, 1], [2, 3], "index"),
    ],
)
def test_sort(fi, fn, bi, ba, fie, fne, bie, bae, according_to):
    HyperrectangleCondition.SORT_ACCORDING_TO = according_to
    cond = HyperrectangleCondition(features_indexes=fi, features_names=fn, bmins=bi, bmaxs=ba)
    assert cond.features_indexes == fie
    assert cond.features_names == fne
    assert cond.bmins == bie
    assert cond.bmaxs == bae
    HyperrectangleCondition.SORT_ACCORDING_TO = "index"


@pytest.mark.parametrize(
    "x, condition, output, error",
    [
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            HyperrectangleCondition([0], bmins=[1], bmaxs=[2]),
            np.array([1, 0, 1]),
            None,
        ),
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            HyperrectangleCondition([1], bmins=[3], bmaxs=[5]),
            np.array([1, 1, 0]),
            None,
        ),
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            HyperrectangleCondition([0, 1], bmins=[1, 3], bmaxs=[2, 5]),
            np.array([1, 0, 0]),
            None,
        ),
        (
            np.array([[1, 3], [3, 4], [2, np.nan]]),
            HyperrectangleCondition([0, 3], bmins=[1, 3], bmaxs=[2, 5]),
            np.array([1, 0, 0]),
            IndexError,
        ),
        (
            pd.DataFrame(data=[[1, 3], [3, 4], [2, np.nan]], columns=["A", "B"]),
            HyperrectangleCondition([0, 3], features_names=["A", "B"], bmins=[1, 3], bmaxs=[2, 5]),
            np.array([1, 0, 0]),
            None,
        ),
        (
            pd.DataFrame(data=[[1, 3], [3, 4], [2, np.nan]], columns=["A", "B"]),
            HyperrectangleCondition([0, 3], features_names=["A", "C"], bmins=[1, 3], bmaxs=[2, 5]),
            np.array([1, 0, 0]),
            IndexError,
        ),
        (
            [[1, 3], [3, 4], [2, np.nan]],
            HyperrectangleCondition([0, 3], features_names=["A", "C"], bmins=[1, 3], bmaxs=[2, 5]),
            np.array([1, 0, 0]),
            AttributeError,
        ),
    ],
)
def test_evaluate(x, condition, output, error):
    if error is not None:
        with pytest.raises(error):
            _ = condition.evaluate(x)
    else:
        res = condition.evaluate(x)
        # noinspection PyUnresolvedReferences
        np.testing.assert_equal(res, output)


@pytest.mark.parametrize(
    "condition1, condition2, output",
    [
        (
            HyperrectangleCondition([0], bmins=[1], bmaxs=[3], features_names=["b"]),
            HyperrectangleCondition([1], bmins=[1], bmaxs=[2], features_names=["a"]),
            HyperrectangleCondition([0, 1], bmins=[1, 1], bmaxs=[2, 3], features_names=["a", "b"]),
        ),
        (
            HyperrectangleCondition([0], bmins=[1], bmaxs=[3], features_names=["b"]),
            HyperrectangleCondition([0], bmins=[1], bmaxs=[2], features_names=["a"]),
            "Some features with different names had same index in both conditions in __and__",
        ),
        (
            HyperrectangleCondition([0, 1], bmins=[0, 0], bmaxs=[10, 2], features_names=["a", "b"]),
            HyperrectangleCondition([0, 2], bmins=[10, 1], bmaxs=[20, 1], features_names=["a", "c"]),
            HyperrectangleCondition([0, 1, 2], bmins=[10, 0, 1], bmaxs=[10, 2, 1], features_names=["a", "b", "c"]),
        ),
        (
            HyperrectangleCondition([0, 1], bmins=[0, 0], bmaxs=[10, 2], features_names=["a", "b"]),
            HyperrectangleCondition([0, 1], bmins=[20, 1], bmaxs=[30, 1], features_names=["a", "c"]),
            "Some features with different names had same index in both conditions in __and__",
        ),
        (
            HyperrectangleCondition([0, 1], bmins=[0, 0], bmaxs=[10, 2], features_names=["a", "b"]),
            HyperrectangleCondition([1, 2], bmins=[20, 1], bmaxs=[30, 1], features_names=["a", "c"]),
            "Some features present in both conditions in __and__ have different indexes",
        ),
        (
            HyperrectangleCondition([0, 1], bmins=[0, 0], bmaxs=[10, 2], features_names=["a", "b"]),
            HyperrectangleCondition([0, 2], bmins=[20, 1], bmaxs=[30, 1], features_names=["a", "c"]),
            HyperrectangleCondition([0, 1, 2], bmins=[20, 0, 1], bmaxs=[10, 2, 1], features_names=["a", "b", "c"]),
        ),
    ],
)
def test_and(condition1, condition2, output):

    if isinstance(output, str):
        with pytest.raises(IndexError) as e:
            _ = condition1 & condition2
        assert output in str(e.value)
    else:
        res = condition1 & condition2
        np.testing.assert_equal(res, output)


@pytest.mark.parametrize(
    "attr, value, impossible_or_raise",
    [
        ("bmins", [0, 1], False),
        ("bmaxs", [0, 1], False),
        ("features_indexes", [0, 2], False),
        ("features_names", ["C", "D"], False),
        ("bmins", [0, 1, 2], f"Condition has 2 features but you gave 3 bmins"),
        ("bmaxs", [0, 1, 2], f"Condition has 2 features but you gave 3 bmaxs"),
        ("features_indexes", [0, 1, 2], f"Condition has 2 features but you gave 3 indexes"),
        ("features_names", ["C", "D", "E"], f"Condition has 2 features but you gave 3 names"),
        ("bmins", [2, 1], True),
        ("bmaxs", [1, 0], True),
    ],
)
def test_set_attr(attr, value, impossible_or_raise):
    cond = HyperrectangleCondition(features_indexes=[0, 1], features_names=["A", "B"], bmins=[0, 1], bmaxs=[1, 2])
    if isinstance(impossible_or_raise, str):
        with pytest.raises(IndexError) as e:
            setattr(cond, attr, value)
        assert impossible_or_raise in str(e.value)
    else:
        setattr(cond, attr, value)
        assert getattr(cond, attr) == value
        assert cond.impossible is impossible_or_raise
