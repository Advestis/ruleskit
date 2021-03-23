import numpy as np
from rule.activation import Activation
import pytest


@pytest.mark.parametrize(
    "vector, val, n",
    [
        (
                np.array([1, 0, 1]),
                5,
                3,
        ),
        (
                np.array([0, 1, 0, 1]),
                5,
                4,
        ),
    ],
)
def test_init(vector, val, n):
    res = Activation(vector)
    np.testing.assert_equal(res.val, val)
    np.testing.assert_equal(res.n, n)


@pytest.mark.parametrize(
    "vector",
    [
        np.array([1, 0, 1]),
        np.array([0, 1, 0, 1]),
    ],
)
def test_get_array(vector):
    res = Activation(vector)
    np.testing.assert_equal(res.get_array(), vector)


@pytest.mark.parametrize(
    "vector, coverage",
    [
        (
                np.array([1, 0, 1]),
                2/3,
        ),
        (
                np.array([0, 1, 0, 1]),
                1/2,
        ),
    ],
)
def test_coverage(vector, coverage):
    res = Activation(vector)
    np.testing.assert_equal(res.calc_coverage_rate(), coverage)


@pytest.mark.parametrize(
    "vector1, vector2, diff",
    [
        (
                np.array([1, 0, 1]),
                np.array([1, 1, 0]),
                np.array([0, 0, 1]),
        ),
    ],
)
def test_diff(vector1, vector2, diff):
    act1 = Activation(vector1)
    act2 = Activation(vector2)
    np.testing.assert_equal((act1 - act2).get_array(), diff)


@pytest.mark.parametrize(
    "vector1, vector2, and_vector",
    [
        (
                np.array([1, 0, 1]),
                np.array([1, 1, 0]),
                np.array([1, 0, 0]),
        ),
    ],
)
def test_diff(vector1, vector2, and_vector):
    act1 = Activation(vector1)
    act2 = Activation(vector2)
    np.testing.assert_equal((act1 & act2).get_array(), and_vector)


@pytest.mark.parametrize(
    "vector1, vector2, add_vector",
    [
        (
                np.array([1, 0, 1]),
                np.array([1, 1, 0]),
                np.array([1, 1, 1]),
        ),
    ],
)
def test_diff(vector1, vector2, add_vector):
    act1 = Activation(vector1)
    act2 = Activation(vector2)
    np.testing.assert_equal((act1 + act2).get_array(), add_vector)
