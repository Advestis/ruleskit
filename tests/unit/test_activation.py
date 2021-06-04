import numpy as np
from ruleskit import Activation
import pytest

Activation.FORCE_STAT = True


@pytest.mark.parametrize(
    "vector, cs, ca, i, n",
    [
        (
            np.array([1, 0, 1]),
            "1,1,2,3",
            np.array([1, 1, 2, 3]),
            5,
            3,
        ),
        (
            np.array([0, 1, 0, 1]),
            "0,1,2,3,4",
            np.array([0, 1, 2, 3, 4]),
            5,
            4,
        ),
    ],
)
def test_init(vector, cs, ca, i, n):
    res = Activation(vector)
    np.testing.assert_equal(res.as_int, i)
    np.testing.assert_equal(res.as_compressed_array, ca)
    np.testing.assert_equal(res.as_compressed_str, cs)
    np.testing.assert_equal(res.length, n)


@pytest.mark.parametrize(
    "vector",
    [
        np.array([1, 0, 1]),
        np.array([0, 1, 0, 1]),
    ],
)
def test_raw(vector):
    res = Activation(vector)
    np.testing.assert_equal(res.raw, vector)


@pytest.mark.parametrize(
    "vector, coverage",
    [
        (
            np.array([1, 0, 1]),
            2 / 3,
        ),
        (
            np.array([0, 1, 0, 1]),
            1 / 2,
        ),
    ],
)
def test_coverage(vector, coverage):
    res = Activation(vector)
    np.testing.assert_equal(res.coverage, coverage)


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
    np.testing.assert_equal((act1 - act2).raw, diff)


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
def test_and(vector1, vector2, and_vector):
    act1 = Activation(vector1)
    act2 = Activation(vector2)
    np.testing.assert_equal((act1 & act2).raw, and_vector)


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
def test_add(vector1, vector2, add_vector):
    act1 = Activation(vector1)
    act2 = Activation(vector2)
    np.testing.assert_equal((act1 + act2).raw, add_vector)


def test_sizes():
    act = Activation(np.array([1, 0, 1]))
    assert act.sizeof_raw > 0
    assert act.sizeof_integer > 0
    assert act.sizeof_compressed_array > 0
    assert act.sizeof_compressed_str > 0


def test_times():
    act = Activation(np.array([1, 0, 1]))
    assert act.time_compress > -1
    assert act.time_decompress > -1
    assert act.time_conversions_to_int > -1
    assert act.time_conversions_from_int > -1
    assert act.decompressions > 0
    assert act.compressions > 0
    assert act.conversions_to_int > 0
    assert act.conversions_from_int > 0
