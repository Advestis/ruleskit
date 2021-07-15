import numpy as np
from ruleskit import Activation
import pytest
from bitarray import bitarray

Activation.FORCE_STAT = True


@pytest.mark.parametrize(
    "vector, cs, ca, b, i, n",
    [
        (
            np.array([1, 0, 1]),
            "1,1,2,3",
            np.array([1, 1, 2, 3]),
            bitarray([1, 0, 1]),
            5,
            3,
        ),
        (
            np.array([0, 1, 0, 1]),
            "0,1,2,3,4",
            np.array([0, 1, 2, 3, 4]),
            bitarray("0101"),
            5,
            4,
        ),
    ],
)
def test_init(vector, cs, ca, b, i, n):
    res = Activation(vector)
    np.testing.assert_equal(res.as_bitarray, b)
    np.testing.assert_equal(res.as_compressed_str, cs)
    np.testing.assert_equal(res.as_compressed_array, ca)
    np.testing.assert_equal(res.length, n)
    Activation.WILL_COMPARE = True
    res = Activation(vector)
    np.testing.assert_equal(res.as_integer, i)
    np.testing.assert_equal(res.as_compressed_str, cs)
    np.testing.assert_equal(res.as_compressed_array, ca)
    np.testing.assert_equal(res.length, n)
    Activation.WILL_COMPARE = False


def test_file():
    exp = np.array([1, 0, 1])
    Activation.USE_FILE = True
    res = Activation(exp, name_for_file="dummy")
    assert res.data.is_file()
    np.testing.assert_equal(exp, res.raw)
    np.testing.assert_equal(exp, res._read())
    Activation.USE_FILE = False
    assert res.ones is not None
    assert res.nones is not None
    assert res.sizeof_raw > 0
    assert res.coverage > 0
    assert res.entropy > 0
    assert res.rel_entropy > 0
    assert res.sizeof_file > 0
    assert res.sizeof_path > 0
    _ = res.as_bitarray
    _ = res.as_integer
    _ = res.as_compressed
    _ = res.as_compressed_array
    _ = res.as_compressed_str
    Activation.clean_files()
    assert not res.data.is_file()


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
    np.testing.assert_equal(Activation.logical_and(act1, act2).raw, and_vector)


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
    added = (act1 + act2).raw
    np.testing.assert_equal(added, add_vector)


def test_sizes():
    act = Activation(np.array([1, 0, 1]))
    assert act.sizeof_raw > 0
    assert act.sizeof_bitarray > 0
    assert act.sizeof_compressed_array > 0
    assert act.sizeof_compressed_str > 0


def test_times():
    act = Activation(np.array([1, 0, 1]))
    assert act.time_raw_to_compressed > -1
    assert act.time_compressed_to_raw > -1
    assert act.time_raw_to_bitarray > -1
    assert act.time_bitarray_to_raw > -1
    assert act.n_compressed_to_raw > 0
    assert act.n_raw_to_compressed > 0
    assert act.n_raw_to_bitarray > 0
    assert act.n_bitarray_to_raw > 0
    Activation.WILL_COMPARE = True
    act = Activation(np.array([1, 0, 1]))
    assert act.time_raw_to_compressed > -1
    assert act.time_compressed_to_raw > -1
    assert act.time_raw_to_integer > -1
    assert act.time_integer_to_raw > -1
    assert act.n_compressed_to_raw > 0
    assert act.n_raw_to_compressed > 0
    assert act.n_raw_to_integer > 0
    assert act.n_integer_to_raw > 0
    Activation.WILL_COMPARE = False
