import numpy as np
from ruleskit import Activation
import pytest
from bitarray import bitarray


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
def test_init_with_raw(clean, vector, cs, ca, b, i, n):
    res = Activation(vector, to_file=False)

    assert res._nones is not None
    assert res._entropy is not None
    assert res._rel_entropy is not None

    assert res.sizeof_path == -1
    assert res.sizeof_file == -1
    assert res.sizeof_raw > 0
    assert res.sizeof_bitarray > 0
    assert res.sizeof_integer == -1
    assert res.sizeof_compressed_str > 0
    assert res.sizeof_compressed_array == -1

    assert res.time_write == -1
    assert res.time_read == -1
    assert res.time_raw_to_compressed > 0
    assert res.time_compressed_to_raw == -1
    assert res.time_raw_to_bitarray > 0
    assert res.time_raw_to_integer == -1
    assert res.time_bitarray_to_raw == -1
    assert res.time_integer_to_raw == -1
    assert res.time_bitarray_to_compressed == -1
    assert res.time_integer_to_compressed == -1
    assert res.time_compressed_to_bitarray == -1
    assert res.time_compressed_to_integer == -1
    assert res.time_bitarray_to_integer == -1
    assert res.time_integer_to_bitarray == -1

    assert res.n_written == 0
    assert res.n_read == 0
    assert res.n_raw_to_compressed == 1
    assert res.n_compressed_to_raw == 0
    assert res.n_raw_to_bitarray == 1
    assert res.n_raw_to_integer == 0
    assert res.n_bitarray_to_raw == 0
    assert res.n_integer_to_raw == 0
    assert res.n_bitarray_to_compressed == 0
    assert res.n_integer_to_compressed == 0
    assert res.n_compressed_to_bitarray == 0
    assert res.n_compressed_to_integer == 0
    assert res.n_bitarray_to_integer == 0
    assert res.n_integer_to_bitarray == 0

    assert res.coverage is not None

    np.testing.assert_equal(res.raw, vector)
    np.testing.assert_equal(res.as_bitarray, b)
    np.testing.assert_equal(res.as_integer, i)
    np.testing.assert_equal(res.as_compressed_str, cs)
    np.testing.assert_equal(res.as_compressed_array, ca)
    np.testing.assert_equal(res.length, n)

    assert res.sizeof_integer > 0
    assert res.sizeof_compressed_array > 0
    assert res.sizeof_bitarray > 0

    assert res.time_compressed_to_bitarray > 0
    assert res.time_compressed_to_raw > 0
    assert res.time_compressed_to_integer > 0

    assert res.n_compressed_to_integer == 1
    assert res.n_compressed_to_bitarray == 1
    assert res.n_compressed_to_raw == 2


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
def test_init_with_integer_not_optimize_will_compare(clean, vector, cs, ca, b, i, n):
    Activation.WILL_COMPARE = True
    with pytest.raises(ValueError):
        _ = Activation(i, optimize=False, to_file=False)
    res = Activation(i, optimize=False, to_file=False, length=n)
    Activation.WILL_COMPARE = False

    assert res._nones is None
    assert res._entropy is None
    assert res._rel_entropy is None

    assert res.sizeof_path == -1
    assert res.sizeof_file == -1
    assert res.sizeof_raw == -1
    assert res.sizeof_bitarray == -1
    assert res.sizeof_integer > 0
    assert res.sizeof_compressed_str == -1
    assert res.sizeof_compressed_array == -1

    assert res.time_write == -1
    assert res.time_read == -1
    assert res.time_raw_to_compressed == -1
    assert res.time_compressed_to_raw == -1
    assert res.time_raw_to_bitarray == -1
    assert res.time_raw_to_integer == -1
    assert res.time_bitarray_to_raw == -1
    assert res.time_integer_to_raw == -1
    assert res.time_bitarray_to_compressed == -1
    assert res.time_integer_to_compressed == -1
    assert res.time_compressed_to_bitarray == -1
    assert res.time_compressed_to_integer == -1
    assert res.time_bitarray_to_integer == -1
    assert res.time_integer_to_bitarray == -1

    assert res.n_written == 0
    assert res.n_read == 0
    assert res.n_raw_to_compressed == 0
    assert res.n_compressed_to_raw == 0
    assert res.n_raw_to_bitarray == 0
    assert res.n_raw_to_integer == 0
    assert res.n_bitarray_to_raw == 0
    assert res.n_integer_to_raw == 0
    assert res.n_bitarray_to_compressed == 0
    assert res.n_integer_to_compressed == 0
    assert res.n_compressed_to_bitarray == 0
    assert res.n_compressed_to_integer == 0
    assert res.n_bitarray_to_integer == 0
    assert res.n_integer_to_bitarray == 0

    assert res.coverage is not None

    np.testing.assert_equal(res.raw, vector)
    np.testing.assert_equal(res.as_bitarray, b)
    np.testing.assert_equal(res.as_integer, i)
    np.testing.assert_equal(res.as_compressed_str, cs)
    np.testing.assert_equal(res.as_compressed_array, ca)
    np.testing.assert_equal(res.length, n)

    assert res.sizeof_integer > 0
    assert res.sizeof_compressed_array > 0
    assert res.sizeof_bitarray > 0
    assert res.sizeof_raw > 0

    assert res.time_integer_to_bitarray > 0
    assert res.time_integer_to_raw > 0
    assert res.time_integer_to_compressed > 0

    assert res.n_integer_to_raw == 4
    assert res.n_integer_to_bitarray == 1
    assert res.n_integer_to_compressed == 1


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
def test_init_with_integer_optimize_will_compare(clean, vector, cs, ca, b, i, n):
    Activation.WILL_COMPARE = True
    res = Activation(i, optimize=True, to_file=False, length=n)
    Activation.WILL_COMPARE = False

    assert res._nones is not None
    assert res._entropy is not None
    assert res._rel_entropy is not None

    assert res.sizeof_path == -1
    assert res.sizeof_file == -1
    assert res.sizeof_raw > 0
    assert res.sizeof_bitarray == -1
    assert res.sizeof_integer > 0
    assert res.sizeof_compressed_str > 0
    assert res.sizeof_compressed_array == -1

    assert res.time_write == -1
    assert res.time_read == -1
    assert res.time_raw_to_compressed > 0
    assert res.time_raw_to_integer == -1
    assert res.time_raw_to_bitarray == -1
    assert res.time_compressed_to_raw == -1
    assert res.time_bitarray_to_raw == -1
    assert res.time_integer_to_raw > 0
    assert res.time_compressed_to_bitarray == -1
    assert res.time_bitarray_to_compressed == -1
    assert res.time_integer_to_compressed > 0
    assert res.time_compressed_to_integer == -1
    assert res.time_bitarray_to_integer == -1
    assert res.time_integer_to_bitarray == -1

    assert res.n_written == 0
    assert res.n_read == 0
    assert res.n_raw_to_compressed == 1
    assert res.n_raw_to_integer == 0
    assert res.n_raw_to_bitarray == 0
    assert res.n_compressed_to_raw == 0
    assert res.n_bitarray_to_raw == 0
    assert res.n_integer_to_raw == 1
    assert res.n_compressed_to_bitarray == 0
    assert res.n_bitarray_to_compressed == 0
    assert res.n_integer_to_compressed == 1
    assert res.n_compressed_to_integer == 0
    assert res.n_bitarray_to_integer == 0
    assert res.n_integer_to_bitarray == 0

    assert res.coverage is not None

    np.testing.assert_equal(res.raw, vector)
    np.testing.assert_equal(res.as_bitarray, b)
    np.testing.assert_equal(res.as_integer, i)
    np.testing.assert_equal(res.as_compressed_str, cs)
    np.testing.assert_equal(res.as_compressed_array, ca)
    np.testing.assert_equal(res.length, n)

    assert res.sizeof_integer > 0
    assert res.sizeof_compressed_array > 0
    assert res.sizeof_bitarray > 0
    assert res.sizeof_raw > 0

    assert res.time_integer_to_bitarray > 0
    assert res.time_integer_to_raw > 0
    assert res.time_integer_to_compressed > 0

    assert res.n_integer_to_raw == 4
    assert res.n_integer_to_bitarray == 1
    assert res.n_integer_to_compressed == 2


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
def test_init_with_integer_not_optimize_will_not_compare(clean, vector, cs, ca, b, i, n):
    res = Activation(i, optimize=False, to_file=False, length=n)

    assert res._nones is None
    assert res._entropy is None
    assert res._rel_entropy is None

    assert res.sizeof_path == -1
    assert res.sizeof_file == -1
    assert res.sizeof_raw == 0
    assert res.sizeof_bitarray > 0
    assert res.sizeof_integer > 0
    assert res.sizeof_compressed_str == -1
    assert res.sizeof_compressed_array == -1

    assert res.time_write == -1
    assert res.time_read == -1
    assert res.time_raw_to_compressed == -1
    assert res.time_raw_to_integer == -1
    assert res.time_raw_to_bitarray == -1
    assert res.time_compressed_to_raw == -1
    assert res.time_bitarray_to_raw == -1
    assert res.time_integer_to_raw == -1
    assert res.time_compressed_to_bitarray == -1
    assert res.time_bitarray_to_compressed == -1
    assert res.time_integer_to_compressed == -1
    assert res.time_compressed_to_integer == -1
    assert res.time_bitarray_to_integer == -1
    assert res.time_integer_to_bitarray > 0

    assert res.n_written == 0
    assert res.n_read == 0
    assert res.n_raw_to_compressed == 0
    assert res.n_raw_to_integer == 0
    assert res.n_raw_to_bitarray == 0
    assert res.n_compressed_to_raw == 0
    assert res.n_bitarray_to_raw == 0
    assert res.n_integer_to_raw == 0
    assert res.n_compressed_to_bitarray == 0
    assert res.n_bitarray_to_compressed == 0
    assert res.n_integer_to_compressed == 0
    assert res.n_compressed_to_integer == 0
    assert res.n_bitarray_to_integer == 0
    assert res.n_integer_to_bitarray == 1

    assert res.coverage is not None

    np.testing.assert_equal(res.raw, vector)
    np.testing.assert_equal(res.as_bitarray, b)
    np.testing.assert_equal(res.as_integer, i)
    np.testing.assert_equal(res.as_compressed_str, cs)
    np.testing.assert_equal(res.as_compressed_array, ca)
    np.testing.assert_equal(res.length, n)

    assert res.sizeof_integer > 0
    assert res.sizeof_compressed_array > 0
    assert res.sizeof_bitarray > 0
    assert res.sizeof_raw > 0

    assert res.time_integer_to_bitarray > 0
    assert res.time_integer_to_raw > 0
    assert res.time_integer_to_compressed > 0

    assert res.n_bitarray_to_raw == 4
    assert res.n_bitarray_to_integer == 1
    assert res.n_bitarray_to_compressed == 2


def test_file(clean):
    exp = np.array([1, 0, 1])
    res = Activation(exp, to_file=True)
    assert res.data.is_file()
    np.testing.assert_equal(exp, res.raw)
    np.testing.assert_equal(exp, res._read())
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
def test_raw(clean, vector):
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
def test_coverage(clean, vector, coverage):
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
def test_diff(clean, vector1, vector2, diff):
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
def test_and(clean, vector1, vector2, and_vector):
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
def test_or(clean, vector1, vector2, add_vector):
    act1 = Activation(vector1)
    act2 = Activation(vector2)
    comb = (act1 | act2).raw
    np.testing.assert_equal(comb, add_vector)


@pytest.mark.parametrize(
    "vector1, vector2, add_vector",
    [
        (
            np.array([1, 0, 1]),
            np.array([1, 1, 0]),
            np.array([0, 1, 1]),
        ),
    ],
)
def test_xor(clean, vector1, vector2, add_vector):
    act1 = Activation(vector1)
    act2 = Activation(vector2)
    comb = (act1 ^ act2).raw
    np.testing.assert_equal(comb, add_vector)


@pytest.mark.parametrize(
    "vector1, vector2, add_vector",
    [
        (
            np.array([1, 0, 1]),
            np.array([1, 1, 0]),
            np.array([0, 0, 1]),
        ),
    ],
)
def test_sub(clean, vector1, vector2, add_vector):
    act1 = Activation(vector1)
    act2 = Activation(vector2)
    comb = (act1 - act2).raw
    np.testing.assert_equal(comb, add_vector)
