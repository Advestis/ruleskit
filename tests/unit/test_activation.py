import itertools

import numpy as np
from ruleskit import Activation
import pytest
import pandas as pd
from pathlib import Path
from bitarray import bitarray

data_path = Path("tests/unit/data")


def make_column(will_compare, optimize, store_raw, to_file, ignore) -> str:
    s = []
    if ignore is None:
        return f"will_compare_{will_compare}_optimize_{optimize}_store_raw_{store_raw}_to_file_{to_file}"
    if "will_compare" not in ignore:
        s.append(f"will_compare_{will_compare}")
    if "optimize" not in ignore:
        s.append(f"optimize_{optimize}")
    if "store_raw" not in ignore:
        s.append(f"store_raw_{store_raw}")
    if "to_file" not in ignore:
        s.append(f"to_file_{to_file}")
    return "_".join(s)


def compare_column(res, df, column):

    print(column)
    for index in df[column].index:
        print(index)
        attr = getattr(res, index)
        expected_attr = df.loc[index, column]
        if index == "data_format":
            assert attr == expected_attr
        elif expected_attr == "=0":
            assert attr == 0
        elif expected_attr == ">0":
            assert attr > 0
        elif expected_attr == "-1":
            assert attr == -1
        elif expected_attr == "None":
            assert attr is None
        elif expected_attr == "not None":
            assert attr is not None
        elif expected_attr == "osef":
            continue
        else:
            assert attr == int(expected_attr)


# noinspection PyTypeChecker
@pytest.mark.parametrize(
    "vector_cs_ca_b_i_n_p_c_o_no, will_compare, optimize, store_raw, to_file, withwhat",
    itertools.product(
        [
            (
                    np.array(
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                    "1,1,275,276",
                    np.array([1, 1, 275, 276]),
                    bitarray(
                        "1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                        "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                        "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000001"),
                    60708402882054033466233184588234965832575213720379360039119137804340758912662765569,
                    276,
                    data_path / "a1.txt",
                    2 / 276,
                    [0, 275],
                    2,
            ),
            (
                np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16",
                np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
                bitarray("0101010101010101"),
                21845,
                16,
                data_path / "a2.txt",
                0.5,
                [1, 3, 5, 7, 9, 11, 13, 15],
                8,
            ),
        ],
        [True, False],
        [True, False],
        [True, False],
        [True, False],
        ["raw", "integer", "bitarray", "compressed_str", "compressed_array", "file"],
    )
)
def test_init(clean, vector_cs_ca_b_i_n_p_c_o_no, will_compare, optimize, store_raw, to_file, withwhat):
    vector, cs, ca, b, i, n, p, c, o, no = vector_cs_ca_b_i_n_p_c_o_no
    print(withwhat, p)
    Activation.WILL_COMPARE = will_compare
    Activation.STORE_RAW = store_raw
    value = vector
    to_file = False
    if withwhat == "integer":
        value = i
        with pytest.raises(ValueError):
            _ = Activation(value, optimize=optimize, to_file=to_file)
    if withwhat == "bitarray":
        value = b
    if withwhat == "compressed_str":
        value = cs
    if withwhat == "compressed_array":
        value = ca
    if withwhat == "file":
        value = p

    expected_df = pd.read_csv(data_path / f"{p.stem}_activation_init_{withwhat}.csv", index_col=0)
    ignore = expected_df.index.name

    column = make_column(will_compare, optimize, store_raw, to_file, ignore)

    res = Activation(value, optimize=optimize, to_file=to_file, length=n)
    Activation.WILL_COMPARE = False
    Activation.STORE_RAW = False

    compare_column(res, expected_df, column)

    np.testing.assert_equal(res.raw, vector)
    np.testing.assert_equal(res.as_bitarray, b)
    np.testing.assert_equal(res.as_integer, i)
    np.testing.assert_equal(res.as_compressed_str, cs)
    np.testing.assert_equal(res.as_compressed_array, ca)
    np.testing.assert_equal(res.length, n)
    assert res.entropy == len(ca)
    assert res.rel_entropy == len(ca) / n
    assert res.coverage == c
    assert res.ones == o
    assert res.nones == no
    if to_file:
        assert res.data.is_file()

    expected_df = pd.read_csv(data_path / f"{p.stem}_activation_init_{withwhat}_after_calls.csv", index_col=0)
    ignore = expected_df.index.name

    column = make_column(will_compare, optimize, store_raw, to_file, ignore)

    compare_column(res, expected_df, column)


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


@pytest.mark.parametrize(
    "vector1, vector2, expected",
    [
        (
            np.array([1, 0, 1]),
            np.array([1, 1, 0]),
            1/3,
        ),
        (
            np.array([1, 0, 1]),
            np.array([1, 0, 0]),
            2/3,
        ),
        (
            np.array([1, 0, 1]),
            np.array([1, 0, 1]),
            1.0,
        ),
        (
            np.array([1, 0, 1]),
            np.array([0, 1, 0]),
            0.0,
        ),
    ],
)
def test_correlation(clean, vector1, vector2, expected):
    assert Activation(vector1, to_file=False).get_correlation(Activation(vector2, to_file=False)) == expected
