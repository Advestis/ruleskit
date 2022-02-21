import pytest
import pandas as pd
import numpy as np
from ruleskit.utils.rfunctions import (
    most_common_class,
    conditional_mean,
    conditional_std,
    mse_function,
    mae_function,
    aae_function,
    calc_regression_criterion,
    calc_classification_criterion,
    success_rate,
)


@pytest.mark.parametrize(
    "activation, y, expected",
    [
        (
            np.array([1, 0, 1]),
            np.array([0, 3, 2]),
            np.array([(0, 0.5), (2, 0.5)]),
        ),
        (
            pd.DataFrame([[1, 1], [0, 1], [1, 0]]),
            np.array([0, 3, 2]),
            pd.DataFrame(index=[0, 2, 3], columns=[0, 1], data=[[0.5, 0.5], [0.5, np.nan], [np.nan, 0.5]]),
        ),
        (
            pd.DataFrame([[1, 1], [0, 1], [1, 0]]),
            np.array(["a", "c", "b"]),
            pd.DataFrame(index=["a", "b", "c"], columns=[0, 1], data=[[0.5, 0.5], [0.5, np.nan], [np.nan, 0.5]]),
        ),
        (
            pd.DataFrame([[1, 1], [0, 1], [1, 0]], columns=["chien", "chat"]),
            np.array(["a", "c", "b"]),
            pd.DataFrame(
                index=["a", "b", "c"], columns=["chien", "chat"], data=[[0.5, 0.5], [0.5, np.nan], [np.nan, 0.5]]
            ),
        ),
    ],
)
def test_most_common_class(activation, y, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_equal(most_common_class(activation, y), expected)
    else:
        pd.testing.assert_frame_equal(most_common_class(activation, y), expected)


@pytest.mark.parametrize(
    "activation, y, expected",
    [
        (
            np.array([1, 0, 1]),
            np.array([-1, 0, 2]),
            0.5,
        ),
        (
            np.array([0, 0, 0]),
            np.array([-1, 0, 2]),
            np.nan,
        ),
        (
            pd.DataFrame([[1, 1, 0], [0, 1, 0], [1, 0, 0]]),
            np.array([-1, 0, 2]),
            pd.Series(index=[0, 1, 2], data=[0.5, -0.5, np.nan]),
        ),
        (
            pd.DataFrame([[1, 1, 0], [0, 1, 0], [1, 0, 0]], columns=["chien", "chat", "cheval"]),
            np.array([-1, 0, 2]),
            pd.Series(index=["chien", "chat", "cheval"], data=[0.5, -0.5, np.nan]),
        ),
    ],
)
def test_conditional_mean(activation, y, expected):
    if isinstance(expected, float):
        if np.isnan(expected):
            assert np.isnan(conditional_mean(activation, y))
        else:
            assert conditional_mean(activation, y) == expected
    else:
        pd.testing.assert_series_equal(conditional_mean(activation, y), expected)


@pytest.mark.parametrize(
    "activation, y, expected",
    [
        (
            np.array([1, 0, 1]),
            np.array([-1, 0, 2]),
            2.121320344,
        ),
        (
            np.array([0, 0, 0]),
            np.array([-1, 0, 2]),
            np.nan,
        ),
        (
            pd.DataFrame([[1, 1, 0], [0, 1, 0], [1, 0, 0]]),
            np.array([-1, 0, 2]),
            pd.Series(index=[0, 1, 2], data=[2.121320344, 0.7071067812, np.nan]),
        ),
        (
            pd.DataFrame([[1, 1, 0], [0, 1, 0], [1, 0, 0]], columns=["chien", "chat", "cheval"]),
            np.array([-1, 0, 2]),
            pd.Series(index=["chien", "chat", "cheval"], data=[2.121320344, 0.7071067812, np.nan]),
        ),
    ],
)
def test_conditional_std(activation, y, expected):
    if isinstance(expected, float):
        if np.isnan(expected):
            assert np.isnan(conditional_std(activation, y))
        else:
            assert round(conditional_std(activation, y), 6) == round(expected, 6)
    else:
        pd.testing.assert_series_equal(conditional_std(activation, y), expected)


@pytest.mark.parametrize(
    "prediction, y, expected",
    [
        (
            np.array([2, np.nan, 2]),
            np.array([-1, 0, 2]),
            4.5,
        ),
        (
            np.array([np.nan, np.nan, np.nan]),
            np.array([-1, 0, 2]),
            np.nan,
        ),
        (
            pd.DataFrame(
                [[2, -1, np.nan],
                 [np.nan, -1, np.nan],
                 [2, np.nan, np.nan]]),
            np.array([-1, 0, 2]),
            pd.Series(index=[0, 1, 2], data=[4.5, 0.5, np.nan]),
        ),
        (
            pd.DataFrame(
                [[2, -1, np.nan],
                 [np.nan, -1, np.nan],
                 [2, np.nan, np.nan]],
                columns=["chien", "chat", "cheval"]
            ),
            np.array([-1, 0, 2]),
            pd.Series(index=["chien", "chat", "cheval"], data=[4.5, 0.5, np.nan]),
        ),
    ],
)
def test_mse_function(prediction, y, expected):
    if isinstance(expected, float):
        if np.isnan(expected):
            assert np.isnan(mse_function(prediction, y))
        else:
            assert round(mse_function(prediction, y), 6) == round(expected, 6)
    else:
        # noinspection PyTypeChecker
        pd.testing.assert_series_equal(mse_function(prediction, y), expected)


@pytest.mark.parametrize(
    "prediction, y, expected",
    [
        (
            np.array([2, np.nan, 2]),
            np.array([-1, 0, 2]),
            1.5,
        ),
        (
            np.array([np.nan, np.nan, np.nan]),
            np.array([-1, 0, 2]),
            np.nan,
        ),
        (
            pd.DataFrame(
                [[2, -1, np.nan],
                 [np.nan, -1, np.nan],
                 [2, np.nan, np.nan]]
            ),
            np.array([-1, 0, 2]),
            pd.Series(index=[0, 1, 2], data=[1.5, 0.5, np.nan]),
        ),
        (
            pd.DataFrame(
                [[2, -1, np.nan],
                 [np.nan, -1, np.nan],
                 [2, np.nan, np.nan]],
                columns=["chien", "chat", "cheval"]
            ),
            np.array([-1, 0, 2]),
            pd.Series(index=["chien", "chat", "cheval"], data=[1.5, 0.5, np.nan]),
        ),
    ],
)
def test_mae_function(prediction, y, expected):
    if isinstance(expected, float):
        if np.isnan(expected):
            assert np.isnan(mae_function(prediction, y))
        else:
            assert round(mae_function(prediction, y), 6) == round(expected, 6)
    else:
        # noinspection PyTypeChecker
        pd.testing.assert_series_equal(mae_function(prediction, y), expected)
