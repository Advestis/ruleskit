import pytest
import pandas as pd
import numpy as np
from ruleskit.utils.rfunctions import (
    class_probabilities,
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
            np.array([1, 0, 1, 1, 1]),
            np.array([0, 3, 2, 100, 2]),
            np.array([(0, 0.25), (2, 0.5), (100, 0.25)]),
        ),
        (
            pd.DataFrame([
                [1, 1],
                [0, 1],
                [1, 0],
                [1, 1],
                [1, 1]
            ]),
            np.array([0, 3, 2, 100, 2]),
            pd.DataFrame(
                index=[0, 2, 3, 100],
                columns=[0, 1],
                data=[
                    [0.25, 0.25],
                    [0.5, 0.25],
                    [np.nan, 0.25],
                    [0.25, 0.25]
                ]),
        ),
        (
            pd.DataFrame([
                [1, 1],
                [0, 1],
                [1, 0],
                [1, 1],
                [1, 1]
            ]),
            np.array(["a", "c", "b", "d", "b"]),
            pd.DataFrame(
                index=["a", "b", "c", "d"],
                columns=[0, 1],
                data=[
                    [0.25, 0.25],
                    [0.5, 0.25],
                    [np.nan, 0.25],
                    [0.25, 0.25]
                ]),
        ),
        (
            pd.DataFrame([
                [1, 1],
                [0, 1],
                [1, 0],
                [1, 1],
                [1, 1]
            ], columns=["chien", "chat"]),
            np.array(["a", "c", "b", "d", "b"]),
            pd.DataFrame(
                index=["a", "b", "c", "d"],
                columns=["chien", "chat"],
                data=[
                    [0.25, 0.25],
                    [0.5, 0.25],
                    [np.nan, 0.25],
                    [0.25, 0.25]
                ]),
        ),
    ],
)
def test_class_probabilities(activation, y, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_equal(class_probabilities(activation, y), expected)
    else:
        pd.testing.assert_frame_equal(class_probabilities(activation, y), expected)


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
            # noinspection PyTypeChecker
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
            # noinspection PyTypeChecker
            assert round(mae_function(prediction, y), 6) == round(expected, 6)
    else:
        # noinspection PyTypeChecker
        pd.testing.assert_series_equal(mae_function(prediction, y), expected)


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
def test_aae_function(prediction, y, expected):
    if isinstance(expected, float):
        if np.isnan(expected):
            assert np.isnan(aae_function(prediction, y))
        else:
            # noinspection PyTypeChecker
            assert round(aae_function(prediction, y), 6) == round(expected, 6)
    else:
        # noinspection PyTypeChecker
        pd.testing.assert_series_equal(aae_function(prediction, y), expected)


@pytest.mark.parametrize(
    "prediction, y, expected, kwargs",
    [
        (
            np.array([2, np.nan, 2]),
            np.array([-1, 0, 2]),
            4.5,
            {}
        ),
        (
            np.array([np.nan, np.nan, np.nan]),
            np.array([-1, 0, 2]),
            np.nan,
            {}
        ),
        (
            pd.DataFrame(
                [[2, -1, np.nan],
                 [np.nan, -1, np.nan],
                 [2, np.nan, np.nan]]
            ),
            np.array([-1, 0, 2]),
            pd.Series(index=[0, 1, 2], data=[4.5, 0.5, np.nan]),
            {}
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
            {}
        ),
        (
            np.array([2, np.nan, 2]),
            np.array([-1, 0, 2]),
            1.5,
            {"method": "mae"}
        ),
        (
            np.array([np.nan, np.nan, np.nan]),
            np.array([-1, 0, 2]),
            np.nan,
            {"method": "mae"}
        ),
        (
            pd.DataFrame(
                [[2, -1, np.nan],
                 [np.nan, -1, np.nan],
                 [2, np.nan, np.nan]]
            ),
            np.array([-1, 0, 2]),
            pd.Series(index=[0, 1, 2], data=[1.5, 0.5, np.nan]),
            {"method": "mae"}
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
            {"method": "mae"}
        ),
        (
            np.array([2, np.nan, 2]),
            np.array([-1, 0, 2]),
            1.5,
            {"method": "aae"}
        ),
        (
            np.array([np.nan, np.nan, np.nan]),
            np.array([-1, 0, 2]),
            np.nan,
            {"method": "aae"}
        ),
        (
            pd.DataFrame(
                [[2, -1, np.nan],
                 [np.nan, -1, np.nan],
                 [2, np.nan, np.nan]]
            ),
            np.array([-1, 0, 2]),
            pd.Series(index=[0, 1, 2], data=[1.5, 0.5, np.nan]),
            {"method": "aae"}
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
            {"method": "aae"}
        ),
    ],
)
def test_calc_regression_criterion(prediction, y, expected, kwargs):
    if isinstance(expected, float):
        if np.isnan(expected):
            assert np.isnan(calc_regression_criterion(prediction, y, **kwargs))
        else:
            # noinspection PyTypeChecker
            assert round(calc_regression_criterion(prediction, y, **kwargs), 6) == round(expected, 6)
    else:
        # noinspection PyTypeChecker
        pd.testing.assert_series_equal(calc_regression_criterion(prediction, y, **kwargs), expected)


@pytest.mark.parametrize(
    "prediction, y, expected",
    [
        (
            2,
            np.array([0, 1, 2, 0]),
            0.25
        ),
        (
            "chien",
            np.array(["chien", "chien", "cheval", "chouette"]),
            0.5
        ),
        (
            pd.Series([2, 0]),
            pd.DataFrame(
                [[0, 0],
                 [np.nan, np.nan],
                 [2, 2],
                 [np.nan, 0]]
            ),
            pd.Series([0.5, 2/3])
        ),
        (
            pd.Series(["chien", "chat"]),
            pd.DataFrame(
                [["chat", "chat"],
                 [np.nan, np.nan],
                 ["chien", "chien"],
                 [np.nan, "chat"]]
            ),
            pd.Series([0.5, 2/3])
        ),
        (
            pd.Series(index=["a", "b"], data=["chien", "chat"]),
            pd.DataFrame(
                [["chat", "chat"],
                 [np.nan, np.nan],
                 ["chien", "chien"],
                 [np.nan, "chat"]],
                columns=["a", "b"]
            ),
            pd.Series([0.5, 2/3], index=["a", "b"])
        ),
    ]
)
def test_success_rate(prediction, y, expected):
    if isinstance(expected, float):
        if np.isnan(expected):
            assert np.isnan(success_rate(prediction, y))
        else:
            # noinspection PyTypeChecker
            assert round(success_rate(prediction, y), 6) == round(expected, 6)
    else:
        # noinspection PyTypeChecker
        pd.testing.assert_series_equal(success_rate(prediction, y), expected)


@pytest.mark.parametrize(
    "activations, prediction, y, expected",
    [
        (
            np.array([1, 0, 1, 0]),
            2,
            np.array([0, 1, 2, 0]),
            0.5
        ),
        (
            np.array([1, 1, 0, 0]),
            "chien",
            np.array(["chien", "chien", "cheval", "chouette"]),
            1.0
        ),
        (
            pd.DataFrame(
                [[1, 1],
                 [0, 0],
                 [1, 1],
                 [0, 1]]
            ),
            pd.Series([2, 0]),
            np.array([0, 1, 2, 0]),
            pd.Series([0.5, 2/3])
        ),
        (
            pd.DataFrame(
                [[1, 0],
                 [0, 1],
                 [1, 1],
                 [0, 1]]
            ),
            pd.Series(["chien", "cheval"]),
            np.array(["chien", "chien", "cheval", "chouette"]),
            pd.Series([0.5, 1/3])
        ),
        (
            pd.DataFrame(
                [[1, 0],
                 [0, 1],
                 [1, 1],
                 [0, 1]],
                columns=["a", "b"]
            ),
            pd.Series(["chien", "cheval"], index=["a", "b"]),
            np.array(["chien", "chien", "cheval", "chouette"]),
            pd.Series([0.5, 1/3], index=["a", "b"])
        ),
    ]
)
def test_calc_classification_criterion(activations, prediction, y, expected):
    if isinstance(expected, float):
        if np.isnan(expected):
            assert np.isnan(calc_classification_criterion(activations, prediction, y))
        else:
            # noinspection PyTypeChecker
            assert round(calc_classification_criterion(activations, prediction, y), 6) == round(expected, 6)
    else:
        # noinspection PyTypeChecker
        pd.testing.assert_series_equal(calc_classification_criterion(activations, prediction, y), expected)
