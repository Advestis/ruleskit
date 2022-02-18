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
            pd.DataFrame([
                [1, 1],
                [0, 1],
                [1, 0]
            ]),
            np.array([0, 3, 2]),
            pd.DataFrame(index=[0, 2, 3], columns=[0, 1], data=[[0.5, 0.5], [0.5, np.nan], [np.nan, 0.5]]),
        ),
        (
            pd.DataFrame([
                [1, 1],
                [0, 1],
                [1, 0]
            ]),
            np.array(["a", "c", "b"]),
            pd.DataFrame(index=["a", "b", "c"], columns=[0, 1], data=[[0.5, 0.5], [0.5, np.nan], [np.nan, 0.5]]),
        ),
    ],
)
def test_most_common_class(activation, y, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_equal(most_common_class(activation, y), expected)
    else:
        pd.testing.assert_frame_equal(most_common_class(activation, y), expected)


