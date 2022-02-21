import numpy as np
from collections import Counter
from typing import Union, Tuple, List
import logging

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
def most_common_class(
    activation: Union[np.ndarray, "pd.DataFrame", None], y: np.ndarray
) -> Union[np.ndarray, "pd.DataFrame"]:
    if activation is None:
        return np.bincount(y).argmax()

    if activation.__class__.__name__ != "DataFrame" and not isinstance(activation, np.ndarray):
        raise TypeError("'activation' in conditional_mean must be None or a np.ndarray or a pd.DataFrame")
    if isinstance(activation, np.ndarray):
        y_conditional = np.extract(activation, y)
        count = Counter(y_conditional)
        n = len(y_conditional)
        prop = [v / n for v in count.values()]
        return np.array([(c, v) for c, v in zip(count.keys(), prop)])
    else:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's stacked activations requies pandas. Please run\npip install pandas")
        y_conditional = (
            activation.mul(pd.Series(y).replace(0, "zero"), axis=0).replace(0, np.nan).replace("", np.nan).replace(
                "zero", 0)
        )
        count = y_conditional.apply(lambda x: x.value_counts())
        count.index = count.index.astype(y.dtype)
        return count.apply(lambda x: x / len(x.dropna()))


# noinspection PyUnresolvedReferences
def conditional_mean(activation: Union[np.ndarray, "pd.DataFrame", None], y: np.ndarray) -> Union[float, "pd.Series"]:
    """Mean of all activated values

    If activation is None, we assume the given y have already been extracted from the activation vector,
    which saves time.
    """
    if activation is None:
        return float(np.nanmean(y))

    if activation.__class__.__name__ != "DataFrame" and not isinstance(activation, np.ndarray):
        raise TypeError("'activation' in conditional_mean must be None or a np.ndarray or a pd.DataFrame")
    if isinstance(activation, np.ndarray):
        y_conditional = np.extract(activation, y)
        non_nans_conditional_y = y_conditional[~np.isnan(y_conditional)]
        if len(non_nans_conditional_y) == 0:
            logger.debug(
                "None of the activated points have a non-nan value in target y." " Conditional mean is set to nan."
            )
            return np.nan
        return float(np.mean(non_nans_conditional_y))
    else:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's stacked activations requies pandas. Please run\npip install pandas")
        y_conditional = (
            activation.mul(pd.Series(y).replace(0, "zero"), axis=0).replace(0, np.nan).replace("", np.nan).replace(
                "zero", 0)
        )
        return y_conditional.mean()


# noinspection PyUnresolvedReferences
def conditional_std(activation: Union[np.ndarray, None], y: np.ndarray) -> Union[float, "pd.Series"]:
    """Standard deviation of all activated values

    If activation is None, we assume the given y have already been extracted from the activation vector,
    which saves time.
    """
    if activation is None:
        return float(np.nanstd(y))

    if activation.__class__.__name__ != "DataFrame" and not isinstance(activation, np.ndarray):
        raise TypeError("'activation' in conditional_mean must be None or a np.ndarray or a pd.DataFrame")
    if isinstance(activation, np.ndarray):
        y_conditional = np.extract(activation, y)
        # ddof ensures numpy uses non-biased estimator of std, like pandas' default
        return float(np.nanstd(y_conditional, ddof=1))
    else:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's stacked activations requies pandas. Please run\npip install pandas")
        y_conditional = (
            activation.mul(pd.Series(y).replace(0, "zero"), axis=0).replace(0, np.nan).replace("", np.nan).replace("zero", 0)
        )
        return y_conditional.std()


# noinspection PyUnresolvedReferences
def mse_function(prediction_vector: Union[np.ndarray, "pd.DataFrame"], y: np.ndarray) -> Union[float, "pd.Series"]:
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)^2 $"

    Parameters
    ----------
    prediction_vector: Union[np.ndarray, "pd.DataFrame"]
      A predictor vector or stacked prediction vectors. It means one or many sparse arrays with two
      different values ymean, if the rule is not active and the prediction is the rule is active.
    y: np.ndarray
        The real target values (real numbers)

    Return
    ------
    criterion: Union[float, "pd.Series"]
        the mean squared error
    """
    if prediction_vector.__class__.__name__ != "DataFrame" and not isinstance(prediction_vector, np.ndarray):
        raise TypeError("'activation' in conditional_mean must be None or a np.ndarray or a pd.DataFrame")
    if isinstance(prediction_vector, np.ndarray):
        if len(prediction_vector) != len(y):
            raise ValueError("Predictions and y must have have the same length")
        error_vector = prediction_vector - y
        criterion = np.nanmean(error_vector ** 2)
        # noinspection PyTypeChecker
        return criterion
    else:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's stacked activations requies pandas. Please run\npip install pandas")
        if len(prediction_vector.index) != len(y):
            raise ValueError("Predictions and y must have the same length")
        error_vector = prediction_vector.sub(y, axis=0)
        return (error_vector ** 2).mean()


# noinspection PyUnresolvedReferences
def mae_function(prediction_vector: Union[np.ndarray, "pd.DataFrame"], y: np.ndarray) -> Union[float, "pd.Series"]:
    """
    Compute the mean absolute error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} |\\hat{y}_i - y_i| $"

    Parameters
    ----------
    prediction_vector: Union[np.ndarray, "pd.DataFrame"]
      A predictor vector or stacked prediction vectors. It means one or many sparse arrays with two
      different values ymean, if the rule is not active and the prediction is the rule is active.
    y: np.ndarray
      The real target values (real numbers)

    Return
    ------
    criterion: Union[float, "pd.Series"]
        the mean absolute error
    """
    if prediction_vector.__class__.__name__ != "DataFrame" and not isinstance(prediction_vector, np.ndarray):
        raise TypeError("'activation' in conditional_mean must be None or a np.ndarray or a pd.DataFrame")
    if isinstance(prediction_vector, np.ndarray):
        if len(prediction_vector) != len(y):
            raise ValueError("The two array must have the same length")
        error_vect = np.abs(prediction_vector - y)
        criterion = np.nanmean(error_vect)
        # noinspection PyTypeChecker
        return criterion
    else:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's stacked activations requies pandas. Please run\npip install pandas")
        if len(prediction_vector.index) != len(y):
            raise ValueError("Predictions and y must have the same length")
        error_vect = prediction_vector.sub(y, axis=0).abs()
        return error_vect.mean()


# noinspection PyUnresolvedReferences
def aae_function(prediction_vector: Union[np.ndarray, "pd.DataFrame"], y: np.ndarray) -> Union[float, "pd.Series"]:
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)$"

    Parameters
    ----------
    prediction_vector: Union[np.ndarray, "pd.DataFrame"]
      A predictor vector or stacked prediction vectors. It means one or many sparse arrays with two
      different values ymean, if the rule is not active and the prediction is the rule is active.
    y: np.ndarray
      The real target values (real numbers)

    Return
    ------
    criterion: Union[float, "pd.Series"]
      the mean squared error, or a Series of mean squared errors
    """
    if prediction_vector.__class__.__name__ != "DataFrame" and not isinstance(prediction_vector, np.ndarray):
        raise TypeError("'activation' in conditional_mean must be None or a np.ndarray or a pd.DataFrame")
    if isinstance(prediction_vector, np.ndarray):
        if len(prediction_vector) != len(y):
            raise ValueError("The two array must have the same length")
        error = np.nanmean(np.abs(prediction_vector - y))
        median = np.nanmean(np.abs(y - np.median(y)))
        return error / median
    else:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's stacked activations requies pandas. Please run\npip install pandas")
        if len(prediction_vector.index) != len(y):
            raise ValueError("Predictions and y must have the same length")
        error_vector = prediction_vector.sub(y, axis=0).abs().mean()
        median = np.mean(np.abs(y - np.median(y)))
        return error_vector / median


# noinspection PyUnresolvedReferences
def calc_regression_criterion(
    prediction_vector: Union[np.ndarray, "pd.DataFrame"], y: np.ndarray, **kwargs
) -> Union[float, "pd.Series"]:
    """
    Compute the criterion

    Parameters
    ----------
    prediction_vector: Union[np.ndarray, "pd.DataFrame"]
      The prediction vector of one rule, or the stacked prediction vectors of a ruleset
    y: np.ndarray
      The real target values (real numbers)
    kwargs:
      Can contain 'method', the method mse_function or mse_function criterion (default is 'mse')

    Returns
    -------
    criterion: Union[float, "pd.Series"]
        Criterion value of one rule, of the Series of the criterion values of several rules.
    """

    method = kwargs.get("method", "mse")

    if method.lower() == "mse":
        criterion = mse_function(prediction_vector, y)
    elif method.lower() == "mae":
        criterion = mae_function(prediction_vector, y)
    elif method.lower() == "aae":
        criterion = aae_function(prediction_vector, y)
    else:
        raise ValueError(f"Unknown criterion: {method}. Please choose among mse, mae and aae")

    return criterion


# noinspection PyUnresolvedReferences
def success_rate(prediction: Union[int, str, "pd.Series"], y: np.ndarray) -> Union[float, "pd.Series"]:
    """
    Parameters
    ----------
    prediction: Union[int, str, "pd.Series"]
      The label prediction, of one rule (int or str) or of a set of rules (pd.Series)
    y: np.ndarray
        The real target classes

    Returns
    -------
      The number of times y equals one rule's prediction (float) or many rules predictions (pd.Series).
    """
    if prediction.__class__.__name__ != "Series" and not isinstance(prediction, (int, str)):
        raise TypeError("'activation' in conditional_mean must be None or a np.ndarray or a pd.DataFrame")
    if isinstance(prediction, (int, str)):
        return sum(prediction == y) / len(y)
    else:
        return prediction.apply(lambda x: (x == y).sum()) / len(y)


# noinspection PyUnresolvedReferences
def calc_classification_criterion(
    activation_vector: Union[np.ndarray, "pd.DataFrame"],
    prediction: Union[int, str, "pd.Series"],
    y: np.ndarray,
    **kwargs,
) -> Union[float, "pd.Series"]:
    """
    Computes the criterion

    Parameters
    ----------
    activation_vector: Union[np.ndarray, "pd.DataFrame"]
      The activation vector of one rule, of the stacked activation vectors of a ruleset
    prediction: Union[int, str, "pd.Series"]
      The label prediction, of one rule (int or str) or of a set of rules (pd.Series)
    y: np.ndarray
        The real target values (real numbers)
    kwargs:
        Can contain 'method', the method mse_function or mse_function criterion (default is 'mse'), and 'cond', whether
        to evaluate the criterion only if the rule is activated (default is True)

    Return
    ------
    Union[float, pd.Series]
      Criterion value of one rule (float) or of a set of rules (pd.Series)
    """

    method = kwargs.get("method", "success_rate")
    cond = kwargs.get("cond", True)

    if cond:
        sub_y = np.extract(activation_vector != 0, y)
    else:
        sub_y = y

    if method.lower() == "success_rate":
        criterion = success_rate(prediction, sub_y)
    else:
        raise ValueError(f"Unknown criterion: {method}. Please choose among:\n* success_rate")

    return criterion
