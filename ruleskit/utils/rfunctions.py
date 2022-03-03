import numpy as np
from collections import Counter
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
def class_probabilities(
    activation: Union[np.ndarray, "pd.DataFrame", None], y: Union[np.ndarray, "pd.Series"]
) -> Union[np.ndarray, "pd.DataFrame"]:
    """Computes the class probability of each rule(s)

    Parameters
    ----------
    activation: Union[np.ndarray, "pd.DataFrame", None]
      Either the activation vector of one rule (np.ndarray) or a DataFrame of activation vectors of many rules (one rule
      is one column)
    y: Union[np.ndarray, "pd.Series"]
      The target classes

    Returns
    -------
    Union[np.ndarray, "pd.DataFrame"]
        If given one activation vector, returns a np.ndarray of the form [(class1, prob 1), ..., (class n, prob n)].
        If given a df of activation vectors, returns a df with the classes as index, the rules as columns and the
        probabilities as values.
    """
    if activation is None:
        return np.bincount(y).argmax()

    if activation.__class__.__name__ != "DataFrame" and not isinstance(activation, np.ndarray):
        raise TypeError("'activation' in most_common_class must be None or a np.ndarray or a pd.DataFrame")
    if isinstance(activation, np.ndarray):
        y_conditional = np.extract(activation, y)
        count = Counter(y_conditional)
        n = len(y_conditional)
        prop = [v / n for v in count.values()]
        proba = np.array([(c, v) for c, v in zip(count.keys(), prop)])
        proba = proba[proba[:, 0].argsort()]
        return proba
    else:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's stacked activations requies pandas. Please run\npip install pandas")
        if y.__class__.__name__ == "DataFrame":
            if len(y.columns) == 1:
                y = y.squeeze()
            else:
                raise ValueError("y must be a 1-D DataFrame or ndarray or pd.Series")
        elif isinstance(y, np.ndarray):
            if len(y.shape) == 1:
                y = pd.Series(y)
            elif y.shape[1] == 1:
                y = pd.Series(y.squeeze())
            else:
                raise ValueError("y must be a 1-D DataFrame or ndarray or pd.Series")
        y_conditional = (
            activation.mul(y.replace(0, "zero"), axis=0)
            .replace(0, np.nan)
            .replace("", np.nan)
            .replace("zero", 0)
        )
        count = y_conditional.apply(lambda x: x.value_counts())
        count.index = count.index.astype(y.dtype)
        return count.apply(lambda x: x / x.dropna().sum())


# noinspection PyUnresolvedReferences
def conditional_mean(
    activation: Union[np.ndarray, "pd.DataFrame", None], y: Union[np.ndarray, "pd.Series"]
) -> Union[float, "pd.Series"]:
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
        return activation.apply(lambda x: np.nanmean(np.extract(x, y)))


# noinspection PyUnresolvedReferences
def conditional_std(
    activation: Union[np.ndarray, "pd.DataFrame", None], y: Union[np.ndarray, "pd.Series"]
) -> Union[float, "pd.Series"]:
    """Standard deviation of all activated values

    If activation is None, we assume the given y have already been extracted from the activation vector,
    which saves time.
    """
    if activation is None:
        return float(np.nanstd(y))

    if activation.__class__.__name__ != "DataFrame" and not isinstance(activation, np.ndarray):
        raise TypeError("'activation' in conditional_std must be None or a np.ndarray or a pd.DataFrame")
    if isinstance(activation, np.ndarray):
        y_conditional = np.extract(activation, y)
        if len(y_conditional) == 0:
            return np.nan
        if len(y_conditional) == 1:
            return 0
        # ddof ensures numpy uses non-biased estimator of std, like pandas' default
        return float(np.nanstd(y_conditional, ddof=1))
    else:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's stacked activations requies pandas. Please run\npip install pandas")
        if y.__class__.__name__ == "DataFrame":
            if len(y.columns) == 1:
                y = y.squeeze()
            else:
                raise ValueError("y must be a 1-D DataFrame or ndarray or pd.Series")
        elif isinstance(y, np.ndarray):
            if len(y.shape) == 1:
                y = pd.Series(y)
            elif y.shape[1] == 1:
                y = pd.Series(y.squeeze())
            else:
                raise ValueError("y must be a 1-D DataFrame or ndarray or pd.Series")
        y_conditional = (
            activation.mul(y.replace(0, "zero"), axis=0)
            .replace(0, np.nan)
            .replace("", np.nan)
            .replace("zero", 0)
        )
        return y_conditional.std()


# noinspection PyUnresolvedReferences
def mse_function(
    prediction_vector: Union[np.ndarray, "pd.DataFrame"], y: Union[np.ndarray, "pd.Series"]
) -> Union[float, "pd.Series"]:
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)^2 $"

    Parameters
    ----------
    prediction_vector: Union[np.ndarray, "pd.DataFrame"]
      A predictor vector or stacked prediction vectors. It means one or many sparse arrays with two
      different values ymean, if the rule is not active and the prediction is the rule is active.
    y: Union[np.ndarray, "pd.Series"]
        The real target values (real numbers)

    Returns
    -------
    criterion: Union[float, "pd.Series"]
        the mean squared error
    """
    if prediction_vector.__class__.__name__ != "DataFrame" and not isinstance(prediction_vector, np.ndarray):
        raise TypeError("'prediction_vector' in mse_function must be a np.ndarray or a pd.DataFrame")
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
def mae_function(
    prediction_vector: Union[np.ndarray, "pd.DataFrame"], y: Union[np.ndarray, "pd.Series"]
) -> Union[float, "pd.Series"]:
    """
    Compute the mean absolute error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} |\\hat{y}_i - y_i| $"

    Parameters
    ----------
    prediction_vector: Union[np.ndarray, "pd.DataFrame"]
      A predictor vector or stacked prediction vectors. It means one or many sparse arrays with two
      different values ymean, if the rule is not active and the prediction is the rule is active.
    y: Union[np.ndarray, "pd.Series"]
      The real target values (real numbers)

    Returns
    -------
    criterion: Union[float, "pd.Series"]
        the mean absolute error
    """
    if prediction_vector.__class__.__name__ != "DataFrame" and not isinstance(prediction_vector, np.ndarray):
        raise TypeError("'prediction_vector' in mae_function must be a np.ndarray or a pd.DataFrame")
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
def aae_function(
    prediction_vector: Union[np.ndarray, "pd.DataFrame"], y: Union[np.ndarray, "pd.Series"]
) -> Union[float, "pd.Series"]:
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)$"

    Parameters
    ----------
    prediction_vector: Union[np.ndarray, "pd.DataFrame"]
      A predictor vector or stacked prediction vectors. It means one or many sparse arrays with two
      different values ymean, if the rule is not active and the prediction is the rule is active.
    y: Union[np.ndarray, "pd.Series"]
      The real target values (real numbers)

    Returns
    -------
    criterion: Union[float, "pd.Series"]
      the mean squared error, or a Series of mean squared errors
    """
    if prediction_vector.__class__.__name__ != "DataFrame" and not isinstance(prediction_vector, np.ndarray):
        raise TypeError("'prediction_vector' in aae_function must be a np.ndarray or a pd.DataFrame")
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
    prediction: Union[np.ndarray, "pd.DataFrame"], y: Union[np.ndarray, "pd.Series"], **kwargs
) -> Union[float, "pd.Series"]:
    """
    Compute the criterion

    Parameters
    ----------
    prediction: Union[np.ndarray, "pd.DataFrame"]
      The prediction vector of one rule, or the stacked prediction vectors of a ruleset
    y: Union[np.ndarray, "pd.Series"]
      The real target values (real numbers)
    kwargs:
      Can contain 'method', the method mse_function or mse_function criterion (default is 'mse')

    Returns
    -------
    criterion: Union[float, "pd.Series"]
        Criterion value of one rule or ruleset, or the Series of the criterion values of several rules
    """

    method = kwargs.get("method", "mse")

    if method.lower() == "mse":
        criterion = mse_function(prediction, y)
    elif method.lower() == "mae":
        criterion = mae_function(prediction, y)
    elif method.lower() == "aae":
        criterion = aae_function(prediction, y)
    else:
        raise ValueError(f"Unknown criterion: {method}. Please choose among mse, mae and aae")

    return criterion


# noinspection PyUnresolvedReferences
def success_rate(
    prediction: Union[float, int, str, np.integer, np.float, "pd.Series"], y: Union[np.ndarray, "pd.DataFrame"]
) -> Union[float, "pd.Series"]:
    """
    Returns the number fraction of y that equal the prediction.

    Parameters
    ----------
    prediction: Union[int, np.integer, np.float, str, "pd.Series"]
        The label prediction, of one rule (int, np.integer, np.float or str) or of a set of rules (pd.Series)
    y: Union[np.ndarray, "pd.DataFrame"]
        The real target points activated by the rule (np.ndarray, without nans) or the rules
        (pd.DataFrame, can contain nans even alongside strings)

    Returns
    -------
      The fraction of y that equal one rule or ruleset's prediction (float) or many rules predictions (pd.Series).
    """
    if prediction.__class__.__name__ != "Series" and not isinstance(
        prediction, (float, int, np.integer, np.float, str)
    ):
        raise TypeError("'prediction' in success_rate must be an integer, a string or a pd.Series of one of those.")
    if isinstance(prediction, (float, int, np.integer, np.float, str)):
        if len(y) == 0:
            return np.nan
        return sum(prediction == y) / len(y)
    elif isinstance(y, np.ndarray) and prediction.__class__.__name__ == "Series":
        if len(y) == 0:
            return prediction.__class__(dtype=int)
        return sum(prediction == y) / len(prediction.dropna())
    else:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's stacked activations requies pandas. Please run\npip install pandas")

        if not y.__class__.__name__ == "DataFrame":
            raise TypeError(
                f"If passing several predictions as a Series, then activated Y must be a DataFrame, not {type(y)}"
            )
        if len(y.index) == 0:
            return y.__class__(dtype=int)
        activated_points = (~y.isnull()).sum()
        correctly_predicted = y == prediction
        return correctly_predicted.sum() / activated_points


# noinspection PyUnresolvedReferences
def calc_classification_criterion(
    activation_vector: Union[np.ndarray, "pd.DataFrame"],
    prediction: Union[float, int, np.integer, np.float, str, "pd.Series"],
    y: Union[np.ndarray, "pd.Series"],
    **kwargs,
) -> Union[float, "pd.Series"]:
    """
    Computes the criterion

    Parameters
    ----------
    activation_vector: Union[np.ndarray, "pd.DataFrame"]
        The activation vector of one rule, or of one ruleset, or the stacked activation vectors of a ruleset
    prediction: Union[float, int, np.integer, np.float, str, "pd.Series"]
        The label prediction, of one rule (int, np.integer, np.float or str) or of a set of rules (pd.Series), or
        the label prediction of one ruleset at each observations (pd.Series)
    y: Union[np.ndarray, "pd.Series"]
        The real target values
    kwargs:
        Can contain 'method', indicating how to evaluate the criterion. For now, one can use:\n * success_rate (default)

    Returns
    -------
    Union[float, pd.Series]
        Criterion value of one rule or ruleset (float) or of a set of rules (pd.Series)
    """

    method = kwargs.get("method", "success_rate")

    if activation_vector.__class__.__name__ != "DataFrame" and not isinstance(activation_vector, np.ndarray):
        raise TypeError("'activation_vector' in calc_classification_criterion must be a np.ndarray or a pd.DataFrame")

    if isinstance(activation_vector, np.ndarray):
        y_conditional = np.extract(activation_vector != 0, y)
        if prediction.__class__.__name__ == "Series":
            prediction = prediction[activation_vector != 0]
        if method.lower() == "success_rate":
            return success_rate(prediction, y_conditional)
        else:
            raise ValueError(f"Unknown criterion: {method}. Please choose among:\n* success_rate")
    else:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's stacked activations requies pandas. Please run\npip install pandas")

        if not prediction.__class__.__name__ == "Series":
            raise TypeError(
                "If passing several activation vector as a DataFrame, then prediction must be a Series,"
                f" not {type(prediction)}"
            )
        if y.__class__.__name__ == "DataFrame":
            if len(y.columns) == 1:
                y = y.squeeze()
            else:
                raise ValueError("y must be a 1-D DataFrame or ndarray or pd.Series")
        elif isinstance(y, np.ndarray):
            if len(y.shape) == 1:
                y = pd.Series(y)
            elif y.shape[1] == 1:
                y = pd.Series(y.squeeze())
            else:
                raise ValueError("y must be a 1-D DataFrame or ndarray or pd.Series")
        y_conditional = (
            activation_vector.mul(y.replace(0, "zero"), axis=0)
            .replace(0, np.nan)
            .replace("", np.nan)
            .replace("zero", 0)
        )
        return success_rate(prediction, y_conditional)


# noinspection PyUnresolvedReferences
def init_weights(
    prediction_vectors: "pd.DataFrame", weights: "pd.DataFrame"
) -> "pd.DataFrame":
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("RuleSet's stacked activations requies pandas. Please run\npip install pandas")
    if weights.empty:
        raise ValueError("Weights not None but empty : can not evaluate prediction")
    weights = weights.replace(0, np.nan).fillna(value=np.nan).dropna(axis=1, how="all")
    absent_rules = prediction_vectors.loc[:, ~prediction_vectors.columns.isin(weights.columns)].columns
    present_rules = prediction_vectors.loc[:, prediction_vectors.columns.isin(weights.columns)].columns
    if len(absent_rules) > 0:
        s = (
            "Some rules given in the prediction vector did not have a weight and will be ignored in the"
            " computation of the ruleset prediction. The concerned rules are :"
        )
        s = "\n".join([s] + list(absent_rules.astype(str)))
        logger.warning(s)
    prediction_vectors = prediction_vectors[present_rules]
    weights = (~prediction_vectors.isna() * 1).replace(0, np.nan) * weights
    if prediction_vectors.empty:
        raise ValueError(
            "No rules had non-zero/non-NaN weights, or all predictions left after applying weights where NaN"
        )
    return prediction_vectors, weights


# noinspection PyUnresolvedReferences
def calc_ruleset_prediction_weighted_regressor(
        prediction_vectors: "pd.DataFrame", weights: "pd.DataFrame"
) -> float:
    prediction_vectors, weights = init_weights(prediction_vectors, weights)
    idx = prediction_vectors.index
    return ((prediction_vectors * weights).sum(axis=1) / weights.sum(axis=1)).reindex(idx)


# noinspection PyUnresolvedReferences
def calc_ruleset_prediction_equally_weighted_regressor(prediction_vectors: "pd.DataFrame") -> float:
    return prediction_vectors.mean(axis=1)


# noinspection PyUnresolvedReferences
def calc_ruleset_prediction_weighted_classificator(
    prediction_vectors: "pd.DataFrame", weights: Optional["pd.DataFrame"]
) -> Union[str, int]:
    prediction_vectors, weights = init_weights(prediction_vectors, weights)
    # noinspection PyUnresolvedReferences
    mask = (weights.T == weights.max(axis=1)).T
    prediction_vectors = prediction_vectors[mask]
    return calc_ruleset_prediction_equally_weighted_classificator(prediction_vectors)


# noinspection PyUnresolvedReferences
def calc_ruleset_prediction_equally_weighted_classificator(prediction_vectors: "pd.DataFrame") -> Union[str, int]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("RuleSet's stacked activations requies pandas. Please run\npip install pandas")
    idx = prediction_vectors.index
    if pd.api.types.is_string_dtype(prediction_vectors.dtypes.iloc[0]):
        most_freq_pred = prediction_vectors.fillna(value=np.nan).replace("nan", np.nan).mode(axis=1)
    else:
        most_freq_pred = prediction_vectors.mode(axis=1)
    most_freq_pred = most_freq_pred.loc[most_freq_pred.count(axis=1) == 1].dropna(axis=1).squeeze()
    most_freq_pred.name = None
    return most_freq_pred.reindex(idx)
