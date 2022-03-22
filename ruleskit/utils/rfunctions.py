import numpy as np
from collections import Counter
from typing import Union, Tuple, List, Optional
import logging
import pandas as pd


logger = logging.getLogger(__name__)


def class_probabilities(
    activation: Union[np.ndarray, pd.DataFrame, None], y: Union[np.ndarray, pd.Series]
) -> Union[np.ndarray, pd.DataFrame]:
    """Computes the class probability of each rule(s)

    Parameters
    ----------
    activation: Union[np.ndarray, pd.DataFrame, None]
      Either the activation vector of one rule (np.ndarray) or a DataFrame of activation vectors of many rules (one rule
      is one column)
    y: Union[np.ndarray, pd.Series]
      The target classes

    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        If given one activation vector, returns a np.ndarray of the form [(class1, prob 1), ..., (class n, prob n)].
        If given a df of activation vectors, returns a df with the classes as index, the rules as columns and the
        probabilities as values.
    """
    if activation is None:
        return np.bincount(y).argmax()

    if not isinstance(activation, pd.DataFrame) and not isinstance(activation, np.ndarray):
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
        if isinstance(y, pd.DataFrame):
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
            activation.mul(y.replace(0, "zero"), axis=0).replace(0, np.nan).replace("", np.nan).replace("zero", 0)
        )
        count = y_conditional.apply(lambda x: x.value_counts())
        count.index = count.index.astype(y.dtype)
        return count.apply(lambda x: x / x.dropna().sum())


def conditional_mean(
    activation: Union[np.ndarray, pd.DataFrame, None], y: Union[np.ndarray, pd.Series]
) -> Union[float, pd.Series]:
    """Mean of all activated values

    If activation is None, we assume the given y have already been extracted from the activation vector,
    which saves time.
    """
    if activation is None:
        return float(np.nanmean(y))

    if not isinstance(activation, pd.DataFrame) and not isinstance(activation, np.ndarray):
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
        return activation.apply(lambda x: np.nanmean(np.extract(x, y)))


def conditional_std(
    activation: Union[np.ndarray, pd.DataFrame, None], y: Union[np.ndarray, pd.Series]
) -> Union[float, pd.Series]:
    """Standard deviation of all activated values

    If activation is None, we assume the given y have already been extracted from the activation vector,
    which saves time.
    """
    if activation is None:
        return float(np.nanstd(y))

    if not isinstance(activation, pd.DataFrame) and not isinstance(activation, np.ndarray):
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
        if isinstance(y, pd.DataFrame):
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
            activation.mul(y.replace(0, "zero"), axis=0).replace(0, np.nan).replace("", np.nan).replace("zero", 0)
        )
        return y_conditional.std()


def mse_function(
    prediction_vector: Union[pd.Series, pd.DataFrame], y: Union[np.ndarray, pd.Series]
) -> Union[float, pd.Series]:
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)^2 $"

    Parameters
    ----------
    prediction_vector: Union[pd.Series, pd.DataFrame]
      A predictor vector or stacked prediction vectors. It means one or many sparse arrays with two
      different values ymean, if the rule is not active and the prediction is the rule is active.
    y: Union[np.ndarray, pd.Series]
        The real target values (real numbers)

    Returns
    -------
    criterion: Union[float, pd.Series]
        the mean squared error
    """
    if not isinstance(prediction_vector, pd.DataFrame) and not isinstance(prediction_vector, pd.Series):
        raise TypeError("'prediction_vector' in mse_function must be a pd.Series or a pd.DataFrame")
    if isinstance(prediction_vector, np.ndarray):
        if len(prediction_vector) != len(y):
            raise ValueError("Predictions and y must have have the same length")
        error_vector = prediction_vector - y
        criterion = np.nanmean(error_vector ** 2)
        # noinspection PyTypeChecker
        return criterion
    else:
        if len(prediction_vector.index) != len(y):
            raise ValueError("Predictions and y must have the same length")
        error_vector = prediction_vector.sub(y, axis=0)
        return (error_vector ** 2).mean()


def mse_norm(
    prediction_vector: Union[pd.Series, pd.DataFrame], y: Union[np.ndarray, pd.Series]
) -> Union[float, pd.Series]:
    return mse_function(prediction_vector, y) / np.mean((np.mean(y) - y) ** 2)


def mae_function(
    prediction_vector: Union[pd.Series, pd.DataFrame], y: Union[np.ndarray, pd.Series]
) -> Union[float, pd.Series]:
    """
    Compute the mean absolute error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} |\\hat{y}_i - y_i| $"

    Parameters
    ----------
    prediction_vector: Union[pd.Series, pd.DataFrame]
      A predictor vector or stacked prediction vectors. It means one or many sparse arrays with two
      different values ymean, if the rule is not active and the prediction is the rule is active.
    y: Union[np.ndarray, pd.Series]
      The real target values (real numbers)

    Returns
    -------
    criterion: Union[float, pd.Series]
        the mean absolute error
    """
    if not isinstance(prediction_vector, pd.DataFrame) and not isinstance(prediction_vector, pd.Series):
        raise TypeError("'prediction_vector' in mae_function must be a pd.Series or a pd.DataFrame")
    if isinstance(prediction_vector, np.ndarray):
        if len(prediction_vector) != len(y):
            raise ValueError("The two array must have the same length")
        error_vect = np.abs(prediction_vector - y)
        criterion = np.nanmean(error_vect)
        # noinspection PyTypeChecker
        return criterion
    else:
        if len(prediction_vector.index) != len(y):
            raise ValueError("Predictions and y must have the same length")
        error_vect = prediction_vector.sub(y, axis=0).abs()
        return error_vect.mean()


def mae_norm(
    prediction_vector: Union[pd.Series, pd.DataFrame], y: Union[np.ndarray, pd.Series]
) -> Union[float, pd.Series]:
    return mae_function(prediction_vector, y) / np.mean(np.mean(y) - y)


def aae_function(
    prediction_vector: Union[pd.Series, pd.DataFrame], y: Union[np.ndarray, pd.Series]
) -> Union[float, pd.Series]:
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)$"

    Parameters
    ----------
    prediction_vector: Union[pd.Series, pd.DataFrame]
      A predictor vector or stacked prediction vectors. It means one or many sparse arrays with two
      different values ymean, if the rule is not active and the prediction is the rule is active.
    y: Union[np.ndarray, pd.Series]
      The real target values (real numbers)

    Returns
    -------
    criterion: Union[float, pd.Series]
      the mean squared error, or a Series of mean squared errors
    """
    if not isinstance(prediction_vector, pd.DataFrame) and not isinstance(prediction_vector, pd.Series):
        raise TypeError("'prediction_vector' in aae_function must be a pd.Series or a pd.DataFrame")
    if isinstance(prediction_vector, np.ndarray):
        if len(prediction_vector) != len(y):
            raise ValueError("The two array must have the same length")
        error = np.nanmean(np.abs(prediction_vector - y))
        median = np.nanmean(np.abs(y - np.median(y)))
        return error / median
    else:
        if len(prediction_vector.index) != len(y):
            raise ValueError("Predictions and y must have the same length")
        error_vector = prediction_vector.sub(y, axis=0).abs().mean()
        median = np.mean(np.abs(y - np.median(y)))
        return error_vector / median


def calc_regression_criterion(
    prediction: Union[pd.Series, pd.DataFrame], y: Union[np.ndarray, pd.Series], **kwargs
) -> Union[float, pd.Series]:
    """
    Compute the criterion

    Parameters
    ----------
    prediction: Union[pd.Series, pd.DataFrame]
      The prediction vector of one rule, or the stacked prediction vectors of a ruleset
    y: Union[np.ndarray, pd.Series]
      The real target values (real numbers)
    kwargs:
      Can contain 'criterion_method', the criterion_method mse_function or mse_function criterion (default is 'mse')

    Returns
    -------
    criterion: Union[float, pd.Series]
        Criterion value of one rule or ruleset, or the Series of the criterion values of several rules
    """

    criterion_method = kwargs.get("criterion_method", "mse_norm")

    if isinstance(y, pd.Series):
        y = y.values

    if criterion_method.lower() == "mse":
        criterion = mse_function(prediction, y)
    elif criterion_method.lower() == "mse_norm":
        criterion = mse_norm(prediction, y)
    elif criterion_method.lower() == "mae":
        criterion = mae_function(prediction, y)
    elif criterion_method.lower() == "mae_norm":
        criterion = mae_norm(prediction, y)
    elif criterion_method.lower() == "aae":
        criterion = aae_function(prediction, y)
    else:
        raise ValueError(f"Unknown criterion: {criterion_method}. Please choose among mse, mae and aae")

    return criterion


def success_rate(
    prediction: Union[float, int, str, np.integer, np.float, pd.Series], y: Union[np.ndarray, pd.DataFrame]
) -> Union[float, pd.Series]:
    """
    Returns the number fraction of y that equal the prediction.

    Parameters
    ----------
    prediction: Union[int, np.integer, np.float, str, pd.Series]
        The label prediction, of one rule (int, np.integer, np.float or str) or of a set of rules (pd.Series),
        of a prediction vector (pd.Series)
    y: Union[np.ndarray, pd.DataFrame]
        The real target points activated by the rule (np.ndarray, without nans) or the rules
        (pd.DataFrame, can contain nans even alongside strings)

    Returns
    -------
    Union[float, pd.Series]
        The fraction of y that equal one rule or ruleset's prediction (float) or many rules predictions (pd.Series).
    """
    if not isinstance(prediction, (pd.Series, float, int, np.integer, np.float, str)):
        raise TypeError(
            "'prediction' in success_rate must be an integer, a string or a pd.Series/np.ndarray of one of those."
        )
    if isinstance(prediction, (float, int, np.integer, np.float, str)):
        if len(y) == 0:
            return np.nan
        return sum(prediction == y) / len(y)
    elif isinstance(y, np.ndarray) and isinstance(prediction, pd.Series):
        if len(y) == 0:
            return pd.Series(dtype=int)
        return sum(prediction == y) / len(prediction.dropna())
    else:
        if not isinstance(y, pd.DataFrame):
            raise TypeError(
                f"If passing several predictions as a Series, then activated Y must be a DataFrame, not {type(y)}"
            )
        if len(y.index) == 0:
            return pd.Series(dtype=int)
        activated_points = (~y.isnull()).sum()
        correctly_predicted = y == prediction
        return correctly_predicted.sum() / activated_points


def calc_classification_criterion(
    activation_vector: Union[np.ndarray, pd.DataFrame],
    prediction: Union[float, int, np.integer, np.float, str, pd.Series],
    y: Union[np.ndarray, pd.Series],
    **kwargs,
) -> Union[float, pd.Series]:
    """
    Computes the criterion

    Parameters
    ----------
    activation_vector: Union[np.ndarray, pd.DataFrame]
        The activation vector of one rule, or of one ruleset, or the stacked activation vectors of a ruleset
    prediction: Union[float, int, np.integer, np.float, str, pd.Series]
        The label prediction, of one rule (int, np.integer, np.float or str) or of a set of rules (pd.Series), or
        the label prediction of one ruleset at each observations (pd.Series)
    y: Union[np.ndarray, pd.Series, pd.DataFrame]
        The real target values. If a dataframe is given, it must have one column only.
    kwargs:
        Can contain 'method', indicating how to evaluate the criterion. For now, one can use:\n * success_rate (default)

    Returns
    -------
    Union[float, pd.Series]
        Criterion value of one rule or ruleset (float) or of a set of rules (pd.Series)
    """

    method = kwargs.get("criterion_method", "success_rate")

    if not isinstance(activation_vector, pd.DataFrame) and not isinstance(activation_vector, np.ndarray):
        raise TypeError("'activation_vector' in calc_classification_criterion must be a np.ndarray or a pd.DataFrame")

    if isinstance(prediction, np.ndarray):
        raise TypeError("Prediction should not be a numpy array")

    if isinstance(activation_vector, np.ndarray):
        y_conditional = np.extract(activation_vector != 0, y)
        if isinstance(prediction, pd.Series):
            if len(prediction) != len(activation_vector):
                raise IndexError("Activation vector and prediction vector should have the same length")
            prediction = prediction[activation_vector != 0]
        if method.lower() == "success_rate":
            return success_rate(prediction, y_conditional)
        else:
            raise ValueError(f"Unknown criterion: {method}. Please choose among:\n* success_rate")
    else:
        if not isinstance(prediction, pd.Series):
            raise TypeError(
                "If passing several activation vector as a DataFrame, then prediction must be a pd.Series,"
                f" not {type(prediction)}"
            )
        if isinstance(y, pd.DataFrame):
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


def init_weights_stacked(prediction_vectors: pd.DataFrame, weights: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
def calc_ruleset_prediction_weighted_regressor_unstacked(
    rules: List["Rule"],
    weights: pd.Series,
    xs: Optional[Union[pd.DataFrame, np.ndarray]] = None,
) -> pd.Series:
    if len(rules) == 0:
        return pd.Series(dtype=int)
    cum_pred = None
    cum_w = None
    for rule in rules:
        rulename = str(rule.condition)
        if rulename not in weights.index:
            logger.warning(f"Rule '{rulename}' has no weight and is ignored in ruleset's prediction")
            continue
        if weights[rulename] == 0:
            logger.warning(f"Rule '{rulename}' has a null weight and is ignored in ruleset's prediction")
            continue
        if np.isnan(weights[rulename]):
            logger.warning(f"Rule '{rulename}' has a nan weight and is ignored in ruleset's prediction")
            continue
        if xs is None:
            activation = rule.activation
        else:
            activation = rule.evaluate_activation(xs).raw
        if activation is None:
            continue
        if len(activation) == 0:
            continue
        if cum_pred is None:
            cum_pred = rule.calc_prediction_vector(activation=activation) * weights[rulename]
            cum_w = activation * weights[rulename]
        else:
            cum_pred = (
                (cum_pred.replace(0, "zero").replace(np.nan, 0) + activation * rule.prediction * weights[rulename])
                .replace(0, np.nan)
                .replace("zero", 0)
            )
            cum_w += activation * weights[rulename]
    if cum_pred is None:
        raise ValueError(
            "No rules had non-zero/non-NaN weights, or all predictions left after applying weights where NaN"
        )
    return cum_pred / cum_w


def calc_ruleset_prediction_weighted_regressor_stacked(
    prediction_vectors: pd.DataFrame, weights: pd.DataFrame
) -> pd.Series:
    prediction_vectors, weights = init_weights_stacked(prediction_vectors, weights)
    idx = prediction_vectors.index
    return ((prediction_vectors * weights).sum(axis=1) / weights.sum(axis=1)).reindex(idx)


# noinspection PyUnresolvedReferences
def calc_ruleset_prediction_equally_weighted_regressor_unstacked(
    rules: List["Rule"], xs: Optional[Union[pd.DataFrame, np.ndarray]] = None
) -> pd.Series:
    if len(rules) == 0:
        return pd.Series(dtype=int)
    cum_pred = None
    cum_w = None
    for rule in rules:
        if xs is None:
            activation = rule.activation
        else:
            activation = rule.evaluate_activation(xs).raw
        if activation is None:
            continue
        if len(activation) == 0:
            continue
        if cum_pred is None:
            cum_pred = rule.calc_prediction_vector(activation=activation)
            cum_w = activation
        else:
            cum_pred = (
                (cum_pred.replace(0, "zero").replace(np.nan, 0) + activation * rule.prediction)
                .replace(0, np.nan)
                .replace("zero", 0)
            )
            cum_w += activation
    if cum_pred is None:
        raise ValueError(
            "No rules had non-zero/non-NaN weights, or all predictions left after applying weights where NaN"
        )
    return cum_pred / cum_w


def calc_ruleset_prediction_equally_weighted_regressor_stacked(prediction_vectors: pd.DataFrame) -> pd.Series:
    return prediction_vectors.mean(axis=1)


# noinspection PyUnresolvedReferences
def calc_ruleset_prediction_weighted_classificator_unstacked(
    rules: List["Rule"], weights: pd.Series, xs: Optional[Union[pd.DataFrame, np.ndarray]] = None
) -> pd.Series:
    if len(rules) == 0:
        return pd.Series(dtype=int)

    def get_max(s: pd.Series):
        ps = s.index
        m = s.max()
        if isinstance(ps[0], str):
            s = (s == m).astype(int).replace(0, np.nan)
            s[~s.isna()] = ps[~s.isna()]
            return s
        else:
            return (s == m).astype(int).replace(0, np.nan) * ps

    preds = None
    for rule in rules:
        rulename = str(rule.condition)
        if rulename not in weights.index:
            logger.warning(f"Rule '{rulename}' has no weight and is ignored in ruleset's prediction")
            continue
        if weights[rulename] == 0:
            logger.warning(f"Rule '{rulename}' has a null weight and is ignored in ruleset's prediction")
            continue
        if np.isnan(weights[rulename]):
            logger.warning(f"Rule '{rulename}' has a nan weight and is ignored in ruleset's prediction")
            continue
        if xs is None:
            activation = rule.activation
            nones = rule.nones
        else:
            activation = rule.evaluate_activation(xs)
            nones = activation.nones
            activation = activation.raw
        if activation is None:
            continue
        if nones == 0:
            continue
        if preds is None:
            preds = pd.DataFrame(index=[rule.prediction], data=[activation * weights[rulename]])
        else:
            if rule.prediction not in preds.index:
                preds.loc[rule.prediction] = activation * weights[rulename]
            else:
                preds.loc[rule.prediction] += activation * weights[rulename]
    if preds is None:
        raise ValueError(
            "No rules had non-zero/non-NaN weights, or all predictions left after applying weights where NaN"
        )
    best_preds = preds.max().replace(0, np.nan)
    best_preds = best_preds[(preds == best_preds).sum() == 1]
    preds.loc[:, ~preds.columns.isin(best_preds.index)] = np.nan
    preds2 = preds.dropna(axis=1).apply(get_max)
    if isinstance(preds.index[0], str):
        preds2 = preds2.bfill().ffill().iloc[0].reindex(preds.columns)
        preds2.name = None
        return preds2
    else:
        return preds2.sum().reindex(preds.columns)


# noinspection PyUnresolvedReferences
def calc_ruleset_prediction_equally_weighted_classificator_unstacked(
    rules: List["Rule"],
    xs: Optional[Union[pd.DataFrame, np.ndarray]] = None,
) -> pd.Series:
    return calc_ruleset_prediction_weighted_classificator_unstacked(
        rules, pd.Series({str(r.condition): 1 for r in rules}), xs=xs
    )


def calc_ruleset_prediction_weighted_classificator_stacked(
    prediction_vectors: pd.DataFrame, weights: pd.DataFrame
) -> pd.Series:
    prediction_vectors, weights = init_weights_stacked(prediction_vectors, weights)
    # noinspection PyUnresolvedReferences
    mask = (weights.T == weights.max(axis=1)).T
    prediction_vectors = prediction_vectors[mask]
    return calc_ruleset_prediction_equally_weighted_classificator_stacked(prediction_vectors)


def calc_ruleset_prediction_equally_weighted_classificator_stacked(prediction_vectors: pd.DataFrame) -> pd.Series:
    idx = prediction_vectors.index
    if pd.api.types.is_string_dtype(prediction_vectors.dtypes.iloc[0]):
        most_freq_pred = prediction_vectors.fillna(value=np.nan).replace("nan", np.nan).mode(axis=1)
    else:
        most_freq_pred = prediction_vectors.mode(axis=1)
    most_freq_pred = most_freq_pred.loc[most_freq_pred.count(axis=1) == 1].dropna(axis=1).squeeze()
    # Happens if only on point is activated, in which case 'squeeze' will only extract this point's prediction as a
    # float, int or str
    if not isinstance(most_freq_pred, pd.Series):
        most_freq_pred = pd.Series([most_freq_pred])
    most_freq_pred.name = None
    return most_freq_pred.reindex(idx)


def calc_zscore_external(
    prediction: Union[pd.Series, float], nones: Union[pd.Series, int], y: np.ndarray, horizon: int = 1
) -> Union[None, float, pd.Series]:
    if isinstance(nones, int) and nones == 0:
        return np.nan
    num = abs(prediction - np.nanmean(y))
    deno = np.sqrt(horizon / nones) * np.nanstd(y)
    return num / deno
