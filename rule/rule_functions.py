import numpy as np


def coverage(activation: np.ndarray) -> float:
    return np.count_nonzero(activation) / len(activation)


def conditional_mean(activation: np.ndarray, y: np.ndarray) -> float:
    y_conditional = np.extract(activation, y)
    cond_mean = np.nanmean(y_conditional)
    return float(cond_mean)


def conditional_std(activation: np.ndarray, y: np.ndarray) -> float:
    y_conditional = np.extract(activation, y)
    cond_std = np.nanstd(y_conditional)
    return float(cond_std)


def mse_function(prediction_vector: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)^2 $"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean squared error
    """
    assert len(prediction_vector) == len(y), \
        'The two array must have the same length'
    error_vector = prediction_vector - y
    criterion = np.nanmean(error_vector ** 2)
    return criterion


def mae_function(prediction_vector: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the mean absolute error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} |\\hat{y}_i - y_i| $"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean absolute error
    """
    assert len(prediction_vector) == len(y), \
        'The two array must have the same length'
    error_vect = np.abs(prediction_vector - y)
    criterion = np.nanmean(error_vect)
    return criterion


def aae_function(prediction_vector: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)$"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean squared error
    """
    assert len(prediction_vector) == len(y), \
        'The two array must have the same length'
    error_vector = np.mean(np.abs(prediction_vector - y))
    median_error = np.mean(np.abs(y - np.median(y)))
    return error_vector / median_error


def calc_criterion(prediction_vector: np.ndarray, y: np.ndarray, method: str, cond: bool = True) -> float:
    """
    Compute the criteria

    Parameters
    ----------
    prediction_vector : {array type}
                        The prediction vector

    y : {array type}
        The real target values (real numbers)

    method : {string type}
             The method mse_function or mse_function criterion

    cond : {boolean type}
            To evaluate the criterion only if the rule is activated

    Return
    ------
    criterion : {float type}
           Criteria value
    """
    if cond:
        sub_y = np.extract(prediction_vector != 0, y)
        sub_pred = np.extract(prediction_vector != 0, prediction_vector)
    else:
        sub_y = y
        sub_pred = prediction_vector

    if method.lower() == 'mse':
        criterion = mse_function(sub_pred, sub_y)

    elif method.lower() == 'mae':
        criterion = mae_function(sub_pred, sub_y)

    elif method.lower() == 'aae':
        criterion = aae_function(sub_pred, sub_y)

    else:
        raise ValueError('Unknown criterion: %s. Please choose among mse, mae and aae' % method)

    return criterion
