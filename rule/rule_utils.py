import numpy as np


def coverage(activation: np.ndarray) -> float:
    return np.count_nonzero(activation) / len(activation)


def insert_zeros(x: np.ndarray, y: np.ndarray):
    return np.array([0] * (len(y) - len(x)) + list(x)), y


def check_lines(x: np.ndarray, y: np.ndarray):
    if x.shape < y.shape:
        return insert_zeros(x, y)
    if y.shape < x.shape:
        return insert_zeros(y, x)
    return x, y


def conditional_mean(activation: np.ndarray, y: np.ndarray) -> float:
    y_conditional = np.extract(activation, y)
    cond_mean = np.nanmean(y_conditional)
    return float(cond_mean)


def conditional_std(activation: np.ndarray, y: np.ndarray) -> float:
    y_conditional = np.extract(activation, y)
    cond_std = np.nanstd(y_conditional)
    return float(cond_std)


def calc_zscore(activation: np.ndarray, y: np.ndarray) -> float:
    num = abs(conditional_mean(activation, y) - np.nanmean(y))
    deno = np.nanstd(y) / np.sqrt(sum(activation))
    return num / deno
