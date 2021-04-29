from abc import ABC
import numpy as np
from typing import Optional
from copy import copy
from .condition import Condition
from .activation import Activation
from .utils import rfunctions as functions


class Rule(ABC):
    def __init__(self, condition: Optional[Condition] = None, activation: Optional[Activation] = None):

        if condition is not None and not isinstance(condition, Condition):
            raise TypeError("Argument 'condition' must derive from Condition or be None.")
        if activation is not None and not isinstance(activation, Activation):
            raise TypeError("Argument 'activation' must derive from Activation or be None.")

        self._condition = condition

        self._activation = activation
        self._coverage = None
        self._prediction = None
        self._std = None
        self._criterion = None

    def __and__(self, other: "Rule") -> "Rule":
        condition = self._condition + other._condition
        activation = self._activation & other._activation
        return self.__class__(condition, activation)

    def __add__(self, other: "Rule") -> "Rule":
        return self & other

    @property
    def condition(self) -> Condition:
        return copy(self._condition)

    @property
    def activation(self) -> np.ndarray:
        """Decompress activation vector

        Returns
        -------
        np.ndarray
            of the form [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
        """
        return self._activation.raw

    @property
    def coverage(self) -> float:
        return self._activation.coverage

    @property
    def prediction(self) -> float:
        return self._prediction

    @property
    def std(self) -> float:
        return self._std

    @property
    def criterion(self) -> float:
        return self._criterion

    def __eq__(self, other) -> bool:
        if not isinstance(other, Rule):
            raise TypeError(f"Can only compare a Rule with another Rule. Tried to compare to {type(other)}.")
        else:
            return self._condition == other._condition

    def __str__(self) -> str:
        prediction = "<prediction unset>"
        if self._prediction is not None:
            prediction = self._prediction
        if self._condition is None:
            return "empty rule"
        return f"If {self._condition.__str__()} Then {prediction}."

    def __hash__(self) -> hash:
        return hash(self._condition)

    def __len__(self):
        return len(self._condition)

    def evaluate(self, xs: np.ndarray) -> Activation:
        return self._condition.evaluate(xs)

    def fit(self, xs: np.ndarray, y: np.ndarray, crit: str = "mse"):
        """Computes activation, prediction, std and criteria of the rule for a given xs and y."""
        self.calc_activation(xs)  # returns Activation
        self.calc_prediction(y)
        self.calc_std(y)
        prediction_vector = self.prediction * self.activation
        self.calc_criterion(prediction_vector, y, crit)

    def calc_activation(self, xs: np.ndarray) -> None:
        self._activation = self.evaluate(xs)

    def calc_prediction(self, y: np.ndarray) -> None:
        """If you do not need to to all 'fit' but only want to compute 'prediction'"""
        if self.activation is None:
            raise ValueError("The activation vector has not been computed yet.")
        self._prediction = functions.conditional_mean(self.activation, y)

    def calc_std(self, y: np.ndarray) -> None:
        """If you do not need to to all 'fit' but only want to compute 'std'"""
        if self.activation is None:
            raise ValueError("The activation vector has not been computed yet.")
        self._std = functions.conditional_std(self.activation, y)

    def calc_criterion(self, p, y, c):
        self._criterion = functions.calc_criterion(p, y, c)

    def predict(self, xs: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns the prediction vector. If xs is not given, will use existing activation vector.
        Will raise ValueError is xs is None and activation is not yet known."""
        if xs is not None:
            self.calc_activation(xs)
        elif self.activation is None:
            raise ValueError("If the activation vector has not been computed yet, xs can not be None.")
        return self._prediction * self.activation
