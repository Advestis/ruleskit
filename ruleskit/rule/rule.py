from abc import ABC
import numpy as np
from typing import Union
from ..condition.condition import Condition
from ..activation.activation import Activation
from . import rule_functions as functions


class Rule(ABC):
    def __init__(self, condition: Union[Condition, None] = None):
        self._condition = condition

        self._activation = None
        self._coverage = None
        self._prediction = None
        self._std = None
        self._criterion = None

    def __and__(self, other: "Rule"):
        condition = self.condition + other.condition
        return Rule(condition)

    def __add__(self, other: "Rule"):
        return self & other

    @property
    def condition(self) -> Condition:
        return self._condition

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
        return self._coverage

    @property
    def prediction(self) -> float:
        return self._prediction

    @property
    def std(self) -> float:
        return self._std

    @property
    def criterion(self) -> float:
        return self._criterion

    @condition.setter
    def condition(self, value: Condition):
        self._condition = value

    @activation.setter
    def activation(self, value: Activation):
        # noinspection PyShadowingNames
        """ Compresses an activation vector into a str(list) describing its variations or an int corresponding to the
         binary representation of the vector

        result will be either a str(list):
            First element of the list is the first value of the array
            last element of the list is the length of the array
            The other elemnts are the coordinates that changed values
        or an int:
            taking the input vector [1 0 0 1 0 0 0 1 1...], converts it to binary string representation :
            "100100011..." then cast it into int using int(s, 2)

        The method will choose which to do based on the size of the compressed list : if it is superior to a certain
        limit, the int willl take less memory and is prefered.

        if an int is passed, just set activation to it and does nothing more

        The limit was estimated from activation vector of 1.8e6 elements, where the int version took 0.25 MB

        Parameters
        ----------
        value: Activation
        """
        self._activation = value

    @coverage.setter
    def coverage(self, value: Union[float, str]):
        if isinstance(value, str):
            value = float(value)
        self._coverage = value

    @prediction.setter
    def prediction(self, value: Union[float, str]):
        if isinstance(value, str):
            value = float(value)
        self._prediction = value

    @std.setter
    def std(self, value: Union[float, str]):
        if isinstance(value, str):
            value = float(value)
        self._std = value

    @criterion.setter
    def criterion(self, value: Union[float, str]):
        if isinstance(value, str):
            value = float(value)
        self._criterion = value

    def __eq__(self, other) -> bool:
        if not isinstance(other, Rule):
            raise TypeError(f"Can only compare a Rule with another Rule. Tried to compare to {type(other)}.")
        else:
            return self._condition == other._condition

    def __str__(self) -> str:
        prediction = "<prediction unset>"
        if self.prediction is not None:
            prediction = self._prediction
        if self._condition is None:
            return "empty rule"
        return f"If {self._condition.__str__()} Then {prediction}."

    def __hash__(self) -> hash:
        return hash(self._condition)

    def __len__(self):
        return len(self._condition)

    def calc_activation(self, xs: np.ndarray) -> Activation:
        return self._condition.evaluate(xs)

    def fit(self, xs: np.ndarray, y: np.ndarray, crit: str = "mse"):
        activation = self.calc_activation(xs)  # returns Activation
        self.activation = activation  # can be int or array
        self.coverage = activation.coverage_rate
        activation_vector = activation.raw

        self.prediction = functions.conditional_mean(activation_vector, y)
        self.std = functions.conditional_std(activation_vector, y)
        prediction_vector = self.prediction * activation_vector
        self.criterion = functions.calc_criterion(prediction_vector, y, crit)

    def predict(self, xs: np.ndarray) -> np.ndarray:
        activation = self.calc_activation(xs)
        return self._prediction * activation.raw
