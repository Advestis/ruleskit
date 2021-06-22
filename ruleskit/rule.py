from abc import ABC
import numpy as np
from typing import Optional, Union
from copy import copy
from time import time
from .condition import Condition
from .activation import Activation
from .utils import rfunctions as functions


class Rule(ABC):
    def __init__(
        self, condition: Optional[Condition] = None, activation: Optional[Activation] = None,
    ):

        if condition is not None and not isinstance(condition, Condition):
            raise TypeError("Argument 'condition' must derive from Condition or be None.")
        if activation is not None and not isinstance(activation, Activation):
            raise TypeError("Argument 'activation' must derive from Activation or be None.")

        self._condition = condition
        self._activation = activation
        self._prediction = None

        self._time_fit = -1
        self._time_calc_activation = -1
        self._time_predict = -1

    def __and__(self, other: "Rule") -> "Rule":
        condition = self._condition + other._condition
        activation = self._activation & other._activation
        return self.__class__(condition, activation)

    def __add__(self, other: "Rule") -> "Rule":
        return NotImplemented("Can not add rules (seen as 'logical OR'). you can use logical AND however.")

    @property
    def condition(self) -> Condition:
        return copy(self._condition)

    @property
    def activation(self) -> Union[None, np.ndarray]:
        """Decompress activation vector

        Returns
        -------
        np.ndarray
            of the form [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
        """
        if self._activation:
            return self._activation.raw
        return None

    @property
    def coverage(self) -> Union[None, float]:
        if self._activation:
            return self._activation.coverage
        return None

    @property
    def prediction(self) -> float:
        return self._prediction

    @property
    def time_fit(self):
        return self._time_fit

    @property
    def time_predict(self):
        return self._time_predict

    @property
    def time_calc_activation(self):
        return self._time_calc_activation

    def __eq__(self, other) -> bool:
        if not isinstance(other, Rule):
            return False
        else:
            return self._condition == other._condition

    def __contains__(self, other: "Rule") -> bool:
        if not self._activation or not other._activation:
            return False
        return other._activation in self._activation

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

    def fit(self, xs: np.ndarray, y: np.ndarray, crit: str = "mse", **kwargs):
        """Computes activation, prediction, std and criteria of the rule for a given xs and y."""
        t0 = time()
        self.calc_activation(xs)  # returns Activation
        self.calc_attributes(xs, y, **kwargs)
        self._time_fit = time() - t0

    def calc_attributes(self, xs: np.ndarray, y: np.ndarray, **kwargs):
        """Implement in daughter class"""
        pass

    def calc_activation(self, xs: np.ndarray) -> None:
        t0 = time()
        self._activation = self.evaluate(xs)
        self._time_calc_activation = time() - t0

    def predict(self, xs: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns the prediction vector. If xs is not given, will use existing activation vector.
        Will raise ValueError is xs is None and activation is not yet known."""
        t0 = time()
        if xs is not None:
            self.calc_activation(xs)
        elif self.activation is None:
            raise ValueError("If the activation vector has not been computed yet, xs can not be None.")
        to_ret = self._prediction * self.activation
        self._time_predict = time() - t0
        return to_ret


class RegressionRule(Rule):
    def __init__(
        self, condition: Optional[Condition] = None, activation: Optional[Activation] = None,
    ):
        super().__init__(condition, activation)

        self._coverage = None
        self._std = None
        self._criterion = None

        # Inspection / Audit attributs
        self._time_calc_criterion = -1
        self._time_calc_prediction = -1
        self._time_calc_std = -1

    @property
    def std(self) -> float:
        return self._std

    @property
    def criterion(self) -> float:
        return self._criterion

    @property
    def time_calc_prediction(self):
        return self._time_calc_prediction

    @property
    def time_calc_criterion(self):
        return self._time_calc_criterion

    @property
    def time_calc_std(self):
        return self._time_calc_std

    def calc_attributes(self, xs: np.ndarray, y: np.ndarray, crit: str = "mse", **kwargs):
        self.calc_activation(xs)
        self.calc_prediction(y)
        self.calc_std(y)
        prediction_vector = self.prediction * self.activation
        self.calc_criterion(prediction_vector, y, crit)

    def calc_activation(self, xs: np.ndarray) -> None:
        t0 = time()
        self._activation = self.evaluate(xs)
        self._time_calc_activation = time() - t0

    def calc_prediction(self, y: np.ndarray) -> None:
        """If you do not need to to all 'fit' but only want to compute 'prediction'"""
        t0 = time()
        if self.activation is None:
            return None
        self._prediction = functions.conditional_mean(self.activation, y)
        self._time_calc_prediction = time() - t0

    def calc_std(self, y: np.ndarray) -> None:
        """If you do not need to to all 'fit' but only want to compute 'std'"""
        t0 = time()
        if self.activation is None:
            return None
        self._std = functions.conditional_std(self.activation, y)
        self._time_calc_std = time() - t0

    def calc_criterion(self, p, y, c, **kwargs):
        t0 = time()
        self._criterion = functions.calc_regression_criterion(p, y, c)
        self._time_calc_criterion = time() - t0

    def predict(self, xs: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns the prediction vector. If xs is not given, will use existing activation vector.
        Will raise ValueError is xs is None and activation is not yet known."""
        t0 = time()
        if xs is not None:
            self.calc_activation(xs)
        elif self.activation is None:
            raise ValueError("If the activation vector has not been computed yet, xs can not be None.")
        to_ret = self._prediction * self.activation
        self._time_predict = time() - t0
        return to_ret


class ClassificationRule(Rule):
    def __init__(
        self, condition: Optional[Condition] = None, activation: Optional[Activation] = None,
    ):
        super().__init__(condition, activation)

        self._criterion = None

        # Inspection / Audit attributs
        self._time_calc_criterion = -1
        self._time_calc_prediction = -1

    @property
    def prediction(self) -> Union[int, str, None]:
        if self._prediction is not None:
            prop = [p[1] for p in self._prediction]
            idx = prop.index(max(prop))
            return self._prediction[idx][0]
        else:
            return None

    @property
    def criterion(self) -> float:
        return self._criterion

    def calc_attributes(self, xs: np.ndarray, y: np.ndarray, crit: str = "success_rate", **kwargs):
        self.calc_prediction(y)
        self.calc_criterion(y, crit)

    def calc_prediction(self, y: np.ndarray) -> None:
        """If you do not need to to all 'fit' but only want to compute 'prediction'"""
        t0 = time()
        if self.activation is None:
            raise ValueError("The activation vector has not been computed yet.")
        self._prediction = functions.most_common_class(self.activation, y)
        self._time_calc_prediction = time() - t0

    def calc_criterion(self, y, c):
        t0 = time()
        self._criterion = functions.calc_classification_criterion(self.activation, self.prediction, y, c)
        self._time_calc_criterion = time() - t0

    def predict(self, xs: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns the prediction vector. If xs is not given, will use existing activation vector.
        Will raise ValueError is xs is None and activation is not yet known."""
        t0 = time()
        if xs is not None:
            self.calc_activation(xs)
        elif self.activation is None:
            raise ValueError("If the activation vector has not been computed yet, xs can not be None.")
        to_ret = np.array([self.prediction if i == 1 else "" for i in self.activation])
        self._time_predict = time() - t0
        return to_ret
