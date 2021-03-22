from abc import ABC
import numpy as np
import sys
from .condition import Condition
from . import rule_functions as functions
from typing import Union


class Rule(ABC):

    SIZE_LIMIT = 0.000000139  # 0.25 / 1.8e6
    EXPECTED_LENGTH = None

    def __init__(self, condition: Union[Condition, None] = None):
        self.act_length = Rule.EXPECTED_LENGTH
        self._condition = condition

        self._activation = None
        self._coverage = None
        self._prediction = None
        self._std = None
        self._criterion = None

    def __and__(self, other: 'Rule'):
        condition = self.condition + other.condition
        return Rule(condition)

    def __add__(self, other: 'Rule'):
        return self & other

    @property
    def condition(self) -> Condition:
        return self._condition

    @property
    def activation_for_comparison(self) -> Union[int, np.ndarray]:
        """Returns a value of the form 45786542 (int) or np.ndarray"""
        if isinstance(self._activation, str):
            return self.activation  # will cast it into np.ndarray
        return self._activation

    @property
    def activation(self) -> np.ndarray:
        """Decompress activation vector

        Returns
        -------
        np.ndarray
            of the form [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
        """
        if isinstance(self._activation, int):
            return int_to_array(self._activation)
        return decompress(self._activation)

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
    def activation(self, value: Union[int, np.ndarray]):
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
        value: np.ndarray
            Of the form [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
        """

        if isinstance(value, int):
            self._activation = value
        else:
            self.act_length = len(value)
            to_ret = compress(value)
            sizeof = sys.getsizeof(to_ret) / 1e6
            if (sizeof / len(value)) > Rule.SIZE_LIMIT:
                self._activation = array_to_int(value)
            else:
                self._activation = to_ret

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

    def calc_activation(self, xs: np.ndarray) -> np.ndarray:
        return self._condition.evaluate(xs)

    def fit(self, xs: np.ndarray, y: np.ndarray, crit: str = 'mse'):
        activation = self.calc_activation(xs)  # returns array
        self.activation = activation  # can be int or array
        correct_length_act, y = functions.check_lines(self.activation, y)
        self.coverage = functions.coverage(correct_length_act)
        self.prediction = functions.conditional_mean(correct_length_act, y)
        self.std = functions.conditional_std(correct_length_act, y)
        prediction_vector = self.prediction * correct_length_act
        self.criterion = functions.calc_criterion(prediction_vector, y, crit)

    def predict(self, xs: np.ndarray) -> np.ndarray:
        activation = self.calc_activation(xs)
        return self._prediction * activation
    
    def compare_activations(self, r: "Rule") -> Union[np.ndarray, int]:
        act1 = self.activation_for_comparison
        act2 = r.activation_for_comparison

        if isinstance(act1, int):
            if not isinstance(act2, int):
                if self.act_length is None:
                    if r.act_length is None:
                        raise ValueError(f"act_length of rule {str(r)} is not set")
                    self.act_length = r.act_length
                return int_to_array(act1, r.act_length) * act2
            return act1 & act2

        if isinstance(act2, int):  # then act1 is array
            if r.act_length is None:
                if self.act_length is None:
                    raise ValueError(f"act_length of rule {str(self)} is not set")
                r.act_length = self.act_length
            return int_to_array(act2, self.act_length) * act1

        return act2 * act1


def array_to_int(value: np.ndarray) -> int:
    """Returns a value of the form 45786542 (int), which is the conversion to int of the binary representation of an
    activation vector"""
    return int("".join(str(i) for i in value), 2)


def int_to_array(value: int, length: int = None) -> np.ndarray:
    """From a value of the form 45786542 (int), which is the conversion to int of the binary representation of an
    activation vector, returns the initial vector.

    There will be a loss of information here since the initial 0s will be lost
    """
    act = np.fromiter(bin(value)[2:], dtype=int)
    if length is None:
        return act
    if len(act) > length:
        raise ValueError("After using int_to_array, I ended up with an activation vector bigger than the specified "
                         "max length. This should not happend as the max length should have been set by the indexing "
                         "of x earlier in your code")
    act_bis = np.zeros(length)
    act_bis[length - len(act):] = act
    act = act_bis
    return act


def compress(value: np.ndarray) -> str:
    to_ret = [value[0]]
    diff_arr = abs(np.diff(value))
    to_ret += list(np.where(diff_arr == 1)[0] + 1)
    to_ret.append(len(value))
    to_ret = str(to_ret).replace(" ", "").replace("[", "").replace("]", "")
    return to_ret


def decompress(value: str) -> np.ndarray:
    value = [int(i) for i in value.split(",")]
    new_value = [value[0]] * value[-1]
    for i, position in enumerate(value[1:-1]):
        new_value[position: value[i + 2]] = [value[0] if i % 2 else abs(value[0] - 1)] * (value[i + 2] - position)
    return np.array(new_value)
