import ast
from abc import ABC
from typing import List, Union
import numpy as np
from .activation import Activation


class Condition(ABC):
    def __init__(self, features_indexes: Union[List[int], None] = None, empty: bool = False):
        if empty:
            self._features_indexes = None
        else:
            if features_indexes is None:
                raise ValueError("Must specify features_indexes")
        self._features_indexes = features_indexes

    def __and__(self, other: "Condition") -> "Condition":
        args = [i + j for i, j in zip(self.getattr, other.getattr)]
        to_ret = Condition(features_indexes=args[0], empty=False)
        if len(set(to_ret.features_indexes)) < len(to_ret.features_indexes):
            to_ret.normalize_features_indexes()
        return to_ret

    def __add__(self, other: "Condition") -> "Condition":
        return self & other

    @property
    def getattr(self):
        return [self.features_indexes]

    @property
    def features_indexes(self) -> List[int]:
        return self._features_indexes

    @features_indexes.setter
    def features_indexes(self, value: Union[List[int], str]):
        if isinstance(value, str):
            value = [int(v) for v in value.replace("[", "").replace("]", "").replace(" ", "").split(",")]
        self._features_indexes = value

    def __len__(self):
        return len(self._features_indexes)

    @staticmethod
    def evaluate(xs: np.ndarray) -> Activation:
        """
        Evaluates where a condition if fullfilled

        Parameters
        ----------
        xs: np.ndarray
            shape (n, d), n number of line, d number of features

        Returns
        -------
        activation: Activation
             Shape  (n, 1). The activation vector, filled with 0 where the condition is met and 1 where it is not.
        """
        activation = np.ones(xs.shape[0])
        return Activation(activation)

    def intersect_condition(self, other):
        """To be implemented in daughter class"""
        pass

    def normalize_features_indexes(self):
        self.features_indexes = list(range(len(self.features_indexes)))


class HyperrectangleCondition(Condition):

    SORT_ACCORDING_TO = "index"

    def __init__(
        self,
        features_indexes: Union[List[int], None] = None,
        bmins: Union[List[Union[int, float]], None] = None,
        bmaxs: Union[List[Union[int, float]], None] = None,
        features_names: Union[List[str], None] = None,
        empty: bool = False,
        sort: bool = True,
    ):
        if empty:
            super().__init__(empty=True)
            self._bmins = None
            self._bmaxs = None
            self._features_names = None
        else:
            super().__init__(features_indexes)
            if any([a > b for a, b in zip(bmins, bmaxs)]):
                raise ValueError("Bmin must be smaller or equal than Bmax")
            self._bmins = bmins
            self._bmaxs = bmaxs
            if features_names is not None:
                self._features_names = features_names
            else:
                self._features_names = ["X_" + str(i) for i in self._features_indexes]
            if sort:
                self.sort()

    def __and__(self, other: "HyperrectangleCondition") -> "HyperrectangleCondition":
        args = [i + j for i, j in zip(self.getattr, other.getattr)]
        # noinspection PyTypeChecker
        to_ret = HyperrectangleCondition(
            features_indexes=args[0], bmins=args[1], bmaxs=args[2], features_names=args[3], empty=False,
        )
        if len(set(to_ret.features_indexes)) < len(to_ret.features_indexes):
            to_ret.normalize_features_indexes()
        return to_ret

    def __add__(self, other: "HyperrectangleCondition") -> "HyperrectangleCondition":
        return self & other

    @property
    def getattr(self) -> List[list]:
        return [self.features_indexes, self.bmins, self.bmaxs, self.features_names]

    @property
    def features_names(self) -> List[str]:
        return self._features_names

    @property
    def bmins(self) -> List[Union[int, float]]:
        return self._bmins

    @property
    def bmaxs(self) -> List[Union[int, float]]:
        return self._bmaxs

    @features_names.setter
    def features_names(self, values: Union[List[str], str]):
        if isinstance(values, str):
            values = ast.literal_eval(values)
        self._features_names = values

    @bmins.setter
    def bmins(self, values: Union[List[Union[int, float]], str]):
        if isinstance(values, str):
            values = [int(v) for v in ast.literal_eval(values)]
        self._bmins = values

    @bmaxs.setter
    def bmaxs(self, values: Union[List[Union[int, float]], str]):
        if isinstance(values, str):
            values = [int(v) for v in ast.literal_eval(values)]
        self._bmaxs = values

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self._features_names is None:
            return "empty condition"
        str_output = f"{self._features_names[0]} in [{self._bmins[0]}, {self._bmaxs[0]}]"
        if len(self) > 1:
            for i in range(1, len(self)):
                str_output += " AND "
                str_output += f"{self._features_names[i]} in [{self._bmins[i]}, {self._bmaxs[i]}]"
        return str_output

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        to_hash = [(self._features_names[i], self._bmins[i], self._bmaxs[i]) for i in range(len(self._features_names))]
        to_hash = frozenset(to_hash)
        return hash(to_hash)

    def __getitem__(self, item):
        return (
            self._features_names[item],
            self._features_indexes[item],
            self._bmins[item],
            self._bmaxs[item],
        )

    def __len__(self):
        return len(self._features_names)

    def sort(self):
        if len(self) > 1:
            if HyperrectangleCondition.SORT_ACCORDING_TO == "index":
                self._bmins = [x for _, x in sorted(zip(self._features_indexes, self._bmins))]
                self._bmaxs = [x for _, x in sorted(zip(self._features_indexes, self._bmaxs))]
                self._features_names = [x for _, x in sorted(zip(self._features_indexes, self._features_names))]
                self._features_indexes = sorted(self._features_indexes)
            elif HyperrectangleCondition.SORT_ACCORDING_TO == "name":
                self._bmins = [x for _, x in sorted(zip(self._features_names, self._bmins))]
                self._bmaxs = [x for _, x in sorted(zip(self._features_names, self._bmaxs))]
                self._features_indexes = [x for _, x in sorted(zip(self._features_names, self._features_indexes))]
                self._features_names = sorted(self._features_names)
            else:
                raise ValueError(
                    "HyperrectangleCondition's SORT_ACCORDING_TO"
                    f" can be 'index' or 'name', not {HyperrectangleCondition.SORT_ACCORDING_TO}"
                )

    def evaluate(self, xs: np.ndarray) -> Activation:
        """
        Evaluates where a condition if fullfilled

        Parameters
        ----------
        xs: np.ndarray
            shape (n, d), n number of line, d number of features

        Returns
        -------
        activation: np.ndarray
            Shape  (n, 1). The activation vector, filled with 0 where the condition is met and 1 where it is not.

        Examples
        --------
        >>> xs_ = np.array([[1, 3], [3, 4], [2, np.nan]])
        >>> c1 = HyperrectangleCondition([0], bmins=[1], bmaxs=[2])
        >>> c1.evaluate(xs_)
        np.array([1, 0, 1])
        >>> c2 = HyperrectangleCondition([1], bmins=[3], bmaxs=[5])
        >>> c2.evaluate(xs_)
        np.array([1, 1, 0])
        >>> c3 = HyperrectangleCondition([0, 1], bmins=[1, 3], bmaxs=[2, 5])
        >>> c3.evaluate(xs_)
        np.array([1, 0, 0])
        """
        geq_min = leq_min = not_nan = np.ones(xs.shape[0], dtype=bool)
        for i, j in enumerate(self._features_indexes):
            geq_min &= np.greater_equal(xs[:, j], self._bmins[i])
            leq_min &= np.less_equal(xs[:, j], self._bmaxs[i])
            not_nan &= np.isfinite(xs[:, j])
        activation = geq_min & leq_min & not_nan

        return Activation(activation.astype(int))
