import ast
from abc import ABC
from copy import copy
from typing import List, Union
import numpy as np
from .activation import Activation


class Condition(ABC):

    """Abstract class for Condition object. Used by Rule objets.
    A condition is a list of variable (here represented by their indexes in an array) and of conditions on those
    variables.
    A condition can be "imossible" to meet, in that case self.impossible is True. This is set automatically.

    One can add conditions and use logical AND (&) operations on two conditions (same thing as add). In that case, the
    two conditions are combined into a new one.
    """

    def __init__(self, features_indexes: Union[List[int], None] = None, empty: bool = False):
        if empty:
            self._features_indexes = None
        else:
            if features_indexes is None:
                raise ValueError("Must specify features_indexes")
        self._features_indexes = features_indexes
        self.impossible = False

    def __and__(self, other: "Condition") -> "Condition":
        """To be implemented in daughter classes"""
        pass

    def __add__(self, other: "Condition") -> "Condition":
        return NotImplemented("Can not add conditions (seen as 'logical OR'). You can use logical AND however.")

    @property
    def to_hash(self):
        return ("c",) + tuple(self._features_indexes)

    def __hash__(self):
        return hash(frozenset(self.to_hash))

    @property
    def getattr(self):
        return [self.features_indexes]

    @property
    def features_indexes(self) -> List[int]:
        return self._features_indexes

    @features_indexes.setter
    def features_indexes(self, value: Union[List[int], str]):
        if isinstance(value, str):
            value = ast.literal_eval(value)
        self._features_indexes = value

    def __len__(self):
        return len(self._features_indexes)

    @staticmethod
    def evaluate(xs: np.ndarray) -> np.ndarray:
        """
        Evaluates where a condition if fullfilled. In this abstract class that does not have any acutal condition,
        it is always fullfilled.

        Parameters
        ----------
        xs: np.ndarray
            shape (n, d), n number of line, d number of features

        Returns
        -------
        activation: Activation
             Shape  (n, 1). The activation vector, filled with 1 where the condition is met and 0 where it is not.
        """
        activation = np.ones(xs.shape[0], dtype=np.ubyte)
        return activation

    def normalize_features_indexes(self):
        """In some daughter classes, features indexes are optional. however since the attribute 'features_indexes'
        must be specified, a default value is set automatically"""
        self._features_indexes = list(range(len(self.features_indexes)))


class HyperrectangleCondition(Condition):

    """Condition class for Hyper Rectangle conditions.

    An Hyper Rectangle condition is a condition where each feature is associated to a min and a max (self.bmins and
    self.bmaxs). The condition is met when all features are within their respective bmin and bmax.

    For example, if the condition has :

    features_indexes = [0, 1]
    bminx = [0, 1]
    mnaxs = [0, 2]

    Then the condition is met when feature 0 is equal to 0 and feature 2 is between 1 and 2.

    In this condition, features indexes are optional if features names are given.

    Such a condition can be sorted either according to features indexes (smaller features first) or by features names in
    alphabetical order. This is set through the class attribute SORT_ACCORDING_TO that can be either "index" or "name".
    """

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
            if features_indexes is None:
                if features_names is None:
                    raise ValueError("Must specify at least one of features_indexes and features_names")
                features_indexes = list(range(len(features_names)))
            super().__init__(features_indexes)
            if any([a > b for a, b in zip(bmins, bmaxs)]):
                # If a bmin is above its associated bmax, then the rule is impossible.
                self.impossible = True
            self._bmins = bmins
            self._bmaxs = bmaxs
            if features_names is not None:
                self._features_names = features_names
            else:
                self._features_names = ["X_" + str(i) for i in self._features_indexes]
            if sort:
                self.sort()

    def __and__(self, other: "HyperrectangleCondition") -> "HyperrectangleCondition":

        """Logical and (&) or two HyperrectangleCondition objects

        If the two conditions do not talk about the same features, then the AND is obvious. For common features,
        the bmin of the feature in the new condition is set to be the greatest of bmins in parent conditions, and the
        bmax the smallest of bmaxs.
        This can give impossible conditions : if in condition 1 feature A must be between 0 and 10 and in condition 2 it
        must be between 20 and 30, then in the new condition it must be between 20 and 10, assumin 20 being the minimum.
        In that case, the new condition, upon creation, will have self.impossible = True. This does not corrupt the
        object nor the code : the condition's method "evaluate", which returns the activation vector, will return a
        vector with only zeros since the condition will never be met.
        """

        self_clone = HyperrectangleCondition(
            features_indexes=copy(self.features_indexes),
            bmins=copy(self.bmins),
            bmaxs=copy(self.bmaxs),
            features_names=copy(self.features_names),
        )
        other_clone = HyperrectangleCondition(
            features_indexes=copy(other.features_indexes),
            bmins=copy(other.bmins),
            bmaxs=copy(other.bmaxs),
            features_names=copy(other.features_names),
        )

        common_features = [f for f in self_clone.features_names if f in other_clone.features_names]

        if len(common_features) > 0:

            # If the two conditions have features in common, the new conditons will have as range the intersection of
            # each condition's range for those features. The new condition can possibly never be met.

            common_features_positions_in_self = [self_clone.features_names.index(f) for f in common_features]
            common_features_positions_in_other = [other_clone.features_names.index(f) for f in common_features]

            common_features_bmins_in_self = [self_clone.bmins[i] for i in common_features_positions_in_self]
            common_features_bmins_in_other = [other_clone.bmins[i] for i in common_features_positions_in_other]

            common_features_bmaxs_in_self = [self_clone.bmaxs[i] for i in common_features_positions_in_self]
            common_features_bmaxs_in_other = [other_clone.bmaxs[i] for i in common_features_positions_in_other]

            common_features_bmins = [
                max(bmin0, bmin1) for bmin0, bmin1 in zip(common_features_bmins_in_self, common_features_bmins_in_other)
            ]
            common_features_bmaxs = [
                min(bmax0, bmax1) for bmax0, bmax1 in zip(common_features_bmaxs_in_self, common_features_bmaxs_in_other)
            ]

            other_clone.features_indexes = [
                other_clone.features_indexes[i]
                for i in range(len(other_clone.features_indexes))
                if i not in common_features_positions_in_other
            ]
            other_clone.features_names = [f for f in other_clone.features_names if f not in common_features]
            other_clone.bmins = [
                other_clone.bmins[i]
                for i in range(len(other_clone.bmins))
                if i not in common_features_positions_in_other
            ]
            other_clone.bmaxs = [
                other_clone.bmaxs[i]
                for i in range(len(other_clone.bmaxs))
                if i not in common_features_positions_in_other
            ]

            for i, index in enumerate(common_features_positions_in_self):
                self_clone.bmins[index] = common_features_bmins[i]
                self_clone.bmaxs[index] = common_features_bmaxs[i]

        args = [i + j for i, j in zip(self_clone.getattr, other_clone.getattr)]

        if len(set(args[0])) != len(args[0]):
            args[0] = list(range(len(args[0])))

        # noinspection PyTypeChecker
        to_ret = HyperrectangleCondition(
            features_indexes=args[0],
            bmins=args[1],
            bmaxs=args[2],
            features_names=args[3],
            empty=False,
        )
        return to_ret

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

    @property
    def to_hash(self):
        return ("c",) + tuple(
            (self._features_names[i], self._bmins[i], self._bmaxs[i]) for i in range(len(self._features_names))
        )

    def __hash__(self):
        return hash(frozenset(self.to_hash))

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
                if len(set(self.features_indexes)) != len(self.features_indexes):
                    raise ValueError(
                        "Can not sort HyperrectangleCondition according to index : there are duplicated indexes"
                    )
                self._bmins = [x for _, x in sorted(zip(self._features_indexes, self._bmins))]
                self._bmaxs = [x for _, x in sorted(zip(self._features_indexes, self._bmaxs))]
                self._features_names = [x for _, x in sorted(zip(self._features_indexes, self._features_names))]
                self._features_indexes = sorted(self._features_indexes)
            elif HyperrectangleCondition.SORT_ACCORDING_TO == "name":
                if len(set(self._features_names)) != len(self._features_names):
                    raise ValueError(
                        "Can not sort HyperrectangleCondition according to names : there are duplicated names"
                    )
                self._bmins = [x for _, x in sorted(zip(self._features_names, self._bmins))]
                self._bmaxs = [x for _, x in sorted(zip(self._features_names, self._bmaxs))]
                self._features_indexes = [x for _, x in sorted(zip(self._features_names, self._features_indexes))]
                self._features_names = sorted(self._features_names)
            else:
                raise ValueError(
                    "HyperrectangleCondition's SORT_ACCORDING_TO"
                    f" can be 'index' or 'name', not {HyperrectangleCondition.SORT_ACCORDING_TO}"
                )

    def evaluate(self, xs: np.ndarray) -> np.ndarray:
        """
        Evaluates where a condition if fullfilled

        Parameters
        ----------
        xs: np.ndarray
            shape (n, d), n number of line, d number of features

        Returns
        -------
        activation: np.ndarray
            Shape  (n, 1). The activation vector, filled with 1 where the condition is met and 0 where it is not.

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
        if self.impossible:
            return np.zeros(xs.shape[0], dtype=np.ubyte)
        geq_min = leq_min = not_nan = np.ones(xs.shape[0], dtype=np.ubyte)
        for i, j in enumerate(self._features_indexes):
            geq_min &= np.greater_equal(xs[:, j], self._bmins[i])
            leq_min &= np.less_equal(xs[:, j], self._bmaxs[i])
            not_nan &= np.isfinite(xs[:, j])
        activation = geq_min & leq_min & not_nan

        return activation
