import ast
from abc import ABC
from copy import copy
from typing import List, Union, Tuple
import numpy as np
from numbers import Number
from .activation import Activation
import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    pandas_ok = True
except ImportError:
    class pd:
        DataFrame = None
        Series = None
    pandas_ok = False


class DuplicatedFeatures(Exception):
    pass


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
            if any([not np.issubdtype(type(a), np.integer) for a in features_indexes]):
                raise TypeError(
                    f"features_indexes must be integers. You gave {[(f, type(f)) for f in features_indexes]}"
                )
            self._features_indexes = features_indexes
            if len(set(self._features_indexes)) != len(self._features_indexes):
                raise DuplicatedFeatures

        self.impossible = False

    def sort(self):
        self._features_indexes = sorted(self._features_indexes)

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
    def features_indexes(self) -> List[int]:
        return self._features_indexes

    @features_indexes.setter
    def features_indexes(self, value: Union[List[int], str]):
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if len(set(value)) != len(value):
            raise DuplicatedFeatures
        if len(self) > 0 and len(self) != len(value):
            raise IndexError(f"Condition has {len(self)} features but you gave {len(value)} indexes")
        self._features_indexes = value

    def __len__(self):
        """A Condition's length is the number of features it talks about"""
        return len(self._features_indexes)

    def evaluate(self, xs: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        To be implemented in daughter class.
        Evaluates where a condition if fullfilled. In this abstract class that does not have any acutal condition,
        it is always fullfilled.

        Parameters
        ----------
        xs: Union[pd.DataFrame, np.ndarray]
            shape (n, d), n number of line, d number of features

        Returns
        -------
        activation: Activation
             Shape  (n, 1). The activation vector, filled with 1 where the condition is met and 0 where it is not.
        """
        pass

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
            if bmins is None:
                raise ValueError("bmins can not be None if 'empty' is False")
            if bmaxs is None:
                raise ValueError("bmaxs can not be None if 'empty' is False")
            if features_indexes is None:
                if features_names is None:
                    raise ValueError("Must specify at least one of features_indexes and features_names")
                features_indexes = list(range(len(features_names)))

            length = len(features_indexes)
            if len(bmaxs) != length:
                raise ValueError(f"Specifed {length} features but {len(bmaxs)} bmaxs")
            if len(bmins) != length:
                raise ValueError(f"Specifed {length} features but {len(bmins)} bmins")
            if features_names is not None and len(features_names) != length:
                raise ValueError(f"Specifed {length} features but {len(features_names)} bmaxs")

            if features_names is not None and any([not isinstance(a, str) for a in features_names]):
                raise TypeError(f"Names must be strings. You gave {[(f, type(f)) for f in features_names]}")
            if any([not isinstance(a, Number) for a in bmins]):
                raise TypeError(f"bmins must be integers or floats. You gave {[(f, type(f)) for f in bmins]}")
            if any([not isinstance(a, Number) for a in bmaxs]):
                raise TypeError(f"bmaxs must be integers or floats. You gave {[(f, type(f)) for f in bmaxs]}")

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
            if len(set(self._features_names)) != len(self._features_names):
                raise DuplicatedFeatures
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

        common_features = [f for f in self_clone.features_names if f in other.features_names]

        if len(common_features) > 0:

            # If the two conditions have features in common, the new conditons will have as range the intersection of
            # each condition's range for those features. The new condition can possibly never be met.

            common_features_positions_in_self = [self_clone.features_names.index(f) for f in common_features]
            common_features_positions_in_other = [other.features_names.index(f) for f in common_features]

            (common_features_indexes_in_self, common_features_bmins_in_self, common_features_bmaxs_in_self) = list(zip(
                *[
                    (
                        self_clone.features_indexes[i],
                        self_clone.bmins[i],
                        self_clone.bmaxs[i]
                    )
                    for i in common_features_positions_in_self
                ]
            ))

            (common_features_indexes_in_other, common_features_bmins_in_other, common_features_bmaxs_in_other) = list(
                zip(
                    *[
                        (
                            other.features_indexes[i],
                            other.bmins[i],
                            other.bmaxs[i]
                        )
                        for i in common_features_positions_in_other
                    ]
                )
            )

            if common_features_indexes_in_self != common_features_indexes_in_other:
                raise IndexError("Some features present in both conditions in __and__ have different indexes : \n "
                                 f"{common_features_indexes_in_self}\n "
                                 f"{common_features_indexes_in_other}")

            common_features_bmins = [
                max(bmin0, bmin1) for bmin0, bmin1 in zip(common_features_bmins_in_self, common_features_bmins_in_other)
            ]
            common_features_bmaxs = [
                min(bmax0, bmax1) for bmax0, bmax1 in zip(common_features_bmaxs_in_self, common_features_bmaxs_in_other)
            ]

            features_indexes = [
                other.features_indexes[i]
                for i in range(len(other.features_indexes))
                if i not in common_features_positions_in_other
            ]
            features_names = [f for f in other.features_names if f not in common_features]
            bmins = [
                other.bmins[i]
                for i in range(len(other.bmins))
                if i not in common_features_positions_in_other
            ]
            bmaxs = [
                other.bmaxs[i]
                for i in range(len(other.bmaxs))
                if i not in common_features_positions_in_other
            ]

            other_clone = HyperrectangleCondition(
                features_indexes=features_indexes,
                bmins=bmins,
                bmaxs=bmaxs,
                features_names=features_names,
            )

            for i, index in enumerate(common_features_positions_in_self):
                self_clone.bmins[index] = common_features_bmins[i]
                self_clone.bmaxs[index] = common_features_bmaxs[i]
        else:
            other_clone = other

        args = [i + j for i, j in zip(self_clone.getattr, other_clone.getattr)]
        if len(set(args[0])) != len(args[0]):
            raise IndexError("Some features with different names had same index in both conditions in __and__:\n "
                             f"{args}")

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
    def features_names(self, value: Union[List[str], str]):
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if len(set(value)) != len(value):
            raise DuplicatedFeatures
        if len(self) > 0 and len(self) != len(value):
            raise IndexError(f"Condition has {len(self)} features but you gave {len(value)} names")
        self._features_names = value

    @bmins.setter
    def bmins(self, value: Union[List[Union[int, float]], str]):
        if isinstance(value, str):
            value = [int(v) for v in ast.literal_eval(value)]
        if len(self) > 0 and len(self) != len(value):
            raise IndexError(f"Condition has {len(self)} features but you gave {len(value)} bmins")
        self._bmins = value
        if any([a > b for a, b in zip(self.bmins, self.bmaxs)]):
            # If a bmin is above its associated bmax, then the rule is impossible.
            self.impossible = True

    @bmaxs.setter
    def bmaxs(self, value: Union[List[Union[int, float]], str]):
        if isinstance(value, str):
            value = [int(v) for v in ast.literal_eval(value)]
        if len(self) > 0 and len(self) != len(value):
            raise IndexError(f"Condition has {len(self)} features but you gave {len(value)} bmaxs")
        self._bmaxs = value
        if any([a > b for a, b in zip(self.bmins, self.bmaxs)]):
            # If a bmin is above its associated bmax, then the rule is impossible.
            self.impossible = True

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
        """Two HyperrectangleConditions are equal if they talk about the same features, and if the bmins and bmaxs
        are the same from one rule to another. Features indexes can be different."""
        return self.__hash__() == other.__hash__()

    @property
    def to_hash(self) -> Tuple[str]:
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

    def __len__(self) -> int:
        """A HyperrectangleCondition's length is the number of features it talks about"""
        if self._features_names is not None:
            return len(self._features_names)
        if self._features_indexes is not None:
            return len(self._features_indexes)
        if self._bmins is not None:
            return len(self._bmins)
        if self._bmaxs is not None:
            return len(self._bmaxs)
        return 0

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

    def evaluate(self, xs: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Evaluates where a condition if fullfilled, by returning a vector of the form [0, 1, 0, 0, ...]

        Parameters
        ----------
        xs: Union[pd.DataFrame, np.ndarray]
            shape (n, d), n number of line, d number of features. If is a pd.DataFrame, will use self.features_names to
            select features to use in xs

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

        if isinstance(xs, np.ndarray):
            if any([i >= xs.shape[1] for i in self.features_indexes]):
                raise IndexError("Some features indexes in self are greater than the size of the given xs array")
            geq_min = leq_min = not_nan = np.ones(xs.shape[0], dtype=np.ubyte)
            for i, j in enumerate(self._features_indexes):
                geq_min &= np.greater_equal(xs[:, j], self._bmins[i])
                leq_min &= np.less_equal(xs[:, j], self._bmaxs[i])
                not_nan &= np.isfinite(xs[:, j])
            activation = geq_min & leq_min & not_nan
        elif pandas_ok:
            if not isinstance(xs, pd.DataFrame):
                raise TypeError("xs should be a np.ndarray or a pd.DataFrame object")
            if any([i not in xs.columns for i in self.features_names]):
                raise IndexError("Some features names in self were not in xs DataFrame columns")
            geq_min = leq_min = not_nan = np.ones(xs.shape[0], dtype=np.ubyte)
            for i, n in enumerate(self._features_names):
                geq_min &= np.greater_equal(xs[n], self._bmins[i])
                leq_min &= np.less_equal(xs[n], self._bmaxs[i])
                not_nan &= np.isfinite(xs[n])
            activation = (geq_min & leq_min & not_nan).values
        else:
            raise ImportError("If xs is not a np.ndarray, Condition expects it to be a pd.DataFrame, but pandas is not "
                              "available. Please run\npip install pandas")

        return activation
