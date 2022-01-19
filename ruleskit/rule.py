from abc import ABC
import numpy as np
from typing import Optional, Union, Tuple
from time import time
from .condition import Condition
from .activation import Activation
from .utils import rfunctions as functions
import logging

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
class Rule(ABC):

    """An abstract Rule object.

    A Rule is a condition (represented by any daughter class of ruleskit.Condition), applied on real features and target
    data.
    The Rule contains, in addition to the Condition object, many attributes dependent on the features data, such as
    the activation vector (a 1-D np.ndarray with 0 when the rule is activated - condition is met - and 0 when it is not)
    but also the rule's prediction (computed in the daughter class).

    Daughter classes can remember more attributes (precision, user-definded criterion...).

    Rule also include metrics that can be used for profiling the code : it will remember the time taken to fit the rule
    (fitting is the computation of the rule's attribute from the condition and the features data), the time taken
    to compute the activation vector and the time taken to make a prediction.

    To compute those metrics, one must use the rule's "fit" methods. Once this is done, one cas use the "predict"
     methods on a different set of features data.

    The Rule object can access any attribute of its condition as if it was its own : rule.features_indexes will return
    the features_indexes attribute's value of the condition in the Rule object. See Condition class for more details.

    The Rule object can also access any attribute of its activation vector as if it was its own. See Activation class
    for more details.
    """

    LOCAL_ACTIVATION = True

    def __init__(
        self, condition: Optional[Condition] = None, activation: Optional[Activation] = None,
    ):

        if condition is not None and not isinstance(condition, Condition):
            raise TypeError("Argument 'condition' must derive from Condition or be None.")
        if activation is not None and not isinstance(activation, Activation):
            raise TypeError("Argument 'activation' must derive from Activation or be None.")
        if activation is not None and condition is None:
            raise ValueError("Condition can not be None if activation is not None")

        self._condition = condition
        self._activation = activation

        self._coverage = None
        self._prediction = None

        self._time_fit = -1
        self._time_calc_activation = -1
        self._time_predict = -1

    @property
    def coverage(self) -> float:
        if self._activation is not None:
            self._coverage = self._activation.coverage
            return self._activation.coverage
        return self._coverage

    @coverage.setter
    def coverage(self, value):
        if self._activation is not None:
            self._activation.coverage = value
        self._coverage = value

    def __and__(self, other: "Rule") -> "Rule":
        """Logical AND (&) of two rules. It is simply the logical AND of the two rule's conditions and activations. """
        condition = self._condition & other._condition
        activation = self._activation & other._activation
        return self.__class__(condition, activation)

    def __add__(self, other: "Rule") -> "Rule":
        return NotImplemented("Can not add rules (seen as 'logical OR'). You can use logical AND however.")

    # def __del__(self):
    #     self.del_activation()

    def del_activation(self):
        """Deletes the activation vector's data, but not the object itself, so any computed attributes will remain
        available"""
        if hasattr(self, "_activation") and self._activation is not None:
            self._activation.delete()

    @property
    def activation_available(self) -> bool:
        """Returns True if the rule has an activation vector, and if this Activation's object data is available."""
        if self._activation is None:
            return False
        if self._activation.data_format == "file":
            return self._activation.data.is_file()
        else:
            return self._activation.data is not None

    @property
    def condition(self) -> Condition:
        return self._condition

    @property
    def activation(self) -> Union[None, np.ndarray]:
        """Returns the Activation vector's data in a form of a 1-D np.ndarray, or None if not available.

        Returns
        -------
        np.ndarray
            of the form [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, ...]
        """
        if self._activation:
            return self._activation.raw
        return None

    @property
    def prediction(self) -> float:
        return self._prediction

    @property
    def time_fit(self) -> float:
        """Profiling attribute. Time in seconds taken to fit the rule"""
        return self._time_fit

    @property
    def time_predict(self) -> float:
        """Profiling attribute. Time in seconds taken by the rule to make a prediction"""
        return self._time_predict

    @property
    def time_calc_activation(self) -> float:
        """Profiling attribute. Time in seconds taken to comptue the activation vector"""
        return self._time_calc_activation

    def __getattr__(self, item):
        """If item is not found in self, try to fetch it from its activation or condition."""
        if item == "_activation" or item == "_condition":
            raise AttributeError(f"'Rule' object has no attribute '{item}'.")

        if hasattr(self._activation, item):
            return getattr(self._activation, item)
        if hasattr(self._condition, item):
            return getattr(self._condition, item)
        raise AttributeError(f"'Rule' object has no attribute '{item}'.")

    def __setattr__(self, item, value):
        """If item is private (starts with _), then default behavior. Else, if the item is not yet known by the rule
        but is known by its condition or activation, will set it to the condition or the activation. Else,
        raises AttributeError."""
        if item.startswith("_"):
            super(Rule, self).__setattr__(item, value)
            return
        if not hasattr(self, item):
            if hasattr(self._activation, item):
                setattr(self._activation, item, value)
            elif hasattr(self._condition, item):
                setattr(self._condition, item, value)
            else:
                raise AttributeError(f"Can not set attribute '{item}' in object Rule.")
        else:
            super(Rule, self).__setattr__(item, value)

    def __eq__(self, other) -> bool:
        """Two rules are equal if their conditions are equal."""
        if not isinstance(other, Rule):
            return False
        else:
            return self._condition == other._condition

    def __contains__(self, other: "Rule") -> bool:
        """A Rule contains another Rule if the second rule's activated points are also all activated by the first rule.
        """
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

    @property
    def to_hash(self) -> Tuple[str]:
        return ("r",) + self._condition.to_hash[1:]

    def __hash__(self) -> hash:
        return hash(frozenset(self.to_hash))

    def __len__(self):
        """A Rule's length is the number of features it talks about"""
        return len(self._condition)

    def evaluate(self, xs: Union["pd.DataFrame", np.ndarray]) -> Activation:
        """Computes and returns the activation vector from an array of features.

        Parameters
        ----------
        xs: Union[pd:DataFrame, np.ndarray]
            The features on which the check whether the rule is activated or not. Must be a 2-D np.ndarray
            or pd:DataFrame.

        Returns
        -------
        Activation
        """
        arr = self._condition.evaluate(xs)
        # noinspection PyTypeChecker
        a = Activation(arr, to_file=Rule.LOCAL_ACTIVATION)
        return a

    # noinspection PyUnusedLocal
    def fit(self, xs: Union["pd.DataFrame", np.ndarray], y: np.ndarray, **kwargs):
        """Computes activation, and other criteria dependant on the nature of the daughter class of the Rule,
        for a given xs and y.

        Parameters
        ----------
        xs: Union[pd:DataFrame, np.ndarray]
            The features on which the check whether the rule is activated or not. Must be a 2-D np.ndarray
            or pd:DataFrame.
        y: np.ndarray
            The targets on which to evaluate the rule prediction, and possibly other criteria. Must be a 1-D np.ndarray.
        kwargs: dict
            Other arguments used by daughter class
        """
        t0 = time()
        self.calc_activation(xs)
        self.calc_attributes(xs, y, **kwargs)
        if self.prediction is None:
            raise ValueError("'fit' did not set 'prediction' : did you overload 'calc_attributes' correctly ?")
        self._time_fit = time() - t0

    def calc_attributes(self, xs: Union["pd.DataFrame", np.ndarray], y: Union[np.ndarray, "pd.Series"], **kwargs):
        """Implement in daughter class. Must set self._prediction."""
        raise NotImplementedError("To implement in daughter class")

    def calc_activation(self, xs: Union["pd.DataFrame", np.ndarray]):
        """Uses self.evaluate to set self._activation.

        Parameters
        ----------
        xs: Union[pd:DataFrame, np.ndarray]
            The features on which the check whether the rule is activated or not. Must be a 2-D np.ndarray
            or pd:DataFrame.
        """
        t0 = time()
        self._activation = self.evaluate(xs)
        self._time_calc_activation = time() - t0

    def predict(self, xs: Optional[Union["pd.DataFrame", np.ndarray]] = None) -> Union[np.ndarray, "pd.Series"]:
        """Returns the prediction vector. If xs is not given, will use existing activation vector.
        Will raise ValueError is xs is None and activation is not yet known.

        Parameters
        ----------
        xs: Optional[Union[pd:DataFrame, np.ndarray]]
            The features on which the check whether the rule is activated or not. Must be a 2-D np.ndarray
            or pd:DataFrame. If not specified the rule's activation vector must have been computed already.

        Returns
        -------
        Union[np.ndarray, pd.Series]
            np.nan where rule is not activated, rule's prediction where it is. If xs vas given and it was a dataframe,
            return a pd.Series. Else, a np.ndarray.
        """
        t0 = time()
        if xs is not None:
            self.calc_activation(xs)
        elif self.activation is None:
            raise ValueError("If the activation vector has not been computed yet, xs can not be None.")
        act = self.activation
        to_ret = np.array([np.nan] * len(act))
        if isinstance(self.prediction, str):
            if self.prediction == "nan":
                raise ValueError("Prediction should not be the 'nan' string, it will conflict with NaNs."
                                 "Rename your class.")
            to_ret = to_ret.astype(str)
        to_ret[act == 1] = self.prediction
        self._time_predict = time() - t0
        if xs is not None and not isinstance(xs, np.ndarray):
            return xs.__class__(index=xs.index, data=to_ret).squeeze()  # So not to requier pandas explicitly
        return to_ret

    def get_correlation(self, other: "Rule") -> float:
        """ Computes the correlation between self and other
        Correlation is the number of points in common between the two vectors divided by their length, times the product
        of the rules' signs.
        Both vectors must have the same length.
        """
        if not len(self) == len(other):
            raise ValueError("Both vectors must have the same length")

        sign = (self.prediction / abs(self.prediction)) * (other.prediction / abs(other.prediction))
        return self._activation.get_correlation(other._activation) * sign


# noinspection PyUnresolvedReferences
class RegressionRule(Rule):

    """Rule applied on continuous target data."""

    def __init__(
        self, condition: Optional[Condition] = None, activation: Optional[Activation] = None,
    ):
        super().__init__(condition, activation)
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

    def calc_attributes(self, xs: Union["pd.DataFrame", np.ndarray], y: Union[np.ndarray, "pd.Series"], **kwargs):
        """Computes prediction, standard deviation, and regression criterion
        
        Parameters
        ----------
        xs: Union[pd:DataFrame, np.ndarray]
            The features on which the check whether the rule is activated or not. Must be a 2-D np.ndarray
            or pd:DataFrame.
        y: Union[np.ndarray, pd.Series]
            The targets on which to evaluate the rule prediction, and possibly other criteria. Must be a 1-D np.ndarray
            or pd.Series.
        kwargs: dict
            Arguments for calc_regression_criterion
        """
        self.calc_prediction(y)
        self.calc_std(y)
        prediction_vector = self.prediction * self.activation
        self.calc_criterion(prediction_vector, y, **kwargs)

    def calc_prediction(self, y: [np.ndarray, "pd.Series"]):
        """Computes the mean of all activated points in target y and use it as prediction
        
        Parameters
        ----------
        y: [np.ndarray, pd.Series]
            The targets on which to evaluate the rule prediction, and possibly other criteria. Must be a 1-D np.ndarray
            or pd.Series
        """
        t0 = time()
        if self.activation is None:
            return None
        self._prediction = functions.conditional_mean(self.activation, y)
        self._time_calc_prediction = time() - t0

    def calc_std(self, y: Union[np.ndarray, "pd.Series"]):
        """Computes the standard deviation of all activated points in target y
        
        Parameters
        ----------
        y: Union[np.ndarray, pd.Series]
            The targets on which to evaluate the rule prediction, and possibly other criteria. Must be a 1-D np.ndarray
            or pd.Series.
        """
        t0 = time()
        if self.activation is None:
            return None
        self._std = functions.conditional_std(self.activation, y)
        self._time_calc_std = time() - t0

    def calc_criterion(self, p: Union[np.ndarray, "pd.Series"], y: Union[np.ndarray, "pd.Series"], **kwargs):
        """
        Parameters
        ----------
        p: Union[np.ndarray, pd.Series]
            Prediction vector. Must be a 1-D np.ndarray or pd.Series.
        y: Union[np.ndarray, pd.Series]
            The targets on which to evaluate the rule prediction, and possibly other criteria. Must be a 1-D np.ndarray
            or pd.Series.
        kwargs: dict
            Arguments for calc_regression_criterion
        """
        t0 = time()
        self._criterion = functions.calc_regression_criterion(p, y, **kwargs)
        self._time_calc_criterion = time() - t0


# noinspection PyUnresolvedReferences
class ClassificationRule(Rule):

    """Rule applied on discret target data."""

    def __init__(
        self, condition: Optional[Condition] = None, activation: Optional[Activation] = None,
    ):
        super().__init__(condition, activation)

        self._criterion = None

        self._time_calc_criterion = -1
        self._time_calc_prediction = -1

    @property
    def prediction(self) -> Union[int, str, None]:
        if self._prediction is not None:
            if isinstance(self._prediction, (float, int, str)):
                return self._prediction
            prop = [p[1] for p in self._prediction]
            idx = prop.index(max(prop))
            return self._prediction[idx][0]
        else:
            return None

    @property
    def criterion(self) -> float:
        return self._criterion

    def calc_attributes(self, xs: Union["pd.DataFrame", np.ndarray], y: Union[np.ndarray, "pd.Series"], **kwargs):
        """
        Parameters
        ----------
        xs: Union[pd:DataFrame, np.ndarray]
            The features on which the check whether the rule is activated or not. Must be a 2-D np.ndarray
            or pd:DataFrame.
        y: Union[np.ndarray, pd.Series]
            The targets on which to evaluate the rule prediction, and possibly other criteria. Must be a 1-D np.ndarray
            or pd.Series.
        kwargs: dict
            Arguments for calc_classification_criterion
        """
        self.calc_prediction(y)
        self.calc_criterion(y, **kwargs)

    def calc_prediction(self, y: [np.ndarray, "pd.Series"]):
        """
        Parameters
        ----------
        y: [np.ndarray, pd.Series]
            The targets on which to evaluate the rule prediction, and possibly other criteria. Must be a 1-D np.ndarray
            or pd.Series.
        """
        t0 = time()
        if self.activation is None:
            raise ValueError("The activation vector has not been computed yet.")
        self._prediction = functions.most_common_class(self.activation, y)
        self._time_calc_prediction = time() - t0

    def calc_criterion(self, y: Union[np.ndarray, "pd.Series"], **kwargs):
        """
        Parameters
        ----------
        y: Union[np.ndarray, pd.Series]
            The targets on which to evaluate the rule prediction, and possibly other criteria. Must be a 1-D np.ndarray
            or pd.Series
        kwargs: dict
            Arguments for calc_classification_criterion
        """
        t0 = time()
        self._criterion = functions.calc_classification_criterion(self.activation, self.prediction, y, **kwargs)
        self._time_calc_criterion = time() - t0
