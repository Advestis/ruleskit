import inspect
from abc import ABC
import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
from time import time
from pathlib import Path
from .condition import Condition
from .activation import Activation
from .utils import rfunctions as functions
from .thresholds import Thresholds
import logging

from .utils.rfunctions import calc_zscore_external

logger = logging.getLogger(__name__)


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
    THRESHOLDS = None
    """Thresholds that the Rule must meet to be good. See `ruleskit.thresholds.Thresholds` for more details."""

    condition_index = ["features_names", "features_indexes", "bmins", "bmaxs"]
    rule_index = ["prediction"]
    index = condition_index + rule_index

    attributes_from_test_set = ["test_set_size"]
    attributes_from_train_set = ["train_set_size"]

    fitted_if_has = "prediction"

    daughters = []

    # noinspection PyUnresolvedReferences
    @classmethod
    def SET_THRESHOLDS(cls, path: Union[str, Path, "TransparentPath", None], show=False):
        """Set thresholds globally for all futur Rules"""
        if path is None:
            cls.THRESHOLDS = None
        else:
            cls.THRESHOLDS = Thresholds(path, show)

    def __init__(
        self,
        condition: Optional[Condition] = None,
        activation: Optional[Activation] = None,
    ):

        if condition is not None and not isinstance(condition, Condition):
            raise TypeError("Argument 'condition' must derive from Condition or be None.")
        if activation is not None and not isinstance(activation, Activation):
            raise TypeError("Argument 'activation' must derive from Activation or be None.")
        if activation is not None and condition is None:
            raise ValueError("Condition can not be None if activation is not None")

        self._condition: Optional[Condition] = condition
        self._activation: Optional[Activation] = activation
        self._thresholds: Optional[Thresholds] = self.__class__.THRESHOLDS
        self._good: bool = True
        self._bad_because: Optional[str] = None

        self._coverage: Optional[float] = None
        self._prediction: Optional[Union[float, str, int]] = None
        self._criterion: Optional[float] = None
        self._zscore: Optional[float] = None

        self._time_fit: float = -1
        self._time_eval: float = -1
        self._time_calc_activation: float = -1
        self._time_predict: float = -1
        self._time_calc_prediction: float = -1
        self._time_calc_criterion: float = -1
        self._time_calc_zscore: float = -1
        self._fitted: bool = False
        self._evaluated = False
        self._train_set_size: Optional[int] = None
        self._test_set_size: Optional[int] = None
        if self._activation is not None:
            self.check_thresholds("coverage")

    # noinspection PyUnresolvedReferences
    def set_thresholds(self, path: Union[str, Path, "TransparentPath"], show=False):
        """Set thresholds for this rule only"""
        if path is None:
            self._thresholds.THRESHOLDS = None
        else:
            self._thresholds = Thresholds(path, show)

    def check_thresholds(self, attribute: Optional[str] = None) -> None:
        """If `ruleskit.rule.Rule.THRESHOLDS` is specified, will check that this rule is good regarding those
        thresholds, and set the flags *good* and *bad_because* accordingly

        Parameters
        ----------
        attribute: Optional[str]
            If specified, will only check the threshold of this rule attribute. If not, will test every rule attributes
            for which a threshold is defined.
        """

        if self.__class__.THRESHOLDS is None:
            return

        if attribute is not None:
            if not self.__class__.THRESHOLDS(attribute, self):
                self._bad_because = attribute
                self._good = False
            return

        for attribute in dir(self):
            if attribute.startswith("__"):
                continue
            if not self.__class__.THRESHOLDS(attribute, self):
                self._bad_because = attribute
                self._good = False
                return
        logger.debug(f"Rule {self} is good")

    def __and__(self, other: "Rule") -> "Rule":
        """Logical AND (&) of two rules. It is simply the logical AND of the two rule's conditions and activations."""
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
    def coverage(self) -> float:
        if self._activation is not None:
            self._coverage = self._activation.coverage
            return self._activation.coverage
        return self._coverage

    @coverage.setter
    def coverage(self, value: Union[int, None]):
        if self._activation is not None:
            self._activation._coverage = value
        self._coverage = value

    @property
    def train_set_size(self) -> int:
        return self._train_set_size

    @property
    def test_set_size(self) -> int:
        return self._test_set_size

    @train_set_size.setter
    def train_set_size(self, value: Union[int, None]):
        self._train_set_size = value

    @test_set_size.setter
    def test_set_size(self, value: Union[int, None]):
        self._test_set_size = value

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
    def prediction(self) -> Union[str, float]:
        return self._prediction

    @property
    def thresholds(self) -> Thresholds:
        return self._thresholds

    @property
    def good(self) -> bool:
        return self._good

    @property
    def bad_because(self) -> str:
        return self._bad_because

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
        """
        A Rule contains another Rule if the second rule's activated points are also all activated by the first
        rule.
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

    def evaluate_activation(self, xs: Union[pd.DataFrame, np.ndarray]) -> Activation:
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
        a = Activation(arr, to_file=self.__class__.LOCAL_ACTIVATION)
        return a

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        xs: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        force_if_not_good: bool = False,
        **kwargs
    ):
        """Computes activation and attributes relevant to the train set

        Parameters
        ----------
        y: Union[np.ndarray, pd.Series]
        xs: Union[pd.DataFrame, np.ndarray]
        force_if_not_good: bool
        kwargs: dict
            Additionnal keyword arguments for calc_<any_attribute>
        """

        if "method" in kwargs:
            raise IndexError("Key 'method' can not be given to 'fit'")

        if self._fitted and xs is None:
            return

        t0 = time()

        def launch_method(method, **kw):
            expected_args = list(inspect.signature(method).parameters)
            if "kwargs" not in expected_args:
                kw = {item: kw[item] for item in kw if item in expected_args}
            method(**kw)

        if xs is not None and len(xs) == 0:
            logger.warning("Given xs is empty")
            return

        self.calc_activation(xs=xs)

        for attr in self.__class__.attributes_from_train_set:
            if attr == "activation":
                raise ValueError("'activation' can not be specified in 'attributes_from_train_set'")
            if not self.good and not force_if_not_good:
                setattr(self, f"_{attr}", np.nan)
                continue
            launch_method(getattr(self, f"calc_{attr}"), y=y, xs=xs, **kwargs)
            if self.good:
                self.check_thresholds(attr)

        if self.good:
            self.check_thresholds()
        self.trigger_subattributes_computation()
        self._time_fit = time() - t0
        self._fitted = True

    def eval(
        self,
        y: Union[np.ndarray, pd.Series],
        xs: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        recompute_activation: bool = False,
        force_if_not_good: bool = False,
        **kwargs,
    ):
        """Computes prediction, standard deviation, and regression criterion

        Parameters
        ----------
        y: Union[np.ndarray, pd.Series]
        xs: Union[pd.DataFrame, np.ndarray]
        recompute_activation: bool
            To reset self.activation using the given xs
        force_if_not_good: bool
            If the rule was seen as "bad", eval will not trigger unless this boolean is True (Default value = False)
        kwargs
            Additionnal keyword arguments for calc_<any_attribute>
        """

        if "method" in kwargs:
            raise IndexError("Key 'method' can not be given to 'eval'")

        t0 = time()

        if not self.good and not force_if_not_good:
            self._time_eval = time() - t0
            self._fitted = True
            for attr in self.__class__.attributes_from_test_set:
                setattr(self, f"_{attr}", np.nan)
            return

        def launch_method(method, **kw):
            expected_args = list(inspect.signature(method).parameters)
            if "kwargs" not in expected_args:
                kw = {item: kw[item] for item in kw if item in expected_args}
            method(**kw)

        if xs is not None and len(xs) == 0:
            logger.warning("Given xs is empty")
            return

        if recompute_activation:
            self.calc_activation(xs=xs)
            xs = None

        if not self.activation_available:
            raise ValueError(
                "Must have fitted the rule before calling 'eval', or use 'recompute_activation=True' to recompute it"
                " from given xs"
            )

        if xs is not None:
            activation = self.evaluate_activation(xs)
        else:
            activation = self._activation

        for attr in self.__class__.attributes_from_test_set:
            if attr == "activation":
                raise ValueError("'activation' can not be specified in 'attributes_from_test_set'")
            if not self.good and not force_if_not_good:
                setattr(self, f"_{attr}", np.nan)
                continue
            launch_method(getattr(self, f"calc_{attr}"), y=y, xs=xs, activation=activation, **kwargs)
            if self.good:
                self.check_thresholds(attr)

        if self.good:
            self.check_thresholds()
        self._time_eval = time() - t0
        self._evaluated = True

    def trigger_subattributes_computation(self):
        """Uses getattr(self, attr) to trigger important attributes computation. Important attributes should be
        Ruleset.rule_index"""
        for attr in self.__class__.index:
            _ = getattr(self, attr)

    def calc_prediction_vector(self, activation: Optional[np.ndarray] = None) -> Union[None, pd.Series]:
        if activation is None:
            activation = self.activation
        if activation is None:
            return None
        if self.prediction is None:
            return pd.Series(np.nan * activation)
        if len(self) == 0:
            return pd.Series(dtype=int)
        if isinstance(self.prediction, str):
            pred = pd.Series(np.where(activation == 0, np.nan, activation))
            pred[pred == 1] = self.prediction
            return pred
        else:
            # noinspection PyUnresolvedReferences
            return self.prediction * pd.Series(np.where(activation == 0, np.nan, activation))

    def predict(self, xs: Optional[Union[pd.DataFrame, np.ndarray]] = None) -> Union[np.ndarray, pd.Series]:
        """Returns the prediction vector. If xs is not given, will use existing activation vector.
        Will raise ValueError is xs is None and activation is not yet known.

        Parameters
        ----------
        xs: Optional[Union[pd.DataFrame, np.ndarray]]
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
            act = self.evaluate_activation(xs).raw
        elif self.activation is None:
            raise ValueError("If the activation vector has not been computed yet, xs can not be None.")
        else:
            act = self.activation
        to_ret = np.array([np.nan] * len(act))
        if isinstance(self.prediction, str):
            if self.prediction == "nan":
                raise ValueError(
                    "Prediction should not be the 'nan' string, it will conflict with NaNs. Rename your class."
                )
            to_ret = to_ret.astype(str)
        to_ret[act == 1] = self.prediction
        if xs is not None and not isinstance(xs, np.ndarray):
            to_ret = xs.__class__(index=xs.index, data=to_ret).squeeze()  # So not to requier pandas explicitly
        self._time_predict = time() - t0
        return to_ret

    def get_correlation(self, other: "Rule") -> float:
        """Computes the correlation between self and other
        Correlation is the number of points in common between the two vectors divided by their length, times the product
        of the rules' signs.
        Both vectors must have the same length.
        """
        if not len(self) == len(other):
            raise ValueError("Both vectors must have the same length")

        sign = (self.prediction / abs(self.prediction)) * (other.prediction / abs(other.prediction))
        return self._activation.get_correlation(other._activation) * sign

    def calc_activation(self, xs: Optional[Union[pd.DataFrame, np.ndarray]] = None):
        """Uses self.evaluate to set self._activation.

        Parameters
        ----------
        xs: Union[pd.DataFrame, np.ndarray, None]
            The features on which the check whether the rule is activated or not. Must be a 2-D np.ndarray
            or pd:DataFrame.
        """
        if xs is None:
            if self._activation is None:
                raise ValueError(
                    "If calling calc_activation without specifying xs, activation must have been computed already."
                )
            return
        t0 = time()

        if len(xs) == 0:
            logger.warning("Given xs is empty")
            return
        self._activation = self.evaluate_activation(xs)
        self._time_calc_activation = time() - t0
        self.check_thresholds("coverage")

    def calc_train_set_size(self, y: Union[np.ndarray, pd.Series]):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            self.train_set_size = len(y.index)
        else:
            self.train_set_size = len(y)

    def calc_test_set_size(self, y: Union[np.ndarray, pd.Series]):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            self.test_set_size = len(y.index)
        else:
            self.test_set_size = len(y)


class RegressionRule(Rule):
    """Rule applied on continuous target data."""

    rule_index = Rule.rule_index + ["coverage", "zscore", "criterion", "std"]
    index = Rule.condition_index + rule_index
    attributes_from_train_set = Rule.attributes_from_train_set + ["prediction", "std", "sign"]
    attributes_from_test_set = ["zscore", "criterion"]

    def __init__(
        self,
        condition: Optional[Condition] = None,
        activation: Optional[Activation] = None,
    ):
        super().__init__(condition, activation)
        self._std = None
        self._sign = None

        # Inspection / Audit attributs
        self._time_calc_std = -1

    @property
    def std(self) -> float:
        return self._std

    @property
    def sign(self) -> str:
        return self._sign

    @property
    def criterion(self) -> float:
        # noinspection PyTypeChecker
        return self._criterion

    @property
    def zscore(self) -> float:
        return self._zscore

    @property
    def time_calc_prediction(self) -> float:
        return self._time_calc_prediction

    @property
    def time_calc_criterion(self) -> float:
        return self._time_calc_criterion

    @property
    def time_calc_std(self) -> float:
        return self._time_calc_std

    def calc_prediction(self, y: [np.ndarray, pd.Series], activation: Optional[Activation] = None):
        """Computes the mean of all activated points in target y and use it as prediction

        Parameters
        ----------
        y: [np.ndarray, pd.Series]
            The targets on which to evaluate the rule prediction, and possibly other criteria. Must be a 1-D np.ndarray
            or pd.Series
        activation: Optional[Activation]
            If specified, uses this activation instead of self.activation
        """
        t0 = time()
        if activation is None:
            activation = self._activation
            if activation is None:
                return None
        if not isinstance(activation, Activation):
            raise TypeError("Needs 'Activation' type activation vector")
        activation = activation.raw
        self._prediction = functions.conditional_mean(activation, y)
        self._time_calc_prediction = time() - t0
        self.check_thresholds("prediction")

    def calc_sign(self):
        if self._prediction is None:
            return
        if self._prediction < 0:
            self._sign = "-"
        else:
            self._sign = "+"

    def calc_std(self, y: Union[np.ndarray, pd.Series], activation: Optional[Activation] = None):
        """Computes the standard deviation of all activated points in target y

        Parameters
        ----------
        y: Union[np.ndarray, pd.Series]
            The targets on which to evaluate the rule prediction, and possibly other criteria. Must be a 1-D np.ndarray
            or pd.Series.
        activation: Optional[Activation]
            If specified, uses this activation instead of self.activation
        """
        t0 = time()
        if activation is None:
            activation = self._activation
            if activation is None:
                return None
        if not isinstance(activation, Activation):
            raise TypeError("Needs 'Activation' type activation vector")
        activation = activation.raw
        self._std = functions.conditional_std(activation, y)
        self._time_calc_std = time() - t0
        self.check_thresholds("std")

    def calc_criterion(
            self,
            y: Union[np.ndarray, pd.Series],
            activation: Optional[Activation] = None,
            **kwargs
    ):
        """
        Parameters
        ----------
        y: Union[np.ndarray, pd.Series]
            The targets on which to evaluate the rule prediction, and possibly other criteria. Must be a 1-D np.ndarray
            or pd.Series.
        activation: Optional[Activation]
            If specified, uses this activation instead of self.activation
        kwargs: dict
            Arguments for calc_regression_criterion
        """
        t0 = time()
        if activation is None:
            activation = self._activation
            if activation is None:
                return None
        if not isinstance(activation, Activation):
            raise TypeError("Needs 'Activation' type activation vector")
        activation = activation.raw
        self._criterion = functions.calc_regression_criterion(
            self.calc_prediction_vector(activation=activation), y, **kwargs
        )
        self._time_calc_criterion = time() - t0
        self.check_thresholds("criterion")

    def calc_zscore(self, y: np.ndarray, activation: Optional[Activation] = None, horizon: int = 1):
        t0 = time()
        if activation is None:
            activation = self._activation
            if activation is None:
                return
        if not isinstance(activation, Activation):
            raise TypeError("Needs 'Activation' type activation vector")
        self._zscore = calc_zscore_external(
            prediction=self.prediction, nones=activation.nones, y=y, horizon=horizon
        )
        self.check_thresholds("zscore")
        self._time_calc_zscore = time() - t0


class ClassificationRule(Rule):
    """Rule applied on discret target data."""

    rule_index = Rule.rule_index + ["coverage", "criterion"]
    index = Rule.condition_index + rule_index
    attributes_from_train_set = Rule.attributes_from_train_set + ["prediction"]
    attributes_from_test_set = ["criterion"]

    @property
    def prediction(self) -> Union[int, str, np.integer, np.float, None]:
        """Returns the rule prediction. If rule was fitted alone, the self._prediction should be a np.ndarray
        containing the probability of each class. In that case, the most probable class is returned. If the rule was
        fitted in a stacked fit, then the prediction is already the most probable class and it is just returned."""
        if self._prediction is not None:
            if isinstance(self._prediction, (float, int, str, np.integer, np.float)):
                return self._prediction
            prop = [p[1] for p in self._prediction]
            idx = prop.index(max(prop))
            return self._prediction[idx][0]
        else:
            return None

    @property
    def criterion(self) -> float:
        return self._criterion

    def calc_prediction(self, y: [np.ndarray, pd.Series]):
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
        self._prediction = functions.class_probabilities(self.activation, y)
        self._time_calc_prediction = time() - t0
        self.check_thresholds("prediction")

    def calc_criterion(
            self,
            y: Union[np.ndarray, pd.Series],
            activation: Optional[Activation] = None,
            **kwargs
    ):
        """
        Parameters
        ----------
        y: Union[np.ndarray, pd.Series]
            The targets on which to evaluate the rule prediction, and possibly other criteria. Must be a 1-D np.ndarray
            or pd.Series
        activation: Optional[Activation]
            If specified, uses this activation instead of self.activation
        kwargs: dict
            Arguments for calc_classification_criterion
        """
        t0 = time()
        if activation is None:
            activation = self._activation
            if activation is None:
                return None
        if not isinstance(activation, Activation):
            raise TypeError("Needs 'Activation' type activation vector")
        activation = activation.raw
        self._criterion = functions.calc_classification_criterion(activation, self.prediction, y, **kwargs)
        self._time_calc_criterion = time() - t0
        self.check_thresholds("criterion")
