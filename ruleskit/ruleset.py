import inspect
from abc import ABC
import ast
from copy import copy
from typing import List, Union, Tuple, Any, Optional
from collections import Counter
import numpy as np
import itertools
from collections import OrderedDict

from .condition import HyperrectangleCondition

from .rule import Rule, ClassificationRule, RegressionRule
from .activation import Activation
from .utils import rfunctions as functions
import logging
import warnings

from .utils.rfunctions import (
    calc_ruleset_prediction_weighted_regressor,
    calc_ruleset_prediction_weighted_classificator,
    calc_ruleset_prediction_equally_weighted_classificator,
    calc_ruleset_prediction_equally_weighted_regressor,
)

logger = logging.getLogger(__name__)


class RuleSet(ABC):

    """A set of rules"""

    NLINES = 5  # half how many rules to show in str(self)
    CHECK_DUPLICATED = False
    STACKED_FIT = False
    all_features_indexes = {}

    @staticmethod
    def check_duplicated_rules(rules, name_or_index: str = "index"):
        if name_or_index == "index":
            str_rules = [str(r.features_indexes) + str(r.bmins) + str(r.bmaxs) + str(r.prediction) for r in rules]
        else:
            str_rules = [str(r.features_names) + str(r.bmins) + str(r.bmaxs) + str(r.prediction) for r in rules]
        if len(set(str_rules)) < len(str_rules):
            duplicated = {}
            for r in str_rules:
                if r not in duplicated:
                    duplicated[r] = 0
                duplicated[r] += 1
            duplicated = [
                f"{r}: {duplicated[r]} (positions {[i for i, x in enumerate(str_rules) if x == r]})\n"
                f"   underlying rules : {[str(rules[i]) for i in [i for i, x in enumerate(str_rules) if x == r]]}"
                for r in duplicated
                if duplicated[r] > 1
            ]
            s = "\n -".join(duplicated)
            raise ValueError(f"There are {len(duplicated)} duplicated rules in your ruleset !\n {s}")

    def __init__(
        self,
        rules_list: Union[List[Rule], None] = None,
        remember_activation: bool = True,
        stack_activation: bool = False,
    ):
        """

        Parameters
        ----------
        rules_list: Union[List[Rule], None]
            The list of rules to start with. Can be None, since a RuleSet can be filled after its creation.
        remember_activation: bool
            The activation of the RuleSet is the logical OR of the activation of all its rules. It is only computed
            if remember_activation is True. (default value = True)
        stack_activation: bool
            If True, the RuleSet will keep in memory 2-D np.ndarray containing the activations of all its rules. This
            can take a lot of memory, but can save time if you apply numpy methods on this stacked vector instead of on
            each rule separately. (default value = False)
        """
        self._rules: List[Rule] = []
        self.features_names: List[str] = []
        self.features_indexes: List[int] = []
        self._activation: Optional[Activation] = None
        self._coverage = None  # in case Activation is not available
        self.prediction = None
        self.criterion = None
        self.stacked_activations: Optional[np.ndarray] = None
        self.remember_activation = remember_activation
        self.stack_activation = stack_activation
        self.rule_type = None
        if rules_list is not None:
            names_available = all([hasattr(r.condition, "features_names") for r in self])
            for rule in rules_list:
                if not isinstance(rule, Rule) and rule is not None:
                    raise TypeError(f"Some rules in given iterable were not of type 'Rule' but of type {type(rule)}")
                if rule is not None:
                    self.append(rule, update_activation=False)
            if self.remember_activation:
                self.compute_self_activation()
            if self.stack_activation:
                self.compute_stacked_activation()
            if names_available:
                self.features_names = list(set(itertools.chain(*[rule.features_names for rule in self])))
            self.set_features_indexes()
        if self.__class__.CHECK_DUPLICATED:
            self.check_duplicated_rules(self.rules, name_or_index="name" if len(self.features_names) > 0 else "index")

    # noinspection PyProtectedMember,PyTypeChecker
    def __iadd__(self, other: Union["RuleSet", Rule]):
        """Appends a rule or each rules of another RuleSet to self and updates activation vector and stacked activations
        if needed. Also updates features_indexes, and features_names if possible."""
        if isinstance(other, Rule):
            self._rules.append(other)
        else:
            self._rules += other._rules
        self.features_indexes = list(set(self.features_indexes + other.features_indexes))
        if hasattr(other, "features_names"):
            self.features_names = list(set(self.features_names + other.features_names))
        if self.remember_activation:
            self._update_activation(other)
        if self.stack_activation:
            self._update_stacked_activation(other)
        return self

    def __add__(self, other: Union["RuleSet", Rule]):
        """Returns the RuleSet resulting in appendind a rule or each rules of another RuleSet to self."""
        remember_activation = self.remember_activation
        stack_activation = self.stack_activation
        if isinstance(other, Rule):
            rules = self.rules + [other]
        else:
            remember_activation &= other.remember_activation
            stack_activation &= other.stack_activation
            rules = list(set(self.rules + other.rules))
        return self.__class__(rules, remember_activation=remember_activation, stack_activation=stack_activation)

    def __getattr__(self, item):
        """If item is not found in self, try to fetch it from its activation."""
        if item == "_activation":
            raise AttributeError(f"'RuleSet' object has no attribute '{item}'")

        if self._activation is not None and hasattr(self._activation, item):
            return getattr(self._activation, item)
        raise AttributeError(f"'RuleSet' object has no attribute '{item}'")

    def __len__(self):
        """The length of a RuleSet its the number of rules stored in it."""
        return len(self.rules)

    def __eq__(self, other: "RuleSet"):
        return set(self.rules) == set(other.rules)

    def __iter__(self):
        if hasattr(self, "_rules"):
            return self.rules.__iter__()
        else:
            return [].__iter__()

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self.rules)))
            return self.__class__([self.rules[i] for i in indices])
        return self.rules.__getitem__(key)

    def __str__(self):
        if len(self) < 2 * self.__class__.NLINES:
            return "\n".join([str(self[i]) for i in range(len(self))])
        else:
            return "\n".join(
                [str(self[i]) for i in range(self.__class__.NLINES)]
                + ["..."]
                + [str(self[i]) for i in range(len(self) - self.__class__.NLINES, len(self))]
            )

    def __hash__(self) -> hash:
        return hash(frozenset(self.to_hash))

    # noinspection PyProtectedMember
    def __contains__(self, other: Rule) -> bool:
        """A RuleSet contains another Rule if the two rule's conditions and predictions are the same"""
        name_or_index = "name" if len(self.features_names) > 0 else "index"

        if name_or_index == "index":
            str_rules = [str(r.features_indexes) + str(r.bmins) + str(r.bmaxs) + str(r.prediction) for r in self]
            str_rule = str(other.features_indexes) + str(other.bmins) + str(other.bmaxs) + str(other.prediction)
        else:
            str_rules = [str(r.features_names) + str(r.bmins) + str(r.bmaxs) + str(r.prediction) for r in self]
            str_rule = str(other.features_names) + str(other.bmins) + str(other.bmaxs) + str(other.prediction)

        return str_rule in str_rules

    @property
    def rules(self) -> List[Rule]:
        return self._rules

    @rules.setter
    def rules(self, rules: Union[List[Rule], None]):
        ruleset = RuleSet(rules, remember_activation=self.remember_activation, stack_activation=self.stack_activation)
        self._rules = ruleset._rules
        self.features_names = ruleset.features_names
        self.features_indexes = ruleset.features_indexes
        self.stacked_activations = ruleset.stacked_activations
        self._activation = ruleset._activation

    @property
    def to_hash(self) -> Tuple[str]:
        if len(self) == 0:
            return "rs",
        to_hash = ("rs",)
        for r in self:
            rule_hash = r.to_hash[1:]
            to_hash += rule_hash
        return to_hash

    @property
    def activation_available(self) -> bool:
        """Returns True if the RuleSet has an activation vector, and if this Activation's object data is available."""
        if self._activation is None:
            return False
        if self._activation.data_format == "file":
            return self._activation.data.is_file()
        else:
            return self._activation.data is not None

    @property
    def stacked_activations_available(self) -> bool:
        """Returns True is the RuleSet has its rules' stacked activations."""
        if self.stack_activation is None:
            return False
        return True

    @property
    def activation(self) -> Union[None, np.ndarray]:
        """Returns the Activation vector's data in a form of a 1-D np.ndarray, or None if not available.

        Returns:
        --------
        Union[None, np.ndarray]
            of the form [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, ...]
        """
        if self._activation:
            return self._activation.raw
        return None

    @property
    def ruleset_coverage(self) -> float:
        """Coverage is the fraction of points equal to 1 in the activation vector"""
        if not self.activation_available:
            return self._coverage
        else:
            return self._activation.coverage

    # noinspection PyProtectedMember,PyTypeChecker
    def _update_activation(self, other: Union[Rule, "RuleSet"]):
        """Updates the activation vector of the RuleSet with the activation vector of a new Rule or RuleSet."""
        if other.activation_available:
            if self._activation is None:
                self._activation = Activation(other.activation, to_file=Rule.LOCAL_ACTIVATION)
            else:
                self._activation = self._activation | other._activation

    # noinspection PyProtectedMember,PyTypeChecker
    def _update_stacked_activation(self, other: Union[Rule, "RuleSet"]):
        """Updates the stacked activation vectors of the RuleSet with the activation vector of a new Rule or
        the stacked activation vectors of another RuleSet."""
        if other.activation_available:
            try:
                import pandas as pd
            except ImportError:
                raise ImportError("RuleSet's stacked activations requires pandas. Please run\npip install pandas")

            if self.stacked_activations is None:
                if isinstance(other, Rule):
                    self.stacked_activations = pd.DataFrame(
                        data=np.array(other.activation).T, columns=[str(other.condition)]
                    )
                else:
                    self.stacked_activations = other.stacked_activations
            else:
                if isinstance(other, Rule):
                    self.stacked_activations[str(other.condition)] = other.activation
                else:
                    self.stacked_activations = pd.concat([self.stacked_activations, other.stacked_activations], axis=1)

    def set_features_indexes(self):
        if len(self.__class__.all_features_indexes) > 0:
            self.features_indexes = [self.__class__.all_features_indexes[f] for f in self.features_names]
            for r in self._rules:
                # noinspection PyProtectedMember
                r._condition._features_indexes = [self.__class__.all_features_indexes[f] for f in r.features_names]
        else:
            list(set(itertools.chain(*[rule.features_indexes for rule in self])))

    # noinspection PyProtectedMember,PyUnresolvedReferences
    def fit(
        self,
        y: Union[np.ndarray, "pd.Series"],
        xs: Optional[Union["pd.DataFrame", np.ndarray]] = None,
        **kwargs,
    ) -> List[Rule]:
        """Fits the ruleset on y and xs to produce the rules' activation vectors and attributes relevant to train set.

        Parameters
        ----------
        y: Union[np.ndarray, "pd.Series"]
        xs: Optional[Union["pd.DataFrame", np.ndarray]]
        kwargs

        Returns
        -------
        List[Rule]
            List of rules that were excluded from the ruleset after fitting because they were 'bad'
        """

        def launch_method(method, **kw):
            expected_args = list(inspect.signature(method).parameters)
            kw = {item: kw[item] for item in kw if item in expected_args}
            return method(**kw)

        if len(self) == 0:
            logger.debug("Ruleset is empty. Nothing to fit.")
            return []

        if all([r._fitted for r in self]) and xs is None:
            return []

        if self.__class__.STACKED_FIT:
            try:
                import pandas as pd
            except ImportError:
                raise ImportError("RuleSet's stacked fit requies pandas. Please run\npip install pandas")

            clean_activation = False
            # Activation must always be computed from train set
            if xs is not None:
                clean_activation = not self.stack_activation
                self.stack_activation = True
                self.calc_activation(xs)
                self.stack_activation = not clean_activation
            elif self.stacked_activations is None:
                clean_activation = not self.stack_activation
                self.stack_activation = True
                self.compute_stacked_activation()
                self.stack_activation = not clean_activation

            # noinspection PyUnresolvedReferences
            if isinstance(y, np.ndarray):
                if not len(self.stacked_activations.index) == y.shape[0]:
                    raise IndexError(
                        "Stacked activation and y have different number of rows. Use pd.Series for y to"
                        " reindex stacked activations automatically."
                    )
            else:
                self.stacked_activations.index = y.index

            computed_attrs = {}
            for attr in self.rule_type.attributes_from_train_set:
                if attr == "activation":
                    raise ValueError("'activation' can not be specified in 'attributes_from_train_set'")
                res = launch_method(getattr(self, f"calc_{attr}s"), y=y, xs=xs, **computed_attrs, **kwargs)
                # If res is None, assumes the calc_(attr)s method set the attribute of the rules itself
                if res is not None:
                    computed_attrs[f"{attr}s"] = res
            to_drop = []

            if clean_activation:
                self.del_stacked_activations()
            else:
                if isinstance(y, np.ndarray):
                    self.stacked_activations.index = pd.RangeIndex(y.shape[0])
                else:
                    self.stacked_activations.index = pd.RangeIndex(len(y.index))

            for ir in range(len(self)):
                self._rules[ir]._fitted = True
                for attr in computed_attrs:
                    setattr(self._rules[ir], f"_{attr[:-1]}", computed_attrs[attr].iloc[ir])
                    self._rules[ir].check_thresholds(attr[:-1])
                    if not self._rules[ir].good:
                        to_drop.append(self._rules[ir])
        else:
            [r.fit(xs=xs, y=y, **kwargs) for r in self]
            to_drop = [r for r in self if not r.good]

        if len(to_drop) > 0:
            rules = [r for r in self.rules if r not in to_drop]
            self._rules = []
            if self._activation is not None:
                self._activation.clear()
            self.del_stacked_activations()
            for r in rules:
                self.append(r, update_activation=False)

            # Recompute activation now that bad rules have been droped
            if self.remember_activation:
                self._activation = None
                self.compute_self_activation()
            if self.stack_activation:
                self.stacked_activations = None
                self.compute_stacked_activation()
        # If not bad rules were dropped and stacked fit was not used, still compute self.activation since it has not
        # been done  (needed to set self.coverage), but not stacked (useless)
        elif not self.__class__.STACKED_FIT:
            if self.remember_activation and (self._activation is None or xs is not None):
                self._activation = None
                self.compute_self_activation()
            if self.stack_activation and (self.stacked_activations is None or xs is not None):
                self.stacked_activations = None
                self.compute_stacked_activation()

        return to_drop

    # noinspection PyProtectedMember,PyUnresolvedReferences
    def eval(
        self,
        y: Union[np.ndarray, "pd.Series"],
        xs: Optional[Union["pd.DataFrame", np.ndarray]] = None,
        keep_new_activations: bool = False,
        **kwargs,
    ) -> List[Rule]:
        """Evaluate the ruleset on y and xs to produce the rules' attributes relevant to the test set. Will recompute
        the ruleset's stacked activation vector if using stakced fit.

        Parameters
        ----------
        y: Union[np.ndarray, "pd.Series"]
        xs: Optional[Union["pd.DataFrame", np.ndarray]]
        keep_new_activations: bool
            If True, activation vectgor and stacked activation vectors will be kept as ruleset's attribute and replace
            the ones computed using the train set, if any. Will also change the activation vectors of the rules.
        kwargs


        Returns
        -------
        List[Rule]
            List of rules that were excluded from the ruleset after fitting because they were 'bad'
        """

        def launch_method(method, **kw):
            expected_args = list(inspect.signature(method).parameters)
            kw = {item: kw[item] for item in kw if item in expected_args}
            return method(**kw)

        if len(self) == 0:
            logger.debug("Ruleset is empty. Nothing to fit.")
            return []

        if not all([r._fitted for r in self]):
            raise ValueError("Not all rules of the ruleset were fitted. Please do so before calling ruleset.eval")

        if self.__class__.STACKED_FIT:
            try:
                import pandas as pd
            except ImportError:
                raise ImportError("RuleSet's stacked fit requies pandas. Please run\npip install pandas")

            clean_activation = False

            # If test set is given, compute the activation used to evaluate test relative attributes.
            # Not the same as self.stacked_activation, computed from the train set.
            # Else, we evaluate on the train set, so we use self.stacked_actviation
            if xs is not None:
                if keep_new_activations:
                    clean_activation = not self.stack_activation
                    self.stack_activation = True
                    self.calc_activation(xs=xs)  # Will reset rules' activation vectors too
                    self.stack_activation = not clean_activation
                    stacked_activation = self.stacked_activations
                    activation = self._activation
                else:
                    stacked_activation = self.evaluate_stacked_activation(xs)
                    activation = self.evaluate_self_activation(xs)
            else:
                # If self.stacked_activation is None, compute it from the rules' activations. They must be available.
                if self.stacked_activations is None:
                    # Else, will compute the stacked activation from the current rules activations
                    clean_activation = not self.stack_activation
                    self.stack_activation = True
                    self.compute_stacked_activation()
                    self.compute_self_activation()
                    if self.stacked_activations is None:
                        raise ValueError("Rules activations must have been computed previously.")
                    self.stack_activation = not clean_activation
                stacked_activation = self.stacked_activations
                activation = self._activation

            # noinspection PyUnresolvedReferences
            if isinstance(y, np.ndarray):
                if not len(stacked_activation.index) == y.shape[0]:
                    raise IndexError("Stacked activation and y have different number of rows.")
            else:
                if not len(stacked_activation.index) == len(y.index):
                    raise IndexError("Stacked activation and y have different number of rows.")
                stacked_activation.index = y.index

            if "prediction" in self.rule_type.attributes_from_train_set:
                computed_attrs = {"prediction": pd.Series({str(r.condition): r.prediction for r in self})}
            else:
                computed_attrs = {}
            for attr in self.rule_type.attributes_from_test_set:
                if attr == "activation":
                    raise ValueError("'activation' can not be specified in 'attributes_from_train_set'")
                computed_attrs[f"{attr}s"] = launch_method(
                    getattr(self, f"calc_{attr}s"),
                    y=y,
                    xs=xs,
                    stacked_activation=stacked_activation,
                    activation=activation,
                    **computed_attrs,
                    **kwargs,
                )
            to_drop = []

            if clean_activation:
                self.del_stacked_activations()
            else:
                if isinstance(y, np.ndarray):
                    stacked_activation.index = pd.RangeIndex(y.shape[0])
                else:
                    stacked_activation.index = pd.RangeIndex(len(y.index))

            for ir in range(len(self)):
                self._rules[ir]._evaluated = True
                for attr in computed_attrs:
                    setattr(self._rules[ir], f"_{attr[:-1]}", computed_attrs[attr].iloc[ir])
                    self._rules[ir].check_thresholds(attr[:-1])
                    if not self._rules[ir].good:
                        to_drop.append(self._rules[ir])
        else:
            [r.eval(xs=xs, y=y, recompute_activation=keep_new_activations, **kwargs) for r in self]
            to_drop = [r for r in self if not r.good]

        if len(to_drop) > 0:
            rules = [r for r in self.rules if r not in to_drop]
            self._rules = []
            if self._activation is not None:
                self._activation.clear()
            self.del_stacked_activations()
            for r in rules:
                self.append(r, update_activation=False)

            # Recompute activation now that bad rules have been droped
            if self.remember_activation and (self._activation is None or (xs is not None and keep_new_activations)):
                self._activation = None
                self.compute_self_activation()
            if self.stack_activation and (
                    self.stacked_activations is None or (xs is not None and keep_new_activations)
            ):
                self.stacked_activations = None
                self.compute_stacked_activation()
        # If not bad rules were dropped and stacked fit was not used, still compute self.activation since it has not
        # been done
        elif not self.__class__.STACKED_FIT:
            if self.remember_activation and (self._activation is None or (xs is not None and keep_new_activations)):
                self._activation = None
                self.compute_self_activation()
            if self.stack_activation and (self._activation is None or (xs is not None and keep_new_activations)):
                self._activation = None
                self.compute_stacked_activation()

        return to_drop

    def append(self, rule: Rule, update_activation: bool = True):
        """Appends a new rule to self. The updates of activation vector and the stacked activation vectors can be
        blocked by specifying update_activation=False. Otherwise, will use self.remember_activation and
        self.stack_activation to determine if the updates should be done or not."""
        if not isinstance(rule, Rule):
            raise TypeError(f"RuleSet's append method expects a Rule object, got {type(rule)}")
        if self.rule_type is None:
            self.rule_type = type(rule)
        else:
            if not isinstance(rule, self.rule_type):
                raise TypeError(
                    f"Ruleset previously had rules of type {self.rule_type}, so can not add rule of type {type(rule)}"
                )
        remember_activation = self.remember_activation
        stack_activation = self.stack_activation
        if not update_activation:
            self.remember_activation = False
            self.stack_activation = False
        self.__iadd__(rule)
        self.remember_activation = remember_activation
        self.stack_activation = stack_activation

    def sort(self, criterion: str = None, reverse: bool = False):
        """Sorts the RuleSet.

        * If criterion is not speficied:
            Will sort the rules according to :
                1. The number of features they talk about
                2. For a same number of features (sorted in alphabetical order, or index if names are not available,
                    optionally reversed), the bmins and bmaxs of the rules
        * If criterion is specified, it must be an float or interger attribute of rule, condition or activation. Then
            sorts according to this criterion, optionally reversed.
        """
        if len(self) == 0:
            return

        if criterion is None or criterion == "":
            if not (hasattr(self[0].condition, "bmins") and hasattr(self[0].condition, "bmaxs")):
                return
            # The set of all the features the RuleSet talks about
            which = "index"
            if len(self.features_names) > 0:
                which = "name"
                fnames_or_indexes = list(set([str(r.features_names) for r in self]))
            else:
                fnames_or_indexes = list(set([str(r.features_indexes) for r in self]))
            dict_names = {}
            lmax = 1
            for f in fnames_or_indexes:
                l_ = len(ast.literal_eval(f))
                if l_ > lmax:
                    lmax = l_
                if l_ not in dict_names:
                    dict_names[l_] = []
                dict_names[l_].append(f)
            for l_ in dict_names:
                dict_names[l_].sort(reverse=reverse)
            fnames_or_indexes = []
            for l_ in range(1, lmax + 1):
                if l_ in dict_names:
                    fnames_or_indexes += dict_names[l_]

            rules_by_fnames = OrderedDict({f: [] for f in fnames_or_indexes})
            for rule in self:
                # noinspection PyUnresolvedReferences
                if which == "name":
                    v = str(rule.features_names)
                else:
                    v = str(rule.features_indexes)
                rules_by_fnames[v].append(rule)
            rules_by_fnames = {
                n: sorted(rules_by_fnames[n], key=lambda x: x.condition.bmins + x.condition.bmaxs)
                for n in rules_by_fnames
            }
            self._rules = []
            for n in rules_by_fnames:
                self._rules += rules_by_fnames[n]
        elif hasattr(self[0], criterion):
            self._rules = sorted(self, key=lambda x: getattr(x, criterion), reverse=reverse)
        else:
            raise ValueError(f"Can not sort RuleSet according to criterion {criterion}")
        if self.stack_activation:
            self.stacked_activations = self.stacked_activations[[str(r.condition) for r in self]]

    # noinspection PyProtectedMember,PyUnresolvedReferences
    def evaluate_self_activation(self, xs: Optional[Union[np.ndarray, "pd.DataFrame"]] = None):
        """Computes the activation vector of self from its rules, using time-efficient Activation.multi_logical_or."""
        if len(self) == 0:
            return Activation(np.array([]))
        if xs is not None:
            activations = [r.evaluate_activation(xs) for r in self]
            activations_available = True
        else:
            activations = [r._activation for r in self]
            activations_available = all([r.activation_available for r in self])
        if activations_available:
            if len(self) == 1:
                return Activation(activations[0].raw, optimize=activations[0].optimize, to_file=activations[0].to_file)
            # noinspection PyProtectedMember
            try:
                return Activation.multi_logical_or(activations)
            except MemoryError:
                act = Activation(activations[0], optimize=activations[0].optimize, to_file=activations[0].to_file)
                for a in activations:
                    act = act or a
                return act

    # noinspection PyUnresolvedReferences
    def compute_self_activation(self, xs: Optional[Union[np.ndarray, "pd.DataFrame"]] = None):
        """Computes the activation vector of self from its rules. If xs is specified, uses it to remake the
        rules' activation vectors, but do not set them as the 'activation' attributes of the rules"""
        self._activation = self.evaluate_self_activation(xs=xs)

    # noinspection PyUnresolvedReferences
    def evaluate_stacked_activation(self, xs: Optional[Union[np.ndarray, "pd.DataFrame"]] = None):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's stacked activations requies pandas. Please run\npip install pandas")

        if len(self) == 0:
            return pd.DataFrame(dtype=int)
        if xs is not None:
            return pd.DataFrame({str(r.condition): r.evaluate_activation(xs).raw for r in self})
        activations_available = all([r.activation_available for r in self])
        if activations_available:
            # noinspection PyProtectedMember
            return pd.DataFrame({str(r.condition): r.activation for r in self})

    # noinspection PyUnresolvedReferences
    def compute_stacked_activation(self, xs: Optional[Union[np.ndarray, "pd.DataFrame"]] = None):
        """Computes the stacked activation vectors of self from its rules. If xs is specified, uses it to remake the
        rules' activation vectors, but do not set them as the 'activation' attributes of the rules"""
        self.stacked_activations = self.evaluate_stacked_activation(xs=xs)

    def del_activations(self):
        """Deletes the data, but not the relevent attributes, of the activation vector or each rules in self."""
        for r in self:
            r.del_activation()

    def del_activation(self):
        """Deletes the activation vector's data of self, but not the object itself, so any computed attribute remains
        available"""
        if hasattr(self, "_activation") and self._activation is not None:
            self._activation.delete()

    def del_stacked_activations(self):
        """Deletes stacked activation vectors of self. Set it to None."""
        if hasattr(self, "stacked_activations") and self.stacked_activations is not None:
            del self.stacked_activations
            self.stacked_activations = None

    # noinspection PyUnresolvedReferences
    def evaluate(self, xs: Union["pd.DataFrame", np.ndarray]) -> Activation:
        """Computes and returns the activation vector from an array of features.

        Parameters
        ----------
        xs: Union[pd.DataFrame, np.ndarray]
            The features on which the check whether the rule is activated or not. Must be a 2-D np.ndarray
            or pd.DataFrame.

        Returns
        -------
        Activation
        """
        if len(self) == 0:
            raise ValueError("Can not use evaluate : The ruleset is empty!")
        activations = [rule.evaluate_activation(xs) for rule in self.rules]
        return Activation.multi_logical_or(activations)

    # noinspection PyUnresolvedReferences
    def calc_activation(self, xs: Union[np.ndarray, "pd.DataFrame"]):
        """Uses input xs features data to compute the activation vector of all rules in self, and updates self's
        activation if self.remember_activation is True and stacked activation if self.stack_activation is True

        Parameters
        ----------
        xs: Union[np.ndarray, "pd.DataFrame"]
        """
        if len(self) == 0:
            raise ValueError("Can not use calc_activation : The ruleset is empty!")
        [rule.calc_activation(xs) for rule in self.rules]

        if self.remember_activation:
            self._activation = None
            self.compute_self_activation()
        if self.stack_activation:
            self.stacked_activations = None
            self.compute_stacked_activation()

    def get_features_count(self) -> List[Tuple[Any, int]]:
        """
        Get a counter of all different features in the ruleset. If names are not available, will use indexes.

        Returns:
        --------
        count : List[Tuple[Any, int]]
            Counter of all different features in the ruleset
        """
        # noinspection PyUnresolvedReferences
        if len(self) == 0:
            return []
        if len(self.features_names) > 0:
            var_in = list(itertools.chain(*[rule.features_names for rule in self]))
        else:
            var_in = list(itertools.chain(*[rule.feautres_indexes for rule in self]))
        count = Counter(var_in)

        count = count.most_common()
        return count

    def load(self, path, **kwargs):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("'pandas' is needed to read a RuleSet from a file. Please run\npip install pandas")

        if hasattr(path, "read"):
            rules = path.read(**kwargs)
        else:
            rules = pd.read_csv(path, **kwargs)
        if rules.empty:
            self._rules = []
        else:
            self._rules = [self.series_to_rule(rules.loc[r]) for r in rules.index]

        if len(self._rules) > 0:
            self.rule_type = type(self._rules[0])
        if self.remember_activation:
            self.compute_self_activation()
        if self.stack_activation:
            self.compute_stacked_activation()
        self.features_names = list(set(traverse([rule.features_names for rule in self])))

    def save(self, path):

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("'pandas' is needed to save a RuleSet to a file. Please run\npip install pandas")

        if len(self) == 0:
            if hasattr(path, "write"):
                path.write(pd.DataFrame())
            else:
                pd.DataFrame().to_csv(path)
            return
        if self.rule_type is None:
            raise TypeError("Rule type unset : can not save Ruleset")
        idx = copy(self.rule_type.index)

        dfs = [
            self.rule_to_series(
                (i, r),
                index=idx,
            )
            for i, r in enumerate(self.rules)
        ]
        if len(dfs) > 0:
            df = pd.concat(dfs, axis=1).T
        else:
            df = pd.DataFrame(columns=idx)
        if hasattr(path, "write"):
            path.write(df)
        else:
            df.to_csv(path)

    # noinspection PyProtectedMember,PyUnresolvedReferences
    @staticmethod
    def rule_to_series(irule: Tuple[int, Rule], index: list) -> "pd.Series":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("'pandas' is needed to read a RuleSet from a file. Please run\npip install pandas")
        i = irule[0]
        rule = irule[1]
        if hasattr(rule, "sign"):
            name = f"R_{i}({len(rule)}){rule.sign}"
        else:
            name = f"R_{i}({len(rule)})"
        sr = pd.Series(data=[str(getattr(rule, ind)) for ind in index], name=name, index=index, dtype=str)
        return sr

    # noinspection PyUnresolvedReferences
    @staticmethod
    def series_to_rule(srule: "pd.Series") -> Rule:

        for ind in srule.index:
            if ind == "Unnamed: 0":
                continue
            if ind not in Rule.index and ind not in RegressionRule.index and ind not in ClassificationRule.index:
                raise IndexError(f"Invalid rule attribute '{ind}'")

        if "std" in srule.index:
            rule = RegressionRule()
        elif "criterion" in srule.index:
            rule = ClassificationRule()
        else:
            rule = Rule()
        rule_idx = copy(type(rule).rule_index)
        condition_index = {c: None for c in type(rule).condition_index}

        for rule_ind in srule.index:
            str_value = str(srule[rule_ind])
            if rule_ind in condition_index:
                condition_index[rule_ind] = ast.literal_eval(str_value)
                # setattr(condition, rule_ind, ast.literal_eval(str_value))
            elif rule_ind in rule_idx:
                if rule_ind == "activation":
                    setattr(rule, f"_{rule_ind}", Activation(str_value))
                elif str_value == "nan":
                    setattr(rule, f"_{rule_ind}", np.nan)
                elif rule_ind == "sign":
                    setattr(rule, f"_{rule_ind}", str_value)
                else:
                    if rule_ind == "prediction":  # Prediction can be a str in case of classification
                        try:
                            setattr(rule, f"_{rule_ind}", ast.literal_eval(str_value))
                        except ValueError:
                            setattr(rule, f"_{rule_ind}", str_value)
                    else:
                        setattr(rule, f"_{rule_ind}", ast.literal_eval(str_value))
            else:
                continue

        rule._condition = HyperrectangleCondition(**condition_index)
        if hasattr(rule, rule.__class__.fitted_if_has) and getattr(rule, rule.__class__.fitted_if_has) is not None:
            rule._fitted = True
        return rule

    # noinspection PyUnresolvedReferences
    def calc_predictions(
        self, y: [np.ndarray, "pd.Series"], stacked_activation: Optional["pd.DataFrame"] = None
    ) -> "pd.Series":
        """
        Will compute the prediction of each rule in the ruleset

        This uses the ruleset's stacked activation, so do not use it with too large rulesets otherwise your memory might
        not suffice.

        Parameters
        ----------
        y: [np.ndarray, pd.Series]
            The targets on which to evaluate the rules predictions, and possibly other criteria. Must be a 1-D
            np.ndarray or pd.Series.
        stacked_activation: Optional[pd.DataFrame)
            If specified, uses this activation instead of self.activation

        Returns
        -------
        "pd.Series"
            A pd.Series with the rules as index and the values being the predictions (most probable class in case of
            classification).
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's calc_predictions requires pandas. Please run\npip install pandas")

        if stacked_activation is None:
            stacked_activation = self.stacked_activations
            if stacked_activation is None:
                raise ValueError("Stacked activation vectors are needed")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if len(self) == 0:
                if self.rule_type is None:
                    return pd.DataFrame(dtype=int)
                elif issubclass(self.rule_type, ClassificationRule):
                    return pd.DataFrame(dtype=int)
                elif issubclass(self.rule_type, RegressionRule):
                    return pd.Series(dtype=int)
                else:
                    raise TypeError(f"Unexpected rule type '{self.rule_type}'")

            if issubclass(self.rule_type, ClassificationRule):
                class_probabilities = functions.class_probabilities(stacked_activation, y)
                maxs = class_probabilities.max()
                return class_probabilities[class_probabilities == maxs].apply(
                    lambda x: x.dropna().sort_index().index[0]
                )
            elif issubclass(self.rule_type, RegressionRule):
                return functions.conditional_mean(stacked_activation, y)
            else:
                raise TypeError(f"Unexpected rule type '{self.rule_type}'")

    # noinspection PyUnresolvedReferences
    def calc_stds(
        self, y: [np.ndarray, "pd.Series"], stacked_activation: Optional["pd.DataFrame"] = None
    ) -> "pd.Series":
        """
        Will compute the std of each rule in the ruleset

        This uses the ruleset's stacked activation, so do not use it with too large rulesets otherwise your memory might
        not suffice.

        Parameters
        ----------
        y: [np.ndarray, pd.Series]
          The targets on which to evaluate the rules predictions, and possibly other criteria. Must be a 1-D np.ndarray
          or pd.Series.
        stacked_activation: Optional[pd.DataFrame]
            If specified, uses this activation instead of self.activation

        Returns
        -------
        "pd.Series"
            A pd.Series with the rules as index and the values being the std
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's calc_predictions requires pandas. Please run\npip install pandas")

        if issubclass(self.rule_type, ClassificationRule):
            raise TypeError(f"'std' can not be computed for '{self.rule_type}'")

        if stacked_activation is None:
            stacked_activation = self.stacked_activations
            if stacked_activation is None:
                raise ValueError("Stacked activation vectors are needed")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if len(self) == 0:
                if self.rule_type is None:
                    return pd.DataFrame()
                elif issubclass(self.rule_type, RegressionRule):
                    return pd.Series(dtype=int)
                else:
                    raise TypeError(f"Unexpected rule type '{self.rule_type}'")

            if issubclass(self.rule_type, RegressionRule):
                return functions.conditional_std(stacked_activation, y)
            else:
                raise TypeError(f"Unexpected rule type '{self.rule_type}'")

    def calc_signs(self):
        [r.calc_sign() for r in self]

    # noinspection PyUnresolvedReferences
    def calc_criterions(
        self,
        y: Union[np.ndarray, "pd.Series"],
        prediction: Optional["pd.Series"] = None,
        stacked_activation: Optional["pd.DataFrame"] = None,
        **kwargs,
    ) -> "pd.Series":
        """
        Will compute the criterion of each rule in the ruleset

        This uses the ruleset's stacked activation, so do not use it with too large rulesets otherwise your memory might
        not suffice.

        Parameters
        ----------
        y: [np.ndarray, pd.Series]
            The targets on which to evaluate the rules predictions, and possibly other criteria. Must be a 1-D
            np.ndarray or pd.Series.
        prediction: Optional["pd.Series"]
            Prediction of each rules. If None, will call self.calc_predictions(y)
        stacked_activation: Optional[pd.DataFrame]
            If specified, uses this activation instead of self.stacked_activations
        kwargs

        Returns
        -------
        pd.Series
            Criterion values a set of rules (pd.Series)

        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's calc_criterions requires pandas. Please run\npip install pandas")

        if stacked_activation is None:
            stacked_activation = self.stacked_activations
            if stacked_activation is None:
                raise ValueError("Stacked activation vectors are needed")

        if prediction is None:
            prediction = self.calc_predictions(y=y)
            """unique prediction of each rules in a pd.Series"""

        if self.rule_type is None:
            return pd.Series(dtype=int)
        if issubclass(self.rule_type, ClassificationRule):
            return functions.calc_classification_criterion(stacked_activation, prediction, y, **kwargs)
        elif issubclass(self.rule_type, RegressionRule):
            return functions.calc_regression_criterion(stacked_activation.replace(0, np.nan) * prediction, y, **kwargs)
        else:
            raise TypeError(f"Unexpected rule type '{self.rule_type}'")

    # noinspection PyUnresolvedReferences
    def calc_prediction(
        self, y: Union[np.ndarray, "pd.Series"], weights: Optional[Union["pd.Series", str]] = None
    ) -> "pd.Series":
        """Computes the prediction vector of an entier ruleset. Predictions of rules must have been computed beforehand.

        This uses the ruleset's stacked activation, so do not use it with too large rulesets otherwise your memory might
        not suffice.

        Parameters
        ----------
        y: [np.ndarray, pd.Series]
            The targets on which to evaluate the ruleset predictions. Must be a 1-D np.ndarray or pd.Series.
        weights: Optional[Union["pd.Series", str]]
            Optional weights. If is a pd.Series, expected the index to be the rules names. If is a str, a pd.Series
            will be constructed by fetching each rules' attribute named after the given string
            (ex: it can be 'criterion')

        Returns
        -------
        pd.Series
            The prediction vector of the ruleset
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's calc_prediction requires pandas. Please run\npip install pandas")
        if len(self) == 0:
            return pd.Series(dtype=int)
        if self.rule_type is None:
            return pd.Series(dtype=int)
        if pd.api.types.is_string_dtype(y.dtype):
            prediction_vectors = self.stacked_activations.replace(0, np.nan).replace(1.0, "") + self.calc_predictions(y)
        else:
            prediction_vectors = self.stacked_activations.replace(0, np.nan) * self.calc_predictions(y)
        if prediction_vectors.empty:
            return prediction_vectors
        if weights is not None:
            if isinstance(weights, str):
                weights = pd.Series({str(r.condition): getattr(r, weights) for r in self})
            weights = self.stacked_activations.replace(0, np.nan) * weights
            if issubclass(self.rule_type, RegressionRule):
                return calc_ruleset_prediction_weighted_regressor(
                    prediction_vectors=prediction_vectors, weights=weights
                )
            elif issubclass(self.rule_type, ClassificationRule):
                return calc_ruleset_prediction_weighted_classificator(
                    prediction_vectors=prediction_vectors, weights=weights
                )
            else:
                raise TypeError(f"Unexpected rule type '{self.rule_type}'")
        else:
            if issubclass(self.rule_type, RegressionRule):
                return calc_ruleset_prediction_equally_weighted_regressor(prediction_vectors=prediction_vectors)
            elif issubclass(self.rule_type, ClassificationRule):
                return calc_ruleset_prediction_equally_weighted_classificator(prediction_vectors=prediction_vectors)
            else:
                raise TypeError(f"Unexpected rule type '{self.rule_type}'")

    # noinspection PyUnresolvedReferences
    def calc_criterion(
        self,
        y: Union[np.ndarray, "pd.Series"],
        weights: Optional[Union["pd.Series", str]] = None,
        predictions_vector: Optional["pd.Series"] = None,
        **kwargs,
    ) -> float:
        """Computes the criterion vector of an entier ruleset. Criterions of rules must have been computed  beforehand.

        This uses the ruleset's stacked activation, so do not use it with too large rulesets otherwise your memory might
        not suffice.

        Parameters
        ----------
        y: [np.ndarray, pd.Series]
            The targets on which to evaluate the ruleset criterions. Must be a 1-D np.ndarray or pd.Series.
        weights: Optional[Union["pd.Series", str]]
            Optional weights. If is a pd.Series, expected the index to be the rules names. If is a str, a pd.Series
            will be constructed by fetching each rules' attribute named after the given string
            (ex: it can be 'criterion').
            Useless if predictions_vector is specified.
        predictions_vector: Optional["pd.Series"]
            The vector of predictions of the ruleset. If not specified, is computed using self.calc_prediction
        kwargs

        Returns
        -------
        float
            The criterion of the ruleset
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("RuleSet's calc_criterion requires pandas. Please run\npip install pandas")
        if len(self) == 0:
            return np.nan
        if self.rule_type is None:
            return np.nan
        if predictions_vector is None:
            predictions_vector = self.calc_prediction(y=y, weights=weights)
        if predictions_vector.empty:
            return predictions_vector
        if issubclass(self.rule_type, ClassificationRule):
            if self._activation is None:
                self.compute_self_activation()
            return functions.calc_classification_criterion(self.activation, predictions_vector, y, **kwargs)
        elif issubclass(self.rule_type, RegressionRule):
            # noinspection PyTypeChecker
            return functions.calc_regression_criterion(predictions_vector.values, y, **kwargs)
        else:
            raise TypeError(f"Unexpected rule type '{self.rule_type}'")


def traverse(o, tree_types=(list, tuple, RuleSet)):
    """Yields each elementary elements from nested list or tuple

    Example
    -------
    >>> list(traverse([[[1, 2], 3,], [4, 5]]))
    [1, 2, 3, 4, 5]
    """
    if isinstance(o, tree_types):
        for value in o:
            for subvalue in traverse(value, tree_types):
                yield subvalue
    else:
        yield o
