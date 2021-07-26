from abc import ABC
import operator
import pandas as pd
from typing import List, Union
from functools import reduce
from collections import Counter
import numpy as np
from collections import OrderedDict
from .rule import Rule
from .condition import HyperrectangleCondition
from .activation import Activation


class RuleSet(ABC):

    NLINES = 5

    def __init__(
        self,
        rules_list: Union[List[Rule], None] = None,
        remember_activation: bool = True,
        stack_activation: bool = False,
    ):
        self._rules = []
        self._activation = None
        self.stacked_activations = None
        self.remember_activation = remember_activation
        self.stack_activation = stack_activation
        if rules_list is not None:
            for rule in rules_list:
                if not isinstance(rule, Rule) and rule is not None:
                    raise TypeError(f"Some rules in given iterable were not of type 'Rule' but of type {type(rule)}")
                if rule is not None:
                    self.append(rule, update_activation=False)
                if self.remember_activation:
                    self.compute_self_activation()
                if self.stack_activation:
                    self.compute_stacked_activation()

    # noinspection PyProtectedMember,PyTypeChecker
    def __iadd__(self, other: Union["RuleSet", Rule]):
        if isinstance(other, Rule):
            self._rules.append(other)
        else:
            self._rules += other._rules
        if self.remember_activation:
            self._update_activation(other)
        if self.stack_activation:
            self._update_stacked_activation(other)
        return self

    def __add__(self, other: Union["RuleSet", Rule]):
        remember_activation = self.remember_activation
        stack_activation = self.stack_activation
        if isinstance(other, Rule):
            rules = self.rules + [other]
        else:
            remember_activation &= other.remember_activation
            stack_activation &= other.stack_activation
            rules = list(set(self.rules + other.rules))
        return self.__class__(rules, remember_activation=remember_activation, stack_activation=stack_activation)

    def __len__(self):
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
        if len(self) < 2 * RuleSet.NLINES:
            return "\n".join([str(self[i]) for i in range(len(self))])
        else:
            return "\n".join(
                [str(self[i]) for i in range(RuleSet.NLINES)]
                + ["..."]
                + [str(self[i]) for i in range(len(self) - RuleSet.NLINES, len(self))]
            )

    @property
    def to_hash(self):
        if len(self) == 0:
            return "rs",
        to_hash = ("rs",)
        for r in self:
            rule_hash = r.to_hash[1:]
            to_hash += rule_hash
        return to_hash

    # noinspection PyProtectedMember
    def __hash__(self) -> hash:
        return hash(frozenset(self.to_hash))

    # noinspection PyProtectedMember,PyTypeChecker
    def _update_activation(self, other: Union[Rule, "RuleSet"]):
        if other.activation_available:
            if self._activation is None:
                self._activation = Activation(other.activation, to_file=Rule.LOCAL_ACTIVATION)
            else:
                self._activation = self._activation | other._activation

    # noinspection PyProtectedMember,PyTypeChecker
    def _update_stacked_activation(self, other: Union[Rule, "RuleSet"]):
        if other.activation_available:
            if self._activation is None:
                if isinstance(other, Rule):
                    self.stacked_activations = pd.DataFrame(data=np.array(other.activation).T, columns=[str(other)])
                else:
                    self.stacked_activations = other.stacked_activations
            else:
                if isinstance(other, Rule):
                    self.stacked_activations[str(other)] = other.activation
                else:
                    self.stacked_activations = pd.concat([self.stacked_activations, other.stacked_activations], axis=1)

    def compute_self_activation(self):
        if len(self) == 0:
            return
        activations_available = all([r.activation_available for r in self])
        if activations_available:
            # noinspection PyProtectedMember
            self._activation = Activation.multi_logical_or([r._activation for r in self])

    def compute_stacked_activation(self):
        if len(self) == 0:
            return
        activations_available = all([r.activation_available for r in self])
        if activations_available:
            # noinspection PyProtectedMember
            self.stacked_activations = pd.DataFrame(
                data=np.array([r.activation for r in self]).T, columns=[str(r.condition) for r in self]
            )

    # def __del__(self):
    #     self.del_activations()
    #     self.del_activation()

    def del_activations(self):
        for r in self:
            r.del_activation()

    def del_activation(self):
        """Deletes the activation vector's data, but not the object itself, so any computed attribute will remain
        available"""
        if hasattr(self, "_activation") and self._activation is not None:
            self._activation.delete()

    def del_stacked_activations(self):
        if hasattr(self, "stacked_activations") and self.stacked_activations is not None:
            del self.stacked_activations
            self.stacked_activations = None

    def append(self, rule: Rule, update_activation: bool = True):
        if not isinstance(rule, Rule):
            raise TypeError(f"RuleSet's append method expects a Rule object, got {type(rule)}")
        remember_activation = self.remember_activation
        stack_activation = self.stack_activation
        if not update_activation:
            self.remember_activation = False
            self.stack_activation = False
        self.__iadd__(rule)
        self.remember_activation = remember_activation
        self.stack_activation = stack_activation

    def sort(self, criterion: str = None, reverse: bool = False) -> None:
        if len(self) == 0:
            return
        import ast

        if criterion is None or criterion == "":
            if not (
                hasattr(self[0].condition, "features_names")
                and hasattr(self[0].condition, "bmins")
                and hasattr(self[0].condition, "bmaxs")
            ):
                return
            # noinspection PyUnresolvedReferences
            fnames = list(set([str(r.features_names) for r in self]))
            dict_names = {}
            lmax = 1
            for f in fnames:
                l_ = len(ast.literal_eval(f))
                if l_ > lmax:
                    lmax = l_
                if l_ not in dict_names:
                    dict_names[l_] = []
                dict_names[l_].append(f)
            for l_ in dict_names:
                dict_names[l_].sort(reverse=reverse)
            fnames = []
            for l_ in range(1, lmax + 1):
                if l_ in dict_names:
                    fnames += dict_names[l_]

            rules_by_fnames = OrderedDict({f: [] for f in fnames})
            for rule in self:
                # noinspection PyUnresolvedReferences
                v = str(rule.features_names)
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

    # noinspection PyProtectedMember
    def __contains__(self, other: Union["RuleSet", Rule]) -> bool:
        """A RuleSet contains another Rule or RuleSet if the second Rule or RuleSet activated points are also all
         activated by the first RuleSet."""
        if not self._activation or not other._activation:
            return False
        return other._activation in self._activation

    @property
    def activation_available(self):
        if self._activation is None:
            return False
        if self._activation.data_format == "file":
            return self._activation.data.is_file()
        else:
            return self._activation.data is not None

    @property
    def stacked_activations_available(self):
        if self.stack_activation is None:
            return False
        return True

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
    def rules(self) -> List[Rule]:
        return self._rules

    def calc_activation(self, xs: np.ndarray):
        if len(self) == 0:
            raise ValueError("Can not use calc_activation : The ruleset is empty!")
        [rule.calc_activation(xs) for rule in self.rules]
        self._activation = None
        self.compute_self_activation()

    def calc_stacked_activations(self, xs: np.ndarray):
        if len(self) == 0:
            raise ValueError("Can not use calc_stacked_activations : The ruleset is empty!")
        [rule.calc_activation(xs) for rule in self.rules]
        self._activation = None
        self.compute_stacked_activation()

    @property
    def coverage(self) -> float:
        if not self.activation_available:
            return 0.0
        else:
            return self._activation.coverage

    def get_variables_count(self):
        """
        Get a counter of all different features in the ruleset
        Parameters
        ----------
        Return
        ------
        count : {Counter type}
            Counter of all different features in the ruleset
        """
        # noinspection PyUnresolvedReferences
        var_in = [
            rule.condition.features_names
            if isinstance(rule.condition, HyperrectangleCondition)
            else rule.condition.features_indexes
            for rule in self
        ]
        if len(var_in) > 1:
            var_in = reduce(operator.add, var_in)
        count = Counter(var_in)

        count = count.most_common()
        return count
