from abc import ABC
import operator
from typing import List, Union
from functools import reduce
from collections import Counter
import numpy as np
from collections import OrderedDict
from .rule import Rule
from .condition import HyperrectangleCondition


class RuleSet(ABC):

    NLINES = 5

    def __init__(self, rules_list: Union[List[Rule], None] = None):
        self._rules = []
        if rules_list is not None:
            for rule in rules_list:
                if not isinstance(rule, Rule):
                    raise TypeError(f"Some rules in given iterable were not of type 'Rule' but of type {type(rule)}")
                self._rules.append(rule)

    def __add__(self, other: Union["RuleSet", Rule]):
        if isinstance(other, Rule):
            rules = self.rules + [other]
        else:
            rules = list(set(self.rules + other.rules))
        return self.__class__(rules)

    def __len__(self):
        return len(self.rules)

    def __eq__(self, other: "RuleSet"):
        return set(self.rules) == set(other.rules)

    def __iter__(self):
        return self.rules.__iter__()

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

    def append(self, rule: Rule):
        if not isinstance(rule, Rule):
            raise TypeError(f"RuleSet's append method expects a Rule object, got {type(rule)}")
        self.rules.append(rule)

    def sort(self) -> None:
        if len(self) == 0:
            return
        if not (
            hasattr(self[0].condition, "features_names")
            and hasattr(self[0].condition, "bmins")
            and hasattr(self[0].condition, "bmaxs")
        ):
            return
        # noinspection PyUnresolvedReferences
        fnames = list(set([str(r.features_names) for r in self]))
        fnames.sort(reverse=True)
        rules_by_fnames = OrderedDict({f: [] for f in fnames})
        for rule in self:
            # noinspection PyUnresolvedReferences
            v = str(rule.features_names)
            rules_by_fnames[v].append(rule)
        rules_by_fnames = {
            n: sorted(rules_by_fnames[n], key=lambda x: x.condition.bmins + x.condition.bmaxs) for n in rules_by_fnames
        }
        self._rules = []
        for n in rules_by_fnames:
            self._rules += rules_by_fnames[n]

    @property
    def rules(self) -> List[Rule]:
        return self._rules

    def get_activation(self, xs: np.ndarray = None):
        if len(self) == 0:
            raise ValueError("The rule set is empty!")
        elif len(self) == 1:
            rule = self[0]
            if xs is None:
                # noinspection PyProtectedMember
                rs_activation = rule._activation
            else:
                rs_activation = rule.condition.evaluate(xs)
        else:
            if xs is not None:
                [rule.calc_activation(xs) for rule in self.rules]

            # noinspection PyProtectedMember
            rs_activation = [rule._activation for rule in self.rules]
            if len(rs_activation) > 1:
                rs_activation = reduce(operator.add, rs_activation)

        return rs_activation

    def calc_coverage_rate(self, xs: np.ndarray = None):
        if len(self) == 0:
            return 0.0
        else:
            rs_activation = self.get_activation(xs)
            return rs_activation.coverage

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
