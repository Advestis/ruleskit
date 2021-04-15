from abc import ABC
import operator
from typing import List, Union
from functools import reduce
from collections import Counter
import numpy as np
from .rule import Rule
from .condition import HyperrectangleCondition


class RuleSet(ABC):

    def __init__(self, rules_list: List[Rule] = None):
        if rules_list is None:
            self._rules = []
        else:
            self._rules = rules_list

    def __add__(self, other: Union['RuleSet', Rule]):
        if type(other) == Rule:
            rules = self.rules + [other]
        else:
            rules = list(set(self.rules + other.rules))
        return RuleSet(rules)

    def __len__(self):
        return len(self.rules)

    def __eq__(self, other: 'RuleSet'):
        return set(self.rules) == set(other.rules)

    def __iter__(self):
        return self.rules.__iter__()

    def __getitem__(self, x):
        return self.rules.__getitem__(x)

    @property
    def rules(self) -> List[Rule]:
        return self._rules

    def get_activation(self, xs: np.ndarray = None):
        if len(self) == 0:
            raise ValueError('The rule set is empty!')
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
            activations_list = [rule._activation for rule in self.rules]
            rs_activation = reduce(operator.add, activations_list)

        return rs_activation

    def calc_coverage_rate(self, xs: np.ndarray = None):
        if len(self) == 0:
            return 0.0
        else:
            rs_activation = self.get_activation(xs)
            return rs_activation.coverage_rate

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
        var_in = [rule.condition.features_names if isinstance(rule.condition, HyperrectangleCondition) else
                  rule.condition.features_indexes for rule in self]
        var_in = reduce(operator.add, var_in)
        count = Counter(var_in)

        count = count.most_common()
        return count
