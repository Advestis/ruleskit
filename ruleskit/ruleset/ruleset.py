from abc import ABC
from typing import List
from functools import reduce
import numpy as np
from ..rule.rule import Rule


class RuleSet(ABC):
    def __init__(self, rules_list: List[Rule] = None):
        if rules_list is None:
            self.rules = []
        else:
            self.rules = rules_list

    def __add__(self, other: "RuleSet"):
        rules = list(set(self.rules + other.rules))
        return RuleSet(rules)

    def __len__(self):
        return len(self.rules)

    def __eq__(self, other: "RuleSet"):
        return set(self.rules) == set(other.rules)

    # noinspection PyProtectedMember
    def calc_coverage_rate(self, xs: np.ndarray = None):
        if len(self) == 0:
            return 0.0
        else:
            if xs is None:
                rs_activation = reduce(lambda a, b: a._activation + b._activation, self.rules)
            else:
                rs_activation = reduce(lambda a, b: a._condition.evaluate(xs) + b._condition.evaluate(xs), self.rules)

            # noinspection PyUnresolvedReferences
            return rs_activation.coverage_rate
