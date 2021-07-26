from typing import List, Union
import copy
import numpy as np
from ..rule import RegressionRule, ClassificationRule
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..condition import HyperrectangleCondition

# noinspection PyProtectedMember
from sklearn.tree import _tree


def extract_rules_from_tree(
    tree: Union[DecisionTreeClassifier, DecisionTreeRegressor],
    xmins: Union[List[float], np.ndarray],
    xmaxs: Union[List[float], np.ndarray],
    features_names: List[str] = None,
    get_leaf: bool = False,
) -> Union[List[RegressionRule], List[ClassificationRule]]:
    """To extract rules from a sklearn decision tree

    features are the list of X names.
    In the tree ('decision_tree' later in the code), the list of relevant Xs are represented
    by their position in the list of Xs that was fed to the tree at learning.
    So to access the name of the feature relevant for a given node in the tree,
    one has first to do decision_tree.feature[node] to access the feature index in x_names,
    then do x_names[decision_tree.feature[node]]

    xmin and xmax are Series.
    Index is a X name, values are the min and the man value of the X for a given Y.
    The Y was specified before calling this function so it does not appear here.

    If all_rules_list is not None, it will be appended with the extracted rules,
    and nothing is returned. Else, returns the extracted rules

    Later in the code:
    bmins is, for a given condition, the minimum value that X must have to be selected
    By analogy, bmaxs is the max value.

    Hence bmaxs must always be >= to bmins, and bmins and bmaxs must be inside the interval
    [xmin, xmax].

    When selecting Ys based on conditions, the criterion will be X > bmins and X <= bmaxs.
    bmins is excluded and bmaxs included. So when creating bmins, if it is equal to bmins,
    change it to be a bit less than xmins so that xmins will pass the criterion.

    In the tree, to each node corresponds a maximum value for the associated X (feature).
    This is accessed by decision_tree.threshold[node] and will be the bmaxs of one condition of a
    rule.

    It will later be the bmins of one condition of the "opposite" rule. Indeed if there
    is a condition saying something like 'If X < 3 then do that', there is also a
    condition saying 'if X >= 3 then do this'.

    At the end there will be 2 rules per node. So not only the leaves give a rule, but
    each branching in the tree.
    Rules do not contain any value for Ys. Just conditions on Xs. RICE will set the
    values in calc_stat.

    If print_visitor is specified, will return a string describing what happened step
    by step

    Parameters
    ----------
    tree : Union[sklearn.tree.DecisionTreeRegressor, sklearn.tree.DecisionTreeClassifier]

    features_names: List[str],
        the list of X names

    xmins: Union[List[int], List[float], np.ndarray]
        min values of each xs, one entry per x

    xmaxs: Union[List[int], List[float], np.ndarray]
        max values of each xs, one entry per x

    get_leaf: Boolean type
        To return only the leaf of the tree.

    Returns
    -------
    rule_list: List[rule.Rule]
        The extracted rules for mthe tree.

    """
    decision_tree = tree.tree_
    if features_names is None:
        features_names = ["X_" + str(i) for i in range(tree.n_features_)]

    def visitor(node, depth, condition=None, rules_list=None):
        if rules_list is None:
            rules_list = []
        if decision_tree.feature[node] != _tree.TREE_UNDEFINED:
            # If
            new_condition = HyperrectangleCondition(
                features_indexes=[decision_tree.feature[node]],
                bmins=[xmins[decision_tree.feature[node]]],
                bmaxs=[decision_tree.threshold[node]],
                features_names=[features_names[decision_tree.feature[node]]],
            )

            # If cond is not None, means we are not at the first node,
            # so need to expand the list of conditions, unless the current X is
            # already in our list of conditions. In that case, update the
            # corresponding condition by setting its bmaxs to be the min of the
            # current and new thresh.olds
            if condition is not None:
                if decision_tree.feature[node] not in condition.features_indexes:
                    # Will concatenate current condition list and new conditions by their
                    # attributes (which are lists)
                    new_condition = condition & new_condition
                else:
                    new_bmaxs = decision_tree.threshold[node]
                    new_condition = copy.deepcopy(condition)
                    place = condition.features_indexes.index(decision_tree.feature[node])
                    new_condition.bmaxs[place] = min(new_bmaxs, new_condition.bmaxs[place])

            # Create a new Rule with all the condition and append it to the rules
            # list. So the rule list is actually a history of how one rule evolved.
            if get_leaf is False:
                if isinstance(tree, DecisionTreeClassifier):
                    new_rule = ClassificationRule(copy.deepcopy(new_condition))
                else:
                    new_rule = RegressionRule(copy.deepcopy(new_condition))
                rules_list.append(new_rule)

            # Execute the current function on the left of the current node
            # (the "True" side)
            rules_list = visitor(decision_tree.children_left[node], depth + 1, new_condition, rules_list)

            # At this point, any rule found on the left of the current node will be
            # in rules list and new_cond will contain the corresponding conditions.

            # Create a new condition corresponding to the opposite of the current
            # node's threshold, i.e bmins is now the previous bmaxs
            # Here we do not have to alter bmins, because we explicitly want
            # node_thresh to not pass the criterion since it must have passed it
            # for the opposite condition.
            new_condition = HyperrectangleCondition(
                features_indexes=[decision_tree.feature[node]],
                bmins=[decision_tree.threshold[node]],
                bmaxs=[xmaxs[decision_tree.feature[node]]],
                features_names=[features_names[decision_tree.feature[node]]],
            )

            # Means we are not at first node. Same logic as previously, except this time
            # if Xi is in the list of features then the new bmins is now the threshold
            # and bmaxs is modified to be the X maximum. Then keep the max of current
            # and new bmins as bmins, same for bmaxs.
            # TODO there is something weird here, for X_i will necessarily be in cond, since
            # it was added at the very beginning of the function.
            if condition is not None:
                if decision_tree.feature[node] not in condition.features_indexes:
                    new_condition = condition & new_condition
                else:
                    new_bmins = decision_tree.threshold[node]
                    new_bmaxs = xmaxs[decision_tree.feature[node]]
                    new_condition = copy.deepcopy(condition)
                    place = new_condition.features_indexes.index(decision_tree.feature[node])
                    new_condition.bmins[place] = max(new_bmins, new_condition.bmins[place])
                    new_condition.bmaxs[place] = max(new_bmaxs, new_condition.bmaxs[place])

            # Create the rule for the right side of the node then apply this
            # function on the right side of the node
            if get_leaf is False:
                if isinstance(tree, DecisionTreeClassifier):
                    new_rule = ClassificationRule(copy.deepcopy(new_condition))
                else:
                    new_rule = RegressionRule(copy.deepcopy(new_condition))
                rules_list.append(new_rule)

            rules_list = visitor(decision_tree.children_right[node], depth + 1, new_condition, rules_list)

        elif get_leaf:
            if isinstance(tree, DecisionTreeClassifier):
                rules_list.append(ClassificationRule(copy.deepcopy(condition)))
            else:
                rules_list.append(RegressionRule(copy.deepcopy(condition)))

        return rules_list

    rule_list = visitor(0, 1)
    return rule_list
