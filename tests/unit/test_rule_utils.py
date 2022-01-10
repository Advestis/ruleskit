from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from ruleskit import extract_rules_from_tree, HyperrectangleCondition, Rule, RuleSet
import numpy as np
import pytest


@pytest.mark.parametrize(
    "tree, output, get_leaf",
    [
        (
            DecisionTreeRegressor(max_depth=2),
            RuleSet([
                Rule(
                    HyperrectangleCondition(
                        [1], [-0.126097385560409], [-0.0037617861526086926]
                    )
                ),
                Rule(
                    HyperrectangleCondition(
                        [1, 0],
                        [-0.126097385560409, -0.0902752958985185],
                        [-0.0037617861526086926, 0.0061888848431408405],
                    )
                ),
                Rule(
                    HyperrectangleCondition(
                        [1, 0],
                        [-0.126097385560409, 0.0061888848431408405],
                        [-0.0037617861526086926, 0.17055522598066],
                    )
                ),
                Rule(
                    HyperrectangleCondition(
                        [1], [-0.0037617861526086926], [0.133598980013008]
                    )
                ),
                Rule(
                    HyperrectangleCondition(
                        [1, 0],
                        [-0.0037617861526086926, -0.0902752958985185],
                        [0.133598980013008, 0.014811381697654724],
                    )
                ),
                Rule(
                    HyperrectangleCondition(
                        [1, 0],
                        [-0.0037617861526086926, 0.014811381697654724],
                        [0.133598980013008, 0.17055522598066],
                    )
                ),
            ]),
            False,
        ),
        (
            DecisionTreeRegressor(max_depth=2),
            RuleSet([
                Rule(
                    HyperrectangleCondition(
                        [1, 0],
                        [-0.126097385560409, -0.0902752958985185],
                        [-0.0037617861526086926, 0.0061888848431408405],
                    )
                ),
                Rule(
                    HyperrectangleCondition(
                        [1, 0],
                        [-0.126097385560409, 0.0061888848431408405],
                        [-0.0037617861526086926, 0.17055522598066],
                    )
                ),
                Rule(
                    HyperrectangleCondition(
                        [1, 0],
                        [-0.0037617861526086926, -0.0902752958985185],
                        [0.133598980013008, 0.014811381697654724],
                    )
                ),
                Rule(
                    HyperrectangleCondition(
                        [1, 0],
                        [-0.0037617861526086926, 0.014811381697654724],
                        [0.133598980013008, 0.17055522598066],
                    )
                ),
            ]),
            True,
        ),
    ],
)
def test_extract_rules_from_tree(tree, output, get_leaf):
    diabetes = datasets.load_diabetes()
    data = diabetes.data
    # Keep the two most important variables
    x = data.T[2]
    y = data.T[-2]
    x_vstack = np.vstack((x, y)).T
    # Get the target variable
    y = diabetes.target

    tree.fit(x_vstack, y)
    rule_list = extract_rules_from_tree(
        tree, xmins=x_vstack.min(axis=0), xmaxs=x_vstack.max(axis=0), get_leaf=get_leaf
    )
    np.testing.assert_equal(rule_list, output)


# @pytest.mark.parametrize(
#     "tree",
#     [
#             DecisionTreeRegressor(max_depth=2),
#     ],
# )
# def test_extract_rules_from_tree(tree):
#     diabetes = datasets.load_diabetes()
#     data = diabetes.data
#     # Keep the two most important variables
#     x = data.T[2]
#     y = data.T[-2]
#     X = np.vstack((x, y)).T
#     # Get the target variable
#     y = diabetes.target
#
#     tree.fit(X, y)
#     prediction_tree = tree.predict(X)
#
#     rule_list = extract_rules_from_tree(tree, xmins=X.min(axis=0), xmaxs=X.max(axis=0))
#     [rule.fit(X, y) for rule in rule_list]
#     prediction_rules = np.zeros(len(y))
#     for rule in rule_list:
#         prediction_rules += rule.predict(X)
#
#     np.testing.assert_equal(prediction_rules, prediction_tree)
