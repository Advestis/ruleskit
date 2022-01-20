import math
import logging
from typing import Union
from pathlib import Path
from json import load

logger = logging.getLogger(__name__)


class Thresholds:

    """Implements the notion of thresholds in Rule. From a given json, sets limits that can be used to flag bad rules.

    Attributes
    ----------
    limits: dict
        The limits to use
    path: Union[str, Path, TransparentPath]
    """

    KWOWN_ARGS = {"abs": abs, "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "str": str, "int": int}

    def __init__(self, path: Union[str, Path, "TransparentPath"], show=False):
        self.limits = {}
        self.path = path

        if type(path) == str:
            try:
                from transparentpath import TransparentPath
                path = TransparentPath(path)
            except ImportError:
                path = Path(path)

        if not path.is_file():
            return

        if show:
            logger.info(f"Found threshold file {path}")

        if hasattr(path, "read"):
            self.limits = path.read()
        else:
            with open(path) as opath:
                self.limits = load(opath)

        if show:
            message = "\n".join([f"{i}: {self.limits[i]}" for i in self.limits])
            logger.info(f"Thresholds are \n{message}")
        if "coverage" not in self.limits:
            self.limits["coverage"] = {"min": 0.05}
            logger.info("Coverage limit was not set. Set to minimum 5%.")
        elif "min" not in self.limits["coverage"]:
            self.limits["coverage"]["min"] = 0.05
            logger.info("Coverage lower limit was not set. Set to 5%.")
        elif self.limits["coverage"]["min"] == 0:
            logger.warning("Coverage limit is set to 0.")

    def __call__(self, tkey: str, rule) -> bool:
        if tkey not in self.limits:
            return True
        value = getattr(rule, tkey)
        if callable(value):
            return True
        if value is None:
            return True

        threshold = self.limits[tkey]
        s = tkey
        if "arg" in threshold:
            if threshold["arg"] not in Thresholds.KWOWN_ARGS:
                raise ValueError(f"Unknown arg for threshold : {threshold['arg']}")
            value = Thresholds.KWOWN_ARGS[threshold["arg"]](value)
            s = f"{threshold['arg']}({tkey})"

        for lim in threshold:
            if lim == "arg":
                continue
            if isinstance(threshold[lim], str):
                threshold[lim] = getattr(rule, threshold[lim])
            if not isinstance(threshold[lim], (int, float)):
                raise TypeError(f"Value in threshols can only be float or int, got {type(threshold[lim])}")

        if "min" in threshold and value < threshold["min"]:
            logger.debug(f"Rule {rule} is bad : {s} = {value} < {threshold['min']}")
            return False
        if "max" in threshold and value > threshold["max"]:
            logger.debug(f"Rule {rule} is bad : {s} = {value} > {threshold['max']}")
            return False
        if "equal" in threshold and value != threshold["equal"]:
            logger.debug(f"Rule {rule} is bad : {s} = {value} != {threshold['equal']}")
            return False
        if "different" in threshold and value == threshold["different"]:
            logger.debug(f"Rule {rule} is bad : {s} = {value} == {threshold['different']}")
            return False
        return True
