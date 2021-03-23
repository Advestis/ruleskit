from abc import ABC
from typing import List, Union
import numpy as np


class Activation(ABC):
    """
    """
    def __init__(self, activation: Union[np.ndarray, List[int]] = None, n: int = None, val: int = None):
        if activation is not None:
            if set(activation) != {0, 1}:
                raise ValueError('Activation vector must be a binary vector!')
            self.n = len(activation)
            self.val = int("".join(str(i) for i in activation), 2)
        else:
            if (n is not None) & (val is not None):
                self.n = n
                self.val = val

    def __and__(self, other):
        if self.n != other.n:
            raise ValueError('Activations must have the same length!')
        else:
            val = self.val & other.val
        return Activation(n=self.n, val=val)

    def __or__(self, other):
        if self.n != other.n:
            raise ValueError('Activations must have the same length!')
        else:
            val = self.val or other.val
        return Activation(n=self.n, val=val)

    def __add__(self, other: 'Activation'):
        if self.n != other.n:
            raise ValueError('Activations must have the same length!')
        else:
            val_xor = self.val ^ other.val
            val_and = self.val & other.val
            val = val_xor ^ val_and
        return Activation(n=self.n, val=val)

    def __sub__(self, other: 'Activation'):
        if self.n != other.n:
            raise ValueError('Activations must have the same length!')
        else:
            val = (self.val ^ other.val) & self.val
        return Activation(n=self.n, val=val)

    def __len__(self):
        return self.n

    def get_array(self) -> np.ndarray:
        vector = [int(i) for i in list('{0:0b}'.format(self.val))]
        if len(vector) < self.n:
            vector = [0]*(self.n - len(vector)) + vector
        return np.array(vector)

    def sum_ones(self) -> int:
        return bin(self.val).count("1")

    def calc_coverage_rate(self) -> float:
        return self.sum_ones() / self.n
