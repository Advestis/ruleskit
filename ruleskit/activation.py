from abc import ABC
from typing import Union
import numpy as np
import pandas as pd
import ast
import sys
from .logger.logger import log as logger

MAX_INT_32 = 2 ** 32


class Activation(ABC):

    SIZE_LIMIT = 0.000000139  # 0.25 / 1.8e6. From numerical experiment.
    DTYPE = str

    def __init__(self, activation: Union[np.ndarray, int, str] = None, length: int = None, optimize: bool = True):
        """ Compresses an activation vector into a str(list) describing its variations or an int corresponding to the
         binary representation of the vector

        stored data will be either a str(list):
            Compression is done : First element of the list is the first value of the array last element of the list is
            the length of the array The other elemnts are the coordinates that changed values
        or a np.ndarray:
            same as str(list) but the list is casted into np.array instead of str
        or an int:
            taking the input vector [1 0 0 1 0 0 0 1 1...], converts it to binary string representation :
            "100100011..." then cast it into int using int(s, 2)

        The method will choose how to store the data based on the size (in MB) of the compressed list : if it is
        superior to a certain limit, the int will take less memory and is prefered. If compression is used and
        dtype is np.ndarray, will check that numbers present in the compressed vector can be stored as int32 to gain
        memory. Else, uses int64.

        The limit upon which integer is prefered was estimated from an activation vector of 1.8e6 elements,
        where the int version took 0.25 MB

        Parameters
        ----------
        activation: Union[np.ndarray, int, str]
            If np.ndarray : Of the form [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1], or compressed vector
            If str : compressed vector
            If int : Integer represented by the binary number that is the activation vector
        length: int
            Only valid if 'value' is an integer. An activation vector stored as an integer has lost the information
            about its size : [0 0 0 1 0 0 0 1 1...] to nit gives 100011... which in turn gives back [1 0 0 0 1 1...].
            To get the leading zeros back, one must specify the length of the activation vector.
        optimize: bool
            Only relevent 'value' is an integer. In that case, will check whether using compression saves up memory.
            Else, does not check and uses integer. Note that is optimize is True, entropy is computed.
        """
        self.length = None  # Will be set by init methods
        self._entropy = None  # Will be set if activation is not an integer or if optimize is True
        self.data_format = None  # Will be set by init methods
        self.data = None  # Will be set by init methods
        self._ones = None  # Will be set if "activation" is the raw activation vector
        self._rel_entropy = None  # Will be set if activation is not an integer or if optimize is True
        self._nones = None  # Will be set if activation is not an integer or if optimize is True
        self._coverage = None

        if isinstance(activation, str) and "," not in activation:  # activation is actualy an integer, stored as an int
            activation = int(activation)

        if isinstance(activation, int):
            self._init_with_integer(activation, Activation.DTYPE, length, optimize)

        elif isinstance(activation, str):
            self._init_with_str(activation)
        elif isinstance(activation, np.ndarray):
            if activation[-1] > 1:
                self._init_with_compressed_array(activation)
            else:
                self._init_with_raw(activation, Activation.DTYPE)
        else:
            raise TypeError(f"An activation can only be a np.ndarray, and int or a str. Got {type(activation)}.")

    def _init_with_integer(self, value: int, dtype: type, length: int = None, optimize: bool = True):

        """
        Will set
            if Optimize is True:
              * self._nones (number of ones in the activation)
              * self.length
              * if optimize is True : self._entropy and self._rel_entropy
              * self.data_format to "integer" or "compressed_str" or "compressed_array" depending on what takes less
                memory
              * self.data as an integer, a str or an array
            else:
              * self.data as an integer
              * self.data_format to "integer"

        """

        if length is None:
            raise ValueError("When giving an integer to Activation, you must also specify its length.")

        logger.debug(f"Activation vector is an int")
        self.length = length

        if optimize:
            raw = self._int_to_array(value)
            self._nones = np.count_nonzero(raw == 1)
            compressed = self._compress(raw, dtype=dtype)
            if dtype == str:
                self._entropy = len(ast.literal_eval(compressed)) - 2
            else:
                self._entropy = len(compressed) - 2
            self._rel_entropy = self._entropy / self.length
            sizeof = sys.getsizeof(compressed) / 1e6
            if (sizeof / self.length) > Activation.SIZE_LIMIT:
                self.data = value
                self.data_format = "integer"
            else:
                self.data = compressed
                if dtype == str:
                    self.data_format = "compressed_str"
                else:
                    self.data_format = "compressed_array"
        else:
            self.data = value
            self.data_format = "integer"

    def _init_with_str(self, value: str):
        """
        will set :
          * self.data as a compressed str
          * self.data_format as "compressed_str"
          * self._entropy and self._rel_entropy
          * self.length
        """
        logger.debug(f"Activation vector is a compressed str")
        evaluated = np.array(ast.literal_eval(value))
        self.data = value
        self._entropy = len(evaluated) - 2
        self.length = evaluated[-1]
        self._rel_entropy = self._entropy / self.length
        self.data_format = "compressed_str"

    def _init_with_compressed_array(self, value: np.ndarray):
        """
        will set :
          * self.data as a compressed array
          * self.data_format as "compressed_array"
          * self._entropy and self._rel_entropy
          * self.length
        """
        logger.debug(f"Activation vector is a compressed array")
        self.data = value
        self._entropy = len(value) - 2
        self.length = value[-1]
        self._rel_entropy = self._entropy / self.length
        self.data_format = "compressed_array"

    def _init_with_raw(self, value: np.ndarray, dtype: type):
        """
        will set :
          * self.data as an integer or a compressed array/str depending on what takes less memory and on what dtype is
          * self.data_format as "integer", "compressed_array" or "compressed_str"
          * self._entropy and self._rel_entropy
          * self.length
          * self._nones
        """
        logger.debug(f"Activation vector is raw")
        self.length = len(value)
        self._nones = np.count_nonzero(value == 1)
        compressed = self._compress(value, dtype=dtype)
        if dtype is str:
            self._entropy = len(compressed.split(",")) - 2
        else:
            self._entropy = len(compressed) - 2
        self._rel_entropy = self._entropy / self.length
        sizeof = sys.getsizeof(compressed) / 1e6
        logger.debug(f"Using int activation representation")
        if (sizeof / len(value)) > Activation.SIZE_LIMIT:
            self.data = self._array_to_int(value)
            self.data_format = "integer"
        else:
            logger.debug(f"Using compressed activation representation")
            self.data = compressed
            if dtype == str:
                self.data_format = "compressed_str"
            else:
                self.data_format = "compressed_array"

    def __and__(self, other: "Activation") -> "Activation":
        if self.length != other.length:
            raise ValueError(f"Activations have different lengths. Left is {self.length}, right is {other.length}")

        if self.data_format == "integer" and other.data_format == "integer":
            return Activation(self.data & other.data, length=self.length)
        else:
            return Activation(self.raw * other.raw)

    def __or__(self, other: "Activation") -> "Activation":
        if self.length != other.length:
            raise ValueError(f"Activations have different lengths. Left is {self.length}, right is {other.length}")
        if self.data_format == "integer" and other.data_format == "integer":
            return Activation(self.data or other.data, length=self.length)
        else:
            return Activation(np.logical_or(self.raw, other.raw, length=self.length).astype("int32"))

    def __add__(self, other: "Activation") -> "Activation":
        if self.length != other.length:
            raise ValueError(f"Activations have different lengths. Left is {self.length}, right is {other.length}")
        if self.data_format == "integer" and other.data_format == "integer":
            val_xor = self.data ^ other.data
            val_and = self.data & other.data
            val = val_xor ^ val_and
        else:
            val_xor = np.logical_xor(self.raw, other.raw)
            val_and = self.raw * other.raw
            val = np.logical_xor(val_xor, val_and).astype("int32")
        return Activation(val, length=self.length)

    def __sub__(self, other: "Activation") -> "Activation":
        if self.length != other.length:
            raise ValueError(f"Activations have different lengths. Left is {self.length}, right is {other.length}")
        if self.data_format == "integer" and other.data_format == "integer":
            return Activation((self.data ^ other.data) & self.data, length=self.length)
        else:
            return Activation(np.logical_xor(self.raw, other.raw).astype("int32") * self.raw, length=self.length)

    def __len__(self):
        return self.length

    def _int_to_array(self, value: int = None) -> np.ndarray:
        """From a value of the form 45786542 (int), which is the base 10 representation of the binary form of an
        activation vector, returns the initial vector.
        """
        if value is None:
            act = np.fromiter(bin(self.data)[2:], dtype=int)
        else:
            act = np.fromiter(bin(value)[2:], dtype=int)

        if len(act) > self.length:
            raise ValueError(
                "After using int_to_array, I ended up with an activation vector bigger than the specified "
                "max length. This should not happend as the max length should have been set by the indexing "
                "of x earlier in your code"
            )
        act_bis = np.zeros(self.length)
        act_bis[self.length - len(act):] = act
        return act_bis

    def _decompress(self, value: Union[str, np.ndarray] = None) -> np.ndarray:
        """Will return the original activation vector, and set self._nones and self._ones"""
        if value is None:
            if self.data_format == "compressed_str":
                act = ast.literal_eval(self.data)
            elif self.data_format == "compressed_array":
                act = self.data
            else:
                raise TypeError("Cannot decompress an activation vector which data format is not compressed")
        else:
            if isinstance(value, str):
                act = ast.literal_eval(value)
            else:
                act = value

        length = act[-1]
        s = np.zeros(length)
        ones = []
        n_ones = 0
        previous_value = 0
        previous_index = 0

        compute_nones = self._nones is None
        compute_ones = self._ones is None

        if act[0] == 1:
            previous_value = 1
            s[0] = 1
        if len(act) == 2:
            if act[0] == 1:
                self._nones = 1
                self._ones = [pd.IndexSlice[0:1]]
                return np.array(s, dtype=int)
            else:
                self._nones = 0
                self._ones = []
                return np.array(s, dtype=int)

        for index in act[1:]:
            if previous_value == 0:
                previous_index = index
                previous_value = 1
            else:
                if compute_nones:
                    n_ones += index - previous_index
                if compute_ones:
                    ones.append(pd.IndexSlice[previous_index:index])
                s[previous_index:index] = np.ones(index - previous_index)
                previous_index = index
                previous_value = 0

        if compute_nones:
            self._nones = n_ones
            self._coverage = self._nones / self.length

        if compute_ones:
            self._ones = ones
        return np.array(s, dtype="int32")

    @staticmethod
    def _compress(value: np.ndarray, dtype: type = str) -> Union[np.ndarray, str]:
        """Transforms a raw activation vector to a compressed one.

        A compressed vector is a collection of integers starting by the initial value of the raw vector (0 or 1) and
        ending with its size. The other integers in the compression are the positions in the raw vector where the
        vector value changes. This stores all the information and saves up memory if the vector is constant over
        large periods of time.

        The compressed vector can be stored as a str looking like "0, 12, 456, ..., 47782" or as a numpy array of
        integers. What storage to use is specified by the "dtype" argument.
        """
        value = value.astype(int)
        to_ret = [value[0]]
        diff_arr = abs(np.diff(value))
        to_ret += list(np.where(diff_arr == 1)[0] + 1)
        to_ret.append(len(value))
        if dtype == str:
            to_ret = str(to_ret).replace(" ", "").replace("[", "").replace("]", "")
        else:
            if to_ret[-1] < MAX_INT_32:
                to_ret = np.array(to_ret, dtype="int32")
            else:
                to_ret = np.array(to_ret, dtype="int64")
        return to_ret

    @staticmethod
    def _array_to_int(value: np.ndarray) -> int:
        """Casts a raw activation vector into the integer represented by its binary form

        Examples
        --------
        >>> from ruleskit import Activation
        >>> Activation._array_to_int(np.array([0, 1, 1, 0]))
        6  # the binary number '0110' is 6 in base 10
        """
        return int("".join(str(i) for i in value.astype('int')), 2)

    @property
    def raw(self) -> np.ndarray:
        if self.data_format == "integer":
            return self._int_to_array()
        else:
            return self._decompress()  # will also set self._ones and self._nones

    @property
    def ones(self) -> int:
        """self._ones might not be set since it can only be set when decompressing a compressed vector"""
        if self._ones is None:
            _ = self.raw  # calling raw will compute nones and ones
        return self._ones

    @property
    def nones(self) -> int:
        """self._nones might not be set since it can only be set at object creation if the full array was given"""
        if self._nones is None:
            if self.data_format == "integer":
                self._nones = bin(self.data).count("1")  # faster than calling "raw"
            else:
                _ = self.raw  # calling raw will compute nones

        if self._coverage is None:
            self._coverage = self._nones / self.length
        return self._nones

    @property
    def entropy(self) -> int:
        if self._entropy is None:
            if self.data_format == "integer":
                compressed = self._compress(self.raw)
                self._entropy = len(ast.literal_eval(compressed)) - 2
            else:
                raise ValueError(
                    "Data format is not integer and yet entropy is not set. There is a problem in the "
                    "Activation class, please contact its maintainer."
                )
        if self._rel_entropy is None:
            self._rel_entropy = self._entropy / self.length
        return self._entropy

    @property
    def rel_entropy(self) -> float:
        if self._rel_entropy is None:
            _ = self.entropy  # will set self._rel_entropy
        return self._rel_entropy

    @property
    def coverage(self) -> float:
        if self._coverage is None:
            _ = self.nones  # will set self._coverage
        return self._coverage

    @property
    def as_int(self):
        if self.data_format == "integer":
            return self.data
        else:
            return self._array_to_int(self.raw)

    @property
    def as_compressed_array(self):
        if self.data_format == "compressed_array":
            return self.data
        else:
            return self._compress(self.raw, dtype=np.ndarray)

    @property
    def as_compressed_str(self):
        if self.data_format == "compressed_str":
            return self.data
        else:
            return self._compress(self.raw, dtype=str)
