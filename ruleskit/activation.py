from abc import ABC
from typing import Union
import numpy as np
import pandas as pd
import ast
import sys
from time import time
from bitarray import bitarray
from .logger.logger import log as logger

MAX_INT_32 = 2 ** 32


class Activation(ABC):

    DTYPE = str
    FORCE_STAT = False

    def __init__(
        self, activation: Union[np.ndarray, bitarray, str] = None, optimize: bool = True,
    ):
        """Compresses an activation vector into a str(list) describing its variations or an bitarray of booleans

        stored data will be either a str(list):
            Compression is done : First element of the list is the first value of the array last element of the list is
            the length of the array The other elemnts are the coordinates that changed values
        or a np.ndarray:
            same as str(list) but the list is casted into np.array instead of str
        or an bitarray:
            The input vector [1 0 0 1 0 0 0 1 1...] where each entry uses up one bit of memory

        The method will choose how to store the data based on the size (in MB) of the compressed list : if it is
        superior size in bitarray, it will take less memory and is prefered. If compression is used and
        dtype is np.ndarray, will check that numbers present in the compressed vector can be stored as int32 to gain
        memory. Else, uses int64.

        Parameters
        ----------
        activation: Union[np.ndarray, bitarray, str]
            If np.ndarray : Of the form [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1], or compressed vector
            If str : compressed vector
            If bitarray : same as np.array but takes 64x less memory (each entry is stored in one bit only)
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
        self._time_compressed_to_raw = -1
        self._time_raw_to_compressed = -1
        self._time_bitarray_to_raw = -1
        self._time_raw_to_bitarray = -1
        self._time_compressed_to_bitarray = -1
        self._time_bitarray_to_compressed = -1
        self._n_compressed_to_raw = 0
        self._n_raw_to_compressed = 0
        self._n_bitarray_to_raw = 0
        self._n_raw_to_bitarray = 0
        self._n_bitarray_to_compressed = 0
        self._n_compressed_to_bitarray = 0
        self._sizeof_compressed_array = -1
        self._sizeof_compressed_str = -1
        self._sizeof_bitarray = -1
        self._sizeof_raw = -1

        if isinstance(activation, str) and "," not in activation:
            activation = bitarray(activation)

        if isinstance(activation, bitarray):
            self._init_with_bitarray(activation, Activation.DTYPE, optimize)

        elif isinstance(activation, str):
            self._init_with_str(activation)
        elif isinstance(activation, np.ndarray):
            if activation[-1] > 1:
                self._init_with_compressed_array(activation)
            else:
                self._init_with_raw(activation, Activation.DTYPE)
        else:
            raise TypeError(f"An activation can only be a np.ndarray, and bitarray or a str. Got {type(activation)}.")

    def _init_with_bitarray(self, value: bitarray, dtype: type, optimize: bool = True):

        """
        Will set
            if Optimize is True:
              * self._nones (number of ones in the activation)
              * self.length
              * self._entropy and self._rel_entropy
              * self.data_format to "bitarray" or "compressed_str" or "compressed_array" depending on what takes less
                memory
              * self.data as a bitarray, a str or an array
            else:
              * self.data as a bitarray
              * self.data_format to "bitarray"

        """

        logger.debug(f"Activation vector is a bitarray")
        self.length = len(value)
        self._sizeof_bitarray = sys.getsizeof(value) / 1e6

        if optimize:
            self._nones = value.count(1)
            t0 = time()
            compressed = self._compress(value, dtype=dtype)
            self._time_raw_to_compressed = time() - t0
            self._n_raw_to_compressed += 1
            if isinstance(compressed, str):
                self._sizeof_compressed_str = sys.getsizeof(compressed) / 1e6
                size_compressed = self._sizeof_compressed_str
            else:
                self._sizeof_compressed_array = sys.getsizeof(compressed) / 1e6
                size_compressed = self._sizeof_compressed_array
            if dtype == str:
                self._entropy = len(ast.literal_eval(compressed)) - 2
            else:
                self._entropy = len(compressed) - 2
            self._rel_entropy = self._entropy / self.length
            if size_compressed > self._sizeof_bitarray:
                self.data = value
                self.data_format = "bitarray"
            else:
                self.data = compressed
                if dtype == str:
                    self.data_format = "compressed_str"
                else:
                    self.data_format = "compressed_array"
        else:
            self.data = value
            self.data_format = "bitarray"

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
        self._sizeof_compressed_str = sys.getsizeof(value) / 1e6
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
        self._sizeof_compressed_array = sys.getsizeof(value) / 1e6
        self._entropy = len(value) - 2
        self.length = value[-1]
        self._rel_entropy = self._entropy / self.length
        self.data_format = "compressed_array"

    def _init_with_raw(self, value: np.ndarray, dtype: type):
        """
        will set :
          * self.data as an integer or a compressed array/str depending on what takes less memory and on what dtype is
          * self.data_format as "bitarray", "compressed_array" or "compressed_str"
          * self._entropy and self._rel_entropy
          * self.length
          * self._nones
        """
        logger.debug(f"Activation vector is raw")
        self._sizeof_raw = sys.getsizeof(value) / 1e6
        self.length = len(value)
        self._nones = np.count_nonzero(value == 1)
        t0 = time()
        compressed = self._compress(value, dtype=dtype)
        self._time_raw_to_compressed = time() - t0
        self._n_raw_to_compressed += 1
        if isinstance(compressed, str):
            self._sizeof_compressed_str = sys.getsizeof(compressed) / 1e6
            size_compressed = self._sizeof_compressed_str
        else:
            self._sizeof_compressed_array = sys.getsizeof(compressed) / 1e6
            size_compressed = self._sizeof_compressed_array
        if dtype is str:
            self._entropy = len(compressed.split(",")) - 2
        else:
            self._entropy = len(compressed) - 2
        self._rel_entropy = self._entropy / self.length
        t0 = time()
        inbitarray = self._raw_to_bitarray(value)
        self._time_raw_to_bitarray = time() - t0
        self._n_raw_to_bitarray += 1
        self._sizeof_bitarray = sys.getsizeof(inbitarray) / 1e6
        if sys.getsizeof(inbitarray) > size_compressed:
            logger.debug(f"Using compressed activation representation")
            self.data = compressed
            if dtype == str:
                self.data_format = "compressed_str"
            else:
                self.data_format = "compressed_array"
        else:
            logger.debug(f"Using bitarray activation representation")
            self.data_format = "bitarray"
            self.data = inbitarray

    def __and__(self, other: "Activation") -> "Activation":
        if self.length != other.length:
            raise ValueError(f"Activations have different lengths. Left is {self.length}, right is {other.length}")

        if self.data_format == "bitarray" and other.data_format == "bitarray":
            return Activation(self.data & other.data)
        else:
            return Activation(self.raw * other.raw)

    def __or__(self, other: "Activation") -> "Activation":
        if self.length != other.length:
            raise ValueError(f"Activations have different lengths. Left is {self.length}, right is {other.length}")
        if self.data_format == "bitarray" and other.data_format == "bitarray":
            return Activation(self.data or other.data)
        else:
            return Activation(np.logical_or(self.raw, other.raw, length=self.length).astype("int32"))

    def __add__(self, other: "Activation") -> "Activation":
        if self.length != other.length:
            raise ValueError(f"Activations have different lengths. Left is {self.length}, right is {other.length}")
        if self.data_format == "bitarray" and other.data_format == "bitarray":
            val_xor = self.data ^ other.data
            val_and = self.data & other.data
            val = val_xor ^ val_and
        else:
            val_xor = np.logical_xor(self.raw, other.raw)
            val_and = self.raw * other.raw
            val = np.logical_xor(val_xor, val_and).astype("int32")
        return Activation(val)

    def __sub__(self, other: "Activation") -> "Activation":
        if self.length != other.length:
            raise ValueError(f"Activations have different lengths. Left is {self.length}, right is {other.length}")
        if self.data_format == "bitarray" and other.data_format == "bitarray":
            return Activation((self.data ^ other.data) & self.data)
        else:
            return Activation(np.logical_xor(self.raw, other.raw).astype("int32") * self.raw)

    def __len__(self):
        return self.length

    def _bitarray_to_raw(self, value: bitarray = None, out=True) -> np.ndarray:
        """Transforms a bitarray to a nparray
        """
        t0 = time()
        if value is None:
            out = False
            value = self.data
        if isinstance(value, np.ndarray) and (value[-1] == 0 or value[-1] == 1):
            if not out:
                self._time_bitarray_to_raw = time() - t0
                self._n_bitarray_to_raw += 1
            return value
        elif not isinstance(value, bitarray):
            raise ValueError("Can not apply _bitarray_to_raw on a compressed vector")
        act = np.array(list(value), dtype=np.ushort)

        if self._sizeof_raw == -1 and not out:
            self._sizeof_raw = sys.getsizeof(act) / 1e6
        if self._sizeof_bitarray == -1 and not out:
            self._sizeof_bitarray = sys.getsizeof(value) / 1e6
        if not out:
            self._time_bitarray_to_raw = time() - t0
            self._n_bitarray_to_raw += 1
        return act

    def _decompress(self, value: Union[str, np.ndarray] = None, raw=True, out=True) -> Union[np.ndarray, bitarray]:
        """Will return the original activation vector, and set self._nones and self._ones"""
        t0 = time()

        if value is None:
            out = False
            value = self.data

        if value[-1] == 0 or value[-1] == 1:
            if not out:
                self._time_compressed_to_raw = time() - t0
                self._n_compressed_to_raw += 1
            return value
        elif isinstance(value, bitarray):
            raise ValueError("Can not apply _decompress on a bitarray vector")

        if isinstance(value, str):
            act = ast.literal_eval(value)
            if self._sizeof_compressed_str == -1 and not out:
                self._sizeof_compressed_str = sys.getsizeof(value) / 1e6
        elif isinstance(value, np.ndarray):
            act = value
            if self._sizeof_compressed_array == -1 and not out:
                self._sizeof_compressed_array = sys.getsizeof(value) / 1e6
        else:
            raise TypeError(f"'value' can not be of type {type(value)}")

        length = act[-1]
        s = np.zeros(length, dtype=np.ushort)
        ones = []
        n_ones = 0
        previous_value = 0
        previous_index = 0

        compute_nones = self._nones is None and not out
        compute_ones = self._ones is None and not out

        if act[0] == 1:
            previous_value = 1
            s[0] = 1
        if len(act) == 2:
            if act[0] == 1:
                if compute_nones:
                    self._nones = 1
                if compute_ones:
                    self._ones = [pd.IndexSlice[0:1]]
                s = np.ones(length, dtype=np.ushort)
            else:
                if compute_nones:
                    self._nones = 0
                if compute_ones:
                    self._ones = []

            if raw:
                act = np.array(s, dtype=np.ushort)
            else:
                # noinspection PyTypeChecker
                act = bitarray(s.tolist())

            if not out:
                if self._sizeof_raw == -1:
                    self._sizeof_raw = sys.getsizeof(act) / 1e6
                self._time_compressed_to_raw = time() - t0
                self._n_compressed_to_raw += 1
            return act

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
        if raw:
            act = np.array(s, dtype=np.ushort)
            if not out:
                if self._sizeof_raw == -1:
                    self._sizeof_raw = sys.getsizeof(act) / 1e6
                self._time_compressed_to_raw = time() - t0
                self._n_compressed_to_raw += 1
        else:
            # noinspection PyTypeChecker
            act = bitarray(s.tolist())
            if not out:
                if self._sizeof_bitarray == -1:
                    self._sizeof_bitarray = sys.getsizeof(act) / 1e6
                self._time_compressed_to_bitarray = time() - t0
                self._n_compressed_to_bitarray += 1
        return act

    def __contains__(self, other: "Activation") -> bool:
        # TODO : pytests (for vmargot)
        nones_intersection = (self & other).nones
        if nones_intersection < min(self.nones, other.nones):
            return False
        return True

    @staticmethod
    def _compress(value: Union[np.ndarray, bitarray], dtype: type = str) -> Union[np.ndarray, str]:
        """Transforms a raw or bitarray activation vector to a compressed one.

        A compressed vector is a collection of integers starting by the initial value of the raw vector (0 or 1) and
        ending with its size. The other integers in the compression are the positions in the raw vector where the
        vector value changes. This stores all the information and saves up memory if the vector is constant over
        large periods of time.

        The compressed vector can be stored as a str looking like "0, 12, 456, ..., 47782" or as a numpy array of
        integers. What storage to use is specified by the "dtype" argument.
        """
        if not isinstance(value, (np.ndarray, bitarray)) or (value[-1] != 0 and value[-1] != 1):
            if isinstance(value, str):
                return np.array(value.split(",")).astype(int)
            return value
        if isinstance(value, np.ndarray):
            value = value.astype(int)
        else:
            value = np.diff(np.array(list(value)))
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
    def _raw_to_bitarray(value: np.ndarray) -> bitarray:
        """Casts a raw activation vector into a bitarray
        """
        if isinstance(value, bitarray):
            return value
        elif not isinstance(value, np.ndarray) or (value[-1] != 0 and value[-1] != 1):
            raise ValueError("Can not use _raw_to_bitarray or a compressed vector")
        # noinspection PyTypeChecker
        return bitarray(value.tolist())

    @property
    def raw(self) -> np.ndarray:
        """Returns the raw np.array. Will set relevant sizes if this has not been done yet"""
        if self.data_format == "bitarray":
            return self._bitarray_to_raw()
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
            if self.data_format == "bitarray":
                self._nones = self.data.count(1)  # faster than calling "raw"
            else:
                _ = self.as_bitarray  # calling as_bitarray will compute nones

        if self._coverage is None:
            self._coverage = self._nones / self.length
        return self._nones

    @property
    def entropy(self) -> int:
        if self._entropy is None:
            if self.data_format == "bitarray":
                t0 = time()
                compressed = self._compress(self.data)
                self._time_bitarray_to_compressed = time() - t0
                self._n_bitarray_to_compressed += 1
                if self.data_format == "compressed_str":
                    if self._sizeof_compressed_str == -1:
                        self._sizeof_compressed_str = sys.getsizeof(compressed) / 1e6
                if self.data_format == "compressed_array":
                    if self._sizeof_compressed_array == -1:
                        self._sizeof_compressed_array = sys.getsizeof(compressed) / 1e6
                self._entropy = len(ast.literal_eval(compressed)) - 2
            else:
                raise ValueError(
                    "Data format is not bitarray and yet entropy is not set. There is a problem in the "
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
    def as_bitarray(self):
        if self.data_format == "bitarray":
            if self._sizeof_bitarray == -1:
                self._sizeof_bitarray = sys.getsizeof(self.data)
            return self.data
        else:
            t0 = time()
            to_ret = self._decompress(raw=False)
            self._nones = to_ret.count(1)
            self._time_compressed_to_bitarray = time() - t0
            self._n_compressed_to_bitarray += 1
            if self._sizeof_bitarray == -1:
                self._sizeof_bitarray = sys.getsizeof(to_ret)
            return to_ret

    @property
    def as_compressed(self):
        if self.data_format == "compressed_array":
            if self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(self.data)
            return self.data
        elif self.data_format == "compressed_str":
            if self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(self.data)
            return self.data
        else:
            t0 = time()
            to_ret = self._compress(self.data)
            self._time_bitarray_to_compressed = time() - t0
            self._n_bitarray_to_compressed += 1
            if self.data_format == "compressed_array" and self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(to_ret)
            elif self.data_format == "compressed_str" and self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(self.data)
            return to_ret

    @property
    def as_compressed_array(self):
        if self.data_format == "compressed_array":
            if self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(self.data)
            return self.data
        else:
            t0 = time()
            to_ret = self._compress(self.data, dtype=np.ndarray)
            self._time_bitarray_to_compressed = time() - t0
            self._n_bitarray_to_compressed += 1
            if self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(to_ret)
            return to_ret

    @property
    def as_compressed_str(self):
        if self.data_format == "compressed_str":
            if self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(self.data)
            return self.data
        else:
            t0 = time()
            to_ret = self._compress(self.data, dtype=str)
            self._time_bitarray_to_compressed = time() - t0
            self._n_bitarray_to_compressed += 1
            if self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(to_ret)
            return to_ret

    @property
    def sizeof_raw(self):
        if self._sizeof_raw == -1 and Activation.FORCE_STAT:
            _ = self.raw
            if self._sizeof_raw == -1:
                raise ValueError("Calling 'raw' should have set _sizeof_raw")
        return self._sizeof_raw

    @property
    def sizeof_bitarray(self):
        if self._sizeof_bitarray == -1 and Activation.FORCE_STAT:
            _ = self.as_bitarray
            if self._sizeof_bitarray == -1:
                raise ValueError("Calling 'as_bitarray' should have set _sizeof_bitarray")
        return self._sizeof_bitarray

    @property
    def sizeof_compressed_array(self):
        if self._sizeof_compressed_array == -1 and Activation.FORCE_STAT:
            _ = self.as_compressed_array
            if self._sizeof_compressed_array == -1:
                raise ValueError("Calling 'as_compressed_array' should have set _sizeof_compressed_array")
        return self._sizeof_compressed_array

    @property
    def sizeof_compressed_str(self):
        if self._sizeof_compressed_str == -1 and Activation.FORCE_STAT:
            _ = self.as_compressed_str
            if self._sizeof_compressed_str == -1:
                raise ValueError("Calling 'as_compressed_str' should have set _sizeof_compressed_str")
        return self._sizeof_compressed_str

    @property
    def time_raw_to_compressed(self):
        if self._time_raw_to_compressed == -1 and Activation.FORCE_STAT:
            t0 = time()
            _ = self._compress(self.raw)
            self._time_raw_to_compressed = time() - t0
            self._n_raw_to_compressed += 1
        return self._time_raw_to_compressed

    @property
    def time_raw_to_bitarray(self):
        if self._time_raw_to_bitarray == -1 and Activation.FORCE_STAT:
            t0 = time()
            _ = self._raw_to_bitarray(self.raw)
            self._time_raw_to_bitarray = time() - t0
            self._n_raw_to_bitarray += 1
            if self._time_raw_to_bitarray == -1:
                raise ValueError("Calling '_raw_to_bitarray' should have set _time_raw_to_bitarray")
        return self._time_raw_to_bitarray

    @property
    def time_compressed_to_raw(self):
        if self._time_compressed_to_raw == -1 and Activation.FORCE_STAT:
            _ = self._decompress(self.as_compressed, out=False)
            if self._time_compressed_to_raw == -1:
                raise ValueError("Calling '_decompress' should have set _time_compressed_to_raw")
        return self._time_compressed_to_raw

    @property
    def time_bitarray_to_raw(self):
        if self._time_bitarray_to_raw == -1 and Activation.FORCE_STAT:
            _ = self._bitarray_to_raw(self.as_bitarray, out=False)
            if self._time_bitarray_to_raw == -1:
                raise ValueError("Calling '_bitarray_to_raw' should have set _time_bitarray_to_raw")
        return self._time_bitarray_to_raw

    @property
    def time_compressed_to_bitarray(self):
        if self._time_compressed_to_bitarray == -1 and Activation.FORCE_STAT:
            _ = self._decompress(self.as_bitarray, raw=False)
            if self._time_compressed_to_bitarray == -1:
                raise ValueError("Calling '_decompress' with raw=False should have set _time_compressed_to_bitarray")
        return self._time_compressed_to_bitarray

    @property
    def time_bitarray_to_compress(self):
        if self._time_bitarray_to_compressed == -1 and Activation.FORCE_STAT:
            t0 = time()
            _ = self._compress(self.as_bitarray)
            self._time_bitarray_to_compressed = time() - t0
            self._n_bitarray_to_compressed += 1
        return self._time_bitarray_to_compressed

    @property
    def n_raw_to_compressed(self):
        return self._n_raw_to_compressed

    @property
    def n_compressed_to_raw(self):
        return self._n_compressed_to_raw

    @property
    def n_raw_to_bitarray(self):
        return self._n_raw_to_bitarray

    @property
    def n_bitarray_to_raw(self):
        return self._n_bitarray_to_raw

    @property
    def n_bitarray_to_compressed(self):
        return self._n_bitarray_to_compressed

    @property
    def n_compressed_to_bitarray(self):
        return self._n_compressed_to_bitarray
