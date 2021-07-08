from abc import ABC
from typing import Union
import numpy as np
import ast
import sys
from time import time
from bitarray import bitarray
from tempfile import gettempdir
from pathlib import Path
from .logger.logger import log as logger

MAX_INT_32 = 2 ** 32


class Activation(ABC):

    DTYPE = str
    FORCE_STAT = False
    WILL_COMPARE = False
    DEFAULT_TEMPDIR = Path(gettempdir())

    @classmethod
    def clean_files(cls):
        for path in cls.DEFAULT_TEMPDIR.glob("ACTIVATION_VECTOR_*.txt"):
            path.unlink()

    def __init__(
        self,
        activation: Union[np.ndarray, bitarray, str, int] = None,
        optimize: bool = True,
        length: int = None,
        name_for_file: str = None,
    ):
        """Compresses an activation vector into a str(list) describing its variations or an bitarray of booleans

        stored data will be either a str(list):
            Compression is done : First element of the list is the first value of the array last element of the list is
            the length of the array The other elemnts are the coordinates that changed values
        or a np.ndarray:
            same as str(list) but the list is casted into np.array instead of str
        or an bitarray:
            The input vector [1 0 0 1 0 0 0 1 1...] where each entry uses up one bit of memory
        or an integer
            taking the input vector [1 0 0 1 0 0 0 1 1...], converts it to binary string representation :
            "100100011..." then cast it into int using int(s, 2). This is done if Activation.WILL_COMPARE is True :
            converting to int is slower than to bit array, but comparing ints is faster. Size is equivalent to bitarray.
        or in a file stored locally.
            This is done if name_for_file is not None

        The method will choose how to store the data based on the size (in MB) of the compressed list : if it is
        superior than in bitarray/int, it will take less memory and is prefered. If compression is used and
        dtype is np.ndarray, will check that numbers present in the compressed vector can be stored as int32 to gain
        memory. Else, uses int64.

        Parameters
        ----------
        activation: Union[np.ndarray, bitarray, str, int]
            If np.ndarray : Of the form [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1], or compressed vector
            If str : compressed vector
            If bitarray : same as np.array but takes 64x less memory (each entry is stored in one bit only)
            If int : as an integer
        optimize: bool
            Only relevent 'value' is an bitarray or integer. In that case, will check whether using compression saves up
            memory. Else, does not check and uses bitarray or integer. Note that if optimize is True, entropy is
            computed.
        length: int
            Only valid if 'value' is an integer. An activation vector stored as an integer has lost the information
            about its size : [0 0 0 1 0 0 0 1 1...] to nit gives 100011... which in turn gives back [1 0 0 0 1 1...].
            To get the leading zeros back, one must specify the length of the activation vector.
        name_for_file: str
            If specified, then activation vector is stored in a file in
            Activation.DEFAULT_TEMPDIR / ACTIVATION_VECTOR_name_for_file.txt
        """
        self.length = None  # Will be set by init methods
        self._entropy = None  # Will be set if activation is not an integer or if optimize is True
        self.data_format = None  # Will be set by init methods
        self.data = None  # Will be set by init methods
        self._rel_entropy = None  # Will be set if activation is not an integer or if optimize is True
        self._nones = None  # Will be set if activation is not an integer or if optimize is True
        self._coverage = None
        self._time_write = -1
        self._time_read = -1
        self._time_compressed_to_raw = -1
        self._time_raw_to_compressed = -1
        self._time_raw_to_integer = -1
        self._time_bitarray_to_raw = -1
        self._time_raw_to_bitarray = -1
        self._time_compressed_to_bitarray = -1
        self._time_bitarray_to_compressed = -1
        self._time_integer_to_compressed = -1
        self._time_integer_to_raw = -1
        self._n_written = 0
        self._n_read = 0
        self._n_compressed_to_raw = 0
        self._n_raw_to_compressed = 0
        self._n_bitarray_to_raw = 0
        self._n_integer_to_raw = 0
        self._n_raw_to_bitarray = 0
        self._n_raw_to_integer = 0
        self._n_bitarray_to_compressed = 0
        self._n_integer_to_compressed = 0
        self._n_compressed_to_bitarray = 0
        self._n_compressed_to_integer = 0
        self._sizeof_compressed_array = -1
        self._sizeof_compressed_str = -1
        self._sizeof_bitarray = -1
        self._sizeof_integer = -1
        self._sizeof_raw = -1
        self._sizeof_file = -1
        self._sizeof_path = -1

        if isinstance(activation, str) and "," not in activation:
            if Activation.WILL_COMPARE:
                activation = int(activation, 2)
            else:
                activation = bitarray(activation)

        if isinstance(activation, bitarray):
            self._init_with_bitarray(activation, Activation.DTYPE, optimize)

        elif isinstance(activation, int):
            self._init_with_integer(activation, Activation.DTYPE, length, optimize)

        elif isinstance(activation, str):
            self._init_with_str(activation)

        elif isinstance(activation, np.ndarray):
            if activation[-1] > 1:
                self._init_with_compressed_array(activation)
            else:
                if name_for_file is None:
                    self._init_with_raw(activation, Activation.DTYPE)
                else:
                    self._write(activation, name_for_file)
        else:
            raise TypeError(
                f"An activation can only be a np.ndarray, and bitarray, a str or an integer. Got"
                f" {type(activation)}."
            )

    def _write(self, value: np.ndarray, name: str):

        logger.debug(f"Activation vector is raw, store it in a file")
        if name is None:
            raise ValueError("If storing to file, need to provide a name")
        self._sizeof_raw = sys.getsizeof(value) / 1e6
        self.length = len(value)
        self._nones = np.count_nonzero(value == 1)
        t0 = time()
        self.data = Activation.DEFAULT_TEMPDIR / f"ACTIVATION_VECTOR_{name}.txt"
        self.data_format = "file"
        if self.data.is_file():
            raise FileExistsError(f"There is already an activation vector with the name {name}")
        with open(self.data, "wb") as f:
            # noinspection PyTypeChecker
            np.save(f, value, allow_pickle=False)
        stat = self.data.stat()
        if isinstance(stat, dict):
            self._sizeof_file = stat["st_size"] / 1e6
        else:
            self._sizeof_file = stat.st_size / 1e6
        self._sizeof_path = sys.getsizeof(self.data) / 1e6
        self._time_write = time() - t0
        self._n_written += 1

    def _read(self, path: Path = None, out: bool = True) -> np.ndarray:
        if path is None:
            if not self.data_format == "file":
                raise ValueError("Activation vector was not saved locally : can not read it.")
            path = self.data
        t0 = time()
        with open(path, "rb") as f:
            # noinspection PyTypeChecker
            value = np.load(f)
        if not out:
            if self._sizeof_raw == -1:
                self._sizeof_raw = sys.getsizeof(value) / 1e6
            if self._nones is None:
                self._nones = np.count_nonzero(value == 1)
            self._time_read = time() - t0
            self._n_read += 1
        return value

    def _init_with_bitarray(self, value: bitarray, dtype: type, optimize: bool = True):

        """
        Will set
          * self._nones (number of ones in the activation)
        if Optimize is True:
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
        self._nones = value.count(1)

        if optimize:
            t0 = time()
            compressed = self._compress(value, dtype=dtype)
            self._time_bitarray_to_compressed = time() - t0
            self._n_bitarray_to_compressed += 1
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
        self._sizeof_integer = sys.getsizeof(value) / 1e6

        if optimize:
            raw = self._integer_to_raw(value)
            self._sizeof_raw = sys.getsizeof(raw) / 1e6
            self._nones = np.count_nonzero(raw == 1)
            t0 = time()
            compressed = self._compress(raw, dtype=dtype)
            self._time_integer_to_compressed = time() - t0
            self._n_integer_to_compressed += 1
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
            if size_compressed > self._sizeof_integer:
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
        if value.dtype != np.ubyte:
            value = value.astype(np.ubyte)
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

    @staticmethod
    def logical_and(r1: "Activation", r2: "Activation", name: str = None) -> "Activation":
        if r1.length != r2.length:
            raise ValueError(f"Activations have different lengths. Left is {r1.length}, right is {r2.length}")

        if (r1.data_format == "bitarray" and r2.data_format == "bitarray") or (
            r1.data_format == "integer" and r2.data_format == "integer"
        ):
            return Activation(r1.data & r2.data, name_for_file=name)
        else:
            return Activation(r1.raw * r2.raw, name_for_file=name)

    def __or__(self, other: "Activation") -> "Activation":
        if self.length != other.length:
            raise ValueError(f"Activations have different lengths. Left is {self.length}, right is {other.length}")
        if (self.data_format == "bitarray" and other.data_format == "bitarray") or (
            self.data_format == "integer" and other.data_format == "integer"
        ):
            return Activation(self.data or other.data)
        else:
            return Activation(np.logical_or(self.raw, other.raw, length=self.length).astype("int32"))

    def __add__(self, other: "Activation") -> "Activation":
        if self.length != other.length:
            raise ValueError(f"Activations have different lengths. Left is {self.length}, right is {other.length}")
        if (self.data_format == "bitarray" and other.data_format == "bitarray") or (
            self.data_format == "integer" and other.data_format == "integer"
        ):
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
        if (self.data_format == "bitarray" and other.data_format == "bitarray") or (
            self.data_format == "integer" and other.data_format == "integer"
        ):
            return Activation((self.data ^ other.data) & self.data)
        else:
            return Activation(np.logical_xor(self.raw, other.raw).astype("int32") * self.raw)

    def __len__(self):
        return self.length

    def _integer_to_raw(self, value: Union[int, Path] = None, out: bool = True) -> np.ndarray:
        """From a value of the form 45786542 (int), which is the base 10 representation of the binary form of an
        activation vector, returns the initial vector.
        """
        t0 = time()
        if value is None:
            out = False
            if self.data_format == "file":
                value = self._read(out=False)
            else:
                value = self.data

        if isinstance(value, np.ndarray) and (value[-1] == 0 or value[-1] == 1):
            if not out:
                self._time_integer_to_raw = time() - t0
                self._n_integer_to_raw += 1
            return value

        if isinstance(value, (bitarray, np.ndarray, str, Path)):
            raise TypeError("Can not apply _integer_to_raw on a bitarray, raw, compressed or a path")
        if not isinstance(value, int):
            raise TypeError(f"Invalid format {type(value)}")
        act = np.fromiter(bin(value)[2:], dtype=np.ubyte)
        if self._sizeof_integer == -1 and not out:
            self._sizeof_integer = sys.getsizeof(value) / 1e6

        if len(act) > self.length:
            raise ValueError(
                "After using _integer_to_raw, I ended up with an activation vector bigger than the specified "
                "max length. This should not happend as the max length should have been set by the indexing "
                "of x earlier in your code"
            )
        act_bis = np.zeros(self.length).astype(np.ubyte)
        act_bis[self.length - len(act):] = act

        if not out:
            if self._sizeof_raw == -1:
                self._sizeof_raw = sys.getsizeof(act_bis) / 1e6
            self._time_integer_to_raw = time() - t0
            self._n_integer_to_raw += 1
            if self._nones is None:
                self._nones = np.count_nonzero(act_bis == 1)
        return act_bis

    def _bitarray_to_raw(self, value: Union[bitarray, Path] = None, out=True) -> np.ndarray:
        """Transforms a bitarray to a nparray"""
        t0 = time()
        if value is None:
            out = False
            if self.data_format == "file":
                value = self._read(out=False)
            else:
                value = self.data

        if isinstance(value, np.ndarray) and (value[-1] == 0 or value[-1] == 1):
            if not out:
                self._time_bitarray_to_raw = time() - t0
                self._n_bitarray_to_raw += 1
            return value

        if isinstance(value, (int, np.ndarray, str, Path)):
            raise TypeError("Can not apply _bitarray_to_raw on a raw, integer, compressed or a path")
        if not isinstance(value, bitarray):
            raise TypeError(f"Invalid format {type(value)}")
        act = np.array(list(value), dtype=np.ubyte)

        if not out:
            if self._sizeof_raw == -1:
                self._sizeof_raw = sys.getsizeof(act) / 1e6
            if self._sizeof_bitarray == -1:
                self._sizeof_bitarray = sys.getsizeof(value) / 1e6
            self._time_bitarray_to_raw = time() - t0
            self._n_bitarray_to_raw += 1
        return act

    def _decompress(
        self, value: Union[str, np.ndarray, Path] = None, raw=True, out=True
    ) -> Union[np.ndarray, bitarray]:
        """Will return the original activation vector, and set self._nones

        If raw is True (default), returns it as a np.ndarray, else as a bitarray
        """
        t0 = time()

        if value is None:
            out = False
            if self.data_format == "file":
                value = self._read(out=False)
            else:
                value = self.data

        if isinstance(value, (int, bitarray, Path)):
            raise TypeError("Can not apply _decompress on a bitarray, integer or Path")
        if not isinstance(value, (str, np.ndarray)):
            raise TypeError(f"Invalid format {type(value)}")

        if value[-1] == 0 or value[-1] == 1:
            if not out:
                self._time_compressed_to_raw = time() - t0
                self._n_compressed_to_raw += 1
            return value

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
        s = np.zeros(length, dtype=np.ubyte)
        previous_value = 0
        previous_index = 0

        if act[0] == 1:
            previous_value = 1
            s[0] = 1
        if len(act) == 2:
            if act[0] == 1:
                s = np.ones(length, dtype=np.ubyte)
            if raw:
                act = np.array(s, dtype=np.ubyte)
            else:
                # noinspection PyTypeChecker
                act = bitarray(s.tolist())

            if not out:
                if self._sizeof_raw == -1:
                    self._sizeof_raw = sys.getsizeof(act) / 1e6
                if self._nones is None:
                    self._nones = np.count_nonzero(act == 1)
                self._time_compressed_to_raw = time() - t0
                self._n_compressed_to_raw += 1
            return act

        for index in act[1:]:
            if previous_value == 0:
                previous_index = index
                previous_value = 1
            else:
                s[previous_index:index] = np.ones(index - previous_index)
                previous_index = index
                previous_value = 0

        if raw:
            act = np.array(s, dtype=np.ubyte)
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
        if self._nones is None:
            self._nones = np.count_nonzero(act == 1)
        return act

    def __contains__(self, other: "Activation") -> bool:
        nones_intersection = (Activation.logical_and(self, other)).nones
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
        if isinstance(value, int):
            raise TypeError("Can not compress an integer vector")
        if not isinstance(value, (np.ndarray, bitarray)) or (value[-1] != 0 and value[-1] != 1):
            if isinstance(value, str):
                return np.array(value.split(",")).astype(np.ubyte)
            return value
        if isinstance(value, np.ndarray):
            # not ubyte (unsigned byte), because np.diff will produce negative value that ubyte can not handle
            value = value.astype(np.byte)
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
        """Casts a raw activation vector into a bitarray"""
        if isinstance(value, bitarray):
            return value
        elif not isinstance(value, np.ndarray) or (value[-1] != 0 and value[-1] != 1):
            raise TypeError("Can not use _raw_to_bitarray on a compressed vector")
        elif isinstance(value, int):
            raise TypeError("Can not use _raw_to_bitarray on a integer vector")
        # noinspection PyTypeChecker
        return bitarray(value.tolist())

    @staticmethod
    def _raw_to_integer(value: np.ndarray) -> int:
        """Casts a raw activation vector into the integer represented by its binary form
        Examples
        --------
        >>> from ruleskit import Activation
        >>> Activation._raw_to_integer(np.array([0, 1, 1, 0]))
        6  # the binary number '0110' is 6 in base 10
        """
        if isinstance(value, int):
            return value
        elif not isinstance(value, np.ndarray) or (value[-1] != 0 and value[-1] != 1):
            raise TypeError("Can not use _raw_to_integer or a compressed vector")
        elif isinstance(value, bitarray):
            raise TypeError("Can not use _raw_to_integer on a bitarray vector")
        to_ret = int("".join(str(i) for i in value.astype(np.ubyte)), 2)
        return to_ret

    @property
    def raw(self) -> np.ndarray:
        """Returns the raw np.array. Will set relevant sizes if this has not been done yet"""
        if self.data_format == "bitarray":
            return self._bitarray_to_raw()
        elif self.data_format == "integer":
            return self._integer_to_raw()
        elif self.data_format == "file":
            return self._read(out=False)
        elif "compressed" in self.data_format:
            return self._decompress()
        else:
            raise ValueError(f"Unkown activation format {self.data_format}")

    @property
    def ones(self) -> np.ndarray:
        """ Contrary to other @properties, do not store 'ones' in array. Since it is the list of indexes where the
        vector is one, 'ones' can be big : several MB or more. """
        raw = self.raw
        ones = np.where(raw == 1)[0].tolist()
        return ones

    @property
    def nones(self) -> int:
        """self._nones might not be set since it can only be set at object creation if the full array was given"""
        if self._nones is None:
            _ = self.raw  # calling raw will compute nones and ones

        if self._coverage is None:
            self._coverage = self._nones / self.length
        return self._nones

    @property
    def entropy(self) -> int:
        if self._entropy is None:
            t0 = time()
            fmt = self.data_format
            if self.data_format == "file":
                data = self._read(out=False)
            else:
                data = self.data
            compressed = self._compress(data)

            if fmt == "bitarray":
                self._time_bitarray_to_compressed = time() - t0
                self._n_bitarray_to_compressed += 1
            else:
                self._time_integer_to_compressed = time() - t0
                self._n_integer_to_compressed += 1

            if self.data_format == "compressed_str" and self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(compressed) / 1e6
            if self.data_format == "compressed_array" and self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(compressed) / 1e6

            self._entropy = len(ast.literal_eval(compressed)) - 2
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
        elif self.data_format == "integer":
            raw = self.raw
            t0 = time()
            to_ret = self._raw_to_bitarray(raw)
            self._time_raw_to_bitarray = time() - t0
            self._n_raw_to_bitarray += 1
            if self._sizeof_bitarray == -1:
                self._sizeof_bitarray = sys.getsizeof(to_ret)
            return to_ret
        elif "compressed" in self.data_format:
            t0 = time()
            to_ret = self._decompress(raw=False)
            self._time_compressed_to_bitarray = time() - t0
            self._n_compressed_to_bitarray += 1
            if self._sizeof_bitarray == -1:
                self._sizeof_bitarray = sys.getsizeof(to_ret)
            return to_ret
        elif self.data_format == "file":
            data = self._read(out=False)
            return self._raw_to_bitarray(data)
        else:
            raise ValueError(f"Unkown activation format {self.data_format}")

    @property
    def as_integer(self):
        if self.data_format == "integer":
            if self._sizeof_integer == -1:
                self._sizeof_integer = sys.getsizeof(self.data)
            return self.data
        else:
            t0 = time()
            to_ret = self._raw_to_integer(self.raw)
            self._time_raw_to_integer = time() - t0
            self._n_raw_to_integer += 1
            if self._sizeof_integer == -1:
                self._sizeof_integer = sys.getsizeof(to_ret)
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
        elif self.data_format == "integer":
            raw = self._integer_to_raw()
            self._nones = np.count_nonzero(raw == 1)
            t0 = time()
            to_ret = self._compress(raw)
            self._time_raw_to_compressed = time() - t0
            self._n_raw_to_compressed += 1
            if self.data_format == "compressed_array" and self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(to_ret)
            elif self.data_format == "compressed_str" and self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(self.data)
            return to_ret
        elif self.data_format == "bitarray":
            t0 = time()
            to_ret = self._compress(self.data)
            self._time_bitarray_to_compressed = time() - t0
            self._n_bitarray_to_compressed += 1
            if self.data_format == "compressed_array" and self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(to_ret)
            elif self.data_format == "compressed_str" and self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(self.data)
            return to_ret
        elif self.data_format == "file":
            data = self._read(out=False)
            to_ret = self._compress(data)
            return to_ret
        else:
            raise ValueError(f"Unkown activation format {self.data_format}")

    @property
    def as_compressed_array(self):
        if self.data_format == "compressed_array":
            if self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(self.data)
            return self.data
        if self.data_format == "compressed_str":
            to_ret = np.array(ast.literal_eval(self.data))
            if self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(to_ret)
            return to_ret
        elif self.data_format == "integer":
            raw = self._integer_to_raw()
            self._nones = np.count_nonzero(raw == 1)
            t0 = time()
            to_ret = self._compress(raw, dtype=np.ndarray)
            self._time_raw_to_compressed = time() - t0
            self._n_raw_to_compressed += 1
            if self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(to_ret)
            return to_ret
        elif self.data_format == "bitarray":
            t0 = time()
            to_ret = self._compress(self.data, dtype=np.ndarray)
            self._time_bitarray_to_compressed = time() - t0
            self._n_bitarray_to_compressed += 1
            if self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(to_ret)
            return to_ret
        elif self.data_format == "file":
            data = self._read(out=False)
            to_ret = self._compress(data, dtype=np.ndarray)
            return to_ret
        else:
            raise ValueError(f"Unkown activation format {self.data_format}")

    @property
    def as_compressed_str(self):
        if self.data_format == "compressed_str":
            if self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(self.data)
            return self.data
        if self.data_format == "compressed_array":
            to_ret = str(self.data).replace(" ", "").replace("[", "").replace("]", "")
            if self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(to_ret)
            return to_ret
        elif self.data_format == "integer":
            raw = self._integer_to_raw()
            self._nones = np.count_nonzero(raw == 1)
            t0 = time()
            to_ret = self._compress(raw, dtype=str)
            self._time_raw_to_compressed = time() - t0
            self._n_raw_to_compressed += 1
            if self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(to_ret)
            return to_ret
        elif self.data_format == "bitarray":
            t0 = time()
            to_ret = self._compress(self.data, dtype=str)
            self._time_bitarray_to_compressed = time() - t0
            self._n_bitarray_to_compressed += 1
            if self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(to_ret)
            return
        elif self.data_format == "file":
            data = self._read(out=False)
            to_ret = self._compress(data, dtype=str)
            return to_ret
        else:
            raise ValueError(f"Unkown activation format {self.data_format}")

    @property
    def sizeof_path(self):  # Can not force stat for file
        return self._sizeof_path

    @property
    def sizeof_file(self):  # Can not force stat for file
        return self._sizeof_file

    @property
    def sizeof_raw(self):
        if self._sizeof_raw == -1 and Activation.FORCE_STAT:
            _ = self.raw
        return self._sizeof_raw

    @property
    def sizeof_bitarray(self):
        if self._sizeof_bitarray == -1 and Activation.FORCE_STAT:
            _ = self.as_bitarray
        return self._sizeof_bitarray

    @property
    def sizeof_integer(self):
        if self._sizeof_integer == -1 and Activation.FORCE_STAT:
            _ = self.as_integer
        return self._sizeof_integer

    @property
    def sizeof_compressed_array(self):
        if self._sizeof_compressed_array == -1 and Activation.FORCE_STAT:
            _ = self.as_compressed_array
        return self._sizeof_compressed_array

    @property
    def sizeof_compressed_str(self):
        if self._sizeof_compressed_str == -1 and Activation.FORCE_STAT:
            _ = self.as_compressed_str
        return self._sizeof_compressed_str

    @property
    def time_write(self):  # Can not force write : would need to provide a name
        return self._time_write

    @property
    def time_read(self):  # Can not force read : file might not exist
        return self._time_read

    @property
    def time_raw_to_compressed(self):
        if self._time_raw_to_compressed == -1 and Activation.FORCE_STAT:
            t0 = time()
            _ = self._compress(self.raw)
            self._time_raw_to_compressed = time() - t0
            self._n_raw_to_compressed += 1
        return self._time_raw_to_compressed

    @property
    def time_raw_to_integer(self):
        if self._time_raw_to_integer == -1 and Activation.FORCE_STAT:
            t0 = time()
            _ = self._raw_to_integer(self.raw)
            self._time_raw_to_integer = time() - t0
            self._n_raw_to_integer += 1
        return self._time_raw_to_integer

    @property
    def time_raw_to_bitarray(self):
        if self._time_raw_to_bitarray == -1 and Activation.FORCE_STAT:
            t0 = time()
            _ = self._raw_to_bitarray(self.raw)
            self._time_raw_to_bitarray = time() - t0
            self._n_raw_to_bitarray += 1
        return self._time_raw_to_bitarray

    @property
    def time_compressed_to_raw(self):
        if self._time_compressed_to_raw == -1 and Activation.FORCE_STAT:
            _ = self._decompress(self.as_compressed, out=False)
        return self._time_compressed_to_raw

    @property
    def time_bitarray_to_raw(self):
        if self._time_bitarray_to_raw == -1 and Activation.FORCE_STAT:
            _ = self._bitarray_to_raw(self.as_bitarray, out=False)
        return self._time_bitarray_to_raw

    @property
    def time_integer_to_raw(self):
        if self._time_integer_to_raw == -1 and Activation.FORCE_STAT:
            _ = self._integer_to_raw(self.as_integer, out=False)
        return self._time_integer_to_raw

    @property
    def time_compressed_to_bitarray(self):
        if self._time_compressed_to_bitarray == -1 and Activation.FORCE_STAT:
            _ = self._decompress(self.as_bitarray, raw=False)
        return self._time_compressed_to_bitarray

    @property
    def time_bitarray_to_compressed(self):
        if self._time_bitarray_to_compressed == -1 and Activation.FORCE_STAT:
            b = self.as_bitarray
            t0 = time()
            _ = self._compress(b)
            self._time_bitarray_to_compressed = time() - t0
            self._n_bitarray_to_compressed += 1
        return self._time_bitarray_to_compressed

    @property
    def time_integer_to_compressed(self):
        if self._time_integer_to_compressed == -1 and Activation.FORCE_STAT:
            i = self.as_integer
            t0 = time()
            _ = self._compress(i)
            self._time_integer_to_compressed = time() - t0
            self._n_integer_to_compressed += 1
        return self._time_integer_to_compressed

    @property
    def n_written(self):
        return self._n_written

    @property
    def n_read(self):
        return self._n_read

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
    def n_raw_to_integer(self):
        return self._n_raw_to_integer

    @property
    def n_bitarray_to_raw(self):
        return self._n_bitarray_to_raw

    @property
    def n_integer_to_raw(self):
        return self._n_integer_to_raw

    @property
    def n_bitarray_to_compressed(self):
        return self._n_bitarray_to_compressed

    @property
    def n_integer_to_compressed(self):
        return self._n_integer_to_compressed

    @property
    def n_compressed_to_bitarray(self):
        return self._n_compressed_to_bitarray

    @property
    def n_compressed_to_integer(self):
        return self._n_compressed_to_integer
