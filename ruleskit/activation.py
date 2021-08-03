from abc import ABC
from typing import Union, List, Optional
import numpy as np
import ast
import sys
from copy import copy
import random
from time import time
from bitarray import bitarray
from tempfile import gettempdir
from pathlib import Path
from .logger.logger import log as logger

try:
    from transparentpath import TransparentPath
except ImportError:
    TransparentPath = None

MAX_INT_32 = 2 ** 32


class Activation(ABC):

    """An activation vector is a 1-D list of 0 and 1 reprenseting the activation of a rule. Each element corresponds to
    a line in the features and targets data (typically, a date, or a pair date-object if the data is multi-indexed),
    and the vector contains 0 if the condition of the rule is not met at this index in the data and 1 if it is.

    When working with lots of data and lots of rules, activation vectors can be hard to keep in memory and can represent
    bottlenecks in computation time. This class is designed to reduce the size in memory of the vector and to minimize
    the computation time.
    """

    # dtype for the compressed format. Can be str or np.ndarray
    DTYPE = str
    # When calling a profiling attribute that is None, will force the call of the method to compute them if FORCE_STAT
    # is True
    FORCE_STAT = False
    # Should be set to True if the program will do ogical AND between activation vectors in integer format at some point
    WILL_COMPARE = False
    # If activation vectors are to be stored on disk instead of RAM, root directory to store them
    DEFAULT_TEMPDIR = Path(gettempdir())
    # If no formatting should be done, just store raw np.ndarray in RAM
    STORE_RAW = False

    @classmethod
    def clean_files(cls):
        """Removes activation vector files, if any."""
        for path in cls.DEFAULT_TEMPDIR.glob("ACTIVATION_VECTOR_*.txt"):
            path.unlink()

    def __init__(
        self,
        activation: Union[np.ndarray, bitarray, str, int, Path],
        optimize: bool = True,
        length: Optional[int] = None,
        to_file: bool = True,
    ):
        """
        An activation vector is an array of 0 and 1, possibly millions of points. The whole purpose of this class is
        to efficiently store the vector and allow to use logcial AND easily between two vectors, no matter the stored
        format.

        The possible storing format are : compressed string or array, integer, bitarray or written on disk.
        Activations are written on disk only if 'to_file' is True. In that case, the raw np.ndarray vector is stored
        using np.save.
        If the vector is instantiated with any of compressed string or array, integer, bitarray, then this format is
        privileged for storing the vector (unless to_file is True), but some other flags like 'optimize' and
        'Activation.WILL_COMPARE' can determine the chosen format.

        If the raw np.ndarray or a path to it was given and if Activation.STORE_RAW is True, then no formatting is done
        and the raw np.ndarray is kept in RAM. This flag has no effect if activation is not the raw np.ndarray nor a
        path to it.

        Compression : can be either compressed str(list) or np.ndarray, depending on Activation.DTYPE:
            Compressed data is : First element of the list is the first value of the array, last element of the list is
            the length of the array. The other elements are the coordinates where the array changed values. This is the
            best solution regarding the RAM if the vector does not change often. However, it is often slower than the
            other methods, particularly if the code will apply logical AND between vectors since it requieres a
            decompression of both vectors.
            The vector is stored that was if compression is more memory-efficient than integer or bitarray format
            OR if optimize is False, or if 'activation' is already a compressed vector.
            It is never stored this way if to_file is True.
        Bitarray :
            The input vector [1 0 0 1 0 0 0 1 1...] where each entry uses up one bit of memory. Takes more RAM than
            compressed format if the vector does not change often, but is much quicker, both in conversion and in
            computing a logical AND.
            The vector is stored that way if it takes less RAM than compressed and if optimize is True and if
            Activation.WILL_COMPARE is False (converting to bitarray is faster than to integer, but computing
            logical AND on integers is faster). Size is equivalent to integer : one bit per entry. It is also stored
            this way if 'activation' is a string and Activation.WILL_COMPARE is False.
            It is never stored this way if to_file is True.
        Integer :
            taking the input vector [1 0 0 1 0 0 0 1 1...], converts it to binary string representation :
            "100100011..." then casts it into int using int(s, 2).
            The vector is stored that way if it takes less RAM than compressed and if optimize is True and if
            Activation.WILL_COMPARE is True. It is also stored this way if 'activation' is a string and
            Activation.WILL_COMPARE is True. It is never stored this way if to_file is True.
        File stored locally :
            This is done if to_file is True. It is often the best solution, and is the default one. Indeed the I/O
            operations using np.save and np.load are very fast, and the only thing in RAM is the path to the vector's
            file. One need enough disk space of course. Using this method will save the np.array with dtype=np.ubyte,
            so taking one byte (8 bits) per entry.

        If compression is used and dtype is np.ndarray, will check that numbers present in the compressed vector can be
        stored as int32 to gain memory. Else, uses int64.

        Parameters:
        -----------
        activation: Union[np.ndarray, bitarray, str, int, Path]
            If np.ndarray : Of the form [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1], or compressed vector
            If str : compressed vector, or '010011100011111', or path
            If bitarray : same as np.array but takes 8x less memory (each entry is stored in one bit only)
            If int : as an integer
            If Path : read raw np.ndarray vector from path.
        optimize: bool
            Only relevent if 'value' is a bitarray or an integer. In that case, will check whether using compression
            saves up memory. Else, does not check and uses bitarray or integer. Note that if optimize is True, entropy
            is computed no matter the chosen format.
        length: Optional[int]
            Only valid if 'value' is an integer. An activation vector stored as an integer has lost the information
            about its size : [0 0 0 1 0 0 0 1 1...] to int gives 100011... which in turn gives back [1 0 0 0 1 1...].
            To get the leading zeros back, one must specify the length of the activation vector.
        to_file: bool
            If True, then activation vector is stored in a file in
            Activation.DEFAULT_TEMPDIR / ACTIVATION_VECTOR_available_number.txt (default value = True)
        """

        # Analytical attributes
        self._entropy = None  # Will be set if activation is not an integer or if optimize is True
        self._rel_entropy = None  # Will be set if activation is not an integer or if optimize is True
        self._nones = None  # Will be set if activation is not an integer or if optimize is True
        self._coverage = None

        # Format attributes
        self.optimize = optimize
        self.length = None  # Will be set by init methods
        self.data_format = None  # Will be set by init methods
        self.data = None  # Will be set by init methods

        """
        Profiling attribtues. All times are in seconds, all sizes in MB. Attributes starting by '_n_' are counts of
        how many times this or that method triggered.
        """

        self._time_write = -1
        self._time_read = -1
        self._time_compressed_to_raw = -1
        self._time_raw_to_compressed = -1
        self._time_raw_to_integer = -1
        self._time_bitarray_to_raw = -1
        self._time_raw_to_bitarray = -1
        self._time_compressed_to_bitarray = -1
        self._time_compressed_to_integer = -1
        self._time_bitarray_to_compressed = -1
        self._time_integer_to_compressed = -1
        self._time_integer_to_raw = -1
        self._time_bitarray_to_integer = -1
        self._time_integer_to_bitarray = -1
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
        self._n_bitarray_to_integer = 0
        self._n_integer_to_bitarray = 0
        self._sizeof_compressed_array = -1
        self._sizeof_compressed_str = -1
        self._sizeof_bitarray = -1
        self._sizeof_integer = -1
        self._sizeof_raw = -1
        self._sizeof_file = -1
        self._sizeof_path = -1

        # does not use instance to avoid conflicts if using TransparentPath
        if type(activation) == "str" and "," not in activation:
            if any([c in activation for c in "abcdefhijklmnopqrstuvwxy./\\"]):  # is a file path
                if TransparentPath is not None:
                    # noinspection PyCallingNonCallable
                    activation = TransparentPath(activation)
                else:
                    activation = Path(activation)
            else:
                if Activation.WILL_COMPARE:
                    activation = int(activation, 2)
                else:
                    activation = bitarray(activation)
        elif isinstance(activation, int) and not Activation.WILL_COMPARE:
            if length is None:
                raise ValueError("When giving an integer to Activation, you must also specify its length.")
            self._sizeof_integer = sys.getsizeof(activation)
            t0 = time()
            s = bin(activation)[2:]
            if len(s) != length:
                s = "0" * (length - len(s)) + s
            activation = bitarray(s)
            self._time_integer_to_bitarray = time() - t0
            self._n_integer_to_bitarray += 1
            self._sizeof_bitarray = sys.getsizeof(activation)
        elif isinstance(activation, bitarray) and Activation.WILL_COMPARE:
            self._sizeof_bitarray = sys.getsizeof(activation)
            self._nones = activation.count(1)
            t0 = time()
            if length is None:
                length = len(activation)
            activation = int(
                str(activation).replace("bitarray(", "").replace(")", "").replace("'", "").replace('"', ""), 2
            )
            self._time_bitarray_to_integer = time() - t0
            self._n_bitarray_to_integer += 1
            self._sizeof_integer = sys.getsizeof(activation)

        # noinspection PyTypeChecker
        if isinstance(activation, Path) or TransparentPath is not None and isinstance(activation, TransparentPath):
            activation = self._read(activation, out=False)

        self._init_with_any(activation, length, to_file)

    def __copy__(self) -> "Activation":
        if self.data_format == "integer":
            return Activation(copy(self.data), optimize=self.optimize, length=self.length)
        return Activation(copy(self.data), optimize=self.optimize, to_file=self.data_format == "file")

    # def __del__(self):
    #     if hasattr(self, "data") and hasattr(self, "data_format") and self.data_format == "file":
    #         if self.data.is_file():
    #             self.data.unlink()

    def _init_with_any(self, activation: Union[np.ndarray, bitarray, int, str], length: int, to_file: bool):
        if isinstance(activation, bitarray):
            self._init_with_bitarray(activation, Activation.DTYPE)

        elif isinstance(activation, int):
            self._init_with_integer(activation, Activation.DTYPE, length)

        elif isinstance(activation, str):
            self._init_with_str(activation)

        elif isinstance(activation, np.ndarray):
            if activation[-1] > 1:
                self._init_with_compressed_array(activation)
            else:
                if to_file:
                    if not self._write(self.to_raw_from_any(activation, out=False)):
                        self._init_with_any(activation, length=length, to_file=False)
                else:
                    self._init_with_raw(activation, Activation.DTYPE)
        else:
            raise TypeError(
                f"An activation can only be a np.ndarray, and bitarray, a str or an integer. Got"
                f" {type(activation)}."
            )

    def to_raw_from_any(self, activation, out: bool = True) -> np.ndarray:
        """Converts any format among integer, bitarray, compressed string or compressed array to raw activation
        vector np.ndarray"""
        if isinstance(activation, bitarray):
            raw = self._bitarray_to_raw(activation, out=out)
        elif isinstance(activation, int):
            raw = self._integer_to_raw(activation, out=out)
        elif isinstance(activation, (str, np.ndarray)):
            raw = self._decompress(activation, raw=True, out=out)
        else:
            raise TypeError(
                f"An activation can only be a np.ndarray, and bitarray, a str or an integer. Got"
                f" {type(activation)}."
            )
        return raw

    def delete(self):
        """Deletes the activation vector's data, either by deleting the local file or by calling del on self.data. In
        the later case, self.data is reset to None."""
        if self.data_format == "file":
            if self.data.is_file():
                self.data.unlink()
        else:
            del self.data
            self.data = None

    """Init methods"""

    def _write(self, value: np.ndarray):
        """Writes the activation vector's raw np.ndarray to a file in Activation.DEFAULT_TEMPDIR under the name
        ACTIVATION_VECTOR_{n}.txt, where n is a random integer chosen among available numbers from 0 to 1e64.

        Will set:
             * self._sizeof_raw
             * self.length
             * self._nones
             * self._sizeof_file
             * self._sizeof_path
             * self_time_write.
             * self.data to the path to the file
             * self.data_format as "file"
        Will iterate self._n_written.
        """
        if value.dtype != np.ubyte:  # Saves memory
            value = value.astype(np.ubyte)
        logger.debug(f"Activation vector is raw, store it in a file")
        self._sizeof_raw = sys.getsizeof(value) / 1e6
        self.length = len(value)
        self._nones = np.count_nonzero(value == 1)
        t0 = time()
        arange = list((0, int(1e64)))
        number = random.randint(*arange)
        data = Activation.DEFAULT_TEMPDIR / f"ACTIVATION_VECTOR_{number}.txt"
        attempts = 0
        while data.is_file():
            if attempts > 99:
                logger.warning(
                    "Failed to save activation vector locally after 100 attempts at finding an available"
                    " name. Will keep it in RAM."
                )
                return False
            number += 1
            attempts += 1
            arange.remove(number)
            number = random.randint(*arange)
            data = Activation.DEFAULT_TEMPDIR / f"ACTIVATION_VECTOR_{number}.txt"
        data.touch()
        self.data = data
        self.data_format = "file"
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
        return True

    def _init_with_bitarray(self, value: bitarray, dtype: type):

        """
        Will set
            * self._nones (number of ones in the activation)
            * self.length
            * self._sizeof_bitarray
            if self.optimize is True:
                  * self._entropy and self._rel_entropy
                  * self.data_format to "bitarray" or "compressed_str" or "compressed_array" depending on what takes
                    less memory
                  * self.data as a bitarray, a str or an array
                  Will iterate self._n_bitarray_to_compressed
            else:
                  * self.data as a bitarray
                  * self.data_format to "bitarray"
        """

        logger.debug(f"Activation vector is a bitarray")
        self.length = len(value)
        self._sizeof_bitarray = sys.getsizeof(value) / 1e6
        self._nones = value.count(1)

        if self.optimize:
            t0 = time()
            raw = self._bitarray_to_raw(value, out=False)
            t1 = time()
            compressed = self._compress(raw, dtype=dtype)
            self._time_raw_to_compressed = time() - t1
            self._n_raw_to_compressed += 1
            self._time_bitarray_to_compressed = time() - t0
            self._n_bitarray_to_compressed += 1
            if isinstance(compressed, str):
                self._sizeof_compressed_str = sys.getsizeof(compressed) / 1e6
                size_compressed = self._sizeof_compressed_str
            else:
                self._sizeof_compressed_array = sys.getsizeof(compressed) / 1e6
                size_compressed = self._sizeof_compressed_array
            if dtype == str:
                self._entropy = len(ast.literal_eval(compressed))
            else:
                self._entropy = len(compressed)
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

    def _init_with_integer(self, value: int, dtype: type, length: Optional[int] = None):

        """
        Will set
            * self.length
            * self._sizeof_integer
            if self.optimize is True:
                * self._nones (number of ones in the activation)
                * self._entropy and self._rel_entropy
                * self.data_format to "integer" or "compressed_str" or "compressed_array" depending on what takes less
                memory
                * self.data as an integer, a str or an np.ndarray
                Will iterate self._n_integer_to_compressed
            else:
                * self.data as an integer
                * self.data_format to "integer"
        """

        if length is None:
            raise ValueError("When giving an integer to Activation, you must also specify its length.")

        logger.debug(f"Activation vector is an int")
        self.length = length
        self._sizeof_integer = sys.getsizeof(value) / 1e6

        if self.optimize:
            t0 = time()
            raw = self._integer_to_raw(value, out=False)
            self._nones = np.count_nonzero(raw == 1)
            t1 = time()
            compressed = self._compress(raw, dtype=dtype)
            self._time_raw_to_compressed = time() - t1
            self._n_raw_to_compressed += 1
            self._time_integer_to_compressed = time() - t0
            self._n_integer_to_compressed += 1
            if isinstance(compressed, str):
                self._sizeof_compressed_str = sys.getsizeof(compressed) / 1e6
                size_compressed = self._sizeof_compressed_str
            else:
                self._sizeof_compressed_array = sys.getsizeof(compressed) / 1e6
                size_compressed = self._sizeof_compressed_array
            if dtype == str:
                self._entropy = len(ast.literal_eval(compressed))
            else:
                self._entropy = len(compressed)
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
            * self._sizeof_compressed_str
            * self.data_format as "compressed_str"
            * self._entropy and self._rel_entropy
            * self.length
        """
        logger.debug(f"Activation vector is a compressed str")
        evaluated = np.array(ast.literal_eval(value))
        self.data = value
        self._sizeof_compressed_str = sys.getsizeof(value) / 1e6
        self._entropy = len(evaluated)
        self.length = evaluated[-1]
        self._rel_entropy = self._entropy / self.length
        self.data_format = "compressed_str"

    def _init_with_compressed_array(self, value: np.ndarray):
        """
        will set :
            * self.data as a compressed array
            * self._sizeof_compressed_array
            * self.data_format as "compressed_array"
            * self._entropy and self._rel_entropy
            * self.length
        """
        logger.debug(f"Activation vector is a compressed array")
        self.data = value
        self._sizeof_compressed_array = sys.getsizeof(value) / 1e6
        self._entropy = len(value)
        self.length = value[-1]
        self._rel_entropy = self._entropy / self.length
        self.data_format = "compressed_array"

    def _init_with_raw(self, value: np.ndarray, dtype: type):
        """
        will set :
            * self.data as an integer or a compressed array/str depending on what takes less memory and on what dtype is
            * self._sizeof_compressed_str or self._sizeof_compressed_array if Activation.STORE_RAW is False
            * self._sizeof_raw
            * self._time_raw_to_compressed if Activation.STORE_RAW is False
            * self._time_raw_to_bitarray or integer if Activation.STORE_RAW is False
            * self._sizeof_bitarray or integer if Activation.STORE_RAW is False
            * self.data_format as "bitarray", "integer", "raw" or "compressed_array" or "compressed_str"
            * self._entropy and self._rel_entropy
            * self.length
            * self._nones
        Will iterate :
            * self._n_raw_to_bitarray or integer if Activation.STORE_RAW is False
            * self._n_raw_to_compressed if Activation.STORE_RAW is False
        """
        if value.dtype != np.ubyte:
            value = value.astype(np.ubyte)
        logger.debug(f"Activation vector is raw")
        self._sizeof_raw = sys.getsizeof(value) / 1e6
        self.length = len(value)
        self._nones = np.count_nonzero(value == 1)

        if Activation.STORE_RAW:
            self.data = value
            self.data_format = "raw"
            return

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
            self._entropy = len(ast.literal_eval(compressed))
        else:
            self._entropy = len(compressed)
        self._rel_entropy = self._entropy / self.length
        t0 = time()
        if Activation.WILL_COMPARE:
            inbit = self._raw_to_integer(value)
            self._time_raw_to_integer = time() - t0
            self._n_raw_to_integer += 1
            self._sizeof_integer = sys.getsizeof(inbit) / 1e6
            thesize = self._sizeof_integer
            s = f"Using integer activation representation"
            fmt = "integer"
        else:
            inbit = self._raw_to_bitarray(value)
            self._time_raw_to_bitarray = time() - t0
            self._n_raw_to_bitarray += 1
            self._sizeof_bitarray = sys.getsizeof(inbit) / 1e6
            thesize = self._sizeof_bitarray
            s = f"Using bitarray activation representation"
            fmt = "bitarray"
        if thesize > size_compressed:
            logger.debug(f"Using compressed activation representation")
            self.data = compressed
            if dtype == str:
                self.data_format = "compressed_str"
            else:
                self.data_format = "compressed_array"
        else:
            logger.debug(s)
            self.data_format = fmt
            self.data = inbit

    """ Binary operators """

    def __and__(self, a2: "Activation") -> "Activation":
        """logical AND of two activation vectors. Only valid if both have the same length. Both vectors DO NOT need
        to have the same format."""
        if self.length != a2.length:
            raise ValueError(f"Activations have different lengths. Left is {self.length}, right is {a2.length}")

        if (self.data_format == "bitarray" and a2.data_format == "bitarray") or (
            self.data_format == "integer" and a2.data_format == "integer"
        ):  # gains time by not using raw
            return Activation(self.data & a2.data, optimize=self.optimize and a2.optimize, to_file=False)
        else:
            return Activation(
                self.raw * a2.raw,
                optimize=self.optimize and a2.optimize,
                to_file=self.data_format == "file" and a2.data_format == "file",
            )

    @staticmethod
    def multi_logical_and(acs: List["Activation"], asarray: bool = False) -> Union["Activation", np.ndarray]:
        """Do LOGICAL AND on many activation vectors at once. Uses raw np.ndarrays to gain time.
        If asarray is True, does not cast the result into an Activation object but returns the raw np.ndarray."""
        if len(acs) == 1:
            res = acs[0].raw
        else:
            res = np.vstack([a.raw for a in acs]).all(axis=0).astype(np.ubyte)
        if asarray:
            return res
        return Activation(
            res,
            optimize=all([a.optimize for a in acs]),
            to_file=all([a.data_format == "file" for a in acs]),
        )

    def __or__(self, a2: "Activation") -> "Activation":
        """LOGICAL OR of two activation vectors. Only valid if both have the same length. Both vectors DO NOT need
        to have the same format."""
        if self.length != a2.length:
            raise ValueError(f"Activations have different lengths. Left is {self.length}, right is {a2.length}")

        if (self.data_format == "bitarray" and a2.data_format == "bitarray") or (
            self.data_format == "integer" and a2.data_format == "integer"
        ):  # gains time by not using raw
            return Activation(self.data | a2.data, optimize=self.optimize and a2.optimize, to_file=False)
        else:
            return Activation(
                self.raw | a2.raw,
                optimize=self.optimize and a2.optimize,
                to_file=self.data_format == "file" and a2.data_format == "file",
            )

    @staticmethod
    def multi_logical_or(acs: List["Activation"], asarray: bool = False) -> Union["Activation", np.ndarray]:
        """Do LOGICAL OR on many activation vectors at once. Uses raw np.ndarrays to gain time.
        If asarray is True, does not cast the result into an Activation object but returns the raw np.ndarray."""
        if len(acs) == 1:
            res = acs[0].raw
        else:
            res = np.vstack([a.raw for a in acs]).any(axis=0).astype(np.ubyte)
        if asarray:
            return res
        return Activation(
            res,
            optimize=all([a.optimize for a in acs]),
            to_file=all([a.data_format == "file" for a in acs]),
        )

    def __xor__(self, a2: "Activation") -> "Activation":
        """Logcial EXCLUSIVE OR of two activation vectors. Only valid if both have the same length. Both vectors DO NOT
        need to have the same format."""
        if self.length != a2.length:
            raise ValueError(f"Activations have different lengths. Left is {self.length}, right is {a2.length}")

        if (self.data_format == "bitarray" and a2.data_format == "bitarray") or (
            self.data_format == "integer" and a2.data_format == "integer"
        ):  # gains time by not using raw
            return Activation(self.data ^ a2.data, optimize=self.optimize and a2.optimize, to_file=False)
        else:
            return Activation(
                self.raw ^ a2.raw,
                optimize=self.optimize and a2.optimize,
                to_file=self.data_format == "file" and a2.data_format == "file",
            )

    @staticmethod
    def multi_logical_xor(acs: List["Activation"], asarray: bool = False) -> Union["Activation", np.ndarray]:
        """Do LOGICAL EXCLUSIVE OR on many activation vectors at once. Uses raw np.ndarrays to gain time.
        If asarray is True, does not cast the result into an Activation object but returns the raw np.ndarray."""
        if len(acs) == 1:
            return Activation(acs[0].raw, optimize=acs[0].optimize, to_file=acs[0].data_format == "file")
        lor = Activation.multi_logical_or(acs, True)
        nland = -(Activation.multi_logical_and(acs, True) - 1)
        res = lor & nland.astype(np.ubyte)
        if asarray:
            return res
        return Activation(
            res,
            optimize=all([a.optimize for a in acs]),
            to_file=all([a.data_format == "file" for a in acs]),
        )

    def __add__(self, other: "Activation") -> "Activation":
        """Synonym of logical OR"""
        return self or other

    def __sub__(self, other: "Activation") -> "Activation":
        """Logical EXCLUSIVE OR then logical AND"""
        return (self ^ other) & self

    def __len__(self):
        """Number of points in the vector"""
        return self.length

    def __contains__(self, other: "Activation") -> bool:
        intersection = self & other
        nones_intersection = intersection.nones
        intersection.delete()
        if nones_intersection < min(self.nones, other.nones):
            return False
        return True

    """ Conversions to raw methods"""

    def _read(self, path: Optional[Path] = None, out: bool = True) -> np.ndarray:
        """Read a raw activation vector's np.ndarray, either from given path, or from self.data. In that case, will
        raise ValueError if self.data_format is not "file".

        If out is not True or value is None, will set:
            * self._sizeof_raw
            * self._nones
            * self._time_read
            * self._sizeof_file
            * self._sizeof_path
            Will iterate self._n_read
        """
        if path is None:
            out = False
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
            stat = path.stat()
            if isinstance(stat, dict):
                self._sizeof_file = stat["st_size"] / 1e6
            else:
                self._sizeof_file = stat.st_size / 1e6
            self._sizeof_path = sys.getsizeof(self.data) / 1e6
            self._time_read = time() - t0
            self._n_read += 1
        return value

    def _integer_to_raw(self, value: Optional[int] = None, out: bool = True) -> np.ndarray:
        """From a value of the form 45786542 (int), which is the base 10 representation of the binary form of an
        activation vector, returns the raw np.ndarray vector of the form [1, 0, 0, 1, 1, 0, ...].

        If out is not True or value is None, will set:
            * self._sizeof_integer
            * self._sizeof_raw
            * self._time_integer_to_raw
            * self._nones
            Will iterate :
                * self._n_integer_to_raw
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
        if not out:
            self._sizeof_integer = sys.getsizeof(value) / 1e6

        if len(act) > self.length:
            raise ValueError(
                "After using _integer_to_raw, I ended up with an activation vector bigger than the specified "
                "max length. This should not happend as the max length should have been set by the indexing "
                "of x earlier in your code"
            )
        act_bis = np.zeros(self.length).astype(np.ubyte)
        act_bis[self.length - len(act) :] = act

        if not out:
            self._sizeof_raw = sys.getsizeof(act_bis) / 1e6
            self._time_integer_to_raw = time() - t0
            self._n_integer_to_raw += 1
            if self._nones is None:
                self._nones = np.count_nonzero(act_bis == 1)
        return act_bis

    def _bitarray_to_raw(self, value: Union[bitarray, Path] = None, out=True) -> np.ndarray:
        """Transforms a bitarray to a np.ndarray

        If out is not True or value is None, will set:
            * self._sizeof_bitarray
            * self._sizeof_raw
            * self._time_bitarray_to_raw
            * self._nones
            Will iterate :
                * self._n_bitarray_to_raw
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
    ) -> Union[np.ndarray, bitarray, int]:
        """From a compressed array of either str or np.ndarray format, will return the raw np.ndarray vector of the form
        [0, 1, 1, 1, 0, 0, 1, ...]

        If raw is True (default), returns it as a np.ndarray, else as a bitarray

        If out is False or value is None, will set:
            * self._time_compressed_to_raw
            * self._sizeof_compressed_str or _sizeof_compressed_array
            * self._sizeof_raw
            * self._nones
            Will iterate :
                * self._n_compressed_to_raw
        raw if False and Activation.WILL_COMPARE is True/False:
            * self._time_compressed_to_integer/bitarray
            * self._sizeof_nteger/bitarray
            Will iterate :
                * self._n_compressed_to_nteger/bitarray
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

        if value[-1] == 0 or value[-1] == 1:  # is already raw
            if not out:
                self._sizeof_raw = sys.getsizeof(value)
                if self._nones is None:
                    self._nones = np.count_nonzero(value == 1)
            return value

        if isinstance(value, str):
            act = ast.literal_eval(value)
            if not out:
                self._sizeof_compressed_str = sys.getsizeof(value) / 1e6
        elif isinstance(value, np.ndarray):
            act = value
            if not out:
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
        else:
            for index in act[1:]:
                if previous_value == 0:
                    previous_index = index
                    previous_value = 1
                else:
                    s[previous_index:index] = np.ones(index - previous_index)
                    previous_index = index
                    previous_value = 0
        s = np.array(s, dtype=np.ubyte)

        if not out:
            if self._sizeof_raw == -1:
                self._sizeof_raw = sys.getsizeof(s) / 1e6
            self._time_compressed_to_raw = time() - t0
            self._n_compressed_to_raw += 1
            if self._nones is None:
                self._nones = np.count_nonzero(s == 1)

        if raw:
            act = s
            fmt = "raw"
        elif Activation.WILL_COMPARE:
            t1 = time()
            act = self._raw_to_integer(s)
            if not out:
                self._time_raw_to_integer = time() - t1
                self._n_raw_to_integer += 1
            fmt = "integer"
        else:
            t1 = time()
            # noinspection PyTypeChecker
            act = self._raw_to_bitarray(s)
            if not out:
                self._time_raw_to_bitarray = time() - t1
                self._n_raw_to_bitarray += 1
            fmt = "bitarray"

        if not out:
            if fmt == "integer":
                # noinspection PyTypeChecker
                if self._sizeof_integer == -1:
                    self._sizeof_integer = sys.getsizeof(act) / 1e6
                self._time_compressed_to_integer = time() - t0
                self._n_compressed_to_integer += 1
            elif fmt == "bitarray":
                # noinspection PyTypeChecker
                if self._sizeof_bitarray == -1:
                    self._sizeof_bitarray = sys.getsizeof(act) / 1e6
                self._time_compressed_to_bitarray = time() - t0
                self._n_compressed_to_bitarray += 1
        return act

    """ Conversions from raw methods"""

    @staticmethod
    def _compress(value: Union[np.ndarray, str], dtype: type = str) -> Union[np.ndarray, str]:
        """Transforms a raw or bitarray activation vector to a compressed one.

        A compressed vector is a collection of integers starting by the initial value of the raw vector (0 or 1) and
        ending with its length. The other integers in the compression are the positions in the raw vector where the
        vector value changes. This stores all the information and saves up memory if the vector is constant over
        large periods of time.

        The compressed vector can be stored as a str looking like "0, 12, 456, ..., 47782" or as a numpy array of
        integers. What storage to use is specified by the "dtype" argument.
        """
        if isinstance(value, (int, bitarray)):
            raise TypeError("Can not compress an integer vector")
        if not isinstance(value, np.ndarray) or (value[-1] != 0 and value[-1] != 1):
            if isinstance(value, str) and dtype != str:
                return np.array(value.split(",")).astype(np.ubyte)
            return value
        if isinstance(value, np.ndarray):
            # not ubyte (unsigned byte), because np.diff will produce negative value that ubyte can not handle
            value = value.astype(np.byte)
        else:
            raise TypeError(f"Can not compress a {type(value)}")
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
        """Casts a raw activation vector into a bitarray, dividing its size in MO by 8."""
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
        """Casts a raw activation vector into the integer represented by its binary form, dividing its size in MO by 8.

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

    """properties"""

    @property
    def ones(self) -> np.ndarray:
        """ones is the list of indexes where the vector is 1
        Contrary to other @properties, do not store 'ones' in self, for it can be quit large : several MB or more.
        When running codes with millions of vector, this can be problematic."""
        raw = self.raw
        ones = np.where(raw == 1)[0].tolist()
        return ones

    @property
    def nones(self) -> int:
        """nones is the number of points where the activation vector is 1
        self._nones might not be set since it can only be set at object creation if the full array was given or
        accessed at some point
        """
        if self._nones is None:
            _ = self.raw  # calling raw will compute nones

        if self._coverage is None:
            self._coverage = self._nones / self.length
        return self._nones

    @property
    def entropy(self) -> int:
        """The entropy of the vector is the length of its compressed reprensentation : the number of times it switched
        from 1 to 0 and vice versa, plus 1 for the information about its initial value, and again 1 for the Information
        about its size.
        Like self._nones, it might not have been computed yet. In that case, compute it.

        If entropy is ocmputed here, will set:
            * self._time_bitarray_to_compressed or self._time_integer_to_compressed
            * self._sizeof_compressed_str or self._sizeof_compressed_array
            * self._entropy (you don't say)
            * self._rel_entropy
            Will iterate :
                * self._n_bitarray_to_compressed or self._n_integer_to_compressed
        """
        if self._entropy is None:
            t0 = time()
            fmt = self.data_format
            if self.data_format == "file":
                data = self._read(out=False)
                fmt = "raw"
            else:
                data = self.data

            if fmt == "bitarray":
                data = self._bitarray_to_raw(data, out=False)
                compressed = self._compress(data)
                self._time_bitarray_to_compressed = time() - t0
                self._n_bitarray_to_compressed += 1
            elif fmt == "integer":
                data = self._integer_to_raw(data, out=False)
                compressed = self._compress(data)
                self._time_integer_to_compressed = time() - t0
                self._n_integer_to_compressed += 1
            elif fmt == "raw":
                compressed = self._compress(data)
                self._time_raw_to_compressed = time() - t0
                self._n_raw_to_compressed += 1
            else:
                raise ValueError(f"Entropy should have been computed already if format is {fmt}")

            if self.data_format == "compressed_str":
                self._sizeof_compressed_str = sys.getsizeof(compressed) / 1e6
            if self.data_format == "compressed_array":
                self._sizeof_compressed_array = sys.getsizeof(compressed) / 1e6

            self._entropy = len(ast.literal_eval(compressed))
        if self._rel_entropy is None:
            self._rel_entropy = self._entropy / self.length
        return self._entropy

    @property
    def rel_entropy(self) -> float:
        """Relative entropy is the entropy divided by the length of the raw np.ndarray vector
        Like self._nones, it might not have been computed yet. In that case, compute it.
        """
        if self._rel_entropy is None:
            _ = self.entropy  # will set self._rel_entropy
        return self._rel_entropy

    @property
    def coverage(self) -> float:
        """Coverage is the fraction of points equal to 1 in the vector
        Like self._nones, it might not have been computed yet. In that case, compute it.
        """
        if self._coverage is None:
            _ = self.nones  # will set self._coverage
        return self._coverage

    """Method to access various formats"""

    @property
    def raw(self) -> np.ndarray:
        """Returns the raw np.ndarray. Will set relevant profiling attributes."""
        if self.data_format == "raw":
            return self.data
        elif self.data_format == "bitarray":
            return self._bitarray_to_raw(out=False)
        elif self.data_format == "integer":
            return self._integer_to_raw(out=False)
        elif self.data_format == "file":
            return self._read(out=False)
        elif "compressed" in self.data_format:
            return self._decompress(out=False)
        else:
            raise ValueError(f"Unkown activation format {self.data_format}")

    @property
    def as_bitarray(self) -> bitarray:
        """Returns the bitarray representation of the vector"""
        if self.data_format == "file":
            data = self._read(out=False)
            t0 = time()
            to_ret = self._raw_to_bitarray(data)
            self._time_raw_to_bitarray = time() - t0
            self._n_raw_to_bitarray += 1
            if self._sizeof_bitarray == -1:
                self._sizeof_bitarray = sys.getsizeof(to_ret)
            return to_ret
        elif self.data_format == "raw":
            t0 = time()
            to_ret = self._raw_to_bitarray(self.data)
            self._time_raw_to_bitarray = time() - t0
            self._n_raw_to_bitarray += 1
            if self._sizeof_bitarray == -1:
                self._sizeof_bitarray = sys.getsizeof(to_ret)
            return to_ret
        elif self.data_format == "bitarray":
            return self.data
        elif self.data_format == "integer":
            t0 = time()
            s = bin(self.data)[2:]
            if len(s) != self.length:
                s = "0" * (self.length - len(s)) + s
            to_ret = bitarray(s)
            self._time_integer_to_bitarray = time() - t0
            self._n_integer_to_bitarray += 1
            if self._sizeof_bitarray == -1:
                self._sizeof_bitarray = sys.getsizeof(to_ret)
            return to_ret
        elif "compressed" in self.data_format:
            wc = Activation.WILL_COMPARE
            Activation.WILL_COMPARE = False
            to_ret = self._decompress(raw=False, out=False)
            Activation.WILL_COMPARE = wc
            return to_ret
        else:
            raise ValueError(f"Unkown activation format {self.data_format}")

    @property
    def as_integer(self) -> int:
        """Returns the integer representation of the vector. Will set relevant profiling attributes."""

        if self.data_format == "file":
            data = self._read(out=False)
            t0 = time()
            to_ret = self._raw_to_integer(data)
            self._time_raw_to_integer = time() - t0
            self._n_raw_to_integer += 1
            if self._sizeof_integer == -1:
                self._sizeof_integer = sys.getsizeof(to_ret)
            return to_ret
        elif self.data_format == "raw":
            t0 = time()
            to_ret = self._raw_to_integer(self.data)
            self._time_raw_to_integer = time() - t0
            self._n_raw_to_integer += 1
            if self._sizeof_integer == -1:
                self._sizeof_integer = sys.getsizeof(to_ret)
            return to_ret
        elif self.data_format == "integer":
            return self.data
        elif self.data_format == "bitarray":
            t0 = time()
            to_ret = int(str(self.data).replace("bitarray(", "").replace(")", "").replace("'", "").replace('"', ""), 2)
            self._time_bitarray_to_integer = time() - t0
            self._n_bitarray_to_integer += 1
            self._sizeof_integer = sys.getsizeof(to_ret)
            return to_ret
        elif "compressed" in self.data_format:
            wc = Activation.WILL_COMPARE
            Activation.WILL_COMPARE = True
            to_ret = self._decompress(raw=False, out=False)
            Activation.WILL_COMPARE = wc
            return to_ret
        else:
            raise ValueError(f"Unkown activation format {self.data_format}")

    @property
    def as_compressed(self) -> Union[str]:
        """Returns the compressed str representation of the vector.
        Will set relevant profiling attributes."""
        return self.as_compressed_str

    @property
    def as_compressed_array(self) -> np.ndarray:
        """Returns the compressed array representation of the vector. Will set relevant profiling attributes."""
        if self.data_format == "raw":
            t0 = time()
            to_ret = self._compress(self.data, dtype=np.ndarray)
            self._time_raw_to_compressed = time() - t0
            self._n_raw_to_compressed += 1
            if self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(to_ret)
        elif self.data_format == "compressed_array":
            to_ret = self.data
            if self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(to_ret)
        elif self.data_format == "compressed_str":
            to_ret = np.array(ast.literal_eval(self.data))
            if self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(to_ret)
        elif self.data_format == "integer":
            t0 = time()
            raw = self.raw
            t1 = time()
            to_ret = self._compress(raw, dtype=np.ndarray)
            self._time_raw_to_compressed = time() - t1
            self._n_raw_to_compressed += 1
            self._time_integer_to_compressed = time() - t0
            self._n_integer_to_compressed += 1
            if self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(to_ret)
        elif self.data_format == "bitarray":
            t0 = time()
            raw = self.raw
            t1 = time()
            to_ret = self._compress(raw, dtype=np.ndarray)
            self._time_raw_to_compressed = time() - t1
            self._n_raw_to_compressed += 1
            self._time_bitarray_to_compressed = time() - t0
            self._n_bitarray_to_compressed += 1
            if self._sizeof_compressed_array == -1:
                self._sizeof_compressed_array = sys.getsizeof(to_ret)
        elif self.data_format == "file":
            data = self._read(out=False)
            t0 = time()
            to_ret = self._compress(data, dtype=np.ndarray)
            self._time_raw_to_compressed = time() - t0
            self._n_raw_to_compressed += 1
        else:
            raise ValueError(f"Unkown activation format {self.data_format}")
        self._entropy = len(to_ret)
        if self._rel_entropy is None:
            self._rel_entropy = self._entropy / self.length
        return to_ret

    @property
    def as_compressed_str(self) -> str:
        """Returns the compressed str representation of the vector. Will set relevant profiling attributes."""
        if self.data_format == "raw":
            t0 = time()
            to_ret = self._compress(self.data, dtype=str)
            self._time_raw_to_compressed = time() - t0
            self._n_raw_to_compressed += 1
            if self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(to_ret)
        elif self.data_format == "compressed_str":
            to_ret = self.data
            if self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(to_ret)
        elif self.data_format == "compressed_array":
            to_ret = str(self.data.tolist()).replace(" ", "").replace("[", "").replace("]", "")
            if self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(to_ret)
        elif self.data_format == "integer":
            t0 = time()
            raw = self.raw
            t1 = time()
            to_ret = self._compress(raw, dtype=str)
            self._time_raw_to_compressed = time() - t1
            self._n_raw_to_compressed += 1
            self._time_integer_to_compressed = time() - t0
            self._n_integer_to_compressed += 1
            if self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(to_ret)
        elif self.data_format == "bitarray":
            t0 = time()
            raw = self.raw
            t1 = time()
            to_ret = self._compress(raw, dtype=str)
            self._time_raw_to_compressed = time() - t1
            self._n_raw_to_compressed += 1
            self._time_bitarray_to_compressed = time() - t0
            self._n_bitarray_to_compressed += 1
            if self._sizeof_compressed_str == -1:
                self._sizeof_compressed_str = sys.getsizeof(to_ret)
        elif self.data_format == "file":
            t0 = time()
            data = self._read(out=False)
            to_ret = self._compress(data, dtype=str)
            self._time_raw_to_compressed = time() - t0
            self._n_raw_to_compressed += 1
        else:
            raise ValueError(f"Unkown activation format {self.data_format}")
        self._entropy = len(ast.literal_eval(to_ret))
        if self._rel_entropy is None:
            self._rel_entropy = self._entropy / self.length
        return to_ret

    @property
    def sizeof_path(self) -> float:
        """Returns the size in MB of the path object, or -1 if it does not exist.
        In that case and if Activation.FORCE_STAT is True, will force the object to write the file to compute the
         relevant profiling attributes."""
        if self._sizeof_path == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self._write(self.raw)
            self.data.unlink()
            self.data = data
            self.data_format = fmt
        return self._sizeof_path

    @property
    def sizeof_file(self) -> float:
        """Returns the size in MB of the path object, or -1 if it does not exist. In that case and if
        Activation.FORCE_STAT is True, will force the object to write the file to compute the relevant profiling
        attributes."""
        if self._sizeof_file == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self._write(self.raw)
            self.data.unlink()
            self.data = data
            self.data_format = fmt
        return self._sizeof_file

    @property
    def sizeof_raw(self) -> float:
        """Returns the size in MB of raw np.ndarray vector, or -1 if it does not exist. In that case and if
        Activation.FORCE_STAT is True, will force the object to call self.raw to compute the relevant profiling
        attributes."""
        if self._sizeof_raw == -1 and Activation.FORCE_STAT:
            _ = self.raw
        return self._sizeof_raw

    @property
    def sizeof_bitarray(self) -> float:
        """Returns the size in MB of the vector in bitarray, or -1 if it does not exist. In that case and if
        Activation.FORCE_STAT is True, will force the object to call self.as_bitarray to compute the relevant
        profiling attributes."""
        if self._sizeof_bitarray == -1 and Activation.FORCE_STAT:
            _ = self.as_bitarray
        return self._sizeof_bitarray

    @property
    def sizeof_integer(self) -> float:
        """Returns the size in MB of the vector in integer, or -1 if it does not exist. In that case and if
        Activation.FORCE_STAT is True, will force the object to call self.as_integer to compute the relevant
        profiling attributes."""
        if self._sizeof_integer == -1 and Activation.FORCE_STAT:
            _ = self.as_integer
        return self._sizeof_integer

    @property
    def sizeof_compressed_array(self) -> float:
        """Returns the size in MB of the vector in compressed array, or -1 if it does not exist. In that case and if
        Activation.FORCE_STAT is True, will force the object to call self.as_compressed_array to compute the relevant
        profiling attributes."""
        if self._sizeof_compressed_array == -1 and Activation.FORCE_STAT:
            _ = self.as_compressed_array
        return self._sizeof_compressed_array

    @property
    def sizeof_compressed_str(self) -> float:
        """Returns the size in MB of the vector in compressed str, or -1 if it does not exist. In that case and if
        Activation.FORCE_STAT is True, will force the object to call self.as_compressed_str to compute the relevant
        profiling attributes."""
        if self._sizeof_compressed_str == -1 and Activation.FORCE_STAT:
            _ = self.as_compressed_str
        return self._sizeof_compressed_str

    @property
    def time_write(self) -> float:
        """Returns the time in seconds to write the vector to a file, or -1 if it does not exist. In that case and
        if Activation.FORCE_STAT is True, will force the object to write the file to compute the relevant profiling
        attributes."""
        if self._time_write == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self._write(self.raw)
            self.data.unlink()
            self.data = data
            self.data_format = fmt
        return self._time_write

    @property
    def time_read(self) -> float:
        """Returns the time in seconds to read the vector to a file, or -1 if it does not exist. In that case and if
        Activation.FORCE_STAT is True, will force the object to write the file to compute the relevant profiling
        attributes."""
        if self._time_read == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self._write(self.raw)
            _ = self._read(self.data, out=False)
            self.data.unlink()
            self.data = data
            self.data_format = fmt
        return self._time_read

    @property
    def time_raw_to_compressed(self) -> float:
        """Returns the time in seconds to compress the vector, or -1 if it does not exist. In that case and if
        Activation.FORCE_STAT is True, will force the object to call self._compress to compute the relevant profiling
        attributes."""
        if self._time_raw_to_compressed == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self.data = self.raw
            self.data_format = "raw"
            _ = self.as_compressed
            self.data = data
            self.data_format = fmt
        return self._time_raw_to_compressed

    @property
    def time_raw_to_integer(self) -> float:
        """Returns the time in seconds go from raw to integer, or -1 if it does not exist. In that case and if
        Activation.FORCE_STAT is True, will force the object to call self._raw_to_integer to compute the relevant
        profiling attributes."""
        if self._time_raw_to_integer == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self.data = self.raw
            self.data_format = "raw"
            _ = self.as_integer
            self.data = data
            self.data_format = fmt
        return self._time_raw_to_integer

    @property
    def time_raw_to_bitarray(self) -> float:
        """Returns the time in seconds go from raw to bitarray, or -1 if it does not exist. In that case and if
        Activation.FORCE_STAT is True, will force the object to call self._raw_to_bitarray to compute the relevant
        profiling attributes."""
        if self._time_raw_to_bitarray == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self.data = self.raw
            self.data_format = "raw"
            _ = self.as_bitarray
            self.data = data
            self.data_format = fmt
        return self._time_raw_to_bitarray

    @property
    def time_compressed_to_raw(self) -> float:
        """Returns the time in seconds to compress the vector, or -1 if it does not exist. In that case and if
        Activation.FORCE_STAT is True, will force the object to call self._decompress to compute the relevant
        profiling attributes."""
        if self._time_compressed_to_raw == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self.data = self.as_compressed
            if isinstance(self.data, str):
                self.data_format = "compressed_str"
            else:
                self.data_format = "compressed_array"
            _ = self.raw
            self.data = data
            self.data_format = fmt
        return self._time_compressed_to_raw

    @property
    def time_bitarray_to_raw(self) -> float:
        """Returns the time in seconds to go from bitarray to raw, or -1 if it does not exist. In that case and if
        Activation.FORCE_STAT is True, will force the object to call self._bitarray_to_raw to compute the relevant
        profiling attributes."""
        if self._time_bitarray_to_raw == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self.data = self.as_bitarray
            self.data_format = "bitarray"
            _ = self.raw
            self.data = data
            self.data_format = fmt
        return self._time_bitarray_to_raw

    @property
    def time_integer_to_raw(self) -> float:
        """Returns the time in seconds to go from integer to raw, or -1 if it does not exist. In that case and if
        Activation.FORCE_STAT is True, will force the object to call self._integer_to_raw to compute the relevant
        profiling attributes."""
        if self._time_integer_to_raw == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self.data = self.as_integer
            self.data_format = "integer"
            _ = self.raw
            self.data = data
            self.data_format = fmt
        return self._time_integer_to_raw

    @property
    def time_compressed_to_bitarray(self) -> float:
        """Returns the time in seconds to go from compressed to bitarray, or -1 if it does not exist. In that case
        and if Activation.FORCE_STAT is True, will force the object to call self._decompress on self.as_compressed with
         'raw=False' to compute the relevant profiling attributes."""
        if self._time_compressed_to_bitarray == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self.data = self.as_compressed
            if isinstance(self.data, str):
                self.data_format = "compressed_str"
            else:
                self.data_format = "compressed_array"
            _ = self.as_bitarray
            self.data = data
            self.data_format = fmt
        return self._time_compressed_to_bitarray

    @property
    def time_bitarray_to_compressed(self) -> float:
        """Returns the time in seconds to go from bitarray to compressed, or -1 if it does not exist. In that case
        and if Activation.FORCE_STAT is True, will force the object to call self._compress on self.as_bitarray to
        compute the relevant profiling attributes."""
        if self._time_bitarray_to_compressed == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self.data = self.as_bitarray
            self.data_format = "bitarray"
            _ = self.as_compressed
            self.data = data
            self.data_format = fmt
        return self._time_bitarray_to_compressed

    @property
    def time_integer_to_compressed(self) -> float:
        """Returns the time in seconds to go from integer to compressed, or -1 if it does not exist. In that case
        and if Activation.FORCE_STAT is True, will force the object to call self._compress on
        self._integer_to_raw(self.as_integer) to compute the relevant profiling attributes."""
        if self._time_integer_to_compressed == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self.data = self.as_integer
            self.data_format = "integer"
            _ = self.as_compressed
            self.data = data
            self.data_format = fmt
        return self._time_integer_to_compressed

    @property
    def time_bitarray_to_integer(self) -> float:
        """Returns the time in seconds to go from bitarray to integer, or -1 if it does not exist. In that case
        and if Activation.FORCE_STAT is True, will force the object to call self._compress on
        self._integer_to_raw(self.as_integer) to compute the relevant profiling attributes."""
        if self._time_bitarray_to_integer == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self.data = self.as_bitarray
            self.data_format = "bitarray"
            _ = self.as_integer
            self.data = data
            self.data_format = fmt
        return self._time_bitarray_to_integer

    @property
    def time_integer_to_bitarray(self) -> float:
        """Returns the time in seconds to go from bitarray to integer, or -1 if it does not exist. In that case
        and if Activation.FORCE_STAT is True, will force the object to call self.as_integer on
        self to compute the relevant profiling attributes."""
        if self._time_integer_to_bitarray == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self.data = self.as_integer
            self.data_format = "integer"
            _ = self.as_bitarray
            self.data = data
            self.data_format = fmt
        return self._time_integer_to_bitarray

    @property
    def time_compressed_to_integer(self) -> float:
        """Returns the time in seconds to go from compressed to integer, or -1 if it does not exist. In that case
        and if Activation.FORCE_STAT is True, will force the object to call self.as_compress on
        to compute the relevant profiling attributes."""
        if self._time_compressed_to_integer == -1 and Activation.FORCE_STAT:
            data = self.data
            fmt = self.data_format
            self.data = self.as_compressed
            if isinstance(self.data, str):
                self.data_format = "compressed_str"
            else:
                self.data_format = "compressed_array"
            _ = self.as_integer
            self.data = data
            self.data_format = fmt
        return self._time_compressed_to_integer

    @property
    def n_written(self) -> int:
        """Returns the number of time the vector was written to disk."""
        return self._n_written

    @property
    def n_read(self) -> int:
        """Returns the number of time the vector was read from disk."""
        return self._n_read

    @property
    def n_raw_to_compressed(self) -> int:
        """Returns the number of time the vector was compressed from raw."""
        return self._n_raw_to_compressed

    @property
    def n_compressed_to_raw(self) -> int:
        """Returns the number of time the vector was decompressed to raw."""
        return self._n_compressed_to_raw

    @property
    def n_raw_to_bitarray(self) -> int:
        """Returns the number of time the vector was formatted as bitarray."""
        return self._n_raw_to_bitarray

    @property
    def n_raw_to_integer(self) -> int:
        """Returns the number of time the vector was formatted as integer."""
        return self._n_raw_to_integer

    @property
    def n_bitarray_to_raw(self) -> int:
        """Returns the number of time the vector was deformatted from bitarray."""
        return self._n_bitarray_to_raw

    @property
    def n_integer_to_raw(self) -> int:
        """Returns the number of time the vector was deformatted from integer."""
        return self._n_integer_to_raw

    @property
    def n_bitarray_to_compressed(self) -> int:
        """Returns the number of time the vector was compressed from bitarray."""
        return self._n_bitarray_to_compressed

    @property
    def n_integer_to_compressed(self) -> int:
        """Returns the number of time the vector was compressed from integer."""
        return self._n_integer_to_compressed

    @property
    def n_compressed_to_bitarray(self) -> int:
        """Returns the number of time the vector was decompressed to bitarray."""
        return self._n_compressed_to_bitarray

    @property
    def n_compressed_to_integer(self) -> int:
        """Returns the number of time the vector was decompressed to integer."""
        return self._n_compressed_to_integer

    @property
    def n_bitarray_to_integer(self) -> int:
        """Returns the number of time the vector was decompressed to integer."""
        return self._n_bitarray_to_integer

    @property
    def n_integer_to_bitarray(self) -> int:
        """Returns the number of time the vector was decompressed to integer."""
        return self._n_integer_to_bitarray
