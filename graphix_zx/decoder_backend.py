"""Decoder backend modules."""

import operator
from abc import ABC, abstractmethod
from functools import reduce


class BaseDecoder(ABC):
    """Base class for decoders."""

    @abstractmethod
    def decode(self, intput_cbits: dict[int, bool], output_cbits: list[int]) -> dict[int, bool]:
        """Decode the input classical bits to a single bit.

        Parameters
        ----------
        input_cbits : dict[int, bool]
            A dictionary mapping bit positions to their boolean values.
        output_cbits : list[int]
            A list of bit positions to be decoded.

        Returns
        -------
        dict[int, bool]
            Decoded results as a dictionary mapping bit positions to their boolean values.
        """


class XORDecoder(BaseDecoder):
    """XOR decoder class."""

    def decode(self, input_cbits: dict[int, bool], output_cbits: list[int]) -> dict[int, bool]:  # noqa: PLR6301
        """Decode the input classical bits to a single bit.

        Parameters
        ----------
        input_cbits : dict[int, bool]
            A dictionary mapping bit positions to their boolean values.
        output_cbits : list[int]
            A list of bit positions to be decoded.

        Returns
        -------
        dict[int, bool]
            The results of XOR operation on the input bits.
        """
        result = reduce(operator.xor, input_cbits.values(), False)
        return dict.fromkeys(output_cbits, result)
