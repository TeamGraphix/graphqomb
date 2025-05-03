"""Decoder backend modules."""

import operator
from abc import ABC, abstractmethod
from functools import reduce
from typing import Any


class BaseDecoder(ABC):
    @abstractmethod
    def decode(self, cbits: dict[int, bool]) -> Any:
        pass


class XORDecoder(BaseDecoder):
    def decode(self, input_cbits: dict[int, bool]) -> bool:
        """Decode the input classical bits to a single bit.

        Parameters
        ----------
        input_cbits : dict[int, bool]
            A dictionary mapping bit positions to their boolean values.

        Returns
        -------
        bool
            The result of XOR operation on the input bits.
        """
        return reduce(operator.xor, input_cbits.values(), False)
