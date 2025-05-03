# test_xor_decoder.py
from __future__ import annotations

import pytest

from graphix_zx.decoder_backend import XORDecoder


@pytest.mark.parametrize(
    ("input_cbits", "expected"),
    [
        ({}, False),
        ({0: False}, False),
        ({0: True}, True),
        ({0: True, 1: True}, False),
        ({0: True, 1: False}, True),
        ({0: False, 1: False, 2: False}, False),
        ({0: True, 1: True, 2: True}, True),
    ],
)
def test_xor_value(input_cbits: dict[int, bool], expected: bool) -> None:
    dec = XORDecoder()
    out = dec.decode(input_cbits, output_cbits=[99])
    assert list(out.values()) == [expected]


@pytest.mark.parametrize(
    "output_cbits",
    [
        [3],
        [3, 4, 5],
        [3, 3, 4, 4, 5],
    ],
)
def test_output_keys_are_set_identically(output_cbits: list[int]) -> None:
    dec = XORDecoder()
    result_dict = dec.decode({0: True, 1: False}, output_cbits=output_cbits)
    assert set(result_dict.keys()) == set(output_cbits)
    assert len(set(result_dict.values())) == 1


def test_decode_returns_new_dict_instance() -> None:
    dec = XORDecoder()
    res1 = dec.decode({0: True}, output_cbits=[1])
    res2 = dec.decode({0: False}, output_cbits=[1])
    assert isinstance(res1, dict)
    assert isinstance(res2, dict)
    assert res1 is not res2
