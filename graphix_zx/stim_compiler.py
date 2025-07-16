"""Pattern to stim compiler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix_zx.command import E, M, N

if TYPE_CHECKING:
    from graphix_zx.pattern import Pattern


def stim_compile(pattern: Pattern) -> str:
    """Compile a pattern to stim format.

    Parameters
    ----------
    pattern : `Pattern`
        The pattern to compile.

    Returns
    -------
    `str`
        The compiled stim string.
    """
    stim_str = ""
    for cmd in pattern:
        if isinstance(cmd, N):
            # prepare node in |+> state
            stim_str += f"H {cmd.node}\n"
        if isinstance(cmd, E):
            q1, q2 = cmd.nodes
            stim_str += f"CZ {q1} {q2}\n"
        if isinstance(cmd, M):
            # need X/Z switch
            stim_str += f"MX {cmd.node}\n"

    return stim_str.strip()
