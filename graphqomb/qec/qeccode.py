"""QEC Code object."""

from __future__ import annotations

from collections.abc import Mapping

from scipy.sparse import csr_matrix


class StabilizerCode:
    """A stabilizer code."""

    def __init__(
        self,
        stabilizer_matrix: csr_matrix,
        *,
        stabilizer_coords: Mapping[int, tuple[float, ...]] | None = None,
        qubit_coords: Mapping[int, tuple[float, ...]] | None = None,
    ) -> None:

        self.hx = csr_matrix(stabilizer_matrix[:, : stabilizer_matrix.shape[1] // 2])
        self.hz = csr_matrix(stabilizer_matrix[:, stabilizer_matrix.shape[1] // 2 :])

        self.stabilizer_coord = stabilizer_coords
        self.qubit_coord = qubit_coords

    @property
    def num_stabilizers(self) -> int:
        """Return the number of stabilizers."""
        return self.hx.shape[0]

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits."""
        return self.hx.shape[1]
