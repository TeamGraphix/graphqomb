from typing import Set, List, Tuple

import numpy as np

from graphix_zx.zxgraphstate import ZXGraphState
from graphix_zx.circuit import MBQCCircuit, circuit2graph
from graphix_zx.gates import CZ, CNOT, H, PhaseGadget


def extract_circuit_graph_state(d: ZXGraphState) -> ZXGraphState:
    """
    Extract a circuit from a ZXGraphState in MBQC+LC form with gflow and return
    the corresponding circuit-like graph state (as ZXGraphState).

    Implements Algorithm 2 (Circuit Extraction) from "There and back again" (Appendix D). 
    """
    # Step 0: Bring into phase-gadget form
    d.convert_to_phase_gadget()

    # Initialize circuit and frontier
    inputs = sorted(d.input_nodes)
    n_qubits = len(inputs)
    circ = MBQCCircuit(n_qubits)
    frontier: Set[int] = set(d.output_nodes)

    # Process initial frontier
    _process_frontier(d, frontier, circ)

    # Main extraction loop
    while True:
        # Remaining unextracted vertices
        remaining = set(d.physical_nodes) - frontier
        if not remaining:
            break
        _update_frontier(d, frontier, circ)

    # Final SWAP/H corrections if needed
    _finalize_extraction(d, frontier, circ)
    _revert_circuit(d, circ)

    # Convert MBQCCircuit back to ZXGraphState
    # graph, _ = circuit2graph(circ)
    # return graph
    
    return circ


def _process_frontier(d: ZXGraphState, frontier: Set[int], circ: MBQCCircuit) -> None:
    """
    Process the frontier: extract local Cliffords and CZ between frontier vertices. 
    """
    lc = d.local_clifford
    for v in sorted(frontier):
        # Extract any local Clifford on v
        if v in lc.keys():
            # to be implemented: add local Clifford gates
            pass
        # Extract any CZ edges between frontier vertices
        for w in list(d.get_neighbors(v) & frontier):
            circ.cz(v, w)
            d.remove_physical_edge(v, w)


def _update_frontier(d: ZXGraphState, frontier: Set[int], circ: MBQCCircuit) -> None:
    """
    Update the frontier by Gaussian elimination or pivots, then extract new frontier vertices. 
    """
    # Build bipartite adjacency: frontier vs neighbors
    neigh = sorted(set().union(*(d.get_neighbors(v) for v in frontier)))
    M = np.zeros((len(frontier), len(neigh)), dtype=int)
    for i, v in enumerate(sorted(frontier)):
        for j, u in enumerate(neigh):
            if u in d.get_neighbors(v):
                M[i, j] = 1
    # Gaussian eliminate over GF(2)
    M_red, row_ops = _gauss_elim(M)
    # Identify rows with single 1
    vs: List[int] = []
    for i, row in enumerate(M_red):
        if row.sum() == 1:
            col = int(np.nonzero(row)[0][0])
            vs.append(neigh[col])

    if not vs:
        # Step 4: pivot YZ vertices adjacent to frontier
        # to be implemented
        pass
        # for u in list(d.physical_nodes - frontier):
        #     if d.meas_bases[u].plane.name == 'YZ' and d.get_neighbors(u) & frontier:
        #         w = next(iter(d.get_neighbors(u) & frontier))
        #         d.pivot(u, w)
        #         _process_frontier(d, frontier, circ)
        # return

    # Apply recorded CNOT row operations
    for r1, r2 in row_ops:
        circ.cnot(sorted(frontier)[r1], sorted(frontier)[r2]) # CNOT is not implemented in MBQCCircuit
        # Update graph accordingly: add edge or local complement as needed
        d.apply_cnot(sorted(frontier)[r1], sorted(frontier)[r2])

    # Extract new frontier vertices
    for v in vs:
        # unique neighbor in frontier
        w = next(iter(d.get_neighbors(v) & frontier))
        circ.add_gate(H(), [w])
        circ.add_gate(PhaseGadget(d.meas_bases[v].angle), [w])
        frontier.remove(w)
        frontier.add(v)
        _process_frontier(d, frontier, circ)


def _gauss_elim(M: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    """
    Perform Gaussian elimination over GF(2), returning reduced matrix and list of row ops. 
    """
    M = M.copy() % 2
    n, m = M.shape
    row_ops: List[Tuple[int,int]] = []
    pivot_row = 0
    for col in range(m):
        # find pivot
        for r in range(pivot_row, n):
            if M[r, col] == 1:
                M[[pivot_row, r]] = M[[r, pivot_row]]
                if r != pivot_row:
                    row_ops.append((pivot_row, r))
                break
        else:
            continue
        # eliminate other rows
        for r in range(n):
            if r != pivot_row and M[r, col] == 1:
                M[r] ^= M[pivot_row]
                row_ops.append((r, pivot_row))
        pivot_row += 1
        if pivot_row == n:
            break
    return M, row_ops


def _finalize_extraction(d: ZXGraphState, frontier: Set[int], circ: MBQCCircuit) -> None:
    """
    Extract final Hadamards or SWAPs to align frontier to inputs. 
    """
    # to be implemented
    
    # # Handle any remaining Hadamard on outputs
    # for v in sorted(frontier):
    #     if d.has_hadamard_on_output(v):
    #         circ.add_gate(H(), [v])
    # # Permute frontier to match inputs via SWAPs
    # perm = d.compute_permutation(list(frontier), list(d.input_nodes))
    # for (q1, q2) in perm:
    #     circ.add_gate(CNOT(), [q1, q2])  # SWAP as three CNOTs omitted for brevity

def _revert_circuit(d: ZXGraphState, circ: MBQCCircuit) -> None:
    """
    Revert the circuit.
    """
    # to be implemented
    pass