from __future__ import annotations

import networkx as nx
import numpy as np
import sympy as sp

from linalg import MatGF2


def find_flow(
    graph: nx.Graph,
    input: set[int],
    output: set[int],
    meas_planes: dict[int, str],
) -> tuple[dict[int, set[int]], dict[int, int]]:
    raise NotImplementedError


def find_gflow(
    graph: nx.Graph,
    input: set[int],
    output: set[int],
    meas_planes: dict[int, str],
) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Maximally delayed gflow finding algorithm

    For open graph g with input, output, and measurement planes, this returns maximally delayed gflow.

    gflow consist of function g(i) where i is the qubit labels,
    and strict partial ordering < or layers labels l_k where each element
    specify the order of qubits to be measured to maintain determinism in MBQC.
    In practice, we must measure qubits in order specified in array l_k (increasing order
    of l_k from 1), and for each measurements of qubit i we must perform corrections on
    qubits in g(i), depending on the measurement outcome.

    For more details of gflow, see Browne et al., NJP 9, 250 (2007).
    We use the extended gflow finding algorithm in Backens et al., Quantum 5, 421 (2021).

    Parameters
    ----------
    graph: nx.Graph
        graph (incl. in and out)
    input: set
        set of node labels for input
    output: set
        set of node labels for output
    meas_planes: dict
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.

    Returns
    -------
    g: dict
        gflow function. g[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    """
    l_k = dict()
    g = dict()
    for node in graph.nodes:
        l_k[node] = 0
    return gflowaux(graph, input, output, meas_planes, 1, l_k, g)


def find_pflow(
    graph: nx.Graph,
    input: set[int],
    output: set[int],
    meas_planes: dict[int, str],
    meas_angles: dict[int, float],
) -> tuple[dict[int, set[int]], dict[int, int]]:
    raise NotImplementedError


def gflowaux(
    graph: nx.Graph,
    input: set[int],
    output: set[int],
    meas_planes: dict[int, str],
    k: int,
    l_k: dict[int, int],
    g: dict[int, set[int]],
):
    """Function to find one layer of the gflow.

    Ref: Backens et al., Quantum 5, 421 (2021).

    Parameters
    ----------
    graph: nx.Graph
        graph (incl. in and out)
    input: set
        set of node labels for input
    output: set
        set of node labels for output
    meas_planes: dict
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.
    k: int
        current layer number.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    g: dict
        gflow function. g[i] is the set of qubits to be corrected for the measurement of qubit i.

    Returns
    -------
    g: dict
        gflow function. g[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    """

    nodes = set(graph.nodes)
    if output == nodes:
        return g, l_k
    non_output = nodes - output
    correction_candidate = output - input
    adj_mat, node_order_list = get_adjacency_matrix(graph)
    node_order_row = node_order_list.copy()
    node_order_row.sort()
    node_order_col = node_order_list.copy()
    node_order_col.sort()
    for out in output:
        adj_mat.remove_row(node_order_row.index(out))
        node_order_row.remove(out)
    adj_mat_row_reduced = adj_mat.copy()  # later use for construct RHS
    for node in nodes - correction_candidate:
        adj_mat.remove_col(node_order_col.index(node))
        node_order_col.remove(node)

    b = MatGF2(np.zeros((adj_mat.data.shape[0], len(non_output)), dtype=int))
    for i_row in range(len(node_order_row)):
        node = node_order_row[i_row]
        vec = MatGF2(np.zeros(len(node_order_row), dtype=int))
        if meas_planes[node] == "XY":
            vec.data[i_row] = 1
        elif meas_planes[node] == "XZ":
            vec.data[i_row] = 1
            vec_add = adj_mat_row_reduced.data[:, node_order_list.index(node)]
            vec = vec + vec_add
        elif meas_planes[node] == "YZ":
            vec.data = adj_mat_row_reduced.data[:, node_order_list.index(node)].reshape(vec.data.shape)
        b.data[:, i_row] = vec.data
    adj_mat, b, _, col_permutation = adj_mat.forward_eliminate(b)
    x, kernels = adj_mat.backward_substitute(b)

    corrected_nodes = set()
    for i_row in range(len(node_order_row)):
        non_out_node = node_order_row[i_row]
        x_col = x[:, i_row]
        if 0 in x_col.shape or x_col[0] == sp.nan:  # no solution
            continue

        sol_list = [x_col[i].subs(zip(kernels, [sp.false] * len(kernels))) for i in range(len(x_col))]
        sol = np.array(sol_list)
        sol_index = sol.nonzero()[0]
        g[non_out_node] = set(node_order_col[col_permutation.index(i)] for i in sol_index)
        if meas_planes[non_out_node] in ["XZ", "YZ"]:
            g[non_out_node] |= {non_out_node}

    if len(corrected_nodes) == 0:
        if output == nodes:
            return g, l_k
        else:
            return None, None
    else:
        return gflowaux(
            graph,
            input,
            output | corrected_nodes,
            meas_planes,
            k + 1,
            l_k,
            g,
        )


def check_causality(
    graph: nx.Graph,
    input: set[int],
    output: set[int],
    meas_planes: dict[int, str],
    meas_angles: dict[int, float],
    gflow: dict[int, set[int]],
) -> bool:
    raise NotImplementedError


# NOTE: want to include Pauli simplification effect
def check_stablizers(
    graph: nx.Graph,
    input: set[int],
    output: set[int],
    meas_planes: dict[int, str],
    meas_angles: dict[int, float],
    gflow: dict[int, set[int]],
) -> bool:
    raise NotImplementedError


def get_adjacency_matrix(graph: nx.Graph) -> tuple[MatGF2, list[int]]:
    """Get adjacency matrix of the graph

    Returns
    -------
    adjacency_matrix: graphix.linalg.MatGF2
        adjacency matrix of the graph. the matrix is defined on GF(2) field.
    node_list: list
        ordered list of nodes. node_list[i] is the node label of i-th row/column of the adjacency matrix.

    """
    node_list = list(graph.nodes)
    node_list.sort()
    adjacency_matrix = nx.to_numpy_array(graph, nodelist=node_list)
    adjacency_matrix = MatGF2(adjacency_matrix.astype(int))
    return adjacency_matrix, node_list
