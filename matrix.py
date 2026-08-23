"""MNA system assembly: node bookkeeping, ground elimination, and solve.

The unknown vector is ``x = [v_1, ..., v_n, i_V1, ..., i_Vm]`` where node 0
("0" or "gnd") is ground and is never given a row/column -- it is *removed*
from the system rather than pinned to zero, which is what keeps the matrix
nonsingular.
"""

import numpy as np

from .elements import VoltageSource

GROUND_NAMES = {"0", "gnd"}


class CircuitError(ValueError):
    """Raised for structural problems (floating nodes, singular systems)."""


class Circuit:
    def __init__(self):
        self.elements = []
        self._node_order = []  # first-seen order, excluding ground
        self._node_index = {}  # name -> row index
        self._vsrc_order = []  # voltage source names, in add() order
        self._vsrc_index = {}  # name -> extra-unknown index (offset by n_nodes)

    def add(self, element):
        for node in (element.n1, element.n2):
            self._register_node(node)
        if isinstance(element, VoltageSource):
            if element.name in self._vsrc_index:
                raise CircuitError(f"duplicate voltage source name {element.name!r}")
            self._vsrc_index[element.name] = len(self._vsrc_order)
            self._vsrc_order.append(element.name)
        self.elements.append(element)
        return element

    def _register_node(self, name):
        if self._is_ground(name):
            return
        if name not in self._node_index:
            self._node_index[name] = len(self._node_order)
            self._node_order.append(name)

    @staticmethod
    def _is_ground(name):
        return str(name).lower() in GROUND_NAMES

    @property
    def n_nodes(self):
        return len(self._node_order)

    @property
    def n_vsrc(self):
        return len(self._vsrc_order)

    def _node_idx(self, name):
        if self._is_ground(name):
            return None
        return self._node_index[name]

    def _vsrc_idx(self, name):
        return self.n_nodes + self._vsrc_index[name]

    def build(self):
        """Assemble and return the dense system (A, z)."""
        n = self.n_nodes + self.n_vsrc
        if n == 0:
            raise CircuitError("circuit has no nodes")
        A = np.zeros((n, n))
        z = np.zeros(n)
        for element in self.elements:
            element.stamp(A, z, self._node_idx, self._vsrc_idx)
        return A, z

    def solve(self):
        """Solve the DC operating point.

        Returns (node_voltages, vsrc_currents), both dicts keyed by name.
        Ground ("0"/"gnd") is not included in node_voltages; it is 0V by
        definition.
        """
        A, z = self.build()
        self._check_floating_nodes(A)
        try:
            x = np.linalg.solve(A, z)
        except np.linalg.LinAlgError as exc:
            raise CircuitError(
                "singular MNA matrix -- check for a loop of voltage sources "
                "or a node connected only to voltage/current sources with no "
                "resistive path to ground"
            ) from exc

        voltages = {name: x[i] for name, i in self._node_index.items()}
        currents = {name: x[self.n_nodes + i] for name, i in self._vsrc_index.items()}
        return voltages, currents

    def _check_floating_nodes(self, A):
        """Give a specific error for the most common singular case.

        A node whose row is entirely zero in the conductance/incidence
        block has no element touching it at all -- numpy would just report
        a generic LinAlgError, which is useless for debugging a netlist.
        """
        for name, i in self._node_index.items():
            if not np.any(A[i, :]) and not np.any(A[:, i]):
                raise CircuitError(f"node {name!r} is floating (not connected to any element)")
