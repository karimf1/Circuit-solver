"""Two-terminal linear elements and their MNA stamps.

Each element knows only how to add its contribution to the system matrix
``A`` and RHS vector ``z`` given a node-name -> row-index lookup. It does not
know about the rest of the circuit, so these stamps can be (and are) unit
tested in isolation against a hand-built index map before any parser exists.

Sign conventions (standard SPICE):
  - Current sources ``I n+ n- value``: current flows from n+ to n- *inside*
    the source, i.e. out of n+ and into n- in the external circuit.
  - Voltage sources ``V n+ n- value``: adds an extra current unknown
    ``i_k`` (the current flowing from n+ to n- through the source, i.e. out
    of the source into the external circuit at n+).
"""

from dataclasses import dataclass


@dataclass
class Resistor:
    name: str
    n1: str
    n2: str
    value: float  # ohms

    def stamp(self, A, z, idx, vidx):
        g = 1.0 / self.value
        i, j = idx(self.n1), idx(self.n2)
        if i is not None:
            A[i, i] += g
        if j is not None:
            A[j, j] += g
        if i is not None and j is not None:
            A[i, j] -= g
            A[j, i] -= g


@dataclass
class CurrentSource:
    name: str
    n1: str
    n2: str
    value: float  # amps, flowing n1 -> n2 through the source

    def stamp(self, A, z, idx, vidx):
        i, j = idx(self.n1), idx(self.n2)
        if i is not None:
            z[i] -= self.value
        if j is not None:
            z[j] += self.value


@dataclass
class VoltageSource:
    name: str
    n1: str
    n2: str
    value: float  # volts, n1 - n2

    def stamp(self, A, z, idx, vidx):
        i, j = idx(self.n1), idx(self.n2)
        k = vidx(self.name)
        if i is not None:
            A[i, k] += 1.0
            A[k, i] += 1.0
        if j is not None:
            A[j, k] -= 1.0
            A[k, j] -= 1.0
        z[k] = self.value
