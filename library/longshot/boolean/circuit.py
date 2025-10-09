"""Truth table representations for Boolean functions using PyTorch tensors."""
import torch
from typing import Callable, Iterable, Sequence, Iterator
from .decision_tree import DecisionTree
from .._core import MAX_VARS, _MonotonicBooleanFunction, _CppDecisionTree


class Circuit:
    """
    A circuit representing a Boolean function with truth table representation.

    Truth tables can be stored in two formats:
    - For n ≤ 6 variables: packed into a 64-bit integer where bit k represents output for input k
    - For n > 6 variables: stored as a tensor of uint64 values

    Attributes:
        vars: Set of variable indices used in this circuit
        table: PyTorch tensor storing the truth table values
    """

    def __init__(
        self, 
        num_vars: int,
        involved_vars: Iterable, 
        truth_table: torch.Tensor | None = None
    ):
        """
        Initialize a circuit with the given truth table tensor.

        Args:
            involved_vars: Iterable of variable indices used in this circuit
            truth_table: Optional PyTorch tensor storing truth table values
        """
        self.num_vars = num_vars
        self.vars = set(involved_vars)
        self.table = truth_table

    def __xor__(self, other: "Circuit") -> "Circuit":
        """Compute XOR of two circuits."""
        if self.num_vars != other.num_vars:
            raise ValueError("Cannot XOR circuits with different number of variables")
        return Circuit(self.num_vars, self.vars.union(other.vars), self.table ^ other.table)

    def __and__(self, other: "Circuit") -> "Circuit":
        """Compute AND of two circuits."""
        if self.num_vars != other.num_vars:
            raise ValueError("Cannot XOR circuits with different number of variables")
        return Circuit(self.num_vars, self.vars.union(other.vars), self.table & other.table)

    def __or__(self, other: "Circuit") -> "Circuit":
        """Compute OR of two circuits."""
        if self.num_vars != other.num_vars:
            raise ValueError("Cannot XOR circuits with different number of variables")
        return Circuit(self.num_vars, self.vars.union(other.vars), self.table | other.table)

    def __sub__(self, other: "Circuit") -> "Circuit":
        """Compute set difference (A AND NOT B) of two circuits."""
        if self.num_vars != other.num_vars:
            raise ValueError("Cannot XOR circuits with different number of variables")
        return Circuit(self.num_vars, self.vars.union(other.vars), self.table & ~other.table)

    def __invert__(self) -> "Circuit":
        """Compute NOT (negation) of the circuit."""
        return Circuit(self.num_vars, self.vars, ~self.table)
    
    @property
    def width(self) -> int:
        """Return the width of the circuit."""
        return len(self.vars)
    
    def __getitem__(self, idx: int) -> int:
        """Get the output value for a specific input index."""
        if self.num_vars <= 6:
            return (self.table.item() >> idx) & 1
        else:
            chunk_idx = idx // 64  # idx // 64
            bit_idx = idx % 64    # idx % 64
            return (self.table[chunk_idx].item() >> bit_idx) & 1
    
    def __str__(self):
        ts = '\n\t' + ',\n\t'.join([f'{bin(i)}={self[i]}' for i in range(1 << self.num_vars)])
        return f"Circuit(vars={self.vars}, table={ts})"
    

def to_signed(n: int, bits: int = 64) -> int:
    return n if n < (1 << (bits - 1)) else n - (1 << bits)

def flatten_args(args: Sequence[Circuit | Sequence[Circuit] | Iterator[Circuit]]) -> list[Circuit]:
    """Flatten nested sequences of Circuit objects into a single list."""
    flat_list = []
    for arg in args:
        if isinstance(arg, Circuit):
            flat_list.append(arg)
        elif isinstance(arg, Sequence):
            flat_list.extend(arg)
        elif isinstance(arg, Iterator):
            flat_list.extend(list(arg))
        else:
            raise ValueError("Arguments must be Circuit or Sequence of Circuit")
    return flat_list

def VAR_factory(
    num_vars: int,
    device: torch.device | str | None = None
) -> Callable[[int], Circuit]:
    """
    Create a factory function for generating single-variable circuits.

    Args:
        num_vars: Total number of variables (must be in range [1, MAX_VARS])
        device: PyTorch device for tensor allocation

    Returns:
        A callable VAR(vidx) that creates a circuit for variable vidx

    Raises:
        ValueError: If num_vars is not in range [1, MAX_VARS]
    """
    if num_vars <= 0 or num_vars > MAX_VARS:
        raise ValueError(f"num_vars must be in the range [1, {MAX_VARS}]")

    def VAR(vidx: int) -> Circuit:
        """
        Generate a circuit for a single variable.

        Args:
            vidx: Variable index (0-based)

        Returns:
            Circuit representing the variable

        Raises:
            ValueError: If vidx is out of range
        """
        if vidx < 0 or vidx >= num_vars:
            raise ValueError(f"vidx must be in the range [0, {num_vars - 1}]")

        if num_vars <= 6:
            # For n <= 6, we can pack the truth table into a 64-bit integer
            # The k-th bit represents the output when input = k
            # For variable vidx, it's 1 when bit vidx of k is 1
            x = 0
            for k in range(1 << num_vars):  # 2^num_vars entries
                if (k >> vidx) & 1:  # Check if vidx-th bit of k is 1
                    x |= (1 << k)  # Set k-th bit in the truth table

            tensor = torch.tensor(
                [to_signed(x)],
                dtype=torch.int64,
                device=device
            )
        else:
            if vidx < 6:
                x = 0
                for k in range(1 << 6):  # 2^6 = 64 entries
                    if (k >> vidx) & 1:  # Check if vidx-th bit of k is 1
                        x |= (1 << k)  # Set k-th bit in the truth table
                        
                t = torch.zeros(
                    (1 << (num_vars - 6),), 
                    dtype=torch.int64, 
                    device=device
                )
                tensor = torch.fill(t, to_signed(x))
            else:
                # torch.arange and torch.where don't support uint64, use int64 then convert
                x = torch.arange(
                    0,
                    1 << (num_vars - 6),
                    dtype=torch.int64,
                    device=device
                )
                mask = (1 << (vidx - 6))
                # -1 in int64 has all bits set, which becomes (1<<64)-1 when cast to uint64
                ones = torch.tensor(-1, dtype=torch.int64, device=device)
                zeros = torch.tensor(0, dtype=torch.int64, device=device)
                tensor = torch.where((x & mask) > 0, ones, zeros)

        return Circuit(num_vars, [vidx], tensor)
    
    return VAR

def XOR(*args: Sequence[Circuit | Sequence[Circuit] | Iterator[Circuit]]) -> Circuit:
    """
    Compute the XOR (exclusive OR) of multiple circuits.

    Args:
        *args: Variable number of Circuit objects, sequences of Circuits, or iterators of Circuits

    Returns:
        Circuit representing the XOR of all inputs

    Raises:
        ValueError: If no arguments are provided
    """
    circuits = flatten_args(args)
    if not circuits:
        raise ValueError("XOR requires at least one argument")
    result = circuits[0]
    for tt in circuits[1:]:
        result = result ^ tt
    return result

def AND(*args: Sequence[Circuit | Sequence[Circuit] | Iterator[Circuit]]) -> Circuit:
    """
    Compute the AND (conjunction) of multiple circuits.

    Args:
        *args: Variable number of Circuit objects, sequences of Circuits, or iterators of Circuits

    Returns:
        Circuit representing the AND of all inputs

    Raises:
        ValueError: If no arguments are provided
    """
    circuits = flatten_args(args)
    if not circuits:
        raise ValueError("AND requires at least one argument")
    result = circuits[0]
    for tt in circuits[1:]:
        result = result & tt
    return result

def OR(*args: Sequence[Circuit | Sequence[Circuit] | Iterator[Circuit]]) -> Circuit:
    """
    Compute the OR (disjunction) of multiple circuits.

    Args:
        *args: Variable number of Circuit objects, sequences of Circuits, or iterators of Circuits

    Returns:
        Circuit representing the OR of all inputs

    Raises:
        ValueError: If no arguments are provided
    """
    circuits = flatten_args(args)
    if not circuits:
        raise ValueError("OR requires at least one argument")
    result = circuits[0]
    for tt in circuits[1:]:
        result = result | tt
    return result


def avgQ(
    circuit: Circuit,
    build_tree: bool = False
) -> float | tuple[float, DecisionTree]:
    """
    Calculate the average query complexity for the given circuit.

    The average query complexity (avgQ) measures the expected number of variable
    queries needed to determine the output of a Boolean function, averaged over
    all possible inputs.

    Args:
        circuit: The circuit to analyze
        build_tree: If True, also construct and return the decision tree

    Returns:
        If build_tree is False, returns the average query complexity as a float.
        If build_tree is True, returns a tuple of (avgQ value, DecisionTree object).

    Raises:
        May raise exceptions from MonotonicBooleanFunction construction or avgQ computation
    """
    n = circuit.num_vars
    table = circuit.table.cpu().to(torch.uint64).numpy()
    bf = _MonotonicBooleanFunction(n, table)
    ctree = _CppDecisionTree() if build_tree else None
    qv = bf.avgQ(ctree)

    if build_tree:
       return qv, DecisionTree(ctree)

    return qv