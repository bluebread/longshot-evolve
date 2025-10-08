"""Truth table representations for Boolean functions using PyTorch tensors."""
import torch
from typing import Callable, Iterable
from decision_tree import DecisionTree
from .._core import MAX_VARS, MonotonicBooleanFunction

class CircuitComponent:
    """
    A circuit component representing a Boolean function with truth table representation.

    Truth tables can be stored in two formats:
    - For n ≤ 6 variables: packed into a 64-bit integer where bit k represents output for input k
    - For n > 6 variables: stored as a tensor of uint64 values

    Attributes:
        vars: Set of variable indices used in this circuit component
        table: PyTorch tensor storing the truth table values
    """

    def __init__(self, involved_vars: Iterable, truth_table: torch.Tensor | None = None):
        """
        Initialize a circuit component with the given truth table tensor.

        Args:
            involved_vars: Iterable of variable indices used in this circuit component
            truth_table: Optional PyTorch tensor storing truth table values
        """
        self.vars = set(involved_vars)
        self.table = truth_table

    def __xor__(self, other: "CircuitComponent") -> "CircuitComponent":
        """Compute XOR of two circuit components."""
        return CircuitComponent(self.vars.union(other.vars), self.table ^ other.table)

    def __and__(self, other: "CircuitComponent") -> "CircuitComponent":
        """Compute AND of two circuit components."""
        return CircuitComponent(self.vars.union(other.vars), self.table & other.table)

    def __or__(self, other: "CircuitComponent") -> "CircuitComponent":
        """Compute OR of two circuit components."""
        return CircuitComponent(self.vars.union(other.vars), self.table | other.table)

    def __sub__(self, other: "CircuitComponent") -> "CircuitComponent":
        """Compute set difference (A AND NOT B) of two circuit components."""
        return CircuitComponent(self.vars.union(other.vars), self.table & ~other.table)

    def __invert__(self) -> "CircuitComponent":
        """Compute NOT (negation) of the circuit component."""
        return CircuitComponent(self.vars, ~self.table)
    
    @property
    def num_vars(self) -> int:
        """Return the number of variables in the circuit component."""
        return len(self.vars)
    
    def avgQ(self, build_tree: bool = False) -> float | tuple[float, DecisionTree]:
        pass    
        # ctree = _CppDecisionTree() if build_tree else None
        # qv = self._bf.avgQ(ctree)
        
        # if build_tree:
        #    return qv, DecisionTree(ctree)
       
        # return qv
    

def single_var_template(
    num_vars: int,
    device: torch.device | str | None = None
) -> Callable[[int], CircuitComponent]:
    """
    Create a factory function for generating single-variable circuit components.

    Args:
        num_vars: Total number of variables (must be in range [1, MAX_VARS])
        device: PyTorch device for tensor allocation

    Returns:
        A callable VAR(vidx) that creates a circuit component for variable vidx

    Raises:
        ValueError: If num_vars is not in range [1, MAX_VARS]
    """
    if num_vars <= 0 or num_vars > MAX_VARS:
        raise ValueError(f"num_vars must be in the range [1, {MAX_VARS}]")

    def VAR(vidx: int) -> CircuitComponent:
        """
        Generate a circuit component for a single variable.

        Args:
            vidx: Variable index (0-based)

        Returns:
            CircuitComponent representing the variable

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
                x,
                dtype=torch.uint64,
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
                    dtype=torch.uint64, 
                    device=device
                )
                tensor = torch.fill(t, x)
            else:
                x = torch.arange(
                    0, 
                    1 << (num_vars - 6), 
                    dtype=torch.uint64, 
                    device=device
                )
                mask = (1 << (vidx - 6))
                ones = torch.tensor((1 << 64) - 1, dtype=torch.uint64, device=device)
                zeros = torch.tensor(0, dtype=torch.uint64, device=device)
                tensor = torch.where((x & mask) > 0, ones, zeros)

        return CircuitComponent(tensor)
    
    return VAR

def XOR(*args: list[CircuitComponent]) -> CircuitComponent:
    """
    Compute the XOR (exclusive OR) of multiple circuit components.

    Args:
        *args: Variable number of CircuitComponent objects

    Returns:
        CircuitComponent representing the XOR of all inputs

    Raises:
        ValueError: If no arguments are provided
    """
    if not args:
        raise ValueError("XOR requires at least one argument")
    result = args[0]
    for tt in args[1:]:
        result = result ^ tt
    return result

def AND(*args: list[CircuitComponent]) -> CircuitComponent:
    """
    Compute the AND (conjunction) of multiple circuit components.

    Args:
        *args: Variable number of CircuitComponent objects

    Returns:
        CircuitComponent representing the AND of all inputs

    Raises:
        ValueError: If no arguments are provided
    """
    if not args:
        raise ValueError("AND requires at least one argument")
    result = args[0]
    for tt in args[1:]:
        result = result & tt
    return result

def OR(*args: list[CircuitComponent]) -> CircuitComponent:
    """
    Compute the OR (disjunction) of multiple circuit components.

    Args:
        *args: Variable number of CircuitComponent objects

    Returns:
        CircuitComponent representing the OR of all inputs

    Raises:
        ValueError: If no arguments are provided
    """
    if not args:
        raise ValueError("OR requires at least one argument")
    result = args[0]
    for tt in args[1:]:
        result = result | tt
    return result
