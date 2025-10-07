"""Truth table representations for Boolean functions using PyTorch tensors."""
import torch
from typing import Callable
from .._core import MAX_VARS

class TruthTable:
    """
    A truth table representation of a Boolean function.

    Truth tables can be stored in two formats:
    - For n ≤ 6 variables: packed into a 64-bit integer where bit k represents output for input k
    - For n > 6 variables: stored as a tensor of uint64 values

    Attributes:
        tensor: PyTorch tensor storing the truth table values
    """

    def __init__(self, tensor: torch.Tensor | None = None):
        """Initialize a truth table with the given tensor."""
        self.tensor = tensor

    def __xor__(self, other: "TruthTable") -> "TruthTable":
        """Compute XOR of two truth tables."""
        return TruthTable(self.tensor ^ other.tensor)

    def __and__(self, other: "TruthTable") -> "TruthTable":
        """Compute AND of two truth tables."""
        return TruthTable(self.tensor & other.tensor)

    def __or__(self, other: "TruthTable") -> "TruthTable":
        """Compute OR of two truth tables."""
        return TruthTable(self.tensor | other.tensor)

    def __sub__(self, other: "TruthTable") -> "TruthTable":
        """Compute set difference (A AND NOT B) of two truth tables."""
        return TruthTable(self.tensor & ~other.tensor)

    def __invert__(self) -> "TruthTable":
        """Compute NOT (negation) of the truth table."""
        return TruthTable(~self.tensor)


def single_var_template(
    num_vars: int,
    device: torch.device | str | None = None
) -> Callable[[int], TruthTable]:
    """
    Create a factory function for generating single-variable truth tables.

    Args:
        num_vars: Total number of variables (must be in range [1, MAX_VARS])
        device: PyTorch device for tensor allocation

    Returns:
        A callable VAR(vidx) that creates a truth table for variable vidx

    Raises:
        ValueError: If num_vars is not in range [1, MAX_VARS]
    """
    if num_vars <= 0 or num_vars > MAX_VARS:
        raise ValueError(f"num_vars must be in the range [1, {MAX_VARS}]")

    def VAR(vidx: int) -> TruthTable:
        """
        Generate a truth table for a single variable.

        Args:
            vidx: Variable index (0-based)

        Returns:
            TruthTable representing the variable

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
                        
                tensor = torch.fill(
                    (1 << (num_vars - 6),),
                    x, 
                    dtype=torch.uint64, 
                    device=device
                )
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

        return TruthTable(tensor)
    
    return VAR

def XOR(*args: list[TruthTable]) -> TruthTable:
    """
    Compute the XOR (exclusive OR) of multiple truth tables.

    Args:
        *args: Variable number of TruthTable objects

    Returns:
        TruthTable representing the XOR of all inputs

    Raises:
        ValueError: If no arguments are provided
    """
    if not args:
        raise ValueError("XOR requires at least one argument")
    result = args[0]
    for tt in args[1:]:
        result = result ^ tt
    return result

def AND(*args: list[TruthTable]) -> TruthTable:
    """
    Compute the AND (conjunction) of multiple truth tables.

    Args:
        *args: Variable number of TruthTable objects

    Returns:
        TruthTable representing the AND of all inputs

    Raises:
        ValueError: If no arguments are provided
    """
    if not args:
        raise ValueError("AND requires at least one argument")
    result = args[0]
    for tt in args[1:]:
        result = result & tt
    return result

def OR(*args: list[TruthTable]) -> TruthTable:
    """
    Compute the OR (disjunction) of multiple truth tables.

    Args:
        *args: Variable number of TruthTable objects

    Returns:
        TruthTable representing the OR of all inputs

    Raises:
        ValueError: If no arguments are provided
    """
    if not args:
        raise ValueError("OR requires at least one argument")
    result = args[0]
    for tt in args[1:]:
        result = result | tt
    return result
