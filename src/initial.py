from longshot import XOR, AND, OR

# EVOLVE-BLOCK-START

def construct_formula(n, w):
    """
    Parameters:
        n: Number of input variables (the function will be f: {0,1}^n -> {0,1})
        w: Width constraint - each returned Circuit must have width at most w

    Returns:
        A list of Circuit objects, each representing an circuit of width ≤ w.
        These will be combined with OR to form the final DNF formula.
    """
    # This DNF formula has avgQ at least w but doesn't utilize
    # other n - w variables well to achieve higher complexity.
    return [XOR([VAR(i) for i in range(w)])]

# EVOLVE-BLOCK-END

from longshot import Circuit, VAR_factory, avgQ

# This part remains fixed (not evolved)
def run_experiment(n: int, w: int) -> tuple[int, int, float, list[Circuit]]:
    """Run experiment to construct and evaluate a DNF formula.

    Constructs a DNF formula from circuits with specified constraints and
    evaluates its average-case query complexity (avgQ).

    Parameters:
        n: Number of input variables (the function will be f: {0,1}^n -> {0,1})
        w: Width constraint - each circuit must have width at most w

    Returns:
        tuple containing:
            - n: Number of variables (echoed back)
            - w: Width constraint (echoed back)
            - q: Average-case query complexity (avgQ) of the constructed DNF formula
            - circuits: List of Circuit objects that form the DNF formula
    """
    global VAR
    VAR = VAR_factory(n)
    
    circuits = construct_formula(n, w)
    dnf = OR(circuits)
    q = avgQ(dnf)
    
    return n, w, q, circuits