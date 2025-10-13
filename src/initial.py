from longshot import XOR, AND, OR

# EVOLVE-BLOCK-START

def construct_formula(n, w):
    """"""
    # This DNF formula has avgQ at least w but doesn't utilize
    # other n - w variables well to achieve higher complexity. 
    return [XOR([VAR(i) for i in range(w)])]

# EVOLVE-BLOCK-END

from longshot import Circuit, VAR_factory

# This part remains fixed (not evolved)
def run_experiment(n: int, w: int) -> list[Circuit]:
    """"""
    global VAR
    VAR = VAR_factory(n)
    
    return construct_formula(n, w)