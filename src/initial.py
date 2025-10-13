from longshot import XOR, AND, OR

# EVOLVE-BLOCK-START

def construct_formula(n, w):
    """"""
    return VAR(0)

# EVOLVE-BLOCK-END

from longshot import Circuit, VAR_factory

# This part remains fixed (not evolved)
def run_experiment(n: int, w: int) -> list[Circuit]:
    """"""
    global VAR
    VAR = VAR_factory(n)
    
    return construct_formula(n, w)