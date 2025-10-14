#!/usr/bin/env python3
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


search_task_sys_msg = """

You are an expert mathematician specializing in boolean function analysis and circuit complexity. Your task is to construct DNF formulas with high average-case deterministic query complexity (avgQ). Here I introduce the background and describe your task.

# Background

## Definition

Let B = {0, 1} and f : B^n -> B be a boolean function, which outputs either 0 or 1 on n-bit inputs. The weight, denoted by wt(f), is the number of inputs on which f outputs 1.

[Def 1.] The average-case deterministic query complexity of a boolean function f: B^n -> B under a uniform input distribution is defined by avgQ(f) = min_T E_{x ~ B^n} [cost(T, x)], where T is taken over all zero-error deterministic decision trees that compute f. 

## Examples

### AND, OR

Define AND_n(x) = x_1 ⋀ ... ⋀ x_n  and OR_n(x) = x_1 V ... V x_n.

[Prop 2.] avgQ(f) = 2(1 - 1/2^n) for any n-variable boolean function f with wt(f) = 1.

KEY INSIGHT: Functions with weight 1 have avgQ approaching 2 as n increases.

[Cor 3.] avgQ(AND_n) = avgQ(OR_n) = 2(1 - 1/2^n).

### Threshold and Majority functions

[Def 4.] The threshold function is defined by THR_{n, t}(x) = 1 if and only if |x| >= t, where |x| is the number of 1's in x.
[Def 5.] The majority function is defined by MAJ_n(x) = 1 if and only if |x| >= n/2.

PROPERTY: THR_{n,t} can be represented by a CNF/DNF formula of width min(t, n-t), and MAJ_n can be
represented by a CNF/DNF formula of width ceil(n/2).

WHY: For threshold functions (a similar argument applies to majority):
  - DNF representation: "At least t variables are 1" = OR of all t-sized AND terms. Width = t.
  - CNF representation: "At most n-t variables are 0" = AND of all (n-t)-sized OR clauses. Width = n-t.
  - We can choose the better representation, giving width = min(t, n-t).

[Prop 6.] Let k = floor(n/2). For any n >= 1, avgQ(MAJ_n) = n - n * C(2*k, k) / 4^k = n - Theta(sqrt(n)).

### Tribes function

[Def 7.] The tribes function with parameters w and s, denoted by TRIBES_{w,s}, is defined by TRIBES_{w,s}(x) = (x_{1,1} ⋀ ... ⋀ x_{1,w}) V (x_{2,1} ⋀ ... ⋀ x_{2,w}) V ... V (x_{s,1} ⋀ ... ⋀ x_{s,w}), where n = w * s and the variables are partitioned into s disjoint blocks of size w.

[Prop 8.] For any w, s >= 1, avgQ(TRIBES_{w,s}) = (1 - (1 - 2^{-w})^s) * 2(2^w - 1).

[Cor 9.] avgQ(TRIBES_{w,s}) = Theta(s) when s < 2^w ln 4. Otherwise, avgQ(TRIBES_{w,s}) = Theta(2^w). 

## Mayor known results

[Thm 10.] For every CNF formula F of w width and n variables, we have avgQ(F) <= n(1 - log (n/w) / O(w)) + w / n.

[Thm 11.] For any integer 2 log n <= w <= n, there exists a boolean function f : B^n -> B computable by a DNF formula of width w and size ceil(2^w / w) such that avgQ(F) = n (1 - log (n) / Theta(w)).

PROOF SKETCH: Construction via function composition.

Key observation: OR_n has avgQ = O(1) under uniform distribution, but avgQ = n(1 - o(1)) under
a p-biased distribution when p = o(1/n) (each input bit is 1 with small probability p).

Construction: Take a hard function g with avgQ(g) ≈ n(1 - o(1)) and small weight (acts as a p-biased bit).
Compose: f = OR(g_1, g_2, ..., g_k) where each g_i is a copy of g on disjoint variable sets.
Under uniform distribution, f inherits the hardness from the biased-OR structure, achieving
avgQ(f) = n(1 - log(n)/Theta(w)). The DNF formula comes from expanding OR of DNFs.


# Your Task

Your goal is to write a Python function that constructs a DNF formula with maximum avgQ complexity.

## Function Signature

You must implement a function with this signature:

```python
def construct_formula(n: int, w: int) -> list[Circuit]:
    \"""
    Parameters:
        n: Number of input variables (the function will be f: {0,1}^n -> {0,1})
        w: Width constraint - each returned Circuit must have width at most w

    Returns:
        A list of Circuit objects, each representing an circuit of width ≤ w.
        These will be combined with OR to form the final DNF formula.
    \"""
    # Your implementation here
    pass
```

## Construction Strategy

Your function returns a list of Circuit objects, where each circuit uses at most `w` of VAR(0), ..., VAR(n-1) variables. These circuits will be combined with OR to form: f = OR(circuit_1, circuit_2, ..., circuit_k)

**Key insight**: Any circuit that uses at most w variables can be converted to a DNF formula of width at most w. This is because we can enumerate all inputs where the circuit outputs 1, and create an AND-term for each satisfying assignment. Therefore, your returned circuits don't need to be simple AND-terms - they can be arbitrary boolean functions on w of n variables!

This construction strategy is similar to Theorem 11: compose hard functions using OR to build a function that is hard to query under uniform distribution.

## Available Tools: The Circuit Class

The Circuit class provides these operations:

```python
from longshot.boolean import VAR_factory, AND, OR

# Create variables
VAR = VAR_factory(n)  # Factory for creating n variables
x0 = VAR(0)           # Variable x_0
x1 = VAR(1)           # Variable x_1

# Boolean operations
circuit = x0 & x1 & x2              # x_0 ∧ x_1 ∧ x_2, equivalent to AND([x0, x1, x2])
circuit = x0 | x1 | x2              # x_0 ∨ x_1, equivalent to OR([x0, x1, x2])
circuit = x0 - x1                   # x_0 ∧ ¬x_1, equivalent to AND([x0, ~x1])
circuit = ~ x0                      # ¬x_0

# Operations can be nested
circuit = AND([x0, OR([x1, x2])])     # x_0 ∧ (x_1 ∨ x_2)

# Useful methods on Circuit
width = circuit.width                # Get the width (number of variables used in this circuit)

# Note: You are NOT allowed to use Circuit class directly. 
```

## Allowed Libraries and Determinism

You may use standard Python libraries that help with construction, such as:
- `itertools` (combinations, permutations, product, etc.)
- `math` (floor, ceil, log, etc.)
- Any deterministic computational tools

**IMPORTANT**: DO NOT use any randomness (random, numpy.random, etc.). Your construction must be completely deterministic - given the same (n, w), it must always produce the same formula.

## Example: Constructing Majority

```python
def construct_majority(n: int, w: int) -> list[Circuit]:
    VAR = VAR_factory(n)
    variables = [VAR(i) for i in range(n)]

    # Majority: OR of all (n//2 + 1)-sized AND terms
    from itertools import combinations
    threshold = n // 2 + 1

    # Each combination becomes an AND term
    terms = [AND(list(combo)) for combo in combinations(variables, threshold)]

    return terms
```

## Evaluation and Scoring

Your function will be tested on various values of n with w ranging from (n - floor(log n)) to n. For each test case (n, w), your formula f is scored using:

  score(f) = s1 + s2 + s3

where:
  - Q(n,w) = n(1 - log(n/w) / w)  [theoretical upper bound]
  - eps = 0.01
  - C = 1 / eps
  - s1 = C * avgQ(f) / Q(n,w)
  - s2 = 1 / max(eps, Q(n,w) - avgQ(f) + eps)
  - s3 = C * (exp(avgQ(f) - Q(n,w)) - 1)  if avgQ(f) >= Q(n,w), else 0

Higher avgQ(f) yields higher scores. The bonus term rewards formulas that exceed the known bounds.

**IMPORTANT**: If any returned circuit uses more than w variables (circuit.width() > w) or more than n variables are used in circuits, the score will be set to 0. Always ensure each circuit respects the constraints!

## Guidelines and Tips

1. **Use all variables**: Every variable from 0 to n-1 should appear in at least one circuit. Unused variables waste potential complexity.

2. **Maximize circuit width**: Each circuit should use exactly w variables (not less). A circuit using fewer than w variables will have lower avgQ than one using w variables, so you're leaving complexity on the table.

3. **Be creative**: The best known result is n(1 - log(n)/Theta(w)) - can you find one that matches or exceeds the upper bound?

Your goal: Discover DNF formulas with avgQ as close to the upper bound as possible while respecting the width constraint w.
"""


job_config = LocalJobConfig(eval_program_path="evaluate.py")

strategy = "weighted"

if strategy == "uniform":
    # 1. Uniform from correct programs
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=0.0,
        exploitation_ratio=1.0,
    )
elif strategy == "hill_climbing":
    # 2. Hill Climbing (Always from the Best)
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=100.0,
        exploitation_ratio=1.0,
    )
elif strategy == "weighted":
    # 3. Weighted Prioritization
    parent_config = dict(
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )
elif strategy == "power_law":
    # 4. Power-Law Prioritization
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=1.0,
        exploitation_ratio=0.2,
    )
elif strategy == "power_law_high":
    # 4. Power-Law Prioritization
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=2.0,
        exploitation_ratio=0.2,
    )
elif strategy == "beam_search":
    # 5. Beam Search
    parent_config = dict(
        parent_selection_strategy="beam_search",
        num_beams=10,
    )
    
db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=5,
    # Inspiration parameters
    elite_selection_ratio=0.6,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    # Island migration parameters
    migration_interval=10,
    migration_rate=0.1,  # chance to migrate program to random island
    island_elitism=True,  # Island elite is protected from migration
    **parent_config,
)


evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=30,
    max_parallel_jobs=16,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=[
        # "gemini-2.5-pro",
        # "gemini-2.5-flash",
        # "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        "o4-mini",
        # "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],
        reasoning_efforts=["auto", "low", "medium"],
        max_tokens=32768,
    ),
    meta_rec_interval=10,
    meta_llm_models=["gpt-5-nano"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    embedding_model="text-embedding-3-small",
    code_embed_sim_threshold=0.97,
    novelty_llm_models=["gpt-5-nano"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
    init_program_path="initial.py",
    results_dir="results",
)


def main():
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()


if __name__ == "__main__":
    results_data = main()
