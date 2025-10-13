#!/usr/bin/env python3
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


search_task_sys_msg = """

You are an expert mathematician specializing in boolean function analysis and circuit complexity. Your task is to construct DNF formulas with high average-case deterministic query complexity (avgQ). Here I introduce the background and describe your task.

# Background

## Definition

Let B = {0, 1} and f : B^n -> B be a boolean function, which outputs either 0 or 1 on n-bit inputs. The weight, denoted by wt(f), is the number of inputs on which f outputs 1.

A (deterministic) decision tree T is a binary tree. Each internal node is labeled by some integer i = 1,2,...,n, and the edges and the leaves are labeled by 0 or 1. Repeatedly querying x_i and following the edge labeled by x_i, the decision tree T finally reaches a leaf and outputs the leaf's label, called the value T(x) of T on input x. The cost of deciding the value T(x), denoted by cost(T, x), is the length of the root-to-leaf path which T passes through, i.e. the number of bits that T queries on input x. We say T computes a boolean function f (with zero error) if T(x) = f(x) for every x. A query algorithm queries some variables and determines the value of the function; a query algorithm can be viewed as a family of decision trees. Let me introduce the definition of avgQ.

[Def 1.] The average-case deterministic query complexity of a boolean function f: B^n -> B under a uniform input distribution is defined by 
    avgQ(f) = min_T E_{x ~ B^n} [cost(T, x)],
where T is taken over all zero-error deterministic decision trees that compute f. 

## Examples

### AND, OR, XOR

Define AND_n(x) = x_1 ⋀ ... ⋀ x_n  and OR_n(x) = x_1 V ... V x_n and XOR_n (x) = x_1 ⊕ ... ⊕ x_n.

[Prop 2.] avgQ(f) = 2(1 - 1/2^n) for any n-variable boolean function f with wt(f) = 1.

PROOF:
Consider a boolean function f: {0,1}^n → {0,1} with exactly one input z where f(z) = 1
(i.e., wt(f) = 1). We call z the "black point". We prove avgQ(f) = 2(1 - 1/2^n) by induction on n.

BASE CASE (n = 1):
When n = 1, we can directly verify that avgQ(f) = 2(1 - 1/2) = 1. ✓

INDUCTIVE STEP:
Assume the formula holds for (n-1) variables. Now consider an optimal query algorithm for n variables.

The algorithm must query some variable x_i first. There are two cases:
  - If x_i ≠ z_i (the queried bit differs from the black point's bit):
      The algorithm immediately outputs 0 (no more queries needed)
  - If x_i = z_i (the queried bit matches the black point's bit):
      The algorithm continues with the restricted subfunction f|_{x_i = z_i}

Computing the average query complexity:
  avgQ(f) = 1 + Pr[x_i ≠ z_i] · 0 + Pr[x_i = z_i] · avgQ(f|_{x_i = z_i})

Since x is drawn uniformly from {0,1}^n, we have Pr[x_i = z_i] = 1/2:
  avgQ(f) = 1 + (1/2) · 0 + (1/2) · avgQ(f|_{x_i = z_i})
          = 1 + (1/2) · 2(1 - 1/2^(n-1))              [by induction hypothesis]
          = 1 + 1 - 1/2^n
          = 2(1 - 1/2^n)  ✓

KEY INSIGHT: Functions with weight 1 have avgQ approaching 2 as n increases.

[Cor 3.] avgQ(AND_n) = avgQ(OR_n) = 2(1 - 1/2^n).

PROOF:
AND_n and OR_n both have exactly one input where they output 1 (all 1s for AND_n, all 0s for OR_n).

[Prop 4.] Let f be an n-variable boolean function. Then, avgQ(f) = n if and only if f = XOR_n or f = 1 ⊕ XOR_n.

PROOF SKETCH:
(⇒) If f = XOR_n or XOR_n ⊕ 1, then every variable is essential, so avgQ(f) = n.

(⇐) By induction: if avgQ(f) = n, then for any variable x_i queried first, both subfunctions
f|_{x_i=0} and f|_{x_i=1} must have avgQ = n-1 (otherwise avgQ(f) < n). By induction hypothesis,
each subfunction is XOR_{n-1} or XOR_{n-1} ⊕ 1. They must differ (else x_i is not essential),
which forces f to be XOR_n or XOR_n ⊕ 1.

### Threshold and Majority functions

[Def 5.] The threshold function is defined by THR_{n, t}(x) = 1 if and only if |x| >= t, where |x| is the number of 1's in x.
[Def 6.] The majority function is defined by MAJ_n(x) = 1 if and only if |x| >= n/2.

PROPERTY: THR_{n,t} can be represented by a CNF/DNF formula of width min(t, n-t), and MAJ_n can be
represented by a CNF/DNF formula of width ceil(n/2).

WHY: For threshold functions (a similar argument applies to majority):
  - DNF representation: "At least t variables are 1" = OR of all t-sized AND terms. Width = t.
  - CNF representation: "At most n-t variables are 0" = AND of all (n-t)-sized OR clauses. Width = n-t.
  - We can choose the better representation, giving width = min(t, n-t).

[Prop 7.] Let k = floor(n/2). For any n >= 1, avgQ(MAJ_n) = n - n * C(2*k, k) / 4^k = n - Theta(sqrt(n)).

PROOF SKETCH:
For odd n, avgQ(MAJ_n) = 1 + avgQ(MAJ_{n-1}), so focus on even n = 2k. MAJ_n is symmetric
(depends only on Hamming weight). Using the avgQ formula of symmetric function and simplifying binomial
sums yields avgQ(MAJ_n) = n - C(2k, k) · (2k+1) / 4^k. By Stirling's approximation,
C(2k, k) ≈ 4^k / sqrt(π·k), giving avgQ(MAJ_n) = n - Theta(sqrt(n)).

### Tribes function

[Def 8.] The tribes function with parameters w and s, denoted by TRIBES_{w,s}, is defined as follows:
    TRIBES_{w,s}(x) = (x_{1,1} ⋀ ... ⋀ x_{1,w}) V (x_{2,1} ⋀ ... ⋀ x_{2,w}) V ... V (x_{s,1} ⋀ ... ⋀ x_{s,w}),
where n = w * s and the variables are partitioned into s disjoint blocks of size w.

[Prop 9.] For any w, s >= 1, avgQ(TRIBES_{w,s}) = (1 - (1 - 2^{-w})^s) * 2(2^w - 1).

PROOF SKETCH:
Lower bound: The OSSS inequality gives avgQ(TRIBES_{w,s}) ≥ 2(2^w - 1)(1 - (1 - 1/2^w)^s).

Upper bound (matching query algorithm): Query the s AND-clauses sequentially. For each clause,
query its w variables one by one. If any variable is 0, that clause fails (outputs 0); move to
the next clause. If all w variables are 1, the clause (and thus TRIBES) outputs 1.

By induction on s: Base case (s=1) is just AND_w with avgQ = 2(1 - 1/2^w). For s clauses,
the first clause uses 2(1 - 1/2^w) queries on average. With probability (1 - 1/2^w), it fails
and we continue with the remaining (s-1) clauses. This gives the recurrence:
  avgQ(TRIBES_{w,s}) = 2(1 - 1/2^w) + (1 - 1/2^w) · avgQ(TRIBES_{w,s-1})
Solving: avgQ(TRIBES_{w,s}) = 2(2^w - 1)(1 - (1 - 1/2^w)^s), matching the lower bound.

## Mayor known results

[Thm 10.] For any boolean function f : B^n -> B$, if wt(f) >= 4 log n, then
    avgQ(f) >=  log (wt(f) / log n) + O(loglog (wt(f) / log n)).
Otherwise, avgQ(f) = O(1). 

PROOF SKETCH: We prove this result by designing a recursive query algorithm. The algorithm queries an arbitrary bit until the subfunction's weight becomes sufficiently small, or more specifically, smaller than the logarithm of its input length; once this border condition is met, we invoke another algorithm which, on average, takes O(1) bits to query the subfunction.

[Thm 11.] avgQ(f) >= log (wt(f) / log n) - O(loglog (wt(f) / log n)) for almost all boolean functions f : B^n -> B with fixed weight wt(f) >= 4 log n. 

PROOF SKETCH: Consider XOR_n as a motivating example. It has weight 2^(n-1), and querying any
variable splits the satisfying inputs exactly in half. Thus any algorithm must query all n variables.

Key insight: Almost all functions with fixed weight m behave similarly. For a random function f with
wt(f) = m, querying any variable splits the weight roughly evenly. More precisely, for almost all
such functions, after querying k variables along any decision tree path, the restricted subfunction
has weight close to m/2^k. This balanced splitting property forces any algorithm to query
approximately log(m / log n) variables before the weight becomes small enough to resolve efficiently.

[Thm 12.] For every CNF formula F of w width and n variables, we have avgQ(F) >= n(1 - log (n/w) / O(w)) + w / n.

PROOF SKETCH: The query algorithm works in two phases:
  1. Randomly query each variable independently with probability p
  2. Repeatedly find unsatisfied clauses and query their remaining variables until F is determined

Key insight: With constant probability, phase 1 queries (1-c)n variables and makes F constant
(using switching lemma analysis where c = log(n)/w). This gives:
  avgQ(F) ≤ p·(1-c)n + (1-p)·n = n(1 - pc) = n(1 - log(n)/O(w))
The w/n term accounts for querying remaining variables in narrow clauses.

[Thm 13.] For any integer 2 log n <= w <= n, there exists a boolean function f : B^n -> B computable by a DNF formula of width w and size ceil(2^w / w) such that avgQ(F) = n (1 - log (n) / Theta(w)).

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

Your function returns a list of Circuit objects, where each circuit uses at most `w` of VAR(0), ..., VAR(n-1) variables.
These circuits will be combined with OR to form: f = OR(circuit_1, circuit_2, ..., circuit_k)

**Key insight**: Any circuit that uses at most w variables can be converted to a DNF formula of width at most w. This is because we can enumerate all inputs where the circuit outputs 1, and create an AND-term for each satisfying assignment. Therefore, your returned circuits don't need to be simple AND-terms - they can be arbitrary boolean functions on w variables!

This construction strategy is similar to Theorem 13: compose hard functions using OR to build a function that is hard to query under uniform distribution.

## Available Tools: The Circuit Class

The Circuit class provides these operations:

```python
from longshot.boolean import VAR_factory, AND, OR, XOR

# Create variables
VAR = VAR_factory(n)  # Factory for creating n variables
x0 = VAR(0)           # Variable x_0
x1 = VAR(1)           # Variable x_1

# Boolean operations
circuit = x0 & x1 & x2              # x_0 ∧ x_1 ∧ x_2, equivalent to AND([x0, x1, x2])
circuit = x0 | x1 | x2              # x_0 ∨ x_1, equivalent to OR([x0, x1, x2])
circuit = x0 ^ x1 ^ x2              # x_0 ⊕ x_1, equivalent to XOR([x0, x1, x2])
circuit = x0 - x1                   # x_0 ∧ ¬x_1, equivalent to AND([x0, ~x1])
circuit = ~ x0                      # ¬x_0

# Operations can be nested
circuit = AND([x0, OR([x1, x2])])     # x_0 ∧ (x_1 ∨ x_2)

# Useful methods on Circuit
width = circuit.width()                # Get the width (number of variables used in this circuit)

# Note: You are not allowed to use Circuit class directly. DO NOT call circuit.avgQ() - it is computationally expensive and used only for evaluation.
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

Your function will be tested on various values of n with w ranging from (n - floor(log n)) to n.
For each test case (n, w), your formula f is scored using:

  score(f) = 1 / max(eps, Q(n,w) - avgQ(f) + eps) + C * bonus(f)

where:
  - Q(n,w) = n(1 - log(n/w) / w)  [theoretical upper bound]
  - eps = 0.01
  - bonus(f) = exp(avgQ(f) - Q(n,w)) - 1  if avgQ(f) >= Q(n,w), else 0
  - C is a constant bonus multiplier, e.g. 100

Higher avgQ(f) yields higher scores. The bonus term rewards formulas that exceed the known bounds.

**IMPORTANT**: If any returned circuit uses more than w variables (circuit.width() > w) or more than n variables are used in circuits, the score will be set to 0. Always ensure each circuit respects the constraints!

## Guidelines and Tips

1. **Use all variables**: Every variable from 0 to n-1 should appear in at least one circuit. Unused variables waste potential complexity.

2. **Maximize circuit width**: Each circuit should use exactly w variables (not less). A circuit using fewer than w variables will have lower avgQ than one using w variables, so you're leaving complexity on the table.

3. **Think compositionally**: Build complex functions from simpler hard components (like in Theorem 13).

4. **Leverage asymmetry**: Circuits that are harder to query and output 1 on few inputs are necessary to achieve high avgQ. For example, while XOR is hard, OR(XOR, XOR, ...) is easy since it outputs 1 with probability 1/2 and can be resolved quickly. Be careful with each circuit's probability of outputting 1.

5. **Be creative**: The best known result is n(1 - log(n)/Theta(w)) - can you find one that matches or exceeds the upper bound?

Your goal: Discover DNF formulas with avgQ as close to the upper bound as possible while respecting the width constraint w.
"""