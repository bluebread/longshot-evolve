# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LongshotEvolve is a boolean formula generation system that uses AlphaEvolve (via the `shinka` framework) to automatically evolve DNF (Disjunctive Normal Form) formulas with high average-case deterministic query complexity (avgQ). The project combines:

- **Longshot library** (`library/`): A C++/Python hybrid library for boolean circuit analysis and avgQ computation
- **Evolution system** (`src/`): Python scripts that evolve formulas using LLM-guided mutations

## Core Architecture

### The Evolution Loop

The evolution system operates on a target function `construct_formula(n, w)` that lives in `src/initial.py` between `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` markers. The workflow is:

1. **Initial formula** (`src/initial.py`): Contains the starting `construct_formula` implementation
2. **Evolution runner** (`src/run_evo.py`): Configures and runs the AlphaEvolve system with:
   - Task description (search_task_sys_msg): Detailed mathematical background on avgQ complexity theory
   - Evolution configuration: LLM models, patch types, parent selection strategies
   - Database configuration: Multi-island evolution with archive and migration
3. **Evaluator** (`src/evaluate.py`): Evaluates candidate formulas by:
   - Running `run_experiment(n, w)` for various (n, w) test cases
   - Validating circuits respect width constraints and variable usage
   - Computing a three-component score: s1 (ratio to theoretical max), s2 (inverse gap penalty), s3 (exponential bonus for exceeding bounds)

### The Longshot Library

The library provides boolean circuit manipulation with avgQ calculation:

- **Circuit class** (`library/longshot/boolean/circuit.py`): Represents boolean functions as truth tables
  - Truth tables stored as PyTorch tensors (packed for n≤6, tensor arrays for n>6)
  - Supports operations: `&` (AND), `|` (OR), `^` (XOR), `~` (NOT), `-` (AND-NOT)
  - Properties: `.vars` (variable indices used), `.width` (number of variables), `.num_vars` (total variables)

- **Variable factory** (`VAR_factory`): Creates variable generators scoped to `n` variables
  ```python
  VAR = VAR_factory(n)  # Create factory for n variables
  x0, x1 = VAR(0), VAR(1)  # Generate variable circuits
  ```

- **Boolean operators**: `AND(circuits)`, `OR(circuits)`, `XOR(circuits)` - accept lists/iterators of circuits

- **avgQ calculation**: C++ backend (`library/longshot/_core`) computes optimal decision tree and returns average query complexity
  - `avgQ(circuit)` returns float
  - `avgQ_with_tree(circuit, build_tree=True)` returns (float, DecisionTree)

**Important limitation**: Maximum `n` is 26 variables (due to 31-bit depth field in C++ implementation). This is exposed via `MAX_VARS` constant.

## Common Development Commands

### Building the Longshot library
```bash
cd library
pip install -e .  # Editable install with C++ compilation
```

The library uses pybind11 to wrap C++ code. Build flags include `-Ofast -fopenmp` for performance.

### Running tests
```bash
cd library
pytest test/test_boolean.py           # Run all tests
pytest test/test_boolean.py::TestXOR  # Run specific test class
pytest -v                              # Verbose output
```

Tests are organized by boolean function type (DNF, CNF, XOR, MAJORITY) and verify avgQ calculations against known theoretical values.

### Running evaluation
```bash
cd src
python evaluate.py --max_num_vars 16 --program_path initial.py --results_dir results
```

Parameters:
- `--max_num_vars`: Maximum n value to test (default: 16)
- `--program_path`: Path to file containing `run_experiment` function
- `--results_dir`: Directory for results (metrics.json, correct.json)

The evaluator tests formulas on (n, w) pairs where n ranges from 3 to `max_num_vars` and w ranges from `n - floor(log n)` to `n` for each n.

### Running evolution
```bash
cd src
python run_evo.py
```

Configuration is embedded in `run_evo.py`. Key parameters:
- `strategy`: Parent selection strategy ("uniform", "hill_climbing", "weighted", "power_law", "beam_search")
- `num_generations`: Evolution iterations
- `max_parallel_jobs`: Parallel evaluation workers
- `llm_models`: List of LLM model names to use
- `patch_types`: Types of code mutations ("diff", "full", "cross")

Results are stored in SQLite database (`evolution_db.sqlite`) with multi-island architecture.

### Cleaning build artifacts
```bash
cd library
make clean  # Removes .so files, __pycache__, .egg-info, build/, .pytest_cache
```

## Key Constraints and Validation

When working with formula construction:

1. **Width constraint**: Each returned Circuit must have `circuit.width <= w`
2. **Variable usage**: All variables from 0 to n-1 should be used across all circuits (warnings issued for unused vars)
3. **Variable indices**: Circuit variables must be in range [0, n) with `max(all_involved_variables) < n`
4. **Determinism**: Constructions must be deterministic (no randomness) - same (n, w) always produces same formula

The evaluator validates these constraints and returns `(is_valid, error_message)`.

## Scoring System

Formulas are scored using a three-component function for each (n, w) test case:

```
Q(n,w) = n * (1 - log(n/w) / w)  # Theoretical upper bound
eps = 0.01
C = 1 / eps = 100

s1 = C * avgQ(f) / Q(n,w)                           # Ratio score
s2 = 1 / max(eps, Q(n,w) - avgQ(f) + eps)           # Proximity penalty
s3 = C * (exp(avgQ(f) - Q(n,w)) - 1) if avgQ(f) >= Q(n,w) else 0  # Bonus for exceeding bounds

score(f) = s1 + s2 + s3
combined_score = sum of scores across all test cases
```

The scoring heavily rewards formulas that approach or exceed the theoretical upper bound Q(n,w).

## Mathematical Background

The task description in `src/run_evo.py` contains extensive background on:
- avgQ definition and decision tree theory
- Known results: AND/OR (~2), XOR (n), MAJORITY (n - Θ(√n)), TRIBES formulas
- Theorem 11: Existence of DNF formulas with avgQ = n(1 - log(n)/Θ(w))
- Construction strategies using function composition

The goal is to discover formulas that match or exceed known bounds while respecting width constraints.

## File Structure Notes

- `library/longshot/__init__.py`: Main exports (Circuit, VAR_factory, avgQ, AND, OR, XOR, etc.)
- `library/longshot/_core`: C++ extension module (compiled from `library/longshot/core/core.cpp`)
- `library/longshot/boolean/decision_tree.py`: DecisionTree wrapper class for C++ decision trees
- `src/initial.py`: Template file - only edit the `construct_formula` function between EVOLVE-BLOCK markers
- Results directory (`results/`) is gitignored

## Known Issues and TODOs

See `library/README.md` for detailed TODO list including:
- Core module has 26-variable limit (31-bit depth field constraint)
- CountingBooleanFunction class is deprecated (will be removed)
- C++ tests need translation to Python
