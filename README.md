# LongshotEvolve

**Automated discovery of boolean formulas with high average-case query complexity using LLM-guided evolution.**

LongshotEvolve combines the [Longshot library](library/) for boolean circuit analysis with the [AlphaEvolve](https://github.com/shinkle-lanl/shinka) framework to automatically evolve DNF (Disjunctive Normal Form) formulas that approach theoretical bounds for average-case deterministic query complexity (avgQ).

## Table of Contents

- [Overview](#overview)
- [What is avgQ?](#what-is-avgq)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Evolution System](#evolution-system)
- [Examples](#examples)
- [Development](#development)
- [Documentation](#documentation)
- [License](#license)

## Overview

Given a boolean function `f: {0,1}^n → {0,1}`, the **average-case deterministic query complexity** (avgQ) measures how many input bits a decision tree must examine on average to compute `f(x)` for a uniformly random input `x`.

**The Challenge**: Can we automatically discover DNF formulas with width constraint `w` that achieve avgQ close to the theoretical upper bound of `n(1 - log(n/w)/Θ(w))`?

**Our Approach**: Use LLM-guided evolutionary search to iteratively improve formula constructions, leveraging:
- The Longshot library's efficient avgQ computation (C++ backend)
- AlphaEvolve's multi-island evolution with cross-pollination
- Mathematical insights about function composition and query complexity

## What is avgQ?

The average-case deterministic query complexity is defined as:

```
avgQ(f) = min_T E_{x ~ {0,1}^n} [cost(T, x)]
```

where `T` ranges over all deterministic decision trees that compute `f` with zero error, and `cost(T, x)` is the number of bits queried on input `x`.

### Known Results

| Function | avgQ | Notes |
|----------|------|-------|
| AND_n, OR_n | ~2 | Constant complexity |
| XOR_n | n | Maximum possible |
| MAJ_n | n - Θ(√n) | Majority function |
| TRIBES_{w,s} | Θ(s) or Θ(2^w) | Depends on parameters |
| CNF/DNF of width w | ≤ n(1 - log(n/w)/O(w)) | Theoretical upper bound |

**Goal**: Discover formulas that match or exceed known constructions.

## Installation

### Prerequisites

- Python 3.8+
- C++ compiler with C++17 support
- OpenMP support (for parallel computation)
- Git

### Install Longshot Library

```bash
cd library
pip install -e .
```

This compiles the C++ backend and installs the Python package in editable mode.

### Install Evolution Dependencies

```bash
pip install shinka  # AlphaEvolve framework
# Additional dependencies for evolution:
# - litellm (for LLM API access)
# - torch (for embeddings)
# - sqlite3 (standard library)
```

### Verify Installation

```bash
cd library
pytest test/test_boolean.py
```

## Quick Start

### Run an Evaluation

Test a formula construction on various (n, w) test cases:

```bash
cd src
python evaluate.py --max_num_vars 16 --program_path initial.py --results_dir results
```

This evaluates the formula in `initial.py` and outputs:
- `results/metrics.json`: Scores for each test case
- `results/correct.json`: Validation results

### Run Evolution

Start the evolutionary search:

```bash
cd src
python run_evo.py
```

The evolution system will:
1. Start with the initial formula in `initial.py`
2. Use LLMs to generate mutations (diff patches, full rewrites, crossovers)
3. Evaluate each candidate formula on test cases
4. Select high-scoring formulas as parents for the next generation
5. Store results in `evolution_db.sqlite`

Progress is logged to console. Evolution runs for the configured number of generations (default: 30).

## Architecture

### Directory Structure

```
longshot-evolve/
├── library/              # Longshot boolean circuit library
│   ├── longshot/
│   │   ├── boolean/     # Python Circuit API
│   │   ├── core/        # C++ avgQ computation
│   │   └── _core.so     # Compiled extension (after build)
│   └── test/            # Library tests
├── src/                 # Evolution system
│   ├── initial.py       # Starting formula (EVOLVED)
│   ├── evaluate.py      # Formula evaluation
│   └── run_evo.py       # Evolution configuration
├── CLAUDE.md           # Documentation for AI assistants
└── README.md           # This file
```

### Key Components

**1. Longshot Library** (`library/`)
- **Circuit class**: Represents boolean functions as truth tables
- **avgQ computation**: C++ implementation using dynamic programming
- **Operations**: AND, OR, XOR, NOT, composition
- **Constraints**: Maximum 26 variables (due to 31-bit depth field)

**2. Evolution System** (`src/`)
- **initial.py**: Contains the evolving `construct_formula(n, w)` function
- **evaluate.py**: Scores formulas using a three-component metric
- **run_evo.py**: Configures LLMs, evolution strategy, database settings

**3. Scoring System**

For each test case (n, w), the score is:

```python
Q = n * (1 - log(n/w) / w)  # Theoretical upper bound
eps = 0.01
C = 1 / eps = 100

s1 = C * avgQ(f) / Q                              # Ratio score
s2 = 1 / max(eps, Q - avgQ(f) + eps)              # Proximity penalty
s3 = C * (exp(avgQ(f) - Q) - 1) if avgQ(f) >= Q else 0  # Bonus

total_score = s1 + s2 + s3
```

Higher avgQ yields higher scores. The bonus term heavily rewards exceeding bounds.

## Evolution System

### How It Works

1. **Initialization**: Start with a seed formula in `initial.py`

2. **Mutation**: LLMs generate three types of patches:
   - **Diff patches** (60%): Incremental modifications
   - **Full rewrites** (30%): Complete reimplementations
   - **Crossovers** (10%): Combine features from two parents

3. **Evaluation**: Each candidate is tested on multiple (n, w) pairs:
   - n ranges from 3 to `max_num_vars`
   - w ranges from `n - floor(log n)` to `n` for each n
   - Validation checks width constraints and variable usage
   - avgQ is computed for valid formulas

4. **Selection**: Parent selection strategies:
   - **Uniform**: Random selection
   - **Hill climbing**: Always select best
   - **Weighted**: Probability proportional to score
   - **Power law**: Emphasize top performers
   - **Beam search**: Keep top-k candidates

5. **Multi-Island Evolution**: Multiple populations evolve independently with periodic migration of top formulas

### Configuration

Key parameters in `run_evo.py`:

```python
evo_config = EvolutionConfig(
    num_generations=30,           # Evolution iterations
    max_parallel_jobs=16,         # Parallel evaluations
    llm_models=["o4-mini", ...],  # LLM models for mutations
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
)

db_config = DatabaseConfig(
    num_islands=2,                # Independent populations
    archive_size=5,               # Top formulas per island
    elite_selection_ratio=0.6,    # Elite parent selection
)
```

## Examples

### Example 1: Simple DNF Formula

```python
from longshot import VAR_factory, AND, OR

def construct_formula(n, w):
    """Create a DNF with all w-sized AND terms."""
    from itertools import combinations

    VAR = VAR_factory(n)
    VAR = [VAR(i) for i in range(n)]

    # Generate all combinations of w variables
    terms = [AND(list(combo)) for combo in combinations(VAR, w)]

    return terms  # Will be combined with OR
```

### Example 2: Using the Library

```python
from longshot import VAR_factory, AND, OR, XOR, avgQ

# Create variables for n=5
VAR = VAR_factory(5)
x0, x1, x2, x3, x4 = VAR(0), VAR(1), VAR(2), VAR(3), VAR(4)

# Build a circuit
majority = (x0 & x1) | (x0 & x2) | (x1 & x2)

# Compute average query complexity
complexity = avgQ(majority)
print(f"avgQ = {complexity}")

# Check circuit properties
print(f"Width: {majority.width}")        # Number of variables used
print(f"Variables: {majority.vars}")     # Which variables used
```

### Example 3: Decision Tree Analysis

```python
from longshot import avgQ_with_tree

circuit = x0 & x1  # AND of two variables
complexity, tree = avgQ_with_tree(circuit, build_tree=True)

print(f"avgQ = {complexity}")
# Can analyze the decision tree structure
```

## Development

### Running Tests

```bash
cd library
pytest test/test_boolean.py -v          # All tests
pytest test/test_boolean.py::TestXOR    # Specific test class
```

### Building the Library

```bash
cd library
pip install -e .                         # Development install
make clean                               # Remove build artifacts
```

### Contributing

The project uses:
- **Code style**: Follow PEP 8 for Python
- **C++ standard**: C++17
- **Testing**: pytest for Python, add tests for new features
- **Documentation**: Update CLAUDE.md for AI-relevant changes

### Known Limitations

- **Maximum n = 26 variables**: Due to 31-bit depth field in C++ implementation
- **Memory usage**: Truth tables use packed representation for n ≤ 6, tensor arrays for n > 6
- **Evaluation time**: avgQ computation is exponential in n (O(n × 2^n))

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive guide for AI coding assistants
- **[library/README.md](library/README.md)**: Longshot library TODO list and development notes
- **[src/run_evo.py](src/run_evo.py)**: Task description and mathematical background (in `search_task_sys_msg`)

### Further Reading

- **avgQ Theory**: See the task description in `run_evo.py` for definitions and known results
- **AlphaEvolve Framework**: https://github.com/shinkle-lanl/shinka
- **Boolean Function Complexity**: Standard texts on computational complexity theory

## Research Context

This project explores the intersection of:
- **Computational complexity theory**: Understanding fundamental limits of computation
- **Automated discovery**: Using LLMs to explore mathematical constructions
- **Evolutionary computation**: Iterative improvement through selection and variation

The goal is to discover new boolean formula constructions that achieve high query complexity, potentially leading to:
- Better understanding of decision tree complexity
- New techniques for function composition
- Insights into the power of evolutionary search for mathematical discovery

## License

[Specify your license here]

## Citation

If you use LongshotEvolve in your research, please cite:

```
[Add citation information]
```

## Contact

[Add contact information or links to issue tracker]

---

**Status**: Active development. The evolution system is functional and discovering interesting formulas. Contributions welcome!
