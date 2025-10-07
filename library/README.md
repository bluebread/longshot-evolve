# Longshot Library

# TODO List

## Critical Bugs

- [O] ~~**Core module input size limitation**: The avgQ calculation uses integer representation where avgQ = depth / 2^n. The `depth` field in [bool.hpp:52](longshot/core/bool.hpp#L52) is 31 bits (`uint32_t depth : 31`). Since max avgQ ≤ n, we have depth ≤ n × 2^n ≤ 2^31, which limits n to 26 (26 × 2^26 ≈ 1.7B < 2^31, but 27 × 2^27 ≈ 3.6B > 2^31). If upgraded to 64-bit depth, n could reach ~58. Need to:~~
  - [O] ~~Hard-code maximum number of input variables (MAX_NUM_VARS = 26 for current 31-bit implementation)~~
  - [O] ~~Add validation in BaseBooleanFunction constructor to reject num_vars > MAX_NUM_VARS~~
  - [O] ~~Expose MAX_NUM_VARS constant to Python interface~~
  - [O] ~~Add clear error messages explaining the limitation~~

## New Features

- [ ] **Add BooleanFunction Python wrapper class**: Create a Python wrapper for C++ MonotonicBooleanFunction with user-friendly interface. Need to:
  - [ ] Create `BooleanFunction` class in Python (in [longshot/boolean/](longshot/boolean/) or new module)
  - [ ] Support initialization from truth table in multiple formats:
    - [ ] Integer representation (e.g., `0b1010` for 4-bit truth table)
    - [ ] List of 0/1 bits (e.g., `[1, 0, 1, 0]`)
  - [ ] Expose `avgQ()` method to calculate complexity
  - [ ] Wrap C++ `_MonotonicBooleanFunction` from [core.cpp](longshot/core/core.cpp)
  - [ ] Add proper error handling and validation
  - [ ] Add documentation and usage examples

- [ ] **PyTorch-based truth table computation**: Implement truth table computation using PyTorch eager execution for LLM-generated boolean programs. Need to:
  - [ ] Design operator/method API for boolean operations that LLM can use (AND, OR, NOT, XOR, etc.)
  - [ ] Implement PyTorch translation layer for each operator
  - [ ] Support automatic truth table generation from operator composition
  - [ ] Enable CPU/GPU backend selection for performance optimization
  - [ ] Create examples showing how LLM writes programs using provided operators
  - [ ] Integrate with BooleanFunction class to pass generated truth tables to C++ avgQ calculator
  - [ ] Add tests for correctness and performance benchmarks

## Refactoring/Cleanup

- [ ] **Remove CountingBooleanFunction**: CountingBooleanFunction class ([bool.hpp:279-325](longshot/core/bool.hpp#L279-L325)) won't be used in future. Need to:
  - [ ] Remove CountingBooleanFunction class definition from [bool.hpp](longshot/core/bool.hpp)
  - [ ] Remove CountingBooleanFunction Python bindings from [core.cpp:77-85](longshot/core/core.cpp#L77-L85)
  - [ ] Remove CountingTruthTable from [truthtable.hpp](longshot/core/truthtable.hpp) if no longer needed
  - [ ] Update tests in [test_core.cpp](test/test_core.cpp) to remove CountingBooleanFunction tests
  - [ ] Keep MonotonicBooleanFunction only

- [ ] **Remove error module**: The custom error module ([longshot/error/](longshot/error/)) is unnecessary. Need to:
  - [ ] Remove [longshot/error/error.py](longshot/error/error.py)
  - [ ] Remove [longshot/error/__init__.py](longshot/error/__init__.py)
  - [ ] Replace custom error classes with Python built-in exceptions (ValueError, TypeError, RuntimeError, etc.)
  - [ ] Update any code that imports from longshot.error to use built-in exceptions
  - [ ] Ensure clearer and more Pythonic error handling throughout the codebase
