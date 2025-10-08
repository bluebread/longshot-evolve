# Longshot Library

# TODO List

## Critical Bugs

- [O] ~~**Core module input size limitation**: The avgQ calculation uses integer representation where avgQ = depth / 2^n. The `depth` field in [bool.hpp:52](longshot/core/bool.hpp#L52) is 31 bits (`uint32_t depth : 31`). Since max avgQ ≤ n, we have depth ≤ n × 2^n ≤ 2^31, which limits n to 26 (26 × 2^26 ≈ 1.7B < 2^31, but 27 × 2^27 ≈ 3.6B > 2^31). If upgraded to 64-bit depth, n could reach ~58. Need to:~~
  - [O] ~~Hard-code maximum number of input variables (MAX_NUM_VARS = 26 for current 31-bit implementation)~~
  - [O] ~~Add validation in BaseBooleanFunction constructor to reject num_vars > MAX_NUM_VARS~~
  - [O] ~~Expose MAX_NUM_VARS constant to Python interface~~
  - [O] ~~Add clear error messages explaining the limitation~~

## New Features

- [ ] **PyTorch-based truth table computation**: Implement truth table computation using PyTorch eager execution for LLM-generated boolean programs. Need to:
  - [O] ~~Design operator/method API for boolean operations that LLM can use (AND, OR, NOT, XOR, etc.)~~
  - [O] ~~Expose `avgQ()` method to calculate complexity~~
  - [O] ~~Implement PyTorch translation layer for each operator~~
  - [O] ~~Support automatic truth table generation from operator composition~~
  - [ ] Create examples showing how LLM writes programs using provided operators
  - [ ] Add tests for correctness and performance benchmarks
  
- [ ] **Test extreme cases**: Test extreme cases e.g. n = 26, to verify the performance and correctness.

## Refactoring/Cleanup

- [ ] **Remove CountingBooleanFunction**: CountingBooleanFunction class ([bool.hpp:279-325](longshot/core/bool.hpp#L279-L325)) won't be used in future. Need to:
  - [ ] Remove CountingBooleanFunction class definition from [bool.hpp](longshot/core/bool.hpp)
  - [ ] Remove CountingTruthTable from [truthtable.hpp](longshot/core/truthtable.hpp) if no longer needed
  - [ ] Update tests in [test_core.cpp](test/test_core.cpp) to remove CountingBooleanFunction tests
  - [ ] Keep MonotonicBooleanFunction only

- [O] ~~**Remove error module**: The custom error module ([longshot/error/](longshot/error/)) is unnecessary.~~

- [ ] **Remove unused methods/functions**: Remove all unused methods/functions from the library for clarity.

- [ ] **Translate C++ test code to Python**: Translate C++ test code to Python, since it relies on deprecated methods that will be removed in the future.

