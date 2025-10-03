LongshotEvolve Design Blueprint
=======================================

Welcome to the LongshotEvolve documentation! This document outlines the design and architecture of the LongshotEvolve system, detailing its components, functionalities, and interactions.

Overview
--------

LongshotEvolve is a framework for automated boolean formula generation and optimization using evolutionary algorithms. Built on top of the **OpenEvolve** project, LongshotEvolve provides a specialized toolkit for code agents to construct and optimize computational graphs of boolean formulas, with the primary objective of maximizing average-case query complexity (avgQ).

The system leverages the MAP-Elites algorithm to explore the space of boolean functions, using code agents to dynamically generate circuit designs that balance complexity with computational constraints.

Core Components
---------------

1. Boolean Formula Construction Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LongshotEvolve provides a comprehensive set of tools for dynamically constructing computational graphs of boolean formulas:

**Variables**
  The basic unit representing a boolean variable in the computational graph.

**Operators**
  - ``__xor__()``: XOR operation between two boolean expressions
  - ``__and__()``: AND operation between two boolean expressions
  - ``__or__()``: OR operation between two boolean expressions
  - ``__not__()``: NOT operation (negation) of a boolean expression
  - ``__minus__()``: Set difference operation, where ``A - B`` is equivalent to ``A and not B``

**Built-in N-ary Functions**
  - ``XOR()``: XOR function
  - ``OR()``: OR function
  - ``AND()``: AND function
  - ``MAJORITY()``: Majority voting function

2. Evaluator System
~~~~~~~~~~~~~~~~~~~~

The evaluator component calculates critical metrics for boolean formula optimization:

**Average-Case Query Complexity (avgQ)**
  The primary optimization target. This metric measures the expected number of queries required to evaluate the boolean function in average cases.

**Feature Dimensions**
  Additional metrics used by the MAP-Elites algorithm to characterize and compare different boolean functions, enabling diverse exploration of the solution space.

3. Code Agent
~~~~~~~~~~~~~

The code agent is responsible for generating boolean functions that maximize avgQ. Each generated function is a Python method with the following signature:

.. code-block:: python

    def generated_formula(num_var, width):
        """
        Generate a boolean formula using LongshotEvolve tools.

        Parameters
        ----------
        num_var : int
            Number of variables in the boolean formula
        width : int
            Maximum width of the circuit (constraint on circuit breadth)

        Returns
        -------
        Boolean formula computational graph
        """
        # Use variables, operators, and built-in functions
        # to construct optimized boolean formula
        pass

The agent must balance maximizing avgQ while respecting the width constraint, which limits the maximum breadth of the computational circuit.

Key Concepts
------------

Computational Graph
~~~~~~~~~~~~~~~~~~~

Boolean formulas are represented as computational graphs where:
- Nodes represent variables, operations, or function calls
- Edges represent data dependencies between operations
- The graph structure determines the evaluation complexity

Set Difference Operation
~~~~~~~~~~~~~~~~~~~~~~~~~

The minus operator (``-``) provides a convenient way to express set difference in boolean logic:

.. code-block:: python

    A - B  # Equivalent to: A and not B
    A.__minus__(B)  # Explicit method call

This operation is particularly useful for constructing complex boolean predicates and constraints.

Optimization Objective
~~~~~~~~~~~~~~~~~~~~~~

The primary goal is to maximize **average-case query complexity (avgQ)** subject to constraints:

- **Maximize**: avgQ (higher complexity indicates more interesting/challenging functions)
- **Constraints**:
  - Number of variables: ``num_var``
  - Maximum circuit width: ``width``

The MAP-Elites algorithm explores the solution space by maintaining a diverse archive of solutions characterized by their feature dimensions.
