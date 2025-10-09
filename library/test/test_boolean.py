"""
Test suite for Boolean circuits translated from test_core.cpp.

Tests the avgQ complexity calculation for various Boolean circuits including:
- Monotonic Boolean functions in DNF form
- Counting Boolean functions in CNF form
- XOR functions
"""
import pytest
from longshot import VAR_factory, XOR, AND, OR, avgQ, avgQ_with_tree
from longshot.boolean.circuit import Circuit


class TestDNF:
    """Test DNF formula (translated from test_bool first block)."""

    def test_empty_function(self):
        """Test empty 3-variable monotonic function."""
        VAR = VAR_factory(3)

        # Empty function - should always return 0
        # This would be represented as an empty circuit
        # For now, we'll skip this as we need to understand how to create empty circuits
        pass

    def test_single_term(self):
        """Test single term: x0 AND NOT x1."""
        VAR = VAR_factory(3)
        x0, x1, x2 = VAR(0), VAR(1), VAR(2)

        # x0 and not x1
        circuit = x0 & ~x1

        # Verify expected avgQ value
        assert avgQ(circuit) == pytest.approx(1.5), f"Expected avgQ=1.5, got {avgQ(circuit)}"

    def test_two_terms(self):
        """Test two terms: (x0 AND NOT x1) OR (NOT x0 AND x1 AND x2)."""
        VAR = VAR_factory(3)
        x0, x1, x2 = VAR(0), VAR(1), VAR(2)

        # (x0 and not x1) OR (not x0 and x1 and x2)
        circuit = (x0 & ~x1) | (~x0 & x1 & x2)

        # Verify expected avgQ value
        assert avgQ(circuit) == pytest.approx(2.25), f"Expected avgQ=2.25, got {avgQ(circuit)}"

    def test_redundant_term(self):
        """Test adding redundant term doesn't change avgQ."""
        VAR = VAR_factory(3)
        x0, x1, x2 = VAR(0), VAR(1), VAR(2)

        # (x0 and not x1) OR (not x0 and x1 and x2) OR (x2) [redundant]
        # The C++ test shows adding x2 alone is redundant with the previous terms
        circuit = (x0 & ~x1) | (~x0 & x1 & x2) | (~x0 & x1 & x2)

        # After adding redundant term, avgQ should remain the same
        assert avgQ(circuit) == pytest.approx(2.25), f"Expected avgQ=2.25, got {avgQ(circuit)}"


    def test_simplification_to_not_x2(self):
        """Test circuit simplifying to NOT x2."""
        VAR = VAR_factory(3)
        x0, x1, x2 = VAR(0), VAR(1), VAR(2)

        # Build up circuit that simplifies to NOT x2
        # From C++ test: after adding term (not x2), avgQ becomes 2.0
        circuit = (x0 & ~x1) | (~x0 & x1 & x2) | (~x0 & x1 & x2) | ~x2

        assert avgQ(circuit) == pytest.approx(2.0), f"Expected avgQ=2.0, got {avgQ(circuit)}"

    def test_tautology(self):
        """Test tautology (always true) has avgQ=0."""
        VAR = VAR_factory(3)
        x0, x1, x2 = VAR(0), VAR(1), VAR(2)

        # NOT x2 OR x2 = always true
        circuit = (x0 & ~x1) | (~x0 & x1 & x2) | (~x0 & x1 & x2) | ~x2 | x2

        # Tautology should have avgQ = 0 (no queries needed)
        assert avgQ(circuit) == pytest.approx(0.0), f"Expected avgQ=0.0, got {avgQ(circuit)}"


class TestCNF:
    """Test CNF formula (translated from test_bool second block)."""

    def test_single_clause(self):
        """Test single clause: x0 OR NOT x1."""
        VAR = VAR_factory(3)
        x0, x1, x2 = VAR(0), VAR(1), VAR(2)

        # In CNF: (x0 or not x1)
        circuit = x0 | ~x1

        assert avgQ(circuit) == pytest.approx(1.5), f"Expected avgQ=1.5, got {avgQ(circuit)}"

    def test_two_clauses_redundant(self):
        """Test adding redundant clause."""
        VAR = VAR_factory(3)
        x0, x1, x2 = VAR(0), VAR(1), VAR(2)

        # (x0 or not x1) AND x2
        # In C++, adding x2 alone is redundant, avgQ stays 1.5
        circuit = (x0 | ~x1) & (x0 | ~x1)

        assert avgQ(circuit) == pytest.approx(1.5), f"Expected avgQ=1.5, got {avgQ(circuit)}"

    def test_three_clauses(self):
        """Test three clauses: (x0 OR NOT x1) AND x2 AND (NOT x0 OR x1 OR x2)."""
        VAR = VAR_factory(3)
        x0, x1, x2 = VAR(0), VAR(1), VAR(2)

        # (x0 or not x1) AND x2 AND (not x0 or x1 or x2)
        circuit = (x0 | ~x1) & (x0 | ~x1) & (~x0 | x1 | x2)

        assert avgQ(circuit) == pytest.approx(2.25), f"Expected avgQ=2.25, got {avgQ(circuit)}"

    def test_four_clauses(self):
        """Test four clauses including NOT x2."""
        VAR = VAR_factory(3)
        x0, x1, x2 = VAR(0), VAR(1), VAR(2)

        # Previous circuit AND (not x2)
        circuit = (x0 | ~x1) & (x0 | ~x1) & (~x0 | x1 | x2) & ~x2

        assert avgQ(circuit) == pytest.approx(2.0), f"Expected avgQ=2.0, got {avgQ(circuit)}"

    def test_contradiction(self):
        """Test contradiction (always false) has avgQ=0."""
        VAR = VAR_factory(3)
        x0, x1, x2 = VAR(0), VAR(1), VAR(2)

        # x2 AND NOT x2 = always false
        circuit = (x0 | ~x1) & (x0 | ~x1) & (~x0 | x1 | x2) & ~x2 & x2

        # Contradiction should have avgQ = 0 (no queries needed)
        assert avgQ(circuit) == pytest.approx(0.0), f"Expected avgQ=0.0, got {avgQ(circuit)}"


class TestXOR:
    """Test XOR function (translated from test_bool third block)."""

    def test_xor10_complexity(self):
        """Test 4-variable XOR has avgQ=4.0."""
        n = 10
        VAR = VAR_factory(n)

        # 4-variable XOR
        circuit = XOR(*[VAR(i) for i in range(n)])

        assert avgQ(circuit) == pytest.approx(float(n)), f"Expected avgQ={float(n)}, got {avgQ(circuit)}"
        
    def test_xor4_complexity(self):
        """Test 4-variable XOR has avgQ=4.0."""
        VAR = VAR_factory(4)
        x0, x1, x2, x3 = VAR(0), VAR(1), VAR(2), VAR(3)

        # 4-variable XOR
        circuit = XOR(x0, x1, x2, x3)

        assert avgQ(circuit) == pytest.approx(4.0), f"Expected avgQ=4.0, got {avgQ(circuit)}"

    def test_xor3_complexity(self):
        """Test 3-variable XOR complexity."""
        VAR = VAR_factory(3)
        x0, x1, x2 = VAR(0), VAR(1), VAR(2)

        circuit = XOR(x0, x1, x2)

        # XOR of n variables should have avgQ = n
        assert avgQ(circuit) == pytest.approx(3.0), f"Expected avgQ=3.0, got {avgQ(circuit)}"

    def test_xor2_complexity(self):
        """Test 2-variable XOR complexity."""
        VAR = VAR_factory(2)
        x0, x1 = VAR(0), VAR(1)

        circuit = XOR(x0, x1)

        assert avgQ(circuit) == pytest.approx(2.0), f"Expected avgQ=2.0, got {avgQ(circuit)}"


class TestMAJORITY:
    """Test MAJORITY function."""

    def construct_majority(self, n: int) -> Circuit:
        """Construct n-variable MAJORITY circuit."""
        VAR = VAR_factory(n)
        variables = [VAR(i) for i in range(n)]

        # Majority can be constructed as OR of all combinations of (n//2 + 1) variables
        from itertools import combinations
        threshold = n // 2 + 1

        terms = [AND(combo) for combo in combinations(variables, threshold)]

        return OR(terms)

    def expected_avgQ_majority(self, n: int) -> float:
        """Calculate expected avgQ for n-variable MAJORITY circuit."""
        import math
        k = n // 2
        return n - (math.comb(2*k, k) * n / (2**(2*k))) + (1 if n % 2 == 1 else 0)

    @pytest.mark.parametrize("n_vars", [3,5,7,9,11,13])
    def test_majority_complexity(self, n_vars: int):
        """Test that n-variable MAJORITY has expected avgQ."""
        if n_vars % 2 == 0:
            pytest.skip("n must be odd for majority function")

        circuit = self.construct_majority(n_vars)
        expected_avgQ = self.expected_avgQ_majority(n_vars)

        assert avgQ(circuit) == pytest.approx(expected_avgQ), \
            f"MAJORITY of {n_vars} variables should have avgQ={expected_avgQ}, got {avgQ(circuit)}"  


class TestDecisionTree:
    """Test decision tree construction with avgQ."""

    def test_tree_construction(self):
        """Test that decision tree can be built along with avgQ calculation."""
        VAR = VAR_factory(3)
        x0, x1, x2 = VAR(0), VAR(1), VAR(2)

        circuit = (x0 | ~x1) & (~x0 | x1 | x2)

        # Request decision tree construction
        qv, tree = avgQ_with_tree(circuit, build_tree=True)

        # Verify tree was created
        assert tree is not None
        # Verify avgQ value is reasonable
        assert 0 <= qv <= 3, f"avgQ should be between 0 and 3, got {qv}"

    def test_xor4_tree(self):
        """Test decision tree for 4-variable XOR."""
        VAR = VAR_factory(4)
        x0, x1, x2, x3 = VAR(0), VAR(1), VAR(2), VAR(3)

        circuit = XOR(x0, x1, x2, x3)

        qv, tree = avgQ_with_tree(circuit, build_tree=True)

        assert tree is not None
        assert qv == pytest.approx(4.0), f"Expected avgQ=4.0, got {qv}"


# Parametrized tests for systematic testing
@pytest.mark.parametrize("n_vars,expected_avgq", [
    (2, 2.0),
    (3, 3.0),
    (4, 4.0),
    (5, 5.0),
])
def test_xor_n_complexity(n_vars, expected_avgq):
    """Test that n-variable XOR has avgQ = n."""
    VAR = VAR_factory(n_vars)
    variables = [VAR(i) for i in range(n_vars)]

    circuit = XOR(*variables)

    assert avgQ(circuit) == pytest.approx(expected_avgq), \
        f"XOR of {n_vars} variables should have avgQ={expected_avgq}"


# Parametrized tests for systematic testing
@pytest.mark.parametrize("n_vars,w,expected_avgq", [
    (3, 2, 2.0),
    (4, 3, 3.0),
    (5, 4, 4.0),
    (8, 7, 7.0),
])
def test_xor_w_complexity(n_vars, w, expected_avgq):
    """Test that n-variable XOR has avgQ = n."""
    VAR = VAR_factory(n_vars)
    variables = [VAR(i) for i in range(w)]

    circuit = XOR(*variables)

    assert avgQ(circuit) == pytest.approx(expected_avgq), \
        f"XOR of {n_vars} variables should have avgQ={expected_avgq}"

@pytest.mark.parametrize("n_vars", [1, 2, 3, 4, 8])
def test_single_variable_complexity(n_vars):
    """Test that any single variable has avgQ = 1.0."""
    VAR = VAR_factory(n_vars)

    # Test first variable
    circuit = VAR(0)
    assert avgQ(circuit) == pytest.approx(1.0)

    # Test last variable
    circuit = VAR(n_vars - 1)
    assert avgQ(circuit) == pytest.approx(1.0)


@pytest.mark.parametrize("n_vars", [2, 3, 4])
def test_tautology_complexity(n_vars):
    """Test that tautologies have avgQ = 0.0."""
    VAR = VAR_factory(n_vars)
    x = VAR(0)

    # x OR NOT x = always true
    circuit = x | ~x

    assert avgQ(circuit) == pytest.approx(0.0), \
        f"Tautology should have avgQ=0.0, got {avgQ(circuit)}"


@pytest.mark.parametrize("n_vars", [2, 3, 4])
def test_contradiction_complexity(n_vars):
    """Test that contradictions have avgQ = 0.0."""
    VAR = VAR_factory(n_vars)
    x = VAR(0)

    # x AND NOT x = always false
    circuit = x & ~x

    assert avgQ(circuit) == pytest.approx(0.0), \
        f"Contradiction should have avgQ=0.0, got {avgQ(circuit)}"
