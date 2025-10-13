"""
Evaluator for circle packing example (n=26) with improved timeout handling
"""

import os
import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from math import floor, log2, exp

from shinka.core import run_shinka_eval
from longshot import Circuit, avgQ


def adapted_validate_formula(
    run_output: Tuple[int, int, List[Circuit]],
) -> Tuple[bool, Optional[str]]:
    """
    Validates circle packing results based on the output of 'run_packing'.

    Args:
        run_output: Tuple (centers, radii, reported_sum) from run_packing.

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    n, w, circuits = run_output
    max_width = max([circ.width for circ in circuits])
    max_num_vars = max([circ.num_vars for circ in circuits])
    all_involved_variables = set()
    
    for circ in circuits:
        all_involved_variables = all_involved_variables.union(circ.vars)
    
    if max_width > w:
        msg = (
            f"Formula validation failed: Maximum circuit width ({max_width}) exceeds allowed width (w={w})."
        )
        return False, msg
    
    if max_num_vars > n or max(all_involved_variables) > n:
        msg = (
            f"Formula validation failed: Variable constraint violated. "
            f"Max num_vars={max_num_vars}, max variable index={max(all_involved_variables)}, allowed n={n}."
        )
        return False, msg

    if len(all_involved_variables) != n:
        unused_vars = set(range(n)).difference(all_involved_variables)
        msg = (
            f" Warning: {[f'x{i}' for i in unused_vars]} variables are not used."
        )
    
    return True, "The formula is constructed correctly." + msg

def aggregate_formula_metrics(results: List[Tuple[int, int, List[Circuit]]]) -> Dict[str, Any]:
    """
    Aggregates metrics for formulas.
    """
    def theo_max_avgQ(n: int, w: int) -> float:
        return n * (1 - log2(n/w) / w)
    
    def score(n, w, q: float) -> float:
        maxq = theo_max_avgQ(n, w)
        eps = 0.01
        C = 100
        bonus = C * exp(q - maxq) if q >= maxq else 0.0
        
        return 1 / (max(0, maxq - q) + eps) + bonus
    
    scores = [(n, w, score(n, w, avgQ(circ))) for n, w, circ in results]
    
    metrics = {
        "combined_score": sum([s for _, _, s in scores]),
        "public": {f"n{n}w{w}": s for n, w, s in scores},
        "private": {}
    }
    
    return metrics

def main(max_n: int, program_path: str, results_dir: str):
    """Runs the circle packing evaluation using shinka.eval."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    experiment_kwargs: list[dict[str, int]] = []
    
    for n in range(3, max_n + 1):
        for w in range(n - floor(log2(n)), n + 1):
            experiment_kwargs.append({'n': n, 'w': w})

    num_experiment_runs = len(experiment_kwargs)
    
    # Define a function to obtain parameters for every experiment
    def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
        """Provides keyword arguments for circle packing runs (none needed)."""
        return experiment_kwargs[run_index]

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=num_experiment_runs,
        get_experiment_kwargs=get_experiment_kwargs,
        validate_fn=adapted_validate_formula,
        aggregate_metrics_fn=aggregate_formula_metrics,
    )

    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <string_too_long_to_display>")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Circle packing evaluator using shinka.eval"
    )
    parser.add_argument(
        "--max-num-vars",
        type=int,
        default=20,
        help="The max number of variables"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_experiment')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json)",
    )
    parsed_args = parser.parse_args()
    
    main(
        parsed_args.max_num_vars,
        parsed_args.program_path, 
        parsed_args.results_dir
    )
