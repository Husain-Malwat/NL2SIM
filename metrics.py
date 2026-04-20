"""
metrics.py
==========
All evaluation metrics for the NL2Sim system.

We measure quality at 5 levels:
  1. API Validity Rate      - did the model use real MuMax3 symbols?
  2. Completeness Score     - are all required symbols present?
  3. IR F1 Score            - how well does the IR match the reference?
  4. Execution Pass Rate    - does the script actually run? (requires MuMax3)
  5. Physics Score          - are the results physically sensible?

Usage:
    from evaluation.metrics import compute_all_metrics

    metrics = compute_all_metrics(
        generated_script="...",
        reference_script="...",
        generated_ir={...},
        reference_ir={...},
        category="field_driven",
    )
    print(metrics["api_validity_rate"])   # e.g. 0.95
    print(metrics["completeness_score"])  # e.g. 0.87
    print(metrics["ir_f1"])               # e.g. 0.82
"""

import re
import json
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ontology.mumax3_api import ALL_VALID_SYMBOLS, CATEGORY_REQUIRED, is_valid_symbol
from validation.static_validator import (
    tokenize_script, extract_assignments, extract_function_calls, validate_script
)


# ─── Metric 1: API Validity Rate ─────────────────────────────────────────────

def compute_api_validity(generated_script, category=None):
    """
    What fraction of API-like tokens in the generated script are real MuMax3 symbols?
    
    Returns:
        - api_validity_rate: float 0-1 (1.0 = no hallucinations)
        - hallucinated: list of bad symbols
        - n_checked: how many symbols were checked
    """
    validation = validate_script(generated_script, expected_category=category)

    n_checked     = validation["n_symbols_checked"]
    n_hallucinated = len(validation["hallucinations"])
    n_valid        = n_checked - n_hallucinated

    rate = n_valid / max(1, n_checked)

    return {
        "api_validity_rate": rate,
        "hallucinated":      validation["hallucinations"],
        "n_symbols_checked": n_checked,
        "n_hallucinated":    n_hallucinated,
    }


# ─── Metric 2: Completeness Score ─────────────────────────────────────────────

def compute_completeness(generated_script, category):
    """
    What fraction of required symbols for this category are present?
    
    Returns:
        - completeness_score: float 0-1
        - missing: list of missing required symbols
        - n_required: total required symbols for this category
    """
    if category not in CATEGORY_REQUIRED:
        return {"completeness_score": 1.0, "missing": [], "n_required": 0}

    required = CATEGORY_REQUIRED[category]

    # Find all symbols in the generated script
    validation    = validate_script(generated_script)
    symbols_found = validation["symbols_found"]
    assignments   = extract_assignments(generated_script)
    fn_calls      = extract_function_calls(generated_script)

    all_found = symbols_found | set(assignments.keys()) | set(fn_calls.keys())

    missing = [sym for sym in required if sym not in all_found]
    n_present = len(required) - len(missing)

    return {
        "completeness_score": n_present / len(required),
        "missing":            missing,
        "n_required":         len(required),
        "n_present":          n_present,
    }


# ─── Metric 3: IR F1 Score ────────────────────────────────────────────────────

def flatten_ir(ir, prefix=""):
    """
    Flatten a nested IR dict into (key, value) pairs for comparison.
    
    Example:
        {"material": {"Msat_Am": 800000}} 
        → {"material.Msat_Am": 800000}
    
    We only compare leaf values (not nested dicts).
    """
    flat = {}
    for key, val in ir.items():
        if key.startswith("_"):   # skip metadata fields
            continue
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(val, dict):
            flat.update(flatten_ir(val, prefix=full_key + "."))
        elif isinstance(val, list):
            # For lists, just record whether they're empty or not
            flat[full_key + ".count"] = len(val)
        elif val is not None:
            flat[full_key] = val
    return flat


def are_values_close(v1, v2, rel_tolerance=0.15):
    """
    Check if two values are "close enough".
    For numbers: within 15% relative difference.
    For strings: exact match (case insensitive).
    For booleans: exact match.
    """
    if isinstance(v1, bool) and isinstance(v2, bool):
        return v1 == v2
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        if v1 == 0 and v2 == 0:
            return True
        if v1 == 0 or v2 == 0:
            return False
        return abs(v1 - v2) / max(abs(v1), abs(v2)) <= rel_tolerance
    if isinstance(v1, str) and isinstance(v2, str):
        return v1.lower() == v2.lower()
    return v1 == v2


def compute_ir_f1(generated_ir, reference_ir):
    """
    Compute precision, recall, and F1 by comparing flattened IR fields.
    
    A field is "correct" if:
    - It's present in both IRs, AND
    - The values are close (within 15% for numbers, exact for strings/booleans)
    
    Returns:
        - ir_precision: fraction of generated fields that are correct
        - ir_recall: fraction of reference fields that are covered
        - ir_f1: harmonic mean of precision and recall
    """
    if generated_ir is None:
        return {"ir_precision": 0.0, "ir_recall": 0.0, "ir_f1": 0.0}

    gen_flat = flatten_ir(generated_ir)
    ref_flat = flatten_ir(reference_ir)

    # True positives: fields in both that match
    tp = 0
    for key, ref_val in ref_flat.items():
        gen_val = gen_flat.get(key)
        if gen_val is not None and are_values_close(gen_val, ref_val):
            tp += 1

    # False positives: fields in generated but wrong or absent in reference
    fp = len(gen_flat) - tp

    # False negatives: fields in reference but missing in generated
    fn = len(ref_flat) - tp

    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1        = (2 * precision * recall) / max(1e-9, precision + recall)

    return {
        "ir_precision":    precision,
        "ir_recall":       recall,
        "ir_f1":           f1,
        "ir_tp":           tp,
        "ir_fp":           fp,
        "ir_fn":           fn,
        "n_gen_fields":    len(gen_flat),
        "n_ref_fields":    len(ref_flat),
    }


# ─── Metric 4: Parameter Accuracy ────────────────────────────────────────────

def compute_parameter_accuracy(generated_ir, reference_ir):
    """
    How accurate are the specific physics parameters?
    
    Computes mean absolute percentage error (MAPE) for key numeric parameters.
    
    Returns:
        - parameter_accuracy: dict of {param_name: match_status}
        - mean_relative_error: average relative error across all params
        - exact_match_rate: fraction of params that match exactly (within 5%)
    """
    # The key physical parameters we care about
    key_params = [
        "material.Msat_Am",
        "material.Aex_Jm",
        "material.alpha",
        "material.Ku1_Jm3",
        "material.DMI_Jm2",
        "material.Temp_K",
    ]

    gen_flat = flatten_ir(generated_ir)
    ref_flat = flatten_ir(reference_ir)

    errors     = []
    results    = {}
    n_exact    = 0
    n_checked  = 0

    for param in key_params:
        ref_val = ref_flat.get(param)
        gen_val = gen_flat.get(param)

        if ref_val is None:
            continue  # not in reference, skip

        n_checked += 1

        if gen_val is None:
            results[param] = {"status": "missing", "ref": ref_val, "gen": None, "rel_error": 1.0}
            errors.append(1.0)
            continue

        # Compute relative error
        if ref_val == 0:
            rel_err = 0.0 if gen_val == 0 else 1.0
        else:
            rel_err = abs(gen_val - ref_val) / abs(ref_val)

        status = "exact" if rel_err < 0.05 else ("close" if rel_err < 0.15 else "wrong")
        results[param] = {
            "status":    status,
            "ref":       ref_val,
            "gen":       gen_val,
            "rel_error": rel_err,
        }
        errors.append(rel_err)
        if status == "exact":
            n_exact += 1

    mean_rel_error  = sum(errors) / max(1, len(errors))
    exact_match_rate = n_exact / max(1, n_checked)

    return {
        "parameter_accuracy": results,
        "mean_relative_error": mean_rel_error,
        "exact_match_rate":   exact_match_rate,
        "n_params_checked":   n_checked,
    }


# ─── Metric 5: Category Accuracy ─────────────────────────────────────────────

def compute_category_accuracy(generated_category, reference_category):
    """Did the model correctly identify what type of simulation this is?"""
    return {
        "category_correct": generated_category == reference_category,
        "generated_category": generated_category,
        "reference_category": reference_category,
    }


# ─── Combined Metrics ─────────────────────────────────────────────────────────

def compute_all_metrics(
    generated_script,
    reference_script,
    generated_ir,
    reference_ir,
    category,
    execution_result=None,  # optional: result from actually running MuMax3
):
    """
    Compute all metrics for one (generated, reference) pair.
    
    Args:
        generated_script:   The script our system generated
        reference_script:   The gold-standard script
        generated_ir:       The IR our system constructed
        reference_ir:       The gold-standard IR
        category:           The true simulation category
        execution_result:   Optional dict with execution info
    
    Returns:
        A dict with all metric values, suitable for JSON serialisation
    """
    metrics = {}

    # 1. API validity
    metrics.update(compute_api_validity(generated_script, category))

    # 2. Completeness
    metrics.update(compute_completeness(generated_script, category))

    # 3. IR F1
    metrics.update(compute_ir_f1(generated_ir, reference_ir))

    # 4. Parameter accuracy
    metrics.update(compute_parameter_accuracy(generated_ir, reference_ir))

    # 5. Category accuracy
    generated_category = generated_ir.get("category") if generated_ir else "unknown"
    metrics.update(compute_category_accuracy(generated_category, category))

    # 6. Execution result (if available)
    if execution_result is not None:
        metrics["execution_passed"] = execution_result.get("success", False)
        metrics["execution_time_s"] = execution_result.get("runtime_s", None)
        metrics["physics_score"]    = execution_result.get("physics_score", None)
    else:
        metrics["execution_passed"] = None  # not run yet
        metrics["physics_score"]    = None

    # 7. Static validation
    val = validate_script(generated_script, expected_category=category)
    metrics["static_valid"]     = val["valid"]
    metrics["n_static_errors"]  = len(val["errors"])
    metrics["n_static_warnings"] = len(val["warnings"])

    return metrics


def aggregate_metrics(list_of_metrics):
    """
    Aggregate metrics across multiple samples.
    
    Takes a list of metric dicts (one per sample) and returns
    summary statistics (mean, std, min, max) for each metric.
    """
    if not list_of_metrics:
        return {}

    # Collect all numeric values
    numeric_keys = [
        k for k, v in list_of_metrics[0].items()
        if isinstance(v, (int, float)) and v is not None
    ]

    summary = {}
    for key in numeric_keys:
        values = [m[key] for m in list_of_metrics if m.get(key) is not None]
        if not values:
            continue
        summary[key] = {
            "mean": sum(values) / len(values),
            "std":  math.sqrt(sum((v - sum(values)/len(values))**2 for v in values) / max(1, len(values))),
            "min":  min(values),
            "max":  max(values),
            "n":    len(values),
        }

    # Boolean metrics
    bool_keys = [
        k for k, v in list_of_metrics[0].items()
        if isinstance(v, bool)
    ]
    for key in bool_keys:
        values = [m[key] for m in list_of_metrics if m.get(key) is not None]
        summary[key] = {
            "pass_rate": sum(values) / max(1, len(values)),
            "n":         len(values),
        }

    return summary
