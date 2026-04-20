"""
static_validator.py
===================
Checks a MuMax3 script for errors WITHOUT running it.

This is like a syntax checker + type checker combined.
It looks for:
  1. Hallucinated API calls (symbols not in the MuMax3 API)
  2. Missing required symbols for the simulation category
  3. Unsatisfied dependencies (e.g., DMI set but no Aex)
  4. Parameter values outside physical ranges
  5. Basic structural problems (no grid, no solver call, etc.)

It does NOT guarantee the script will run - only MuMax3 execution can do that.
But it catches the most common errors fast.

Usage:
    from validation.static_validator import validate_script

    result = validate_script(script_text, expected_category="field_driven")
    if result["valid"]:
        print("Looks good!")
    else:
        for error in result["errors"]:
            print("Error:", error)
"""

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ontology.mumax3_api import (
    ALL_VALID_SYMBOLS, CATEGORY_REQUIRED, ALWAYS_REQUIRED,
    check_dependencies, is_valid_symbol
)


# ─── Parameter Range Limits ───────────────────────────────────────────────────
# These define what values are physically reasonable
PARAM_RANGES = {
    "Msat":  (1e3,  2e6,   "A/m"),     # 1 kA/m to 2 MA/m
    "Aex":   (1e-13, 1e-10, "J/m"),    # 0.1 to 100 pJ/m
    "alpha": (1e-5,  1.0,   ""),       # dimensionless
    "Ku1":   (1e2,   1e7,   "J/m3"),   # 100 J/m3 to 10 MJ/m3
    "Dind":  (0,     1e-2,  "J/m2"),   # up to 10 mJ/m2
    "Dbulk": (0,     1e-2,  "J/m2"),
    "Temp":  (0,     2000,  "K"),
}


def tokenize_script(script_text):
    """
    Extract all identifier tokens from a MuMax3 script.
    
    Skips:
    - Comment lines (starting with //)
    - String literals
    - Pure numbers
    
    Returns: list of (line_number, token) tuples
    """
    tokens = []
    for line_num, line in enumerate(script_text.split("\n"), start=1):
        # Strip comments
        line = line.split("//")[0].strip()
        if not line:
            continue

        # Find all identifiers: sequences of letters, digits, underscore
        # that don't start with a digit
        found = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', line)
        for token in found:
            tokens.append((line_num, token))

    return tokens


def extract_assignments(script_text):
    """
    Find all variable assignments like:  Msat = 800e3
    
    Returns: dict mapping variable name -> (value_string, line_number)
    """
    assignments = {}
    for line_num, line in enumerate(script_text.split("\n"), start=1):
        # Remove comments
        line = line.split("//")[0].strip()
        # Match: identifier = number (scientific notation or plain)
        match = re.match(
            r'^([A-Za-z_]\w*)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',
            line
        )
        if match:
            name  = match.group(1)
            value = match.group(2)
            assignments[name] = (value, line_num)
    return assignments


def extract_function_calls(script_text):
    """
    Find all function calls in the script.
    
    Returns: dict mapping function_name -> list of line numbers
    """
    calls = {}
    for line_num, line in enumerate(script_text.split("\n"), start=1):
        line = line.split("//")[0].strip()
        # Match: identifier followed by (
        found = re.findall(r'\b([A-Za-z_]\w*)\s*\(', line)
        for name in found:
            if name not in calls:
                calls[name] = []
            calls[name].append(line_num)
    return calls


def validate_script(script_text, expected_category=None):
    """
    Run all static validation checks on a MuMax3 script.
    
    Args:
        script_text: The MuMax3 script as a string
        expected_category: The expected simulation category (optional).
                           Used to check for category-specific required symbols.
    
    Returns:
        A dict with:
          "valid":           bool - True if no errors found
          "errors":          list of error strings
          "warnings":        list of warning strings
          "symbols_found":   set of API symbols found
          "hallucinations":  list of tokens that are not in the MuMax3 API
          "hallucination_rate": float 0-1
          "missing_required": list of required symbols that are missing
          "param_issues":    list of parameter range violations
    """
    errors   = []
    warnings = []

    # ── Extract tokens, assignments, and function calls ───────────────────────
    all_tokens      = tokenize_script(script_text)
    token_strings   = {t for _, t in all_tokens}
    assignments     = extract_assignments(script_text)
    function_calls  = extract_function_calls(script_text)

    # Find which API symbols are used
    symbols_in_script = token_strings & ALL_VALID_SYMBOLS

    # ── Check 1: Structural Requirements ─────────────────────────────────────
    # Every valid MuMax3 script needs these
    if "SetGridsize" not in symbols_in_script and "SetGridsize" not in function_calls:
        errors.append("MISSING SetGridsize() - every MuMax3 script must set the grid dimensions")

    if "SetCellsize" not in symbols_in_script and "SetCellsize" not in function_calls:
        errors.append("MISSING SetCellsize() - every MuMax3 script must set the cell size")

    if "Msat" not in assignments:
        errors.append("MISSING Msat - saturation magnetisation must be set")

    if "Aex" not in assignments:
        errors.append("MISSING Aex - exchange stiffness must be set")

    if "alpha" not in assignments:
        warnings.append("MISSING alpha - Gilbert damping not set (MuMax3 default is 0.01)")

    # Check that there's some kind of solver call
    has_solver = any(s in function_calls for s in ("run", "relax", "minimize", "RunWhile"))
    if not has_solver:
        errors.append("MISSING solver call - script has no run(), relax(), or minimize()")

    # ── Check 2: Hallucination Detection ─────────────────────────────────────
    # Any identifier that looks like an API call but isn't in the API is a hallucination
    # We filter to things that look like they're being USED (not just variable names)
    # by checking against function calls and known parameter names
    possible_api_calls = set(function_calls.keys()) | set(assignments.keys())

    # Reserved words and common non-API tokens to ignore
    IGNORE_TOKENS = {
        "for", "if", "else", "var", "true", "false", "nil", "return",
        "print", "printf", "sprintf", "fprintln",  # these ARE in API
        # Common variable names people use
        "t", "i", "j", "n", "x", "y", "z", "dt", "B",
    }

    hallucinated = []
    for token in possible_api_calls:
        if token in IGNORE_TOKENS:
            continue
        if token.startswith("_"):  # private variables
            continue
        if not is_valid_symbol(token):
            # Check if it's just a user variable (pure lowercase or common patterns)
            # We flag things that look like they're trying to be API calls
            if token[0].isupper() or "_" in token:
                hallucinated.append(token)

    if hallucinated:
        for h in hallucinated:
            errors.append(f"HALLUCINATION: '{h}' is not a valid MuMax3 API symbol")

    # Compute hallucination rate
    n_api_attempts = len(possible_api_calls - IGNORE_TOKENS)
    hallucination_rate = len(hallucinated) / max(1, n_api_attempts)

    # ── Check 3: Category-Specific Required Symbols ───────────────────────────
    missing_required = []
    if expected_category and expected_category in CATEGORY_REQUIRED:
        required = CATEGORY_REQUIRED[expected_category]
        for sym in required:
            if sym not in symbols_in_script and sym not in function_calls and sym not in assignments:
                missing_required.append(sym)

        if missing_required:
            warnings.append(
                f"For '{expected_category}' simulation, these symbols are typically required "
                f"but not found: {', '.join(missing_required)}"
            )

    # ── Check 4: Dependency Violations ───────────────────────────────────────
    dep_violations = check_dependencies(symbols_in_script | set(assignments.keys()))
    for (symbol, missing_dep) in dep_violations:
        warnings.append(
            f"DEPENDENCY: '{symbol}' is present but '{missing_dep}' is missing. "
            f"'{symbol}' depends on '{missing_dep}' being set."
        )

    # ── Check 5: Parameter Range Validation ───────────────────────────────────
    param_issues = []
    for param, (val_str, line_num) in assignments.items():
        if param in PARAM_RANGES:
            lo, hi, unit = PARAM_RANGES[param]
            try:
                val = float(val_str)
                if val < lo:
                    msg = f"Line {line_num}: {param} = {val:.2e} is below minimum {lo:.2e} {unit}"
                    errors.append(f"RANGE ERROR: {msg}")
                    param_issues.append(msg)
                elif val > hi:
                    msg = f"Line {line_num}: {param} = {val:.2e} is above maximum {hi:.2e} {unit}"
                    errors.append(f"RANGE ERROR: {msg}")
                    param_issues.append(msg)
            except ValueError:
                pass  # value wasn't a simple number (might be an expression)

    # ── Check 6: Autosave Without Run ────────────────────────────────────────
    if "autosave" in function_calls and "run" not in function_calls:
        warnings.append(
            "autosave() is set but no run() call found. "
            "Autosave only works during run(). Did you mean save() for a one-time save?"
        )

    # ── Check 7: Exchange Length vs Cell Size ─────────────────────────────────
    # We can compute this if Msat and Aex are simple numbers
    import math
    MU0 = 1.2566370614e-6

    if "Msat" in assignments and "Aex" in assignments and "SetCellsize" in function_calls:
        try:
            Msat = float(assignments["Msat"][0])
            Aex  = float(assignments["Aex"][0])
            l_ex = math.sqrt(2 * Aex / (MU0 * Msat**2)) * 1e9  # nm

            # Try to extract cell size from SetCellsize call
            cellsize_match = re.search(
                r'SetCellsize\s*\(\s*([0-9eE.+-]+)',
                script_text
            )
            if cellsize_match:
                dx_m  = float(cellsize_match.group(1))
                dx_nm = dx_m * 1e9
                if dx_nm > 2 * l_ex:
                    errors.append(
                        f"PHYSICS: Cell size ({dx_nm:.2f} nm) is larger than "
                        f"2x exchange length ({l_ex:.2f} nm). "
                        f"Exchange physics will be numerically inaccurate."
                    )
                elif dx_nm > l_ex:
                    warnings.append(
                        f"Cell size ({dx_nm:.2f} nm) is close to exchange length "
                        f"({l_ex:.2f} nm). Consider refining the mesh."
                    )
        except (ValueError, ZeroDivisionError):
            pass  # couldn't parse, skip this check

    # ── Compute Final Validity ────────────────────────────────────────────────
    # Script is "valid" if there are no errors (warnings are ok)
    is_valid = len(errors) == 0

    return {
        "valid":             is_valid,
        "errors":            errors,
        "warnings":          warnings,
        "symbols_found":     symbols_in_script,
        "hallucinations":    hallucinated,
        "hallucination_rate": hallucination_rate,
        "missing_required":  missing_required,
        "param_issues":      param_issues,
        "n_symbols_checked": n_api_attempts,
    }


def print_validation_report(validation_result, verbose=True):
    """Pretty-print a validation result."""
    v = validation_result
    status = "✓ VALID" if v["valid"] else "✗ INVALID"
    print(f"\nValidation Result: {status}")
    print(f"  Symbols checked: {v['n_symbols_checked']}")
    print(f"  Hallucinations:  {len(v['hallucinations'])} ({v['hallucination_rate']*100:.1f}%)")

    if v["errors"]:
        print(f"\n  ERRORS ({len(v['errors'])}):")
        for e in v["errors"]:
            print(f"    ✗ {e}")

    if v["warnings"] and verbose:
        print(f"\n  WARNINGS ({len(v['warnings'])}):")
        for w in v["warnings"]:
            print(f"    ⚠ {w}")

    if v["hallucinations"] and verbose:
        print(f"\n  Hallucinated symbols: {v['hallucinations']}")
