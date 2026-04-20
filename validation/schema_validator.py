#!/usr/bin/env python3

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
import math

try:
    import jsonschema
    from jsonschema import validate, ValidationError
except ImportError:
    print("⚠️ jsonschema not installed. Run: pip install jsonschema")
    jsonschema = None


# Load schemas
SCHEMA_DIR = Path(__file__).parent.parent / "data"
DATASET_SCHEMA_PATH = SCHEMA_DIR / "dataset_schema_v1.0.json"
IR_SCHEMA_PATH = SCHEMA_DIR / "ir_schema_v1.0.json"

_DATASET_SCHEMA = None
_IR_SCHEMA = None


def _load_schemas():
    """Load schemas from disk (lazy loading)."""
    global _DATASET_SCHEMA, _IR_SCHEMA
    
    if _DATASET_SCHEMA is None and DATASET_SCHEMA_PATH.exists():
        with open(DATASET_SCHEMA_PATH) as f:
            _DATASET_SCHEMA = json.load(f)
    
    if _IR_SCHEMA is None and IR_SCHEMA_PATH.exists():
        with open(IR_SCHEMA_PATH) as f:
            _IR_SCHEMA = json.load(f)


def validate_dataset_entry(entry: Dict[str, Any]) -> Tuple[bool,List[str]]:
    """
    Validate a dataset entry against dataset_schema_v1.0.json.
    
    Args:
        entry: Dataset entry dictionary
    
    Returns:
        (is_valid, list_of_errors)
    """
    if jsonschema is None:
        return False, ["jsonschema library not installed"]
    
    _load_schemas()
    errors = []
    
    # JSON Schema validation
    if _DATASET_SCHEMA:
        try:
            validate(instance=entry, schema=_DATASET_SCHEMA)
        except ValidationError as e:
            errors.append(f"Schema validation: {e.message}")
    else:
        errors.append("Dataset schema not found at {DATASET_SCHEMA_PATH}")
    
    # Custom validations
    # 1. Content hash matches
    if "mx3_script" in entry and "metadata" in entry:
        computed_hash = hashlib.sha256(entry["mx3_script"].encode()).hexdigest()
        declared_hash = entry["metadata"].get("content_hash", "")
        if declared_hash and computed_hash != declared_hash:
            errors.append(f"Content hash mismatch: declared={declared_hash[:8]}..., computed={computed_hash[:8]}...")
    
    # 2. source_type is valid
    valid_sources = ["official_example", "official_test", "workshop", "community", 
                     "github_search", "paper", "forum", "synthetic"]
    if "metadata" in entry:
        source = entry["metadata"].get("source_type", "")
        if source and source not in valid_sources:
            errors.append(f"Invalid source_type: {source}")
    
    # 3. Complexity is in valid range
    if "metadata" in entry:
        complexity = entry["metadata"].get("complexity", "")
        if complexity and complexity not in ["trivial", "simple", "medium", "complex"]:
            errors.append(f"Invalid complexity: {complexity}")
    
    return (len(errors) == 0, errors)


def validate_ir(ir: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate IR against ir_schema_v1.0.json and physics plausibility.
    
    Args:
        ir: IR dictionary
    
    Returns:
        (is_valid, list_of_errors)
    """
    if jsonschema is None:
        return False, ["jsonschema library not installed"]
    
    _load_schemas()
    errors = []
    warnings = []
    
    # JSON Schema validation
    if _IR_SCHEMA:
        try:
            validate(instance=ir, schema=_IR_SCHEMA)
        except ValidationError as e:
            errors.append(f"Schema validation: {e.message}")
    else:
        errors.append(f"IR schema not found at {IR_SCHEMA_PATH}")
    
    # Physics plausibility checks
    if "mesh" in ir and "materials" in ir:
        phys_errors = _check_physics_plausibility(ir)
        warnings.extend(phys_errors)
    
    return (len(errors) == 0, errors + warnings)


def _check_physics_plausibility(ir: Dict[str, Any]) -> List[str]:
    """
    Check if the IR makes physical sense.
    
    Returns list of warnings (not hard errors).
    """
    warnings = []
    
    # Get material parameters
    materials = ir.get("materials", [])
    if not materials:
        return warnings
    
    global_mat = next((m for m in materials if m.get("region_id", 0) == 0), None)
    if not global_mat:
        return warnings
    
    Msat = global_mat.get("Msat")
    Aex = global_mat.get("Aex")
    alpha = global_mat.get("alpha")
    
    # 1. Parameter ranges
    if Msat is not None:
        if Msat < 1e4 or Msat > 2e6:
            warnings.append(f"⚠️  Msat={Msat:.2e} A/m is outside typical range [1e4, 2e6]")
    
    if Aex is not None:
        if Aex < 1e-13 or Aex > 5e-11:
            warnings.append(f"⚠️  Aex={Aex:.2e} J/m is outside typical range [1e-13, 5e-11]")
    
    if alpha is not None:
        if alpha < 0:
            warnings.append(f"❌ alpha={alpha} cannot be negative")
        elif alpha > 2.0:
            warnings.append(f"⚠️  alpha={alpha} is unusually large (typical: 0.001-1.0)")
    
    # 2. Exchange length vs cell size
    if Msat and Aex and "mesh" in ir and "cell_size" in ir["mesh"]:
        mu0 = 1.2566370614359173e-6
        lex = math.sqrt(2 * Aex / (mu0 * Msat**2))
        
        cell_size = ir["mesh"]["cell_size"]
        max_cell = max(cell_size)
        
        if max_cell > lex:
            warnings.append(f"⚠️  Cell size ({max_cell:.2e} m) > exchange length ({lex:.2e} m). "
                          "Results may be inaccurate. Recommend cell < lex.")
        elif max_cell > lex / 2:
            warnings.append(f"⚠️  Cell size ({max_cell:.2e} m) > lex/2 ({lex/2:.2e} m). "
                          "Consider reducing for better accuracy.")
    
    # 3. Grid size sanity
    if "mesh" in ir and "grid" in ir["mesh"]:
        grid = ir["mesh"]["grid"]
        total_cells = grid[0] * grid[1] * grid[2]
        
        if total_cells < 100:
            warnings.append(f"⚠️  Very small grid ({total_cells} cells). May not capture physics.")
        elif total_cells > 1e7:
            warnings.append(f"⚠️  Very large grid ({total_cells:.0e} cells). May be slow.")
        
        # Prefer powers of 2 or small factors for FFT efficiency
        for i, n in enumerate(grid):
            if n > 1:
                # Check if n has only small prime factors (2, 3, 5, 7)
                temp = n
                for p in [2, 3, 5, 7]:
                    while temp % p == 0:
                        temp //= p
                if temp > 1:
                    warnings.append(f"⚠️  Grid dimension {n} (axis {i}) has large prime factors. "
                                  f"FFT may be slow. Prefer powers of 2 or products of 2,3,5,7.")
    
    # 4. DMI stability condition (if DMI is present)
    Dind = global_mat.get("Dind")
    if Dind and Msat and Aex:
        # DMI/Aex ratio should be reasonable
        D_over_A = abs(Dind) / Aex * 1e-9  # Normalize
        if D_over_A > 10:
            warnings.append(f"⚠️  DMI/Aex ratio is very large ({D_over_A:.2f}). "
                          "May indicate unstable parameters.")
    
    # 5. PMA quality factor (if Ku1 and AnisU present)
    Ku1 = global_mat.get("Ku1")
    AnisU = global_mat.get("AnisU")
    if Ku1 and AnisU and Msat:
        # Check if out-of-plane (AnisU ≈ [0,0,1])
        if abs(AnisU[2]) > 0.9:
            mu0 = 1.2566370614359173e-6
            Q = Ku1 / (0.5 * mu0 * Msat**2)
            if Q < 1:
                warnings.append(f"⚠️  PMA quality factor Q={Q:.2f} < 1. "
                              "Magnetization may not stabilize out-of-plane.")
            elif Q > 100:
                warnings.append(f"⚠️  PMA quality factor Q={Q:.2f} is very high. "
                              "May indicate unrealistic parameters.")
    
    # 6. Temperature vs energy scales
    Temp = global_mat.get("Temp")
    if Temp and Temp > 0:
        kb = 1.380649e-23  # Boltzmann constant
        if Aex:
            E_ex = Aex * 1e9  # Exchange energy scale (J) per nm
            if kb * Temp > E_ex:
                warnings.append(f"⚠️  Thermal energy kT={kb*Temp:.2e} J > exchange energy scale "
                              f"{E_ex:.2e} J. Thermal fluctuations may destroy order.")
    
    return warnings


def validate_file(file_path: Path, schema_type: str = "auto") -> Tuple[bool, List[str]]:
    """
    Validate a JSON file.
    
    Args:
        file_path: Path to JSON file
        schema_type: "dataset", "ir", or "auto" (detect from content)
    
    Returns:
        (is_valid, list_of_errors)
    """
    with open(file_path) as f:
        data = json.load(f)
    
    # Auto-detect schema type
    if schema_type == "auto":
        if "mx3_script" in data and "nl_description" in data:
            schema_type = "dataset"
        elif "mesh" in data and "materials" in data:
            schema_type = "ir"
        else:
            return False, ["Cannot auto-detect schema type"]
    
    if schema_type == "dataset":
        return validate_dataset_entry(data)
    elif schema_type == "ir":
        return validate_ir(data)
    else:
        return False, [f"Unknown schema type: {schema_type}"]


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python schema_validator.py <file.json> [schema_type]")
        print("       schema_type: dataset | ir | auto (default)")
        print("   or: python schema_validator.py --test")
        sys.exit(1)
    
    if sys.argv[1] == "--test":
        # Test IR validation
        test_ir = {
            "mesh": {
                "grid": [128, 64, 1],
                "cell_size": [4e-9, 4e-9, 10e-9]
            },
            "materials": [{
                "region_id": 0,
                "Msat": 800e3,
                "Aex": 13e-12,
                "alpha": 0.02
            }],
            "initial_config": {
                "type": "uniform",
                "params": {"mx": 1, "my": 0, "mz": 0}
            },
            "simulation_type": "relax"
        }
        
        valid, errors = validate_ir(test_ir)
        print(f"IR validation: {'✓ PASS' if valid else '✗ FAIL'}")
        if errors:
            for err in errors:
                print(f"  {err}")
        else:
            print("  No errors or warnings")
    
    else:
        file_path = Path(sys.argv[1])
        schema_type = sys.argv[2] if len(sys.argv) > 2 else "auto"
        
        valid, errors = validate_file(file_path, schema_type)
        
        print(f"{'✓ VALID' if valid else '✗ INVALID'}: {file_path}")
        if errors:
            for err in errors:
                print(f"  {err}")
