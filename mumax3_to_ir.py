"""
mumax3_to_ir.py
===============
Parses a MuMax3 script and extracts an Intermediate Representation (IR).

This is the REVERSE direction of our pipeline:
  MuMax3 script  →  Structured IR (dict)

We use this when processing collected community scripts:
  1. Scraper collects raw .mx3 files
  2. This parser extracts the IR from each one
  3. The IR becomes the "gold standard" for training

The parser works by scanning the script line by line and looking for
known patterns (assignments, function calls). It is NOT a full parser
(we don't build a complete AST) — just enough to extract the physics
information we care about.

Usage:
    from validation.mumax3_to_ir import parse_script

    with open("my_sim.mx3") as f:
        script_text = f.read()

    ir = parse_script(script_text)
    if ir is not None:
        print(ir["category"])       # "field_driven"
        print(ir["material"])       # {"Msat_Am": 800000, ...}
"""

import re
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ontology.mumax3_api import ALL_VALID_SYMBOLS


MU0 = 1.2566370614e-6


# ─── Regex Patterns ───────────────────────────────────────────────────────────
# These match the common MuMax3 syntax patterns

# Matches: VariableName = number
ASSIGNMENT_RE = re.compile(
    r'^([A-Za-z_]\w*)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
)

# Matches: VariableName = vector(x, y, z)
VECTOR_ASSIGN_RE = re.compile(
    r'^([A-Za-z_]\w*)\s*=\s*vector\s*\(\s*'
    r'([+-]?[\d.eE+-]+)\s*,\s*([+-]?[\d.eE+-]+)\s*,\s*([+-]?[\d.eE+-]+)\s*\)'
)

# Matches: SetGridsize(nx, ny, nz)
GRIDSIZE_RE = re.compile(
    r'SetGridsize\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'
)

# Matches: SetCellsize(dx, dy, dz)
CELLSIZE_RE = re.compile(
    r'SetCellsize\s*\(\s*'
    r'([+-]?[\d.eE+-]+)\s*,\s*([+-]?[\d.eE+-]+)\s*,\s*([+-]?[\d.eE+-]+)\s*\)'
)

# Matches: run(duration)
RUN_RE = re.compile(r'\brun\s*\(\s*([+-]?[\d.eE+-]+)\s*\)')

# Matches: autosave(quantity, dt)
AUTOSAVE_RE = re.compile(r'\bautosave\s*\(\s*(\w+)')

# Matches: tableadd(quantity)
TABLEADD_RE = re.compile(r'\btableadd\s*\(\s*(\w+)')

# Matches: tableautosave(dt)
TABLEAUTOSAVE_RE = re.compile(r'\btableautosave\s*\(\s*([+-]?[\d.eE+-]+)\s*\)')

# Matches: m = uniform(mx, my, mz)
UNIFORM_RE = re.compile(r'\bm\s*=\s*uniform\s*\(\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*\)')

# Matches: m = twoDomain(...)
TWODOMAIN_RE = re.compile(r'\bm\s*=\s*twoDomain\s*\(')

# Matches: m = vortex(...)
VORTEX_RE = re.compile(r'\bm\s*=\s*vortex\s*\(')

# Matches: SetGeom(shape(...))
SETGEOM_RE = re.compile(r'\bSetGeom\s*\(\s*(\w+)\s*\(')

# Matches: defRegion(id, shape)
DEFREGION_RE = re.compile(r'\bdefRegion\s*\(')


def clean_line(line):
    """Remove comments and leading/trailing whitespace from a line."""
    return line.split("//")[0].strip()


def try_float(s):
    """Try to parse a string as float. Returns None if it fails."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_script(script_text):
    """
    Parse a MuMax3 script and extract an IR dict.

    Args:
        script_text: String containing the complete .mx3 script

    Returns:
        An IR dict, or None if the script is too incomplete to parse.
        The IR may have None values for fields not found in the script.
    """
    lines = script_text.split("\n")

    # ── Storage for extracted values ──────────────────────────────────────────
    assignments  = {}      # name -> float value
    vectors      = {}      # name -> (x, y, z) tuple
    functions_called = set()  # set of function names called

    grid     = None   # (nx, ny, nz)
    cellsize = None   # (dx, dy, dz) in metres
    run_time = None   # total simulation time in seconds
    autosaved_quantities = []
    table_quantities = []
    table_dt = None
    init_type = None
    init_direction = None
    geometry_shape = "rectangle"  # default

    # ── Line-by-line parsing ──────────────────────────────────────────────────
    for line in lines:
        line_clean = clean_line(line)
        if not line_clean:
            continue

        # Track function calls
        fn_matches = re.findall(r'\b([A-Za-z_]\w*)\s*\(', line_clean)
        for fn in fn_matches:
            functions_called.add(fn)

        # SetGridsize
        m = GRIDSIZE_RE.search(line_clean)
        if m:
            grid = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue

        # SetCellsize
        m = CELLSIZE_RE.search(line_clean)
        if m:
            cellsize = (try_float(m.group(1)), try_float(m.group(2)), try_float(m.group(3)))
            continue

        # Simple assignment: Name = number
        m = ASSIGNMENT_RE.match(line_clean)
        if m:
            name = m.group(1)
            val  = try_float(m.group(2))
            if val is not None:
                assignments[name] = val
            continue

        # Vector assignment: Name = vector(x, y, z)
        m = VECTOR_ASSIGN_RE.match(line_clean)
        if m:
            name = m.group(1)
            vectors[name] = (
                try_float(m.group(2)),
                try_float(m.group(3)),
                try_float(m.group(4)),
            )
            continue

        # run(t)
        m = RUN_RE.search(line_clean)
        if m:
            run_time = try_float(m.group(1))
            continue

        # autosave(quantity, dt)
        m = AUTOSAVE_RE.search(line_clean)
        if m:
            autosaved_quantities.append(m.group(1))
            # Also try to get the dt
            dt_match = re.search(
                r'\bautosave\s*\(\s*\w+\s*,\s*([+-]?[\d.eE+-]+)\s*\)',
                line_clean
            )
            if dt_match:
                assignments["_autosave_dt"] = try_float(dt_match.group(1))

        # tableadd(quantity)
        m = TABLEADD_RE.search(line_clean)
        if m:
            table_quantities.append(m.group(1))

        # tableautosave(dt)
        m = TABLEAUTOSAVE_RE.search(line_clean)
        if m:
            table_dt = try_float(m.group(1))

        # Initial state detection
        if UNIFORM_RE.search(line_clean):
            um = UNIFORM_RE.search(line_clean)
            mx, my, mz = try_float(um.group(1)), try_float(um.group(2)), try_float(um.group(3))
            init_type = "uniform"
            if mz and mz > 0.5:
                init_direction = "+z"
            elif mz and mz < -0.5:
                init_direction = "-z"
            elif mx and mx > 0.5:
                init_direction = "+x"
            elif mx and mx < -0.5:
                init_direction = "-x"

        elif TWODOMAIN_RE.search(line_clean):
            init_type = "two_domain"
            init_direction = "+x"

        elif VORTEX_RE.search(line_clean):
            init_type = "vortex"

        elif "setInShape" in line_clean or "setinshape" in line_clean.lower():
            init_type = "uniform_then_nucleate"

        # Geometry shape
        m = SETGEOM_RE.search(line_clean)
        if m:
            shape_name = m.group(1).lower()
            if "cylinder" in shape_name or "circle" in shape_name:
                geometry_shape = "cylinder"
            elif "ellipse" in shape_name:
                geometry_shape = "ellipse"
            elif "rect" in shape_name:
                geometry_shape = "rectangle"

        # Multi-region check
        if DEFREGION_RE.search(line_clean):
            functions_called.add("defRegion")

    # ── Check we have minimum required info ───────────────────────────────────
    if "Msat" not in assignments or "Aex" not in assignments:
        # Can't build a meaningful IR without material constants
        return None

    if grid is None or cellsize is None:
        return None

    # ── Build material dict ───────────────────────────────────────────────────
    material = {
        "name":     _identify_material(assignments),
        "Msat_Am":  assignments["Msat"],
        "Aex_Jm":   assignments["Aex"],
        "alpha":    assignments.get("alpha"),
        "Ku1_Jm3":  assignments.get("Ku1"),
        "DMI_Jm2":  assignments.get("Dind") or assignments.get("Dbulk"),
        "Temp_K":   assignments.get("Temp"),
    }

    # ── Compute physical dimensions from grid and cellsize ────────────────────
    nx, ny, nz = grid
    dx, dy, dz = cellsize
    size_x_nm = nx * dx * 1e9
    size_y_nm = ny * dy * 1e9
    size_z_nm = nz * dz * 1e9

    geometry = {
        "shape":     geometry_shape,
        "size_x_nm": size_x_nm,
        "size_y_nm": size_y_nm,
        "size_z_nm": size_z_nm,
    }

    domain = {
        "nx": nx, "ny": ny, "nz": nz,
        "dx_m": dx, "dy_m": dy, "dz_m": dz,
        "cell_size_nm": dx * 1e9,
        "exchange_length_nm": _exchange_length(assignments["Msat"], assignments["Aex"]) * 1e9,
    }

    # ── Determine physics flags ───────────────────────────────────────────────
    has_zeeman   = "B_ext" in vectors or "B_ext" in assignments
    has_dmi      = "Dind" in assignments or "Dbulk" in assignments
    has_aniso    = "Ku1" in assignments
    has_stt      = "Jc" in vectors and ("pol" in assignments or "xi" in assignments)
    has_sot      = "Jc" in vectors and ("xi_DL" in assignments)
    has_thermal  = "Temp" in assignments and assignments.get("Temp", 0) > 0
    has_region   = "defRegion" in functions_called

    physics = {
        "use_exchange":   True,
        "use_demag":      True,
        "use_zeeman":     has_zeeman,
        "use_dmi":        has_dmi,
        "use_anisotropy": has_aniso,
        "use_stt":        has_stt,
        "use_sot":        has_sot,
        "use_thermal":    has_thermal,
    }

    # ── Detect category from physics flags ────────────────────────────────────
    category = _classify_category(physics, has_region, functions_called)

    # ── Build excitation dict ─────────────────────────────────────────────────
    excitation = None
    if has_zeeman:
        B_vec = vectors.get("B_ext", (0, 0, 0))
        B_mag = max(abs(v) for v in B_vec if v is not None)
        excitation = {
            "type":  "dc_field",   # simplified — we don't detect pulse() here
            "B_T":   B_mag,
            "direction": _vector_to_direction(B_vec),
        }
    elif has_stt or has_sot:
        Jc_vec = vectors.get("Jc", (0, 0, 0))
        Jc_mag = max(abs(v) for v in Jc_vec if v is not None)
        excitation = {
            "type":   "current",
            "Jc_Am2": Jc_mag,
            "pol":    assignments.get("pol"),
            "xi_DL":  assignments.get("xi_DL"),
        }

    # ── Build solver dict ─────────────────────────────────────────────────────
    dt_save = assignments.get("_autosave_dt") or table_dt

    solver = {
        "mode":      "relax_only" if "run" not in functions_called else "relax_then_run",
        "t_total_s": run_time,
        "dt_save_s": dt_save,
    }

    # ── Build outputs list ────────────────────────────────────────────────────
    outputs = []
    for q in set(autosaved_quantities):
        outputs.append({"quantity": q, "schedule": "autosave"})
    for q in set(table_quantities):
        outputs.append({"quantity": q, "schedule": "table"})

    # ── Initialisation ────────────────────────────────────────────────────────
    initialization = {
        "type":      init_type or "uniform",
        "direction": init_direction or "+x",
    }

    ir = {
        "ir_version":     "1.0",
        "category":       category,
        "domain":         domain,
        "geometry":       geometry,
        "material":       material,
        "physics":        physics,
        "excitation":     excitation,
        "initialization": initialization,
        "solver":         solver,
        "outputs":        outputs,
        "_source":        "parsed",
    }

    return ir


# ─── Helper Functions ──────────────────────────────────────────────────────────

def _exchange_length(Msat, Aex):
    """Exchange length in metres."""
    if Msat <= 0 or Aex <= 0:
        return 1e-9
    return math.sqrt(2 * Aex / (MU0 * Msat ** 2))


def _identify_material(assignments):
    """
    Try to identify the material from Msat and Aex values.
    Uses a tolerance of 10% to match against known materials.
    """
    Msat = assignments.get("Msat", 0)
    Aex  = assignments.get("Aex", 0)

    KNOWN = {
        "Permalloy": (800e3,  13e-12),
        "CoFeB":     (1100e3, 20e-12),
        "YIG":       (140e3,  3.7e-12),
        "Co":        (1400e3, 30e-12),
        "Fe":        (1700e3, 21e-12),
        "Ni":        (490e3,  9e-12),
    }

    for name, (m_ref, a_ref) in KNOWN.items():
        if (abs(Msat - m_ref) / m_ref < 0.10 and
                abs(Aex  - a_ref) / a_ref < 0.10):
            return name

    return "unknown"


def _vector_to_direction(vec):
    """Convert a (x, y, z) vector tuple to a direction string like '+x', '-z'."""
    if vec is None or all(v is None for v in vec):
        return "x"
    x, y, z = (v or 0 for v in vec)
    if abs(x) > abs(y) and abs(x) > abs(z):
        return "+x" if x > 0 else "-x"
    if abs(y) > abs(x) and abs(y) > abs(z):
        return "+y" if y > 0 else "-y"
    return "+z" if z > 0 else "-z"


def _classify_category(physics, has_region, functions_called):
    """Classify the simulation into one of our 8 categories."""
    if has_region:
        return "multi_region"
    if physics["use_sot"]:
        return "sot"
    if physics["use_stt"]:
        return "stt"
    if physics["use_dmi"]:
        return "dmi_skyrmion"
    if physics["use_thermal"]:
        return "thermal"
    if physics["use_zeeman"]:
        # Check if FFT output is saved (suggests frequency analysis)
        if "FFT" in str(functions_called):
            return "fft_analysis"
        return "field_driven"
    return "simple_relax"


# ─── Batch Parser ─────────────────────────────────────────────────────────────

def parse_directory(directory_path, output_jsonl=None):
    """
    Parse all .mx3 files in a directory.

    Returns list of (filename, ir) tuples where ir may be None for failed parses.
    Optionally saves results to a JSONL file.
    """
    import json
    from pathlib import Path

    results = []
    files   = list(Path(directory_path).glob("*.mx3"))
    n_ok    = 0
    n_fail  = 0

    print(f"Parsing {len(files)} .mx3 files in {directory_path}...")

    for path in files:
        script_text = path.read_text(encoding="utf-8", errors="replace")
        ir = parse_script(script_text)

        if ir is not None:
            n_ok += 1
            results.append((str(path), ir))
        else:
            n_fail += 1
            print(f"  FAIL: {path.name} (could not extract Msat/Aex/grid)")

    print(f"Done: {n_ok} parsed, {n_fail} failed ({100*n_fail/max(1,len(files)):.1f}% fail rate)")

    if output_jsonl:
        with open(output_jsonl, "w") as f:
            for fname, ir in results:
                f.write(json.dumps({"file": fname, "ir": ir}) + "\n")
        print(f"Results saved to {output_jsonl}")

    return results
