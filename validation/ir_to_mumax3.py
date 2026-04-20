#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def ir_to_script(ir: Dict[str, Any], add_comments: bool = True) -> str:
    """
    Convert IR JSON to MuMax3 script.
    
    Args:
        ir: Intermediate representation dictionary
        add_comments: If True, add explanatory comments
    
    Returns:
        MuMax3 script as string
    """
    lines = []
    
    if add_comments:
        lines.append("// MuMax3 script generated from IR")
        lines.append("// Auto-generated — may need manual review")
        lines.append("")
    
    # 1. Mesh configuration (REQUIRED)
    lines.extend(_generate_mesh(ir["mesh"], add_comments))
    lines.append("")
    
    # 2. Geometry
    if "geometry" in ir and ir["geometry"]:
        lines.extend(_generate_geometry(ir["geometry"], add_comments))
        lines.append("")
    
    # 3. Material parameters (REQUIRED)
    lines.extend(_generate_materials(ir["materials"], add_comments))
    lines.append("")
    
    # 4. Initial magnetization (REQUIRED)
    lines.extend(_generate_initial_config(ir["initial_config"], add_comments))
    lines.append("")
    
    # 5. Excitations (optional)
    if "excitation" in ir and ir["excitation"]:
        lines.extend(_generate_excitation(ir["excitation"], add_comments))
        lines.append("")
    
    # 6. Extensions (optional)
    if "extensions" in ir and ir["extensions"]:
        lines.extend(_generate_extensions(ir["extensions"], add_comments))
        lines.append("")
    
    # 7. Output configuration
    if "output" in ir and ir["output"]:
        lines.extend(_generate_output(ir["output"], add_comments))
        lines.append("")
    
    # 8. Run/Solver (REQUIRED)
    lines.extend(_generate_run(ir, add_comments))
    
    return "\n".join(lines)


def _generate_mesh(mesh: Dict[str, Any], comments: bool) -> List[str]:
    """Generate mesh configuration."""
    lines = []
    if comments:
        lines.append("// Mesh configuration")
    
    grid = mesh["grid"]
    cell = mesh["cell_size"]
    
    lines.append(f"SetGridsize({grid[0]}, {grid[1]}, {grid[2]})")
    lines.append(f"SetCellsize({cell[0]:.9g}, {cell[1]:.9g}, {cell[2]:.9g})")
    
    if "pbc" in mesh and mesh["pbc"]:
        pbc = mesh["pbc"]
        if any(p > 0 for p in pbc):
            if comments:
                lines.append("// Periodic boundary conditions must be set BEFORE SetGridsize")
            # Note: This should actually come before SetGridsize, but we document it
            lines.insert(0, f"SetPBC({pbc[0]}, {pbc[1]}, {pbc[2]})")
    
    if mesh.get("edge_smooth", 0) > 0:
        lines.append(f"EdgeSmooth = {mesh['edge_smooth']}")
    
    return lines


def _generate_geometry(geom: Dict[str, Any], comments: bool) -> List[str]:
    """Generate geometry definition."""
    lines = []
    
    if "shape" in geom and geom["shape"]:
        if comments:
            lines.append("// Geometry")
        lines.append(f"setgeom({geom['shape']})")
    
    if "regions" in geom and geom["regions"]:
        if comments:
            lines.append("// Region definitions")
        for region in geom["regions"]:
            rid = region["id"]
            shape = region["shape_expr"]
            label = region.get("label", "")
            if label and comments:
                lines.append(f"// Region {rid}: {label}")
            lines.append(f"defregion({rid}, {shape})")
    
    return lines


def _generate_materials(materials: List[Dict[str, Any]], comments: bool) -> List[str]:
    """Generate material parameter assignments."""
    lines = []
    if comments:
        lines.append("// Material parameters")
    
    # Group by region: region 0 (global) first, then others
    global_params = [m for m in materials if m.get("region_id", 0) == 0]
    region_params = [m for m in materials if m.get("region_id", 0) != 0]
    
    # Global parameters
    for mat in global_params:
        if "label" in mat and comments:
            lines.append(f"// {mat['label']}")
        
        # Always required
        lines.append(f"Msat  = {mat['Msat']:.9g}")
        lines.append(f"Aex   = {mat['Aex']:.9g}")
        lines.append(f"alpha = {mat['alpha']:.9g}")
        
        # Optional parameters
        _add_optional_param(lines, mat, "Ku1")
        _add_optional_param(lines, mat, "Ku2")
        _add_optional_param(lines, mat, "Kc1")
        _add_optional_param(lines, mat, "Kc2")
        _add_optional_param(lines, mat, "Kc3")
        _add_optional_param(lines, mat, "Dind")
        _add_optional_param(lines, mat, "Dbulk")
        _add_optional_param(lines, mat, "Temp")
        
        if "AnisU" in mat and mat["AnisU"]:
            u = mat["AnisU"]
            lines.append(f"AnisU = vector({u[0]:.9g}, {u[1]:.9g}, {u[2]:.9g})")
        
        if "AnisC1" in mat and mat["AnisC1"]:
            c = mat["AnisC1"]
            lines.append(f"AnisC1 = vector({c[0]:.9g}, {c[1]:.9g}, {c[2]:.9g})")
        
        if "AnisC2" in mat and mat["AnisC2"]:
            c = mat["AnisC2"]
            lines.append(f"AnisC2 = vector({c[0]:.9g}, {c[1]:.9g}, {c[2]:.9g})")
    
    # Per-region parameters
    for mat in region_params:
        rid = mat["region_id"]
        if "label" in mat and comments:
            lines.append(f"// Region {rid}: {mat['label']}")
        
        lines.append(f"Msat.SetRegion({rid}, {mat['Msat']:.9g})")
        lines.append(f"Aex.SetRegion({rid}, {mat['Aex']:.9g})")
        lines.append(f"alpha.SetRegion({rid}, {mat['alpha']:.9g})")
        
        _add_optional_region_param(lines, mat, rid, "Ku1")
        _add_optional_region_param(lines, mat, rid, "Ku2")
        _add_optional_region_param(lines, mat, rid, "Kc1")
        _add_optional_region_param(lines, mat, rid, "Dind")
        _add_optional_region_param(lines, mat, rid, "Dbulk")
        _add_optional_region_param(lines, mat, rid, "Temp")
        
        if "AnisU" in mat and mat["AnisU"]:
            u = mat["AnisU"]
            lines.append(f"AnisU.SetRegion({rid}, vector({u[0]:.9g}, {u[1]:.9g}, {u[2]:.9g}))")
    
    return lines


def _add_optional_param(lines: List[str], mat: Dict, param: str):
    """Add optional parameter if present."""
    if param in mat and mat[param] is not None:
        lines.append(f"{param} = {mat[param]:.9g}")


def _add_optional_region_param(lines: List[str], mat: Dict, rid: int, param: str):
    """Add optional region parameter if present."""
    if param in mat and mat[param] is not None:
        lines.append(f"{param}.SetRegion({rid}, {mat[param]:.9g})")


def _generate_initial_config(init: Dict[str, Any], comments: bool) -> List[str]:
    """Generate initial magnetization configuration."""
    lines = []
    if comments:
        lines.append("// Initial magnetization")
    
    init_type = init["type"]
    params = init.get("params", {})
    
    if init_type == "uniform":
        mx = params.get("mx", 1)
        my = params.get("my", 0)
        mz = params.get("mz", 0)
        lines.append(f"m = uniform({mx:.9g}, {my:.9g}, {mz:.9g})")
    
    elif init_type == "vortex":
        circ = params.get("circulation", 1)
        pol = params.get("polarization", 1)
        lines.append(f"m = vortex({circ}, {pol})")
    
    elif init_type == "two_domain":
        # TwoDomain(mx1, my1, mz1, mx2, my2, mz2, mx_wall, my_wall, mz_wall)
        domain1 = params.get("domain1", [1, 0, 0])
        domain2 = params.get("domain2", [-1, 0, 0])
        wall = params.get("wall", [0, 1, 0])
        lines.append(f"m = twodomain({domain1[0]:.9g}, {domain1[1]:.9g}, {domain1[2]:.9g}, "
                    f"{wall[0]:.9g}, {wall[1]:.9g}, {wall[2]:.9g}, "
                    f"{domain2[0]:.9g}, {domain2[1]:.9g}, {domain2[2]:.9g})")
    
    elif init_type == "random":
        lines.append("m = randomMag()")
    
    elif init_type == "skyrmion_bloch":
        circ = params.get("circulation", 1)
        pol = params.get("polarization", -1)
        scale = params.get("scale", [1, 1, 1])
        lines.append(f"m = BlochSkyrmion({circ}, {pol})")
        if scale != [1, 1, 1]:
            lines.append(f"m = m.scale({scale[0]:.9g}, {scale[1]:.9g}, {scale[2]:.9g})")
    
    elif init_type == "skyrmion_neel":
        circ = params.get("circulation", 1)
        pol = params.get("polarization", -1)
        scale = params.get("scale", [1, 1, 1])
        lines.append(f"m = NeelSkyrmion({circ}, {pol})")
        if scale != [1, 1, 1]:
            lines.append(f"m = m.scale({scale[0]:.9g}, {scale[1]:.9g}, {scale[2]:.9g})")
    
    elif init_type == "from_file":
        filename = params.get("filename", "init.ovf")
        lines.append(f'm.LoadFile("{filename}")')
    
    elif init_type == "custom":
        # Custom initialization expression
        expr = params.get("expression", "uniform(1, 0, 0)")
        lines.append(f"m = {expr}")
    
    else:
        # Fallback: uniform
        lines.append("m = uniform(1, 0, 0)")
    
    return lines


def _generate_excitation(exc: Dict[str, Any], comments: bool) -> List[str]:
    """Generate excitation (fields, currents)."""
    lines = []
    
    # External field
    if "B_ext" in exc:
        if comments:
            lines.append("// External field")
        b_ext = exc["B_ext"]
        
        if "static" in b_ext and b_ext["static"]:
            b = b_ext["static"]
            lines.append(f"B_ext = vector({b[0]:.9g}, {b[1]:.9g}, {b[2]:.9g})")
        
        if "time_dependent" in b_ext and b_ext["time_dependent"]:
            expr = b_ext["time_dependent"]
            lines.append(f"B_ext = {expr}")
    
    # Spin currents (STT/SOT)
    if "current" in exc:
        if comments:
            lines.append("// Spin-transfer torque / Spin-orbit torque")
        curr = exc["current"]
        
        if "density" in curr:
            j = curr["density"]
            lines.append(f"J = vector({j[0]:.9g}, {j[1]:.9g}, {j[2]:.9g})")
        
        if "Pol" in curr:
            lines.append(f"Pol = {curr['Pol']:.9g}")
        
        if "xi" in curr:  # Zhang-Li non-adiabaticity
            lines.append(f"xi = {curr['xi']:.9g}")
        
        if "Lambda" in curr:  # Slonczewski Lambda
            lines.append(f"Lambda = {curr['Lambda']:.9g}")
        
        if "EpsilonPrime" in curr:
            lines.append(f"epsilonprime = {curr['EpsilonPrime']:.9g}")
        
        if "fixed_layer" in curr:
            fl = curr["fixed_layer"]
            lines.append(f"fixedlayer = vector({fl[0]:.9g}, {fl[1]:.9g}, {fl[2]:.9g})")
    
    return lines


def _generate_extensions(extensions: List[Dict[str, Any]], comments: bool) -> List[str]:
    """Generate extension function calls."""
    lines = []
    if comments:
        lines.append("// Extensions")
    
    for ext in extensions:
        name = ext["name"]
        params = ext.get("params", {})
        
        # Build parameter string
        param_list = []
        for k, v in params.items():
            if isinstance(v, (int, float)):
                param_list.append(f"{v:.9g}")
            elif isinstance(v, str):
                param_list.append(f'"{v}"')
            elif isinstance(v, list):
                param_list.append(f"vector({', '.join(f'{x:.9g}' for x in v)})")
            else:
                param_list.append(str(v))
        
        lines.append(f"{name}({', '.join(param_list)})")
    
    return lines


def _generate_output(output: Dict[str, Any], comments: bool) -> List[str]:
    """Generate output configuration."""
    lines = []
    if comments:
        lines.append("// Output configuration")
    
    # Autosave
    if "autosave" in output:
        for item in output["autosave"]:
            qty = item["quantity"]
            period = item["period"]
            lines.append(f"autosave({qty}, {period:.9g})")
    
    # Table quantities
    if "table_quantities" in output:
        for qty in output["table_quantities"]:
            lines.append(f"TableAdd({qty})")
    
    if "table_period" in output:
        lines.append(f"tableautosave({output['table_period']:.9g})")
    
    # Snapshots
    if "snapshots" in output:
        for qty in output["snapshots"]:
            lines.append(f"save({qty})")
    
    return lines


def _generate_run(ir: Dict[str, Any], comments: bool) -> List[str]:
    """Generate solver/run commands."""
    lines = []
    if comments:
        lines.append("// Run simulation")
    
    sim_type = ir.get("simulation_type", "dynamics")
    run_params = ir.get("run_params", {})
    
    # Solver settings
    if "solver" in run_params:
        lines.append(f"SetSolver({run_params['solver']})")
    
    if "MaxErr" in run_params:
        lines.append(f"MaxErr = {run_params['MaxErr']:.9g}")
    
    if "FixDt" in run_params:
        lines.append(f"FixDt = {run_params['FixDt']:.9g}")
    
    # Main run command
    if sim_type == "relax":
        lines.append("relax()")
    
    elif sim_type == "minimize":
        if "MinimizerStop" in run_params:
            lines.append(f"MinimizerStop = {run_params['MinimizerStop']:.9g}")
        lines.append("minimize()")
    
    elif sim_type == "hysteresis":
        # Hysteresis loop
        loop = run_params.get("loop", {})
        var = loop.get("variable", "B_ext")
        start = loop.get("start", 0)
        stop = loop.get("stop", 0.1)
        step = loop.get("step", 0.001)
        direction = loop.get("direction", "both")
        
        if direction in ["up", "both"]:
            lines.append(f"for B := {start:.9g}; B <= {stop:.9g}; B += {step:.9g} {{")
            lines.append(f"    B_ext = vector(B, 0, 0)")
            lines.append(f"    minimize()")
            lines.append(f"    tablesave()")
            lines.append(f"}}")
        
        if direction == "both":
            lines.append(f"for B := {stop:.9g}; B >= {-stop:.9g}; B -= {step:.9g} {{")
            lines.append(f"    B_ext = vector(B, 0, 0)")
            lines.append(f"    minimize()")
            lines.append(f"    tablesave()")
            lines.append(f"}}")
            lines.append(f"for B := {-stop:.9g}; B <= {stop:.9g}; B += {step:.9g} {{")
            lines.append(f"    B_ext = vector(B, 0, 0)")
            lines.append(f"    minimize()")
            lines.append(f"    tablesave()")
            lines.append(f"}}")
    
    elif sim_type == "dynamics":
        duration = run_params.get("duration", 1e-9)
        lines.append(f"run({duration:.9g})")
    
    else:
        # Default: short run
        duration = run_params.get("duration", 1e-9)
        lines.append(f"run({duration:.9g})")
    
    return lines


def load_and_convert(ir_file: Path, output_file: Optional[Path] = None, 
                      add_comments: bool = True) -> str:
    """
    Load IR from JSON file and convert to MuMax3 script.
    
    Args:
        ir_file: Path to IR JSON file
        output_file: Optional output path for .mx3 script
        add_comments: Add explanatory comments
    
    Returns:
        Generated MuMax3 script
    """
    with open(ir_file) as f:
        ir = json.load(f)
    
    script = ir_to_script(ir, add_comments)
    
    if output_file:
        output_file.write_text(script)
        print(f"✓ Generated {output_file}")
    
    return script


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ir_to_mumax3.py <ir.json> [output.mx3]")
        print("   or: python ir_to_mumax3.py --test")
        sys.exit(1)
    
    if sys.argv[1] == "--test":
        # Test with minimal example
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
                "params": {"mx": 1, "my": 0.1, "mz": 0}
            },
            "simulation_type": "relax"
        }
        
        script = ir_to_script(test_ir)
        print(script)
        print("\n✓ Test conversion successful")
    
    else:
        ir_path = Path(sys.argv[1])
        out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        load_and_convert(ir_path, out_path)
