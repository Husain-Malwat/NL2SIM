import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class MuMax3Parser:
    """Parser for MuMax3 scripts to IR."""
    
    def __init__(self, script: str):
        self.script = script
        self.lines = script.split('\n')
        self.ir: Dict[str, Any] = {}
        self.warnings: List[str] = []
    
    def parse(self) -> Dict[str, Any]:
        """Parse the script and return IR."""
        # Initialize IR structure
        self.ir = {
            "mesh": {},
            "materials": [],
            "initial_config": {},
            "simulation_type": "dynamics",
        }
        
        # Remove comments for easier parsing
        self.clean_script = self._remove_comments(self.script)
        
        # Parse each section
        self._parse_mesh()
        self._parse_geometry()
        self._parse_materials()
        self._parse_initial_config()
        self._parse_excitation()
        self._parse_extensions()
        self._parse_output()
        self._parse_run_params()
        
        return self.ir
    
    def _remove_comments(self, text: str) -> str:
        """Remove // and /* */ comments."""
        # Remove single-line comments
        text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
        # Remove multi-line comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text
    
    def _parse_mesh(self):
        """Extract mesh configuration."""
        # SetGridsize/SetGridSize (case-insensitive)
        grid_match = re.search(r'SetGridsize\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', 
                               self.clean_script, re.IGNORECASE)
        if grid_match:
            self.ir["mesh"]["grid"] = [int(grid_match.group(1)), 
                                       int(grid_match.group(2)), 
                                       int(grid_match.group(3))]
        
        # SetCellsize (case-insensitive)
        cell_match = re.search(r'SetCellsize\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)', 
                               self.clean_script, re.IGNORECASE)
        if cell_match:
            try:
                dx = self._eval_numeric(cell_match.group(1))
                dy = self._eval_numeric(cell_match.group(2))
                dz = self._eval_numeric(cell_match.group(3))
                self.ir["mesh"]["cell_size"] = [dx, dy, dz]
            except:
                self.warnings.append(f"Could not parse cell size: {cell_match.group(0)}")
        
        # SetPBC
        pbc_match = re.search(r'SetPBC\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', 
                              self.clean_script, re.IGNORECASE)
        if pbc_match:
            self.ir["mesh"]["pbc"] = [int(pbc_match.group(1)), 
                                      int(pbc_match.group(2)), 
                                      int(pbc_match.group(3))]
        
        # EdgeSmooth
        edge_match = re.search(r'EdgeSmooth\s*=\s*(\d+)', self.clean_script, re.IGNORECASE)
        if edge_match:
            self.ir["mesh"]["edge_smooth"] = int(edge_match.group(1))
    
    def _parse_geometry(self):
        """Extract geometry definitions."""
        geom = {}
        
        # setgeom() calls
        setgeom_matches = re.findall(r'setgeom\s*\(([^)]+)\)', self.clean_script, re.IGNORECASE)
        if setgeom_matches:
            # Take the last one (most recent geometry setting)
            geom["shape"] = setgeom_matches[-1].strip()
        
        # defregion() calls
        defregion_matches = re.findall(r'defregion\s*\(\s*(\d+)\s*,\s*([^)]+)\)', 
                                        self.clean_script, re.IGNORECASE)
        if defregion_matches:
            regions = []
            for rid, shape_expr in defregion_matches:
                regions.append({
                    "id": int(rid),
                    "shape_expr": shape_expr.strip()
                })
            geom["regions"] = regions
        
        if geom:
            self.ir["geometry"] = geom
    
    def _parse_materials(self):
        """Extract material parameters."""
        materials = []
        
        # Global parameters (region 0)
        global_mat = {"region_id": 0}
        
        # Msat (required)
        msat = self._find_param_value(r'Msat\s*=\s*([^\s;]+)', case_insensitive=True)
        if msat is not None:
            global_mat["Msat"] = msat
        
        # Aex (required)
        aex = self._find_param_value(r'Aex\s*=\s*([^\s;]+)', case_insensitive=True)
        if aex is not None:
            global_mat["Aex"] = aex
        
        # alpha (required)
        alpha = self._find_param_value(r'alpha\s*=\s*([^\s;]+)', case_insensitive=True)
        if alpha is not None:
            global_mat["alpha"] = alpha
        
        # Optional parameters
        ku1 = self._find_param_value(r'Ku1\s*=\s*([^\s;]+)', case_insensitive=True)
        if ku1 is not None:
            global_mat["Ku1"] = ku1
        
        ku2 = self._find_param_value(r'Ku2\s*=\s*([^\s;]+)', case_insensitive=True)
        if ku2 is not None:
            global_mat["Ku2"] = ku2
        
        kc1 = self._find_param_value(r'Kc1\s*=\s*([^\s;]+)', case_insensitive=True)
        if kc1 is not None:
            global_mat["Kc1"] = kc1
        
        dind = self._find_param_value(r'Dind\s*=\s*([^\s;]+)', case_insensitive=True)
        if dind is not None:
            global_mat["Dind"] = dind
        
        dbulk = self._find_param_value(r'Dbulk\s*=\s*([^\s;]+)', case_insensitive=True)
        if dbulk is not None:
            global_mat["Dbulk"] = dbulk
        
        temp = self._find_param_value(r'Temp\s*=\s*([^\s;]+)', case_insensitive=True)
        if temp is not None:
            global_mat["Temp"] = temp
        
        # AnisU vector
        anisu = self._find_vector_value(r'AnisU\s*=\s*vector\s*\(([^)]+)\)', case_insensitive=True)
        if anisu:
            global_mat["AnisU"] = anisu
        
        if len(global_mat) > 1:  # More than just region_id
            materials.append(global_mat)
        
        # Per-region parameters
        region_params = self._find_region_params()
        materials.extend(region_params)
        
        self.ir["materials"] = materials
    
    def _parse_initial_config(self):
        """Extract initial magnetization configuration."""
        init_config = {}
        
        # Look for m = ... patterns
        # uniform
        uniform_match = re.search(r'm\s*=\s*uniform\s*\(\s*([^,)]+)\s*,\s*([^,)]+)\s*,\s*([^)]+)\s*\)', 
                                  self.clean_script, re.IGNORECASE)
        if uniform_match:
            try:
                mx = self._eval_numeric(uniform_match.group(1))
                my = self._eval_numeric(uniform_match.group(2))
                mz = self._eval_numeric(uniform_match.group(3))
                init_config = {
                    "type": "uniform",
                    "params": {"mx": mx, "my": my, "mz": mz}
                }
            except:
                init_config = {"type": "uniform", "params": {}}
        
        # vortex
        vortex_match = re.search(r'm\s*=\s*vortex\s*\(\s*([^,)]+)\s*,\s*([^)]+)\s*\)', 
                                 self.clean_script, re.IGNORECASE)
        if vortex_match:
            try:
                circ = int(self._eval_numeric(vortex_match.group(1)))
                pol = int(self._eval_numeric(vortex_match.group(2)))
                init_config = {
                    "type": "vortex",
                    "params": {"circulation": circ, "polarization": pol}
                }
            except:
                init_config = {"type": "vortex", "params": {}}
        
        # twodomain
        twodomain_match = re.search(r'm\s*=\s*twodomain\s*\(([^)]+)\)', 
                                    self.clean_script, re.IGNORECASE)
        if twodomain_match:
            init_config = {"type": "two_domain", "params": {}}
        
        # randomMag
        if re.search(r'm\s*=\s*randomMag\s*\(\s*\)', self.clean_script, re.IGNORECASE):
            init_config = {"type": "random", "params": {}}
        
        # BlochSkyrmion
        bloch_match = re.search(r'm\s*=\s*BlochSkyrmion\s*\(\s*([^,)]+)\s*,\s*([^)]+)\s*\)', 
                                self.clean_script, re.IGNORECASE)
        if bloch_match:
            try:
                circ = int(self._eval_numeric(bloch_match.group(1)))
                pol = int(self._eval_numeric(bloch_match.group(2)))
                init_config = {
                    "type": "skyrmion_bloch",
                    "params": {"circulation": circ, "polarization": pol}
                }
            except:
                init_config = {"type": "skyrmion_bloch", "params": {}}
        
        # NeelSkyrmion
        neel_match = re.search(r'm\s*=\s*NeelSkyrmion\s*\(\s*([^,)]+)\s*,\s*([^)]+)\s*\)', 
                               self.clean_script, re.IGNORECASE)
        if neel_match:
            try:
                circ = int(self._eval_numeric(neel_match.group(1)))
                pol = int(self._eval_numeric(neel_match.group(2)))
                init_config = {
                    "type": "skyrmion_neel",
                    "params": {"circulation": circ, "polarization": pol}
                }
            except:
                init_config = {"type": "skyrmion_neel", "params": {}}
        
        # LoadFile
        loadfile_match = re.search(r'm\.LoadFile\s*\(\s*["\']([^"\']+)["\']\s*\)', 
                                   self.clean_script, re.IGNORECASE)
        if loadfile_match:
            init_config = {
                "type": "from_file",
                "params": {"filename": loadfile_match.group(1)}
            }
        
        if not init_config:
            init_config = {"type": "uniform", "params": {"mx": 1, "my": 0, "mz": 0}}
        
        self.ir["initial_config"] = init_config
    
    def _parse_excitation(self):
        """Extract excitations (B_ext, current)."""
        exc = {}
        
        # B_ext
        b_ext_static = self._find_vector_value(r'B_ext\s*=\s*vector\s*\(([^)]+)\)', case_insensitive=True)
        if b_ext_static:
            exc["B_ext"] = {"static": b_ext_static}
        
        # Time-dependent B_ext (look for sin, cos, t in expression)
        b_ext_time = re.search(r'B_ext\s*=\s*(.+?)(?:\n|$)', self.clean_script, re.IGNORECASE)
        if b_ext_time and ('sin' in b_ext_time.group(1).lower() or 
                           'cos' in b_ext_time.group(1).lower() or 
                           '*t' in b_ext_time.group(1)):
            if "B_ext" not in exc:
                exc["B_ext"] = {}
            exc["B_ext"]["time_dependent"] = b_ext_time.group(1).strip()
        
        # Current density J
        j_value = self._find_vector_value(r'J\s*=\s*vector\s*\(([^)]+)\)', case_insensitive=True)
        if j_value:
            curr = {"density": j_value}
            
            # STT/SOT parameters
            pol = self._find_param_value(r'Pol\s*=\s*([^\s;]+)', case_insensitive=True)
            if pol is not None:
                curr["Pol"] = pol
            
            xi = self._find_param_value(r'xi\s*=\s*([^\s;]+)', case_insensitive=True)
            if xi is not None:
                curr["xi"] = xi
            
            lambda_val = self._find_param_value(r'Lambda\s*=\s*([^\s;]+)', case_insensitive=True)
            if lambda_val is not None:
                curr["Lambda"] = lambda_val
            
            eps = self._find_param_value(r'epsilonprime\s*=\s*([^\s;]+)', case_insensitive=True)
            if eps is not None:
                curr["EpsilonPrime"] = eps
            
            fixed = self._find_vector_value(r'fixedlayer\s*=\s*vector\s*\(([^)]+)\)', case_insensitive=True)
            if fixed:
                curr["fixed_layer"] = fixed
            
            exc["current"] = curr
        
        if exc:
            self.ir["excitation"] = exc
    
    def _parse_extensions(self):
        """Extract extension function calls."""
        extensions = []
        
        # ext_* function patterns
        ext_pattern = r'(ext_\w+)\s*\(([^)]*)\)'
        for match in re.finditer(ext_pattern, self.clean_script, re.IGNORECASE):
            func_name = match.group(1)
            args = match.group(2)
            extensions.append({
                "name": func_name,
                "params": {}  # TODO: parse args if needed
            })
        
        if extensions:
            self.ir["extensions"] = extensions
    
    def _parse_output(self):
        """Extract output configuration."""
        output = {}
        
        # autosave
        autosave_matches = re.findall(r'autosave\s*\(\s*(\w+)\s*,\s*([^)]+)\s*\)', 
                                       self.clean_script, re.IGNORECASE)
        if autosave_matches:
            autosaves = []
            for qty, period in autosave_matches:
                try:
                    period_val = self._eval_numeric(period)
                    autosaves.append({"quantity": qty, "period": period_val})
                except:
                    pass
            if autosaves:
                output["autosave"] = autosaves
        
        # TableAdd
        tableadd_matches = re.findall(r'TableAdd\s*\(\s*(\w+)\s*\)', 
                                       self.clean_script, re.IGNORECASE)
        if tableadd_matches:
            output["table_quantities"] = tableadd_matches
        
        # tableautosave
        table_period = self._find_param_value(r'tableautosave\s*\(\s*([^)]+)\s*\)', case_insensitive=True)
        if table_period is not None:
            output["table_period"] = table_period
        
        # save() snapshots
        save_matches = re.findall(r'save\s*\(\s*(\w+)\s*\)', self.clean_script, re.IGNORECASE)
        if save_matches:
            output["snapshots"] = list(set(save_matches))  # Deduplicate
        
        if output:
            self.ir["output"] = output
    
    def _parse_run_params(self):
        """Extract run/solver parameters and determine simulation type."""
        run_params = {}
        
        # Check for relax()
        if re.search(r'\brelax\s*\(\s*\)', self.clean_script, re.IGNORECASE):
            self.ir["simulation_type"] = "relax"
        
        # Check for minimize()
        elif re.search(r'\bminimize\s*\(\s*\)', self.clean_script, re.IGNORECASE):
            self.ir["simulation_type"] = "minimize"
            minimizer_stop = self._find_param_value(r'MinimizerStop\s*=\s*([^\s;]+)', case_insensitive=True)
            if minimizer_stop is not None:
                run_params["MinimizerStop"] = minimizer_stop
        
        # Check for run()
        run_match = re.search(r'\brun\s*\(\s*([^)]+)\s*\)', self.clean_script, re.IGNORECASE)
        if run_match:
            self.ir["simulation_type"] = "dynamics"
            try:
                duration = self._eval_numeric(run_match.group(1))
                run_params["duration"] = duration
            except:
                pass
        
        # Check for hysteresis loop pattern
        if re.search(r'for\s+\w+\s*:=.*?B_ext.*?minimize\s*\(\s*\)', self.clean_script, re.IGNORECASE | re.DOTALL):
            self.ir["simulation_type"] = "hysteresis"
        
        # Solver settings
        solver_match = re.search(r'SetSolver\s*\(\s*(\d+)\s*\)', self.clean_script, re.IGNORECASE)
        if solver_match:
            run_params["solver"] = int(solver_match.group(1))
        
        maxerr = self._find_param_value(r'MaxErr\s*=\s*([^\s;]+)', case_insensitive=True)
        if maxerr is not None:
            run_params["MaxErr"] = maxerr
        
        fixdt = self._find_param_value(r'FixDt\s*=\s*([^\s;]+)', case_insensitive=True)
        if fixdt is not None:
            run_params["FixDt"] = fixdt
        
        if run_params:
            self.ir["run_params"] = run_params
    
    def _find_param_value(self, pattern: str, case_insensitive: bool = False) -> Optional[float]:
        """Find and evaluate a parameter value."""
        flags = re.IGNORECASE if case_insensitive else 0
        match = re.search(pattern, self.clean_script, flags)
        if match:
            try:
                return self._eval_numeric(match.group(1))
            except:
                return None
        return None
    
    def _find_vector_value(self, pattern: str, case_insensitive: bool = False) -> Optional[List[float]]:
        """Find and parse a vector value."""
        flags = re.IGNORECASE if case_insensitive else 0
        match = re.search(pattern, self.clean_script, flags)
        if match:
            try:
                components = match.group(1).split(',')
                return [self._eval_numeric(c.strip()) for c in components]
            except:
                return None
        return None
    
    def _find_region_params(self) -> List[Dict[str, Any]]:
        """Find per-region parameter settings."""
        regions = {}
        
        # Pattern: Param.SetRegion(id, value)
        region_pattern = r'(\w+)\.SetRegion\s*\(\s*(\d+)\s*,\s*([^)]+)\s*\)'
        for match in re.finditer(region_pattern, self.clean_script, re.IGNORECASE):
            param = match.group(1)
            rid = int(match.group(2))
            value_str = match.group(3)
            
            if rid not in regions:
                regions[rid] = {"region_id": rid}
            
            try:
                # Check if it's a vector
                if 'vector' in value_str.lower():
                    vec_match = re.search(r'vector\s*\(([^)]+)\)', value_str, re.IGNORECASE)
                    if vec_match:
                        components = vec_match.group(1).split(',')
                        regions[rid][param] = [self._eval_numeric(c.strip()) for c in components]
                else:
                    regions[rid][param] = self._eval_numeric(value_str)
            except:
                pass
        
        return list(regions.values())
    
    def _eval_numeric(self, expr: str) -> float:
        """Safely evaluate a numeric expression."""
        expr = expr.strip()
        
        # Replace common MuMax3 constants
        expr = expr.replace('pi', '3.14159265358979')
        expr = expr.replace('mu0', '1.2566370614359173e-6')
        
        # Handle scientific notation: 1e-9, 1.5e9, etc.
        # Handle arithmetic: *, /, +, -, ()
        # SECURITY: Only allow safe characters
        if not re.match(r'^[0-9eE+\-*/.()\s]+$', expr):
            raise ValueError(f"Unsafe expression: {expr}")
        
        try:
            return float(eval(expr))
        except:
            # If eval fails, try simple float parsing
            return float(expr)


def script_to_ir(script: str) -> Dict[str, Any]:
    """
    Parse MuMax3 script to IR.
    
    Args:
        script: MuMax3 script as string
    
    Returns:
        IR dictionary
    """
    parser = MuMax3Parser(script)
    ir = parser.parse()
    
    if parser.warnings:
        ir["_warnings"] = parser.warnings
    
    return ir


def load_and_parse(script_file: Path, output_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load MuMax3 script and parse to IR.
    
    Args:
        script_file: Path to .mx3 file
        output_file: Optional output path for IR JSON
    
    Returns:
        IR dictionary
    """
    script = script_file.read_text()
    ir = script_to_ir(script)
    
    if output_file:
        output_file.write_text(json.dumps(ir, indent=2))
        print(f"✓ Parsed to {output_file}")
    
    return ir


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mumax3_to_ir.py <script.mx3> [output.json]")
        print("   or: python mumax3_to_ir.py --test")
        sys.exit(1)
    
    if sys.argv[1] == "--test":
        # Test with minimal script
        test_script = """\
SetGridsize(128, 64, 1)
SetCellsize(4e-9, 4e-9, 10e-9)

Msat  = 800e3
Aex   = 13e-12
alpha = 0.02

m = uniform(1, 0.1, 0)
relax()
"""
        ir = script_to_ir(test_script)
        print(json.dumps(ir, indent=2))
        print("\n✓ Test parsing successful")
    
    else:
        script_path = Path(sys.argv[1])
        out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        load_and_parse(script_path, out_path)
