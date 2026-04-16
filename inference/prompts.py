"""
System and user prompts for each inference stage:
intent classification, entity extraction, IR completion, code generation, and self-repair.
"""

# ========== INTENT CLASSIFICATION ==========
INTENT_SYSTEM_ZERO_SHOT = """You are a classifier for micromagnetic simulations.
Classify the user's natural language description into exactly one of these categories:
simple_relax, field_driven, dmi_skyrmion, stt, sot, thermal, multi_region, fft_analysis.
Output only the category name, nothing else."""

INTENT_FEW_SHOT_EXAMPLES = """Examples:
Input: "Relax the magnetisation to equilibrium without any external field"
Output: simple_relax

Input: "Apply a 50 mT magnetic pulse to a Permalloy nanowire and watch domain wall motion"
Output: field_driven

Input: "Simulate skyrmion nucleation using interfacial DMI and SOT"
Output: dmi_skyrmion

Input: "Spin‑transfer torque switching of a CoFeB nanodot"
Output: stt

Input: "Spin‑orbit torque driven domain wall motion"
Output: sot

Input: "Thermal stability of a magnetic bit at 400 K"
Output: thermal

Input: "Multi‑region simulation with two different materials"
Output: multi_region

Input: "Compute the ferromagnetic resonance spectrum"
Output: fft_analysis
"""

def get_intent_prompt(user_input: str, mode: str):
    """Generate system and user prompts for intent classification."""
    if mode == "zero_shot":
        return INTENT_SYSTEM_ZERO_SHOT, user_input
    elif mode == "few_shot":
        # Combine examples + current input
        prompt = f"{INTENT_FEW_SHOT_EXAMPLES}\n\nNow classify this input:\n{user_input}\nOutput only the category name:"
        return "", prompt
    else:  # fine_tuned
        return "", f"Classify the simulation intent:\n{user_input}"

# ========== ENTITY EXTRACTION ==========
ENTITY_SYSTEM_ZERO_SHOT = """You are an entity extractor for micromagnetic simulations.
Extract the following fields from the user's description and return a JSON object.
Fields: material (string), geometry_shape (rectangle|disk|cylinder|ellipse|wire),
size_x_nm (float), size_y_nm (float), size_z_nm (float),
applied_field_B_mT (float, optional), current_density_JA_m2 (float, optional),
temperature_K (float, optional), damping_alpha (float, optional).
If a field is not mentioned, omit it. Output only valid JSON."""

ENTITY_FEW_SHOT_EXAMPLES = """
Example 1:
Input: "Simulate a Permalloy nanowire of size 512x128x5 nm with a 10 mT field along x"
Output: {"material": "Permalloy", "geometry_shape": "wire", "size_x_nm": 512, "size_y_nm": 128, "size_z_nm": 5, "applied_field_B_mT": 10}

Example 2:
Input: "Relax a CoFeB disk of diameter 200 nm and thickness 10 nm, damping 0.01"
Output: {"material": "CoFeB", "geometry_shape": "disk", "size_x_nm": 200, "size_y_nm": 200, "size_z_nm": 10, "damping_alpha": 0.01}
"""

def get_entity_prompt(user_input: str, category: str, mode: str):
    """Generate system and user prompts for entity extraction."""
    base_prompt = f"Simulation category: {category}\nUser description: {user_input}\nExtract entities as JSON."
    if mode == "zero_shot":
        return ENTITY_SYSTEM_ZERO_SHOT, base_prompt
    elif mode == "few_shot":
        full_prompt = f"{ENTITY_FEW_SHOT_EXAMPLES}\n\nNow extract from:\n{user_input}\nCategory: {category}\nOutput JSON only:"
        return "", full_prompt
    else:
        return "", base_prompt

# ========== IR COMPLETION (optional) ==========
IR_COMPLETION_SYSTEM = """You are a physics-aware assistant that fills missing fields in a micromagnetic simulation intermediate representation (IR).
Given a partial IR (JSON), add reasonable default values for missing required fields based on the simulation category and existing fields.
Use standard material parameters from the MuMax3 ontology. Return the complete IR as JSON."""

def get_ir_completion_prompt(partial_ir_json: str, category: str):
    """Generate system and user prompts for IR completion."""
    user_prompt = f"""Partial IR:
{partial_ir_json}

Category: {category}

Add missing fields (e.g., Msat_kApm, Aex_pJm, alpha, cell_size_nm, grid_nx, grid_ny, grid_nz, t_total_ns, dt_save_ps).
Output complete IR as JSON only."""
    return IR_COMPLETION_SYSTEM, user_prompt

# ========== CODE GENERATION (IR → MuMax3) ==========
CODEGEN_SYSTEM_ZERO_SHOT = """You are an expert in MuMax3 scripting.
Convert the given simulation IR (JSON) into a complete, executable MuMax3 .mx3 script.
Follow MuMax3 syntax: SetGridSize, SetCellSize, SetGeom, material parameters (Msat, Aex, alpha),
enable physics (EnableDemag, Dind, etc.), set initial magnetisation, run relax() if needed,
apply excitations, configure outputs (autosave, tableadd), and run dynamics.
Output only the MuMax3 script, no explanations."""

CODEGEN_FEW_SHOT_EXAMPLE = """
Example IR:
{"domain": {"grid": {"nx": 128, "ny": 128, "nz": 1}, "cell": {"dx": 1e-9, "dy": 1e-9, "dz": 1e-9}}, "materials": {"global": {"Msat": 8e5, "Aex": 13e-12, "alpha": 0.01}}, "physics": {"enabled": ["Exchange","Demag"], "parameters": {}}, "solver": {"mode": "relax", "duration": "1e-9"}}

Example output:
SetGridSize(128, 128, 1)
SetCellSize(1e-9, 1e-9, 1e-9)
SetGeom(cylinder(100e-9, 100e-9, 5e-9))
Msat = 800e3
Aex = 13e-12
alpha = 0.01
EnableDemag = true
relax()
"""

def get_codegen_prompt(ir_json: dict, mode: str):
    """Generate system and user prompts for MuMax3 code generation."""
    import json
    ir_str = json.dumps(ir_json, indent=2)
    if mode == "zero_shot":
        return CODEGEN_SYSTEM_ZERO_SHOT, f"IR:\n{ir_str}\n\nGenerate MuMax3 script:"
    elif mode == "few_shot":
        full_prompt = f"{CODEGEN_FEW_SHOT_EXAMPLE}\n\nNow generate script for this IR:\n{ir_str}\n\nMuMax3 script:"
        return "", full_prompt
    else:  # fine_tuned
        return "", f"IR:\n{ir_str}\n\nMuMax3 script:"

# ========== SELF-REPAIR ==========
REPAIR_SYSTEM = """You are a MuMax3 expert. The following generated script failed static analysis.
Errors are listed below. Fix the script to resolve all errors.
Output only the corrected MuMax3 script, no explanations."""

def get_repair_prompt(script: str, errors: list):
    """Generate system and user prompts for script repair."""
    user_prompt = f"""Original script:
{script}

Errors:
{chr(10).join(errors)}

Corrected script:"""
    return REPAIR_SYSTEM, user_prompt
