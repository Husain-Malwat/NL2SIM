"""
IR (Intermediate Representation) builder.
Constructs simulation IR from intent and extracted entities using rule-based logic.
"""
from typing import Dict, Any

def build_initial_ir(intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rule-based IR construction from intent and extracted entities.
    Creates a complete intermediate representation suitable for code generation.
    """
    ir = {
        "intent_summary": intent,
        "domain": {
            "grid": {"nx": 64, "ny": 64, "nz": 1},
            "cell": {"dx": 2e-9, "dy": 2e-9, "dz": 2e-9}
        },
        "geometry": {
            "shapes": ["rectangle"],
            "operations": [],
            "regions": []
        },
        "materials": {
            "global": {},
            "per_region": []
        },
        "physics": {
            "enabled": ["Exchange"],
            "parameters": {}
        },
        "excitations": [],
        "initialization": {
            "type": "uniform",
            "params": {"direction": [1, 0, 0]}
        },
        "solver": {
            "mode": "run",
            "duration": "1e-9",
            "controls": {}
        },
        "outputs": [
            {
                "quantity": "m",
                "schedule": "autosave",
                "period": "100e-12"
            }
        ]
    }

    # Material parameters database (simplified lookup)
    material_db = {
        "permalloy": {"Msat": 800e3, "Aex": 13e-12, "alpha": 0.01},
        "cofeb": {"Msat": 1100e3, "Aex": 20e-12, "alpha": 0.005},
        "cobalt": {"Msat": 1400e3, "Aex": 30e-12, "alpha": 0.01},
        "yig": {"Msat": 140e3, "Aex": 3.7e-12, "alpha": 0.0002},
    }
    
    # Set material from database or use default
    if "material" in entities:
        mat = entities["material"].lower()
        if mat in material_db:
            ir["materials"]["global"] = material_db[mat].copy()
        else:
            # Default values for unknown material
            ir["materials"]["global"] = {"Msat": 800e3, "Aex": 13e-12, "alpha": 0.01}
    else:
        ir["materials"]["global"] = {"Msat": 800e3, "Aex": 13e-12, "alpha": 0.01}

    # Override material parameters with extracted values
    if "damping_alpha" in entities:
        ir["materials"]["global"]["alpha"] = entities["damping_alpha"]
    
    if "applied_field_B_mT" in entities:
        ir["physics"]["enabled"].append("Zeeman")
        ir["physics"]["parameters"]["B_ext"] = [entities["applied_field_B_mT"] * 1e-3, 0, 0]
    
    # Update grid and cell size based on geometry
    if "size_x_nm" in entities:
        nx = int(entities["size_x_nm"] / 2)  # assume 2 nm cells
        ir["domain"]["grid"]["nx"] = max(1, nx)
        ir["domain"]["cell"]["dx"] = entities["size_x_nm"] * 1e-9 / max(1, nx)
    
    if "size_y_nm" in entities:
        ny = int(entities["size_y_nm"] / 2)
        ir["domain"]["grid"]["ny"] = max(1, ny)
        ir["domain"]["cell"]["dy"] = entities["size_y_nm"] * 1e-9 / max(1, ny)
    
    if "size_z_nm" in entities:
        nz = int(entities["size_z_nm"] / 2)
        ir["domain"]["grid"]["nz"] = max(1, nz)
        ir["domain"]["cell"]["dz"] = entities["size_z_nm"] * 1e-9 / max(1, nz)
    
    # Set geometry shape
    if "geometry_shape" in entities:
        ir["geometry"]["shapes"] = [entities["geometry_shape"]]

    # Intent-specific physics and parameters
    if intent == "simple_relax":
        ir["solver"]["mode"] = "relax"
    
    elif intent == "field_driven":
        ir["physics"]["enabled"].append("Zeeman")
        if "applied_field_B_mT" not in entities:
            ir["physics"]["parameters"]["B_ext"] = [50e-3, 0, 0]  # default 50 mT
    
    elif intent == "dmi_skyrmion":
        ir["physics"]["enabled"].extend(["DMI", "Anisotropy"])
        ir["physics"]["parameters"]["Dind"] = 2e-3   # J/m²
        ir["physics"]["parameters"]["Ku1"] = 700e3   # J/m³
    
    elif intent == "stt":
        ir["physics"]["enabled"].append("STT")
        if "current_density_JA_m2" in entities:
            ir["physics"]["parameters"]["Jc"] = entities["current_density_JA_m2"]
        else:
            ir["physics"]["parameters"]["Jc"] = 1e12  # default J/m²
    
    elif intent == "sot":
        ir["physics"]["enabled"].extend(["SOT", "Anisotropy"])
        ir["physics"]["parameters"]["Jc"] = 1e12  # default J/m²
        ir["physics"]["parameters"]["Ku1"] = 500e3
    
    elif intent == "thermal":
        ir["physics"]["enabled"].append("Thermal")
        if "temperature_K" in entities:
            ir["physics"]["parameters"]["Temp"] = entities["temperature_K"]
        else:
            ir["physics"]["parameters"]["Temp"] = 300  # Room temperature default
    
    elif intent == "multi_region":
        ir["geometry"]["operations"].append("multi_region")
        ir["materials"]["per_region"] = [
            {"region": 0, "params": ir["materials"]["global"].copy()},
            {"region": 1, "params": ir["materials"]["global"].copy()}
        ]
    
    elif intent == "fft_analysis":
        ir["solver"]["mode"] = "fft"
        ir["outputs"].append({"quantity": "B_ext_fft", "schedule": "once"})

    return ir
