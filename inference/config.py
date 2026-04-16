"""
Configuration constants for the NL2SIM pipeline.
Includes MuMax3 API symbols, material database, and validation rules.
"""

# Complete set of MuMax3 API symbols (can be expanded)
API_SYMBOLS = {
    # Setup functions
    "SetGridSize", "SetCellSize", "SetGeom", "SetGeomGrid", "SetGeomFromFile",
    
    # Geometry primitives
    "cylinder", "ellipse", "xrange", "universe", "rect", "sphere", "box",
    
    # Material parameters
    "Msat", "Aex", "alpha", "Ku1", "Ku2", "anisU", "EnableDemag", 
    "Dind", "Dbulk", "Jc", "pol", "xi", "lambda", "epsilonprime", "Temp", "B_ext",
    
    # Solver functions
    "relax", "run", "RunWhile", "minimize", "Fixdt", "MinDt",
    
    # Output functions
    "autosave", "save", "tableadd", "tableautosave", "snapshot",
    
    # Physics probes
    "ext_topologicalcharge", "ext_dwpos", "ext_corepos", "m", "uniform", "vortex", "twoDomain",
    
    # Geometry operations
    "setInShape", "setgeom",
    
    # Control statements
    "true", "false", "nil"
}

# Symbols that must appear in every valid script
REQUIRED_SYMBOLS = ["SetGridSize", "SetCellSize", "Msat", "Aex", "alpha"]

# Material database with standard MuMax3 parameters
MATERIAL_DB = {
    "permalloy": {
        "Msat": 800e3,      # A/m
        "Aex": 13e-12,      # J/m
        "alpha": 0.01,      # Damping
    },
    "cofeb": {
        "Msat": 1100e3,
        "Aex": 20e-12,
        "alpha": 0.005,
    },
    "cobalt": {
        "Msat": 1400e3,
        "Aex": 30e-12,
        "alpha": 0.01,
    },
    "yig": {
        "Msat": 140e3,
        "Aex": 3.7e-12,
        "alpha": 0.0002,
    },
    "fe": {
        "Msat": 1700e3,
        "Aex": 21.4e-12,
        "alpha": 0.02,
    },
}

# Default simulation parameters
DEFAULT_GRID = {"nx": 64, "ny": 64, "nz": 1}
DEFAULT_CELL_SIZE = 2e-9  # 2 nm

# Physics parameter ranges (for validation)
PARAM_RANGES = {
    "alpha": (0, 1.0),          # Damping
    "Msat": (1e3, 2000e3),      # Saturation magnetization (A/m)
    "Aex": (1e-13, 100e-12),    # Exchange constant (J/m)
    "Ku1": (-2000e3, 2000e3),   # Uniaxial anisotropy (J/m³)
    "Temp": (0, 1000),          # Temperature (K)
    "Jc": (1e10, 1e13),         # Current density (A/m²)
}

# Exchange length calculation (Lex = sqrt(2*Aex / (mu0 * Msat^2)))
MU0 = 4e-7 * 3.14159265359  # H/m
