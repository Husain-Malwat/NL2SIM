"""
Static analyzer for MuMax3 scripts.
Validates generated scripts for API symbols, required fields, and physics parameter ranges.
"""
import re
from typing import List, Set

class StaticAnalyzer:
    """Validates generated MuMax3 scripts against known API and physics constraints."""
    
    def __init__(self, api_symbols: Set[str], required_symbols: List[str]):
        self.api_symbols = api_symbols
        self.required_symbols = required_symbols

    def validate(self, script: str) -> List[str]:
        """
        Run all validation checks on the script.
        Returns a list of error strings (empty if valid).
        """
        errors = []
        
        # 1. Check API symbols (hallucination detection)
        tokens = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', script))
        # Keywords that are valid even if not in API
        keywords = {'if', 'else', 'for', 'func', 'true', 'false', 'nil', 'return', 'in', 'and', 'or'}
        unknown = tokens - self.api_symbols - keywords
        if unknown:
            errors.append(f"HALLUCINATION: Unknown API symbols: {', '.join(sorted(unknown))}")

        # 2. Check required symbols
        for req in self.required_symbols:
            if req not in script:
                errors.append(f"MISSING_REQUIRED: {req}")

        # 3. Basic range check for damping
        alpha_match = re.search(r'alpha\s*=\s*([0-9.eE+-]+)', script)
        if alpha_match:
            try:
                alpha = float(alpha_match.group(1))
                if alpha <= 0 or alpha > 1.0:
                    errors.append(f"PARAM_RANGE: alpha={alpha} outside (0,1]")
            except ValueError:
                pass

        # 4. Exchange length warning (simplified)
        msat_match = re.search(r'Msat\s*=\s*([0-9.eE+-]+)', script)
        aex_match = re.search(r'Aex\s*=\s*([0-9.eE+-]+)', script)
        cell_match = re.search(r'SetCellSize\s*\(\s*([0-9.eE+-]+)', script)
        if msat_match and aex_match and cell_match:
            try:
                Msat = float(msat_match.group(1))
                Aex = float(aex_match.group(1))
                dx = float(cell_match.group(1))
                mu0 = 4e-7 * 3.1415926535  # approx 1.2566e-6
                lex = (2 * Aex / (mu0 * Msat**2))**0.5
                if dx > lex:
                    errors.append(f"PHYSICS_WARNING: Cell size {dx:.2e} > exchange length {lex:.2e}")
            except (ValueError, ZeroDivisionError):
                pass

        return errors
