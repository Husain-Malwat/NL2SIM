"""
End-to-end NL → MuMax3 pipeline.
Orchestrates intent classification, entity extraction, IR construction, 
code generation, and self-repair with static analysis.
"""
import json
from typing import Optional, List, Dict
from dataclasses import dataclass, field

import llm
from .prompts import (
    get_intent_prompt, get_entity_prompt, get_ir_completion_prompt,
    get_codegen_prompt, get_repair_prompt
)
from .static_analyzer import StaticAnalyzer
from .ir_builder import build_initial_ir
from .config import API_SYMBOLS, REQUIRED_SYMBOLS

@dataclass
class GenerationResult:
    """Encapsulates the output of a complete generation pipeline run."""
    success: bool
    intent: Optional[str] = None
    entities: Optional[Dict] = None
    ir: Optional[Dict] = None
    script: Optional[str] = None
    static_errors: List[str] = field(default_factory=list)
    repaired: bool = False
    final_script: Optional[str] = None

class MuMax3Pipeline:
    """
    Complete pipeline for converting natural language to MuMax3 scripts.
    Loads model once and reuses across multiple inference requests.
    """
    
    def __init__(self, mode: str = "few_shot",
                 use_llm_ir_completion: bool = True,
                 use_llm_codegen: bool = True,
                 max_repair_attempts: int = 1):
        """
        Initialize the pipeline.
        
        Args:
            mode: Inference mode - "zero_shot", "few_shot", or "fine_tuned"
            use_llm_ir_completion: Whether to use LLM for IR field completion
            use_llm_codegen: Whether to use LLM for code generation
            max_repair_attempts: Number of self-repair iterations for failed scripts
        """
        self.mode = mode
        self.use_llm_ir_completion = use_llm_ir_completion
        self.use_llm_codegen = use_llm_codegen
        self.max_repair_attempts = max_repair_attempts
        self.analyzer = StaticAnalyzer(API_SYMBOLS, REQUIRED_SYMBOLS)

    def run(self, user_input: str) -> GenerationResult:
        """
        Execute the full pipeline for a single user query.
        
        Args:
            user_input: Natural language description of the simulation
            
        Returns:
            GenerationResult with success flag, IR, script, and any errors
        """
        result = GenerationResult(success=False)

        # ---- Stage 1: Intent Classification ----
        system, user = get_intent_prompt(user_input, self.mode)
        intent_raw = llm.complete(user, system_prompt=system, temperature=0.0, max_tokens=20)
        result.intent = intent_raw.strip().lower()
        print(f"[Intent] {result.intent}")

        # ---- Stage 2: Entity Extraction ----
        system, user = get_entity_prompt(user_input, result.intent, self.mode)
        entities_raw = llm.complete(user, system_prompt=system, temperature=0.0, max_tokens=512)
        try:
            result.entities = json.loads(entities_raw)
        except json.JSONDecodeError:
            print(f"[Warning] Failed to parse entities JSON: {entities_raw[:100]}")
            result.entities = {}
        print(f"[Entities] {result.entities}")

        # ---- Stage 3: IR Construction & Completion ----
        ir = build_initial_ir(result.intent, result.entities)
        if self.use_llm_ir_completion:
            system, user = get_ir_completion_prompt(json.dumps(ir, indent=2), result.intent)
            completed_str = llm.complete(user, system_prompt=system, temperature=0.2, max_tokens=1024)
            try:
                completed = json.loads(completed_str)
                ir.update(completed)
            except json.JSONDecodeError:
                print(f"[Warning] Failed to parse IR completion JSON")
        result.ir = ir
        print(f"[IR] Created with {len(ir['materials']['global'])} material params")

        # ---- Stage 4: Code Generation ----
        if self.use_llm_codegen:
            system, user = get_codegen_prompt(ir, self.mode)
            script = llm.complete(user, system_prompt=system, temperature=0.2, max_tokens=2048)
            result.script = script
        else:
            # Fallback to template-based script (simplified)
            result.script = self._generate_template_script(ir)
        print(f"[CodeGen] Generated {len(result.script.splitlines())} lines")

        # ---- Stage 5: Static Analysis & Self-Repair ----
        errors = self.analyzer.validate(result.script)
        result.static_errors = errors
        final_script = result.script
        
        for attempt in range(self.max_repair_attempts):
            if not errors:
                break
            print(f"[Repair Attempt {attempt + 1}] Fixing {len(errors)} errors...")
            system, user = get_repair_prompt(final_script, errors)
            repaired = llm.complete(user, system_prompt=system, temperature=0.1, max_tokens=2048)
            final_script = repaired
            errors = self.analyzer.validate(final_script)
            result.repaired = True
        
        result.final_script = final_script
        result.success = len(errors) == 0
        
        if result.success:
            print("[Success] Script generated and validated.")
        else:
            print(f"[Failed] {len(errors)} validation errors remain:")
            for err in errors:
                print(f"  - {err}")
        
        return result

    def _generate_template_script(self, ir: Dict) -> str:
        """Generate a basic MuMax3 script from IR (fallback when LLM codegen is disabled)."""
        lines = [
            "// Generated MuMax3 script (template-based)",
            f"// Intent: {ir['intent_summary']}",
            ""
        ]
        
        # Grid and cell size
        grid = ir['domain']['grid']
        cell = ir['domain']['cell']
        lines.append(f"SetGridSize({grid['nx']}, {grid['ny']}, {grid['nz']})")
        lines.append(f"SetCellSize({cell['dx']:.2e}, {cell['dy']:.2e}, {cell['dz']:.2e})")
        lines.append("")
        
        # Material parameters
        mat = ir['materials']['global']
        if 'Msat' in mat:
            lines.append(f"Msat = {mat['Msat']:.2e}")
        if 'Aex' in mat:
            lines.append(f"Aex = {mat['Aex']:.2e}")
        if 'alpha' in mat:
            lines.append(f"alpha = {mat['alpha']}")
        lines.append("")
        
        # Physics
        if 'Exchange' in ir['physics']['enabled']:
            lines.append("EnableDemag = true")
        lines.append("")
        
        # Solver
        if ir['solver']['mode'] == 'relax':
            lines.append("relax()")
        else:
            duration = ir['solver'].get('duration', '1e-9')
            lines.append(f"run({duration})")
        
        return "\n".join(lines)
