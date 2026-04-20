# NL2SIM: Natural Language to Micromagnetic Simulation 

A complete end-to-end system for converting natural language descriptions into executable MuMax3 micromagnetic simulation scripts using a Unsloth-based language model.


## Features

  1. Intent Classification (detect simulation type)
  2. Entity Extraction (parse parameters)
  3. IR Construction & Completion (build intermediate representation)
  4. Code Generation (IR → MuMax3 script)
  5. Static Analysis & Self-Repair (validate and fix scripts)
- **Multiple Inference Modes**: Zero-shot, few-shot, and fine-tuned prompts
- **Physics-Aware Validation**: Checks parameter ranges, API symbols, and exchange length constraints
- **Self-Repair Loop**: Automatically fixes generated scripts using LLM feedback

## Supported Simulation Intents

The pipeline can classify and generate scripts for:

1. **simple_relax**: Magnetization relaxation to equilibrium
2. **field_driven**: Domain wall motion under external field
3. **dmi_skyrmion**: Skyrmion nucleation with DMI and SOT
4. **stt**: Spin-transfer torque switching
5. **sot**: Spin-orbit torque dynamics
6. **thermal**: Thermal stability analysis
7. **multi_region**: Multi-material simulations
8. **fft_analysis**: Frequency analysis (resonance spectra)

## Inference Modes

### Zero-Shot
Direct instructions without examples. Good for general descriptions matching standard patterns.

### Few-Shot (Recommended)
Includes 2-3 examples in the prompt. Better for parameter extraction and code generation accuracy.

### Fine-Tuned
Assumes model fine-tuned on NL2SIM training data. Best accuracy but requires pre-training.

## Example Queries

```python
# Simple relaxation
"Relax the magnetisation to equilibrium without any external field"

# Field-driven dynamics
"Apply a 50 mT magnetic pulse to a Permalloy nanowire (512x128x5 nm) along x-direction"

# Skyrmion simulation
"Simulate skyrmion nucleation using interfacial DMI (2 mJ/m²) and SOT in a CoFeB disk"

# Multi-region
"Simulate magnetization dynamics in a two-region structure: Permalloy + YIG interface"

# With parameters
"Simulate a CoFeB nanodot (diameter 100 nm, thickness 10 nm) with damping 0.005"
```