# Ablation Study Results   NL2Sim Component Contribution Analysis

## 1. Main Ablation Results

| Config | API Validity | Completeness | IR F1 | Static Valid | Hallucinat. Rate |
|--------|-------------|-------------|-------|-------------|-----------------|
| **A** Full Pipeline | **94.8%** | **82.7%** | **0.807** | **83.7%** | **5.2%** |
| **B** No Phys. Valid. | 94.2% | 81.9% | 0.801 | 82.1% | 5.6% |
| **C** No Static Valid. | 94.8% | 82.7% | 0.807 |   | 5.2% |
| **D** No IR | 81.3% | 68.9% | 0.623 | 61.2% | 18.7% |
| **E** Rule-Based | 87.4% | 71.3% | 0.641 | 72.9% | 9.4% |
| **F** LLM Code Gen | 88.6% | 79.1% | 0.794 | 74.3% | 11.4% |

---

## 2. Component Contribution Analysis

### 2.1 Value of the IR Stage 

**Interpretation**: The IR acts as a structured constraint that forces the LLM to commit 
to a typed representation before generating code. Without this commitment step, the LLM 
produces code that mixes valid and invalid API symbols unpredictably.

The hallucination rate is **3.6× higher** without the IR. This is the strongest single finding 
of the ablation study.

### 2.2 Value of LLM in Stage 1 (A vs E)

**Replacing LLM extraction with rule-based extraction causes significant degradation.**

**Interpretation**: The LLM is most valuable for:
1. Correctly identifying the simulation category (especially for ambiguous phrasing)
2. Extracting numerical parameter values from natural language units
3. Inferring implicit parameters ("Permalloy" → automatically fills in Msat=800kA/m, Aex=13pJ/m)

The rule-based extractor fails on:
- Paraphrased material names ("ferromagnetic strip" vs "nanowire")
- Non-standard unit expressions ("13 picojoules per meter" vs "13 pJ/m")
- Implicit category-to-physics mapping

### 2.3 Value of Template vs LLM Code Generation 

**The deterministic template generator outperforms LLM code generation.**

| Metric | Template (A) | LLM Codegen (F) | Drop |
|--------|-------------|-----------------|------|
| API Validity | 94.8% | 88.6% | −6.2pp |
| Static Valid | 83.7% | 74.3% | −9.4pp |
| Hallucinat. | 5.2% | 11.4% | +6.2pp |
| Completeness | 82.7% | 79.1% | −3.6pp |
| IR F1 | 0.807 | 0.794 | −0.013 |


**Caveat**: The LLM code generator may be superior for complex/unusual scenarios not 
covered by the template. The 5-point drop in completeness (79.1% vs 82.7%) for LLM 
codegen likely reflects the LLM generating more creative but incomplete scripts for 
edge cases where the template would have filled in all required fields automatically.

### 2.4 Value of Physics Validator (A vs B)

**The physics validator has a small but important effect.**

| Metric | With Phys. Valid. (A) | Without (B) | Drop |
|--------|----------------------|-------------|------|
| API Validity | 94.8% | 94.2% | −0.6pp |
| Static Valid | 83.7% | 82.1% | −1.6pp |
| Cell-size compliant | 97.8% | 87.0% | −10.8pp |

The physics validator's main impact is on **cell-size compliance** 
Without it, some generated IRs use cell sizes larger than the exchange length. These would produce scripts that run but generate numerically incorrect results.

The validator prevents 4.3% of samples from entering the corpus with exchange-length 
violations. These violations would not be caught by static validation or even execution 
(MuMax3 doesn't error on unresolved exchange   it just gives wrong physics quietly).

---


### 2.5 Does LLM Temperature Affect Results?

ran Config A with three temperatures:

| Temperature | API Validity | IR F1 | Static Valid |
|------------|-------------|-------|-------------|
| 0.4 | 94.6% | 0.801 | 83.2% |
| **0.5 ** | **94.8%** | **0.807** | **83.7%** |
| 0.8 | 93.1% | 0.788 | 81.4% |
| 1.0 | 89.7% | 0.754 | 76.8% |

Temperature 0.5 provides the best trade-off: low enough to be deterministic for common cases, high enough to avoid degenerate repetition.

---
