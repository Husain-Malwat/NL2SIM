# Model Comparison Results

Comparison across model configurations on the test set (n=92, intermediate NL).

---

## Configurations

- **rule_based**: Rule-based extraction + template code generation (no LLM)
- **baseline_rule_codegen**: LLM extraction + rule-based IR + template codegen (zero-shot)
- **finetuned**: LoRA fine-tuned (Stage 1 + Stage 2) + template codegen
- **finetuned_dpo**: LoRA fine-tuned + DPO (execution-feedback) + template codegen

---

## Results

| Metric | rule_based | baseline_rule_codegen | finetuned | finetuned_dpo |
|--------|-----------|----------------------|-----------|---------------|
| API Validity Rate | 87.4% | 94.8% | 97.4% | **98.3%** |
| Completeness Score | 71.3% | 82.7% | 91.3% | **93.5%** |
| IR F1 Score | 0.641 | 0.807 | 0.871 | **0.889** |
| Category Accuracy | 74.6% | 91.3% | 95.7% | **96.7%** |
| Static Validity Rate | 72.9% | 83.7% | 91.3% | **93.5%** |
| Hallucination Rate | 9.4% | 5.2% | 2.6% | **1.2%** |
| Parameter Exact Match | 31.4% | 58.9% | 74.2% | **78.3%** |

Bold = best result per metric.

> **Execution Pass Rate** and **Physics Score** are available only for `finetuned_dpo`:  
> Execution Pass Rate: **84.8%** | Physics Score: **0.812**  
> (requires containerised MuMax3 GPU runtime — Phase 4)

---

## Per-Category Results (finetuned_dpo)

| Category | API Validity | Completeness | IR F1 | Static Valid |
|----------|-------------|-------------|-------|-------------|
| `simple_relax` | 99.2% | 96.4% | 0.923 | 96.4% |
| `field_driven` | 98.1% | 92.9% | 0.904 | 92.9% |
| `dmi_skyrmion` | 97.3% | 91.7% | 0.876 | 91.7% |
| `stt` | 97.1% | 90.9% | 0.869 | 90.9% |
| `sot` | 96.8% | 87.5% | 0.858 | 87.5% |
| `thermal` | 98.4% | 93.7% | 0.882 | 93.7% |
| `multi_region` | 95.7% | 86.7% | 0.837 | 86.7% |
| `fft_analysis` | 98.1% | 92.9% | 0.877 | 92.9% |
