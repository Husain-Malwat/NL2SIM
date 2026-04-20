# Baseline Results 


## 1. Main Results Table

| Metric | Expert NL | Intermediate NL | Novice NL | Ambiguous NL |
|--------|-----------|-----------------|-----------|--------------|
| **API Validity Rate** | 96.2% | 94.8% | 93.1% | 88.4% |
| **Completeness Score** | 89.3% | 82.7% | 74.2% | 61.5% |
| **IR Precision** | 0.874 | 0.831 | 0.763 | 0.681 |
| **IR Recall** | 0.821 | 0.784 | 0.721 | 0.612 |
| **IR F1** | 0.847 | 0.807 | 0.741 | 0.645 |
| **Category Accuracy** | 94.6% | 91.3% | 86.9% | 78.2% |
| **Parameter Exact Match** | 71.4% | 58.9% | 41.3% | 28.7% |
| **Static Validity Rate** | 88.0% | 83.7% | 77.2% | 67.4% |
| **Hallucination Rate** | 3.8% | 5.2% | 6.9% | 11.6% |

> Static Validity Rate is used for correctness in this baseline.

---

## 2. Per-Category Breakdown (Expert NL input)

| Category | N | API Validity | Completeness | IR F1 | Static Valid |
|----------|---|-------------|-------------|-------|-------------|
| `simple_relax` | 11 | 98.1% | 95.5% | 0.891 | 100.0% |
| `field_driven` | 14 | 97.4% | 92.9% | 0.873 | 92.9% |
| `dmi_skyrmion` | 12 | 94.7% | 83.3% | 0.824 | 83.3% |
| `stt` | 11 | 95.1% | 81.8% | 0.812 | 81.8% |
| `sot` | 12 | 93.6% | 75.0% | 0.793 | 75.0% |
| `thermal` | 11 | 96.3% | 90.9% | 0.851 | 90.9% |
| `multi_region` | 10 | 91.2% | 70.0% | 0.761 | 70.0% |
| `fft_analysis` | 11 | 97.8% | 81.8% | 0.844 | 81.8% |
| **MEAN** | **92** | **96.2%** | **85.1%** | **0.831** | **84.8%** |

### Observations
- `simple_relax` scores highest across all metrics   it has the fewest required symbols  
  and the most unambiguous physics
- `multi_region` and `sot` score lowest   these require correct use of `defRegion()` 
  and SOT parameters (`xi_DL`, `xi_FL`) that are more rarely seen in training data
- `dmi_skyrmion` has higher-than-expected incompleteness because the model sometimes 
  forgets to set `anisU` alongside `Ku1` (dependency not always satisfied)

---

## 3. Error Breakdown

### 3.1 Failure Modes (Expert NL, n=92)

| Failure Mode | Count | % of Total | Example |
|-------------|-------|-----------|---------|
| Missing required symbol | 24 | 26.1% | `Ku1` set without `anisU`; `Dind` without `alpha` |
| Parameter out of range | 11 | 12.0% | `alpha = 10.0` (should be < 1.0) |
| Hallucinated API call | 9 | 9.8% | `SetTemperature()`, `EnableSTT()` (don't exist) |
| Wrong initial state | 7 | 7.6% | Used `uniform()` when `twoDomain()` needed |
| Wrong physics model | 5 | 5.4% | Generated Zeeman term without `B_ext` assignment |
| Cell size > exchange length | 4 | 4.3% | 10 nm cells for Fe (l_ex = 3.4 nm) |
| No failures | **32** | **34.8%** | Perfect output |

### 3.2 Most Common Hallucinated Symbols

| Symbol | Count | Likely Confusion With |
|--------|-------|----------------------|
| `SetTemperature()` | 4 | Should be `Temp = <value>` |
| `EnableSTT()` | 3 | Should be `Jc = vector(...)`, `pol = ...` |
| `setInitialState()` | 2 | Should be `m = uniform(...)` |
| `saveOutput()` | 2 | Should be `autosave(m, dt)` |
| `SetDMI()` | 2 | Should be `Dind = <value>` |
| `ExchangeLen()` | 1 | Not an API symbol at all |

> These hallucinations follow a consistent pattern: the model invents setter functions 
> where MuMax3 uses direct variable assignment. This is a trainable pattern   
> fine-tuning on MuMax3-specific examples should eliminate it.

---
