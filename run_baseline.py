"""
run_baseline.py
===============
Runs the baseline evaluation experiment.

This tests our pipeline (without any fine-tuning) on the test set,
trying all 4 NL granularity levels to see how performance changes
when the input description is more or less detailed.

The baseline uses:
  - Stage 1: LLM-based intent extraction (or rule-based fallback)
  - Stage 2: IR construction from entities
  - Stage 3: Template-based code generation (deterministic)
  - Stage 4: Static validation

Results are saved to results/baseline/ and a summary markdown is generated.

Usage:
    # Run with Gemini (recommended):
    python experiments/run_baseline.py --backend gemini --api-key AIza...

    # Run with vLLM:
    python experiments/run_baseline.py --backend vllm --vllm-url http://localhost:8000

    # Quick test run (first 10 samples only):
    python experiments/run_baseline.py --quick --backend none
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pipeline import NL2SimPipeline
from evaluation.metrics import compute_all_metrics, aggregate_metrics
from evaluation.evaluator import load_test_set, run_evaluation


# These are the 4 NL levels we test
NL_LEVELS = [
    ("nl_expert",       "Expert-level description (all parameters specified)"),
    ("nl_intermediate", "Intermediate description (main physics, no constants)"),
    ("nl_novice",       "Novice description (plain language)"),
    ("nl_ambiguous",    "Ambiguous description (terse keyword label)"),
]


def run_baseline_experiment(
    test_file,
    out_dir,
    backend="none",
    api_key=None,
    vllm_url=None,
    max_samples=None,
    quick=False,
):
    """
    Run the full baseline experiment across all 4 NL levels.
    
    Saves individual result files per level and a combined summary.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BASELINE EXPERIMENT")
    print(f"  Test file: {test_file}")
    print(f"  Backend:   {backend}")
    print(f"  Output:    {out_dir}")
    print("=" * 60)

    # Load test samples
    if not os.path.exists(test_file):
        print(f"\nTest file not found: {test_file}")
        print("Using synthetic demo samples...")
        test_samples = generate_demo_test_set()
    else:
        test_samples = load_test_set(test_file)

    if quick:
        test_samples = test_samples[:10]
        print(f"Quick mode: using first {len(test_samples)} samples")

    if max_samples:
        test_samples = test_samples[:max_samples]

    # Set up pipeline
    pipeline = NL2SimPipeline(
        backend=backend,
        api_key=api_key,
        vllm_url=vllm_url,
        verbose=False,  # suppress per-sample output in batch mode
    )

    # Results across all NL levels
    all_level_results = {}

    for nl_field, nl_description in NL_LEVELS:
        print(f"\n{'─'*60}")
        print(f"Testing NL level: {nl_field}")
        print(f"  Description: {nl_description}")
        print(f"{'─'*60}")

        level_out_file = out_path / f"baseline_{nl_field}.jsonl"

        # Check if this level has a field in the test data
        # (if test data only has "nl_intent", skip other levels)
        has_field = any(nl_field in s for s in test_samples)
        effective_field = nl_field if has_field else "nl_intent"

        if not has_field:
            print(f"  Note: '{nl_field}' not in test data, using 'nl_intent'")

        results = run_evaluation(
            test_samples=test_samples,
            pipeline=pipeline,
            output_file=str(level_out_file),
            max_samples=max_samples,
            nl_field=effective_field,
            resume=True,
        )

        # Aggregate metrics for this level
        all_metrics = [r["metrics"] for r in results if "metrics" in r]
        agg = aggregate_metrics(all_metrics)
        all_level_results[nl_field] = agg

        # Print quick summary
        if "api_validity_rate" in agg:
            print(f"\n  Summary for {nl_field}:")
            print(f"    API Validity:   {agg['api_validity_rate']['mean']*100:.1f}%")
            print(f"    Completeness:   {agg['completeness_score']['mean']*100:.1f}%")
            print(f"    IR F1:          {agg['ir_f1']['mean']:.3f}")

    # Generate markdown summary report
    summary_path = out_path / "baseline_summary.md"
    write_baseline_summary(all_level_results, summary_path, backend)
    print(f"\n{'='*60}")
    print(f"Baseline experiment complete. Summary: {summary_path}")

    return all_level_results


def generate_demo_test_set():
    """Create a small demo test set for when the real test file doesn't exist."""
    samples = [
        {
            "id": f"demo_{i:03d}",
            "nl_intent":        desc,
            "nl_expert":        f"[Expert] {desc} with detailed parameters",
            "nl_intermediate":  desc,
            "nl_novice":        f"I want to simulate {desc.lower()}",
            "nl_ambiguous":     desc.split()[0] + " " + desc.split()[-1],
            "physics_category": cat,
            "mumax3_script": "SetGridsize(256,128,1)\nSetCellsize(2e-9,2e-9,5e-9)\n"
                             f"Msat=800e3\nAex=13e-12\nalpha=0.01\nm=uniform(1,0,0)\nrelax()\n",
            "ir": {"category": cat, "material": {"Msat_Am": 800000, "Aex_Jm": 1.3e-11, "alpha": 0.01}},
        }
        for i, (desc, cat) in enumerate([
            ("Permalloy nanowire domain wall under 10mT field",   "field_driven"),
            ("Permalloy disk energy minimisation",                 "simple_relax"),
            ("CoFeB thin film skyrmion with DMI=2mJ/m2",          "dmi_skyrmion"),
            ("YIG nanodisk spin wave at 5GHz",                     "fft_analysis"),
            ("Permalloy wire under spin-transfer torque",          "stt"),
            ("CoFeB Pt bilayer spin-orbit torque switching",       "sot"),
            ("Permalloy film at 300K thermal fluctuations",        "thermal"),
            ("Permalloy CoFe bilayer region simulation",           "multi_region"),
        ])
    ]
    return samples


def write_baseline_summary(all_level_results, output_path, backend):
    """Write a markdown summary of the baseline results."""
    lines = []
    lines.append("# Baseline Experiment Results")
    lines.append("")
    lines.append(f"**Backend**: {backend}")
    lines.append(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Performance Across NL Granularity Levels")
    lines.append("")
    lines.append("| NL Level | API Validity | Completeness | IR F1 | Static Valid |")
    lines.append("|----------|-------------|-------------|-------|-------------|")

    level_names = {
        "nl_expert":       "Expert",
        "nl_intermediate": "Intermediate",
        "nl_novice":       "Novice",
        "nl_ambiguous":    "Ambiguous",
    }

    for nl_field, agg in all_level_results.items():
        label = level_names.get(nl_field, nl_field)

        api_val  = agg.get("api_validity_rate", {}).get("mean", 0) * 100
        comp_val = agg.get("completeness_score", {}).get("mean", 0) * 100
        f1_val   = agg.get("ir_f1", {}).get("mean", 0)
        val_val  = agg.get("static_valid", {}).get("pass_rate", 0) * 100

        lines.append(
            f"| {label} | {api_val:.1f}% | {comp_val:.1f}% | {f1_val:.3f} | {val_val:.1f}% |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- API Validity measures hallucination rate (higher is better)")
    lines.append("- Completeness measures required symbol coverage")
    lines.append("- IR F1 measures semantic accuracy of the parsed representation")
    lines.append("- Static Valid is the fraction of scripts passing all static checks")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run NL2Sim baseline experiment")
    parser.add_argument("--test-file", default="data/splits/test.jsonl")
    parser.add_argument("--out-dir",   default="results/baseline")
    parser.add_argument("--backend",   choices=["gemini","vllm","none"], default="none")
    parser.add_argument("--api-key",   default=None)
    parser.add_argument("--vllm-url",  default=None)
    parser.add_argument("--max",       type=int, default=None, help="Limit samples")
    parser.add_argument("--quick",     action="store_true", help="First 10 samples only")
    args = parser.parse_args()

    run_baseline_experiment(
        test_file=args.test_file,
        out_dir=args.out_dir,
        backend=args.backend,
        api_key=args.api_key,
        vllm_url=args.vllm_url,
        max_samples=args.max,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
