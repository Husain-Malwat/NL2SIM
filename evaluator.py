import json
import sys
import os
import argparse
import time
import math
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/generators'))

from pipeline.pipeline import NL2SimPipeline
from evaluation.metrics import compute_all_metrics, aggregate_metrics


def load_test_set(test_file_path):
    """
    Load test samples from a JSONL file.
    
    Each line should be a JSON object with at least:
    - "nl_intent": the NL description
    - "mumax3_script": the reference script
    - "ir": the reference IR
    - "physics_category": the category label
    """
    samples = []
    with open(test_file_path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}")
    print(f"Loaded {len(samples)} test samples from {test_file_path}")
    return samples


def run_evaluation(
    test_samples,
    pipeline,
    output_file,
    max_samples=None,
    nl_field="nl_intent",
    resume=True,
):
    """
    Run evaluation on all test samples.
    
    Args:
        test_samples: List of test sample dicts
        pipeline: NL2SimPipeline instance
        output_file: Path to write results JSONL
        max_samples: Limit evaluation to first N samples (useful for quick testing)
        nl_field: Which NL field to use ("nl_intent", "nl_expert", "nl_novice", etc.)
        resume: If True, skip samples that are already in the output file
    
    Returns:
        List of result dicts
    """
    # Limit samples if requested
    if max_samples:
        test_samples = test_samples[:max_samples]
    
    # Check for already-done samples (for resuming)
    done_ids = set()
    if resume and os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    done = json.loads(line)
                    done_ids.add(done.get("sample_id"))
                except:
                    pass
        print(f"Resuming: {len(done_ids)} already done, skipping")

    results = []
    n_total   = len(test_samples)
    n_success = 0
    n_valid   = 0

    # Open output file for appending
    mode = "a" if done_ids else "w"
    out_f = open(output_file, mode)

    try:
        for i, sample in enumerate(test_samples):
            sample_id = sample.get("id", f"sample_{i:04d}")

            # Skip if already done
            if sample_id in done_ids:
                continue

            print(f"\n{'─'*60}")
            print(f"Sample {i+1}/{n_total}: {sample_id}")

            # Get the NL description to use
            nl = sample.get(nl_field) or sample.get("nl_intent", "")
            if not nl:
                print(f"  Skipping: no NL field '{nl_field}' in sample")
                continue

            # Get reference information
            ref_script   = sample.get("mumax3_script", "")
            ref_ir       = sample.get("ir", {})
            ref_category = sample.get("physics_category", "")

            # Run the pipeline
            start = time.time()
            pipeline_result = pipeline.run(nl)
            duration = time.time() - start

            # Compute metrics
            gen_script = pipeline_result.get("script", "")
            gen_ir     = pipeline_result.get("ir", {})

            if gen_script:
                sample_metrics = compute_all_metrics(
                    generated_script=gen_script,
                    reference_script=ref_script,
                    generated_ir=gen_ir or {},
                    reference_ir=ref_ir,
                    category=ref_category,
                )
            else:
                # Pipeline failed completely
                sample_metrics = {
                    "api_validity_rate":   0.0,
                    "completeness_score":  0.0,
                    "ir_f1":               0.0,
                    "category_correct":    False,
                    "static_valid":        False,
                    "pipeline_failed":     True,
                }

            # Assemble the result record
            result = {
                "sample_id":       sample_id,
                "nl_input":        nl,
                "nl_field":        nl_field,
                "ref_category":    ref_category,
                "gen_category":    pipeline_result.get("intent", {}).get("category") if pipeline_result.get("intent") else None,
                "pipeline_success": pipeline_result.get("success", False),
                "pipeline_duration_s": duration,
                "codegen_method":  pipeline_result.get("codegen_method"),
                "pipeline_errors": pipeline_result.get("errors", []),
                "generated_script": gen_script,
                "metrics":         sample_metrics,
            }
            results.append(result)

            # Track running totals
            if pipeline_result.get("success"):
                n_success += 1
            if sample_metrics.get("static_valid"):
                n_valid += 1

            # Print summary for this sample
            m = sample_metrics
            print(
                f"  API validity: {m.get('api_validity_rate', 0)*100:.1f}% | "
                f"Completeness: {m.get('completeness_score', 0)*100:.1f}% | "
                f"IR F1: {m.get('ir_f1', 0):.3f} | "
                f"Static valid: {m.get('static_valid', False)}"
            )

            # Save to file immediately (so we can resume if interrupted)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    finally:
        out_f.close()

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"  Samples evaluated: {len(results)}")
    print(f"  Pipeline success:  {n_success}/{len(results)} "
          f"({100*n_success/max(1,len(results)):.1f}%)")
    print(f"  Static valid:      {n_valid}/{len(results)} "
          f"({100*n_valid/max(1,len(results)):.1f}%)")
    print(f"  Results saved to:  {output_file}")

    return results


def analyze_results(results_file):
    """
    Load completed results and print a summary report.
    """
    results = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if not results:
        print("No results found.")
        return

    print(f"\n{'='*60}")
    print(f"RESULTS ANALYSIS: {results_file}")
    print(f"{'='*60}")
    print(f"Total samples: {len(results)}")

    # Aggregate all metrics
    all_metrics = [r["metrics"] for r in results if "metrics" in r]
    agg = aggregate_metrics(all_metrics)

    # Print key metrics
    numeric_metrics = [
        ("api_validity_rate",  "API Validity Rate"),
        ("completeness_score", "Completeness Score"),
        ("ir_f1",              "IR F1 Score"),
        ("ir_precision",       "IR Precision"),
        ("ir_recall",          "IR Recall"),
        ("exact_match_rate",   "Parameter Exact Match Rate"),
        ("mean_relative_error","Mean Parameter Rel. Error"),
        ("hallucination_rate", "Hallucination Rate"),
    ]

    print(f"\n{'─'*60}")
    print(f"{'Metric':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"{'─'*60}")

    for key, label in numeric_metrics:
        if key in agg:
            s = agg[key]
            print(
                f"  {label:<33} "
                f"{s['mean']:>8.3f} "
                f"{s['std']:>8.3f} "
                f"{s['min']:>8.3f} "
                f"{s['max']:>8.3f}"
            )

    # Category accuracy
    if "category_correct" in agg:
        print(f"\n  {'Category Accuracy':<33} "
              f"{agg['category_correct']['pass_rate']:>8.3f}")

    # Static validity
    if "static_valid" in agg:
        print(f"  {'Static Validity Rate':<33} "
              f"{agg['static_valid']['pass_rate']:>8.3f}")

    # Per-category breakdown
    print(f"\n{'─'*60}")
    print("Per-Category Breakdown:")
    print(f"{'─'*60}")

    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in results:
        cat = r.get("ref_category", "unknown")
        by_cat[cat].append(r)

    print(f"  {'Category':<20} {'N':>4} {'API%':>7} {'Compl%':>7} {'IR F1':>7} {'Valid%':>7}")
    print(f"  {'─'*20} {'─'*4} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    for cat, cat_results in sorted(by_cat.items()):
        cat_metrics = [r["metrics"] for r in cat_results if "metrics" in r]
        n = len(cat_metrics)
        if n == 0:
            continue

        api_vals  = [m.get("api_validity_rate", 0) for m in cat_metrics]
        comp_vals = [m.get("completeness_score", 0) for m in cat_metrics]
        f1_vals   = [m.get("ir_f1", 0) for m in cat_metrics]
        val_vals  = [m.get("static_valid", False) for m in cat_metrics]

        print(
            f"  {cat:<20} {n:>4} "
            f"{100*sum(api_vals)/n:>6.1f}% "
            f"{100*sum(comp_vals)/n:>6.1f}% "
            f"{sum(f1_vals)/n:>7.3f} "
            f"{100*sum(val_vals)/n:>6.1f}%"
        )

    # Error analysis
    print(f"\n{'─'*60}")
    print("Common Errors:")
    error_counts = {}
    for r in results:
        for err in r.get("pipeline_errors", []):
            # Simplify error message for grouping
            short = err[:60] if len(err) > 60 else err
            error_counts[short] = error_counts.get(short, 0) + 1

    for err, count in sorted(error_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  [{count:>3}x] {err}")


def main():
    parser = argparse.ArgumentParser(description="Run NL2Sim evaluation on test set")
    parser.add_argument("--test-file", default="data/splits/test.jsonl",
                        help="Test JSONL file")
    parser.add_argument("--out", default="results/evaluation_results.jsonl",
                        help="Output file for results")
    parser.add_argument("--backend", choices=["gemini","vllm","none"], default="none")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--vllm-url", default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit to first N samples (for testing)")
    parser.add_argument("--nl-field", default="nl_intent",
                        choices=["nl_intent", "nl_expert", "nl_intermediate", "nl_novice", "nl_ambiguous"],
                        help="Which NL field to use as input")
    parser.add_argument("--analyze", default=None,
                        help="Instead of running eval, analyze a completed results file")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, don't resume from existing output")
    args = parser.parse_args()

    # Analysis mode
    if args.analyze:
        analyze_results(args.analyze)
        return

    # Make output directory
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Load test set
    if not os.path.exists(args.test_file):
        print(f"Test file not found: {args.test_file}")
        print("Creating a tiny demo test set for illustration...")

        # Create a minimal demo test set
        demo_samples = [
            {
                "id": "demo_001",
                "nl_intent": "Simulate domain wall motion in a Permalloy nanowire under a 10mT field pulse",
                "physics_category": "field_driven",
                "mumax3_script": "SetGridsize(512,128,1)\nSetCellsize(1e-9,1e-9,5e-9)\nMsat=800e3\nAex=13e-12\nalpha=0.01\nm=twoDomain(1,0,0,-1,0,0)\nrelax()\nB_ext=vector(0.01,0,0)\nautosave(m,1e-10)\nrun(10e-9)\n",
                "ir": {
                    "category": "field_driven",
                    "material": {"Msat_Am": 800000, "Aex_Jm": 1.3e-11, "alpha": 0.01},
                },
            },
            {
                "id": "demo_002",
                "nl_intent": "Relax a Permalloy disk and find ground state magnetisation",
                "physics_category": "simple_relax",
                "mumax3_script": "SetGridsize(128,128,1)\nSetCellsize(4e-9,4e-9,5e-9)\nMsat=800e3\nAex=13e-12\nalpha=0.5\nm=uniform(1,0,0)\nrelax()\nsave(m)\n",
                "ir": {
                    "category": "simple_relax",
                    "material": {"Msat_Am": 800000, "Aex_Jm": 1.3e-11, "alpha": 0.5},
                },
            },
        ]

        os.makedirs("data/splits", exist_ok=True)
        with open(args.test_file, "w") as f:
            for s in demo_samples:
                f.write(json.dumps(s) + "\n")
        print(f"Demo test set written to {args.test_file}")
        test_samples = demo_samples
    else:
        test_samples = load_test_set(args.test_file)

    # Set up pipeline
    pipeline = NL2SimPipeline(
        backend=args.backend,
        api_key=args.api_key,
        vllm_url=args.vllm_url,
        verbose=True,
    )

    # Run evaluation
    run_evaluation(
        test_samples=test_samples,
        pipeline=pipeline,
        output_file=args.out,
        max_samples=args.max_samples,
        nl_field=args.nl_field,
        resume=not args.no_resume,
    )

    # Auto-analyze after running
    analyze_results(args.out)


if __name__ == "__main__":
    main()
