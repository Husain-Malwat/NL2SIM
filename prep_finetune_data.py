"""
prep_finetune_data.py
=====================
Converts the annotated dataset into the format needed for fine-tuning.

We fine-tune in TWO stages:
  Stage 1: NL → IR  (understanding stage)
  Stage 2: IR → MuMax3 script  (code generation stage)

Each stage produces its own training file in instruction-tuning format:
  [
    {
      "instruction": "...",
      "input":       "...",   (can be empty)
      "output":      "..."
    },
    ...
  ]

This format works with most fine-tuning frameworks (Axolotl, LLaMA-Factory, etc.)

Usage:
    python finetuning/prep_finetune_data.py \
        --annotated data/annotated/ \
        --out-dir   data/finetune_ready/ \
        --split-ratio 0.8 0.1 0.1
"""

import json
import os
import sys
import argparse
import random
import hashlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Stage 1 Prompt Templates ─────────────────────────────────────────────────
# These teach the model to extract a structured IR from natural language.
# We have 4 variants because we train on all 4 NL granularity levels.

STAGE1_INSTRUCTION = """\
You are a micromagnetic simulation expert. Read the natural language description \
of a MuMax3 simulation and extract a structured intermediate representation (IR) as JSON.

The IR must include:
- category: the simulation type
- material: Msat, Aex, alpha, and optional Ku1, DMI, Temp
- geometry: shape and dimensions in nm
- physics: which interactions are active
- excitation: applied field or current details
- solver: simulation duration and save interval

Return ONLY the JSON object, no explanation."""


def make_stage1_example(sample, nl_field="nl_intent"):
    """
    Create one Stage 1 training example from an annotated sample.

    Args:
        sample: An annotated dataset sample dict
        nl_field: Which NL field to use as input

    Returns:
        A dict with instruction/input/output, or None if required fields missing
    """
    nl_text = sample.get(nl_field)
    ir_dict = sample.get("ir")

    if not nl_text or not ir_dict:
        return None

    # Make the IR compact and clean for training
    # We remove metadata fields that aren't meaningful for the model to learn
    clean_ir = {
        "category":     ir_dict.get("category", "simple_relax"),
        "material":     ir_dict.get("material", {}),
        "geometry":     ir_dict.get("geometry", {}),
        "physics":      ir_dict.get("physics", {}),
        "excitation":   ir_dict.get("excitation"),
        "initialization": ir_dict.get("initialization", {}),
        "solver":       ir_dict.get("solver", {}),
    }

    # Remove None values to keep the training target clean
    clean_ir = {k: v for k, v in clean_ir.items() if v is not None}

    return {
        "instruction": STAGE1_INSTRUCTION,
        "input":       nl_text,
        "output":      json.dumps(clean_ir, separators=(",", ":"), ensure_ascii=False),
    }


# ─── Stage 2 Prompt Templates ─────────────────────────────────────────────────
# These teach the model to generate syntactically correct MuMax3 code from an IR.

STAGE2_INSTRUCTION = """\
You are a MuMax3 micromagnetics simulation expert. Given a structured simulation \
specification (IR) as JSON, generate a complete, syntactically correct MuMax3 script.

MuMax3 rules:
- Always start with SetGridsize and SetCellsize
- Set material parameters directly: Msat = value (not SetMsat())
- Use relax() before run() to find equilibrium first
- Use autosave(m, dt) for magnetisation snapshots
- Use tableadd(quantity) + tableautosave(dt) for scalar outputs
- End with run(duration) for dynamic simulations

Return ONLY the MuMax3 script text, no explanation, no markdown."""


def make_stage2_example(sample):
    """
    Create one Stage 2 training example from an annotated sample.

    Args:
        sample: An annotated dataset sample dict

    Returns:
        A dict with instruction/input/output, or None if required fields missing
    """
    ir_dict    = sample.get("ir")
    script_txt = sample.get("mumax3_script")

    if not ir_dict or not script_txt:
        return None

    # The input is the clean IR JSON
    clean_ir = {
        "category":     ir_dict.get("category", "simple_relax"),
        "material":     ir_dict.get("material", {}),
        "geometry":     ir_dict.get("geometry", {}),
        "physics":      ir_dict.get("physics", {}),
        "excitation":   ir_dict.get("excitation"),
        "initialization": ir_dict.get("initialization", {}),
        "solver":       ir_dict.get("solver", {}),
    }
    clean_ir = {k: v for k, v in clean_ir.items() if v is not None}

    return {
        "instruction": STAGE2_INSTRUCTION,
        "input":       json.dumps(clean_ir, indent=2, ensure_ascii=False),
        "output":      script_txt.strip(),
    }


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_annotated_samples(source_dir_or_file):
    """
    Load annotated samples from either:
    - A single JSONL file
    - A directory of JSONL files
    """
    samples = []
    path = Path(source_dir_or_file)

    if path.is_file():
        files = [path]
    else:
        files = list(path.glob("*.jsonl"))

    print(f"Loading from {len(files)} file(s)...")

    for f in files:
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    print(f"Loaded {len(samples)} total samples")
    return samples


def deduplicate(samples):
    """
    Remove duplicate samples based on the mumax3_script content hash.
    Keeps the first occurrence of each unique script.
    """
    seen_hashes = set()
    unique = []

    for sample in samples:
        script = sample.get("mumax3_script", "")
        h = hashlib.md5(script.strip().encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(sample)

    print(f"After deduplication: {len(unique)} samples (removed {len(samples)-len(unique)})")
    return unique


def train_val_test_split(samples, ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Split samples into train / val / test sets.

    Uses stratified splitting by physics_category to ensure
    each category is represented in all three splits.
    """
    rng = random.Random(seed)

    # Group by category
    by_category = {}
    for s in samples:
        cat = s.get("physics_category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(s)

    train, val, test = [], [], []

    for cat, cat_samples in by_category.items():
        rng.shuffle(cat_samples)
        n     = len(cat_samples)
        n_tr  = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1]))

        train.extend(cat_samples[:n_tr])
        val.extend(cat_samples[n_tr:n_tr + n_val])
        test.extend(cat_samples[n_tr + n_val:])

    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")
    return train, val, test


# ─── Main Preparation Pipeline ────────────────────────────────────────────────

def prepare_finetune_data(source, out_dir, split_ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Full data preparation pipeline.

    1. Load all annotated samples
    2. Deduplicate
    3. Split into train/val/test
    4. Create Stage 1 training files (NL → IR)
    5. Create Stage 2 training files (IR → script)
    6. Save all split info for reference
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load and clean
    samples = load_annotated_samples(source)
    samples = deduplicate(samples)

    # Split
    train, val, test = train_val_test_split(samples, ratios=split_ratios, seed=seed)

    # Save the raw splits (useful for evaluator.py)
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        split_file = out_path / f"{split_name}.jsonl"
        with open(split_file, "w") as f:
            for s in split_data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Saved {split_name}: {split_file} ({len(split_data)} samples)")

    # Also copy to data/splits/ for the evaluator
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        splits_file = splits_dir / f"{split_name}.jsonl"
        with open(splits_file, "w") as f:
            for s in split_data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ── Stage 1 training examples (NL → IR) ───────────────────────────────────
    print("\nBuilding Stage 1 (NL → IR) training data...")
    stage1_dir = out_path / "stage1_nl_to_ir"
    stage1_dir.mkdir(exist_ok=True)

    NL_FIELDS = ["nl_intent", "nl_expert", "nl_intermediate", "nl_novice", "nl_ambiguous"]

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        examples = []
        for sample in split_data:
            for nl_field in NL_FIELDS:
                # Only use fields that exist in the sample
                if sample.get(nl_field):
                    ex = make_stage1_example(sample, nl_field=nl_field)
                    if ex:
                        ex["_source_id"] = sample.get("id", "")
                        ex["_nl_field"]  = nl_field
                        examples.append(ex)

        out_file = stage1_dir / f"{split_name}.jsonl"
        with open(out_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Stage 1 {split_name}: {len(examples)} examples → {out_file}")

    # ── Stage 2 training examples (IR → script) ───────────────────────────────
    print("\nBuilding Stage 2 (IR → script) training data...")
    stage2_dir = out_path / "stage2_ir_to_script"
    stage2_dir.mkdir(exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        examples = []
        for sample in split_data:
            ex = make_stage2_example(sample)
            if ex:
                ex["_source_id"] = sample.get("id", "")
                examples.append(ex)

        out_file = stage2_dir / f"{split_name}.jsonl"
        with open(out_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Stage 2 {split_name}: {len(examples)} examples → {out_file}")

    # ── Dataset statistics ─────────────────────────────────────────────────────
    stats = {
        "total_samples": len(samples),
        "train_size":    len(train),
        "val_size":      len(val),
        "test_size":     len(test),
        "stage1_train_examples": len(train) * sum(
            1 for s in train[:1] for f in NL_FIELDS if s.get(f)
        ),
        "split_seed": seed,
    }

    stats_file = out_path / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_file}")
    print(f"\nDone! Fine-tuning data ready in {out_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare annotated data for fine-tuning"
    )
    parser.add_argument("--annotated", required=True,
                        help="Path to annotated data (JSONL file or directory)")
    parser.add_argument("--out-dir", default="data/finetune_ready",
                        help="Output directory")
    parser.add_argument("--split-ratio", nargs=3, type=float,
                        default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_finetune_data(
        source=args.annotated,
        out_dir=args.out_dir,
        split_ratios=tuple(args.split_ratio),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
