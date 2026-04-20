"""
train_lora.py
=============
Fine-tune a language model on the NL2Sim dataset using LoRA.

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method.
Instead of updating ALL the model's parameters (which requires huge GPU memory),
LoRA adds small "adapter" matrices to the attention layers and only trains those.

This means we can fine-tune a 7B or 8B parameter model on a single GPU
with 16GB of VRAM.

We fine-tune EACH STAGE SEPARATELY:
  - Stage 1 model: learns to extract IR from natural language
  - Stage 2 model: learns to generate MuMax3 code from IR

Usage:
    # Fine-tune Stage 1 (NL → IR):
    python finetuning/train_lora.py \
        --stage 1 \
        --data data/finetune_ready/stage1_nl_to_ir/ \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --out-dir models/stage1_lora/ \
        --epochs 3

    # Fine-tune Stage 2 (IR → script):
    python finetuning/train_lora.py \
        --stage 2 \
        --data data/finetune_ready/stage2_ir_to_script/ \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --out-dir models/stage2_lora/ \
        --epochs 5

Requirements:
    pip install transformers peft datasets accelerate bitsandbytes

Notes:
    - Uses 4-bit quantization (QLoRA) to reduce GPU memory requirements
    - Default hyperparameters are from the LoRA paper for instruction tuning
    - Training logs to stdout; use --log-file to redirect
"""

import argparse
import json
import os
import sys

# ── Lazy imports: only import training libraries when actually training ─────
# This lets us import this file without having transformers installed
# (useful for dry-run mode and unit tests)
_LIBS_AVAILABLE = True
try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    _LIBS_AVAILABLE = False


# ─── LoRA Hyperparameters ─────────────────────────────────────────────────────
# These are the default settings. Adjust based on your GPU and dataset size.

LORA_CONFIG = {
    # Rank of the LoRA matrices.
    # Higher rank = more parameters = potentially better fit but more memory.
    # 16 is a good default. Use 8 for very limited GPU memory.
    "r": 16,

    # LoRA scaling factor. Usually set to 2*r.
    "lora_alpha": 32,

    # Dropout for regularisation. 0.1 is standard.
    "lora_dropout": 0.1,

    # Which attention matrices to add LoRA to.
    # "q_proj" and "v_proj" are the query and value projection matrices.
    # Adding "k_proj" too gives more capacity but uses more memory.
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
}

TRAINING_DEFAULTS = {
    "learning_rate":    2e-4,    # Higher than full fine-tuning (LoRA can handle it)
    "per_device_batch": 2,       # Batch size per GPU. Increase if you have memory.
    "grad_accumulation": 8,      # Effective batch = per_device * grad_accumulation = 16
    "max_seq_length":   1024,    # Maximum token length (trim longer sequences)
    "warmup_ratio":     0.05,    # 5% warmup steps
    "weight_decay":     0.01,
    "save_steps":       100,     # Save checkpoint every N steps
    "eval_steps":       100,     # Evaluate every N steps
    "logging_steps":    10,
}

# Stage-specific settings
STAGE_CONFIGS = {
    1: {
        "description": "NL → IR (understanding stage)",
        "max_seq_length": 800,   # NL inputs are shorter
        "epochs": 3,
        "learning_rate": 2e-4,
    },
    2: {
        "description": "IR → MuMax3 script (code generation stage)",
        "max_seq_length": 1200,  # IR inputs + script outputs are longer
        "epochs": 5,             # Code generation benefits from more epochs
        "learning_rate": 1e-4,   # Slightly lower for code
    },
}


def load_jsonl_dataset(file_path):
    """Load a JSONL file into a list of dicts."""
    samples = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def format_example_as_prompt(example, tokenizer):
    """
    Convert a training example to the prompt format the model expects.

    We use the Alpaca instruction format:
        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}

    The model is trained to predict everything after "### Response:\n".
    """
    instruction = example.get("instruction", "")
    input_text  = example.get("input", "")
    output_text = example.get("output", "")

    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    full_text = prompt + output_text + tokenizer.eos_token
    return prompt, full_text


def tokenize_example(example, tokenizer, max_length):
    """
    Tokenize one training example.

    We need to:
    1. Tokenize the full sequence (prompt + output)
    2. Create labels where prompt tokens are masked (-100)
       so the model only learns to predict the output, not the prompt.
    """
    prompt, full_text = format_example_as_prompt(example, tokenizer)

    # Tokenize the prompt (to find where output starts)
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    prompt_len = prompt_tokens["input_ids"].shape[1]

    # Tokenize the full sequence
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = full_tokens["input_ids"].squeeze()
    labels    = input_ids.clone()

    # Mask prompt tokens: set to -100 so they're ignored in loss
    labels[:prompt_len] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": full_tokens["attention_mask"].squeeze(),
        "labels":         labels,
    }


def setup_model(base_model_name, use_4bit=True):
    """
    Load the base model with optional 4-bit quantization.

    4-bit quantization (QLoRA) lets us fit large models on smaller GPUs.
    A 7-8B model that normally needs 16GB fits in ~8GB with 4-bit.
    """
    if not _LIBS_AVAILABLE:
        raise ImportError(
            "Training libraries not installed. "
            "Run: pip install transformers peft datasets accelerate bitsandbytes"
        )

    print(f"Loading base model: {base_model_name}")
    print(f"4-bit quantization: {use_4bit}")

    # Configure 4-bit quantization
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",         # NF4 quantization (QLoRA)
            bnb_4bit_use_double_quant=True,    # Nested quantization
        )
    else:
        quant_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS as PAD
    tokenizer.padding_side = "right"           # Pad on right

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto",         # Automatically distribute across GPUs
        trust_remote_code=True,
    )

    return model, tokenizer


def apply_lora(model, lora_config=None):
    """Apply LoRA adapters to the model."""
    if lora_config is None:
        lora_config = LORA_CONFIG

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
    )

    model = get_peft_model(model, config)

    # Print trainable parameter count (just for information)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}% of {total:,})")

    return model


def run_training(
    stage,
    data_dir,
    base_model_name,
    out_dir,
    epochs=None,
    use_4bit=True,
    dry_run=False,
):
    """
    Full fine-tuning pipeline for one stage.

    Args:
        stage:            1 (NL→IR) or 2 (IR→script)
        data_dir:         Directory with train.jsonl and val.jsonl
        base_model_name:  HuggingFace model name or local path
        out_dir:          Where to save the trained adapter
        epochs:           Training epochs (None = use stage default)
        use_4bit:         Use 4-bit quantization (QLoRA)
        dry_run:          If True, just show config without training
    """
    config     = STAGE_CONFIGS[stage]
    max_length = config["max_seq_length"]
    n_epochs   = epochs or config["epochs"]
    lr         = TRAINING_DEFAULTS["learning_rate"]

    print(f"\n{'='*60}")
    print(f"FINE-TUNING STAGE {stage}: {config['description']}")
    print(f"{'='*60}")
    print(f"  Base model:    {base_model_name}")
    print(f"  Data:          {data_dir}")
    print(f"  Output:        {out_dir}")
    print(f"  Epochs:        {n_epochs}")
    print(f"  Max length:    {max_length} tokens")
    print(f"  Learning rate: {lr}")
    print(f"  LoRA rank:     {LORA_CONFIG['r']}")
    print(f"  4-bit QLoRA:   {use_4bit}")

    if dry_run:
        print("\n  [DRY RUN] Not actually training. Remove --dry-run to train.")
        return

    if not _LIBS_AVAILABLE:
        print("\nERROR: Training libraries not installed.")
        print("Install with: pip install transformers peft datasets accelerate bitsandbytes")
        return

    # Load data
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path   = os.path.join(data_dir, "val.jsonl")

    if not os.path.exists(train_path):
        print(f"ERROR: Training data not found at {train_path}")
        print("Run prep_finetune_data.py first.")
        return

    print(f"\nLoading training data from {train_path}...")
    train_data = load_jsonl_dataset(train_path)
    val_data   = load_jsonl_dataset(val_path) if os.path.exists(val_path) else []
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val:   {len(val_data)} examples")

    # Load model and tokenizer
    model, tokenizer = setup_model(base_model_name, use_4bit=use_4bit)
    model = apply_lora(model)

    # Tokenize dataset
    print("\nTokenizing dataset...")
    train_tokenized = [tokenize_example(ex, tokenizer, max_length) for ex in train_data]
    val_tokenized   = [tokenize_example(ex, tokenizer, max_length) for ex in val_data]

    train_dataset = Dataset.from_list(train_tokenized)
    val_dataset   = Dataset.from_list(val_tokenized) if val_tokenized else None

    # Training arguments
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=TRAINING_DEFAULTS["per_device_batch"],
        gradient_accumulation_steps=TRAINING_DEFAULTS["grad_accumulation"],
        learning_rate=lr,
        weight_decay=TRAINING_DEFAULTS["weight_decay"],
        warmup_ratio=TRAINING_DEFAULTS["warmup_ratio"],
        lr_scheduler_type="cosine",
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=TRAINING_DEFAULTS["eval_steps"],
        save_strategy="steps",
        save_steps=TRAINING_DEFAULTS["save_steps"],
        logging_steps=TRAINING_DEFAULTS["logging_steps"],
        fp16=True,           # Use float16 for faster training
        report_to="none",    # Disable WandB logging by default
        load_best_model_at_end=True if val_dataset else False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    # Train!
    print(f"\nStarting training...")
    trainer.train()

    # Save the final model
    print(f"\nSaving model to {out_dir}...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Done! LoRA adapter saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune NL2Sim pipeline stages with LoRA"
    )
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True,
                        help="Which stage to fine-tune: 1=NL→IR, 2=IR→script")
    parser.add_argument("--data", required=True,
                        help="Directory with train.jsonl and val.jsonl")
    parser.add_argument("--base-model",
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--out-dir",
                        help="Output directory (default: models/stage{N}_lora/)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (default: stage-specific)")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (needs more VRAM)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show config without training")
    args = parser.parse_args()

    out_dir = args.out_dir or f"models/stage{args.stage}_lora"

    run_training(
        stage=args.stage,
        data_dir=args.data,
        base_model_name=args.base_model,
        out_dir=out_dir,
        epochs=args.epochs,
        use_4bit=not args.no_4bit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
