"""
Singleton LLM interface using Unsloth FastLanguageModel.
Loads model once at startup, then complete() uses the cached model.
"""
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
from typing import Optional, List, Dict, Tuple

# Global variables (module-level cache)
_model = None
_tokenizer = None

def initialize(model_name: str = "unsloth/gpt-oss-20b",
               max_seq_length: int = 4096,
               load_in_4bit: bool = False,
               full_finetuning: bool = False,
               device_map: str = "auto") -> Tuple:
    """Load model once. Must be called before any complete() call."""
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            dtype=None,                # auto detection
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            full_finetuning=full_finetuning,
            device_map=device_map,
        )
        print(f"Model {model_name} loaded successfully.")
    return _model, _tokenizer

def complete(prompt: str,
             system_prompt: Optional[str] = None,
             temperature: float = 0.0,
             max_tokens: int = 512,
             stream: bool = False) -> str:
    """
    Generate text using the loaded model.
    If system_prompt is provided, it is inserted as a system message.
    """
    global _model, _tokenizer
    if _model is None:
        raise RuntimeError("Model not initialized. Call initialize() first.")

    # Build message list
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Apply chat template
    inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(_model.device)

    # Generation kwargs
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0,
        "top_p": 0.95 if temperature > 0 else 1.0,
    }
    if stream:
        streamer = TextStreamer(_tokenizer)
        gen_kwargs["streamer"] = streamer

    with torch.no_grad():
        outputs = _model.generate(**inputs, **gen_kwargs)

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    response = _tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()
