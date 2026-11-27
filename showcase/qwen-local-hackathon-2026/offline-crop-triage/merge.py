"""Merge a language-only PEFT adapter into the pinned full-precision VLM on CPU."""

import argparse
import json
from pathlib import Path

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    config = json.loads((args.adapter / "adapter_config.json").read_text())
    if config["peft_type"] != "LORA" or config["r"] != 16 or config["lora_alpha"] != 32:
        parser.error("Expected the rank-16, alpha-32 LoRA adapter from train.sh")
    with safe_open(args.adapter / "adapter_model.safetensors", framework="pt") as weights:
        if not weights.keys() or any(".language_model." not in key for key in weights.keys()):
            parser.error("Expected all adapter tensors to belong to the language model")

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float32, device_map={"": "cpu"}
    )
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True, exist_ok=False)
    merged.to(torch.bfloat16).save_pretrained(args.output, safe_serialization=True)
    AutoProcessor.from_pretrained(MODEL, revision=REVISION).save_pretrained(args.output)


if __name__ == "__main__":
    main()
