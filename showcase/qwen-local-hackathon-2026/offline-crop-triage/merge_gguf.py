"""Merge the crop-triage LoRA into a disposable copy of Qwen's official F16 GGUF."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import re

BASE_SHA256 = "27b80f2e258f8372c9ee3a9b2293bd747cc08b18ed99709b57288cf457c42595"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
KEY = re.compile(
    r"base_model\.model\.model\.language_model\."
    r"(layers\.\d+\.(?:self_attn|mlp)\.\w+)\.lora_([AB])\.weight"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("scratch_f16_gguf", type=Path, help="Disposable GGUF copy, overwritten in place")
    args = parser.parse_args()

    import gguf
    import numpy as np
    from safetensors.numpy import load_file

    config = json.loads((args.adapter / "adapter_config.json").read_text())
    if (config["peft_type"], config["r"], config["lora_alpha"], config["bias"]) != ("LORA", 16, 32, "none"):
        parser.error("Expected the rank-16, alpha-32 adapter with no trained bias")
    if config["base_model_name_or_path"].rstrip("/").split("/")[-1] != REVISION:
        parser.error("Adapter base revision differs from the pinned training model")
    for option in ("use_dora", "use_rslora", "use_qalora", "fan_in_fan_out", "lora_bias",
                   "modules_to_save", "rank_pattern", "alpha_pattern"):
        if config.get(option):
            parser.error(f"Unsupported adapter option: {option}")
    with args.scratch_f16_gguf.open("rb") as source:
        if file_digest(source, "sha256").hexdigest() != BASE_SHA256:
            parser.error("Scratch GGUF does not match the official Qwen F16 checksum")

    reader = gguf.GGUFReader(args.scratch_f16_gguf, "r+")
    if reader.fields["general.architecture"].contents() != "qwen3vl":
        parser.error("Expected Qwen3-VL GGUF")
    blocks = reader.fields["qwen3vl.block_count"].contents()
    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3VL, blocks)
    base = {tensor.name: tensor.data for tensor in reader.tensors}
    factors = {}
    for key, value in load_file(args.adapter / "adapter_model.safetensors").items():
        match = KEY.fullmatch(key)
        if match is None or not np.isfinite(value).all():
            parser.error(f"Unexpected adapter tensor: {key}")
        name = name_map.get_name(f"model.{match[1]}.weight", try_suffixes=(".weight",))
        if name is None or name not in base:
            parser.error(f"No matching GGUF tensor for {key}")
        factors.setdefault(name, {})[match[2]] = value
    if len(factors) != blocks * 7:
        parser.error(f"Expected seven modified projections per block, found {len(factors)} tensors")
    for name, pair in factors.items():
        rows, cols = base[name].shape
        if (base[name].dtype != np.float16 or pair.keys() != {"A", "B"}
                or pair["A"].shape != (16, cols) or pair["B"].shape != (rows, 16)):
            parser.error(f"Unexpected format or shape for {name}")
    for name, pair in factors.items():
        weight = base[name]
        delta = pair["B"].astype(np.float32) @ pair["A"].astype(np.float32)
        weight[:] = weight.astype(np.float32) + np.float32(2) * delta
    reader.data.flush()
    print(f"Merged {len(factors)} language tensors into {args.scratch_f16_gguf}")


if __name__ == "__main__":
    main()
