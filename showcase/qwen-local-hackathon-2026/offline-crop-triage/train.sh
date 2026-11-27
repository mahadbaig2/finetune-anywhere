#!/usr/bin/env bash
set -euo pipefail
data=$(realpath "${1:?Pass the prepared data directory}")
out=$(realpath -m "${2:?Pass a new run directory}")
mkdir "$out"
python -m pip freeze > "$out/requirements.lock"
nvidia-smi > "$out/gpu.txt"
cp "$data/split.json" "$out/split.json"
export IMAGE_MAX_TOKEN_NUM=128
steps=()
if [[ -n ${MAX_STEPS:-} ]]; then
    steps=(--max_steps "$MAX_STEPS")
fi
SECONDS=0
swift sft \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --model_revision 89644892e4d85e24eaac8bacfd4f463576704203 --use_hf true \
    --tuner_type lora --target_regex '.*language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)' \
    --freeze_vit true --freeze_aligner true --lora_rank 16 --lora_alpha 32 \
    --quant_method bnb --quant_bits 4 --bnb_4bit_compute_dtype float16 \
    --torch_dtype float16 --attn_impl sdpa --gradient_checkpointing true \
    --dataset "$data/train.jsonl" --val_dataset "$data/dev.jsonl" --split_dataset_ratio 0 \
    --max_length 768 --num_train_epochs 1 --learning_rate 0.0002 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 --seed 42 --data_seed 42 \
    --eval_strategy epoch --save_strategy epoch --save_total_limit 2 \
    --logging_steps 10 --report_to none --output_dir "$out" "${steps[@]}" \
    2>&1 | tee "$out/training.log"
echo "$SECONDS" > "$out/training-seconds.txt"
