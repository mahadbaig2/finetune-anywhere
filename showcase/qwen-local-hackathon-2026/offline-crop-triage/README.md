# Offline Crop Triage

A farmer or extension worker can give Qwen a leaf photo and get an estimated crop, visible symptoms, and a possible diagnosis with supporting observations. The CLI runs on a local llama.cpp server. The training and benchmark data come from PlantVillageVQA.

**Submission draft.** Dataset preparation and the full Colab training run are complete. The entry uses a provisional slug until the participant supplies their name. See the [run log](RUN_LOG.md) for measured results and remaining checks.

## 1. Prepare the data

Run from this entry folder. Use Python 3.11 or newer. In Colab, select a GPU runtime, clone the working branch, and use the same commands in a single `%%bash` cell after changing to this entry directory. The commands use `/content` as an example scratch location. Use `/kaggle/working` on Kaggle and enable Internet access during setup.

```bash
mkdir -p /content/crop-work
curl -fL --retry 3 \
  https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA/resolve/a025a438308eff9a7289231ebc09e27e8ec6678c/PlantVillageVQA.zip \
  -o /content/crop-work/PlantVillageVQA.zip
echo '4b15cc2a1edc1a14313200abadb549076788a5c7025b0d6c98750cf2ef1cfa6f  /content/crop-work/PlantVillageVQA.zip' | sha256sum -c -
python prepare.py /content/crop-work/PlantVillageVQA.zip /content/crop-work/data
```

The archive contains 193,609 QA rows referencing 50,073 unique image IDs. Preparation reads the original annotations, joins actual ZIP members case-insensitively, and creates a new seed-42 split. It assigns repeated IDs and byte-identical copies to the same partition before sampling. The default sample has 4,000 train, 300 dev and 350 test QA rows, with all nine question categories and 38 inferred crop–condition combinations in each split. The train, dev and test subsets contain 3,412, 263 and 309 distinct image IDs respectively. `split.json` records image IDs, category counts and inferred crop/condition coverage. Crop and condition labels inferred from reference answers are used for sampling and omitted from prompts. Some labels are unknown and the assignments have not been independently verified. Exact-byte checks do not identify edited or near-duplicate photos.

The train and dev JSONL store absolute image paths for ms-swift. Prepare these files on the GPU machine. To move the held-out set to the laptop, transfer `test.jsonl` plus the referenced images, keeping the `images/` layout. The following cell packages only the held-out data.

```bash
python - <<'PY'
import json
from pathlib import Path
from zipfile import ZipFile

root = Path('/content/crop-work/data')
test = root / 'test.jsonl'
with ZipFile(root.parent / 'held-out.zip', 'w') as bundle:
    bundle.write(test, test.name)
    for image in sorted({json.loads(line)['image'] for line in test.read_text().splitlines()}):
        bundle.write(root / image, image)
PY
```

PlantVillageVQA is CC BY 4.0. Credit Sakib, Haque, Hossain and Arman and the original PlantVillage corpus by Hughes and Salathé. The source and its attribution guidance are in the [dataset card](https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA).

## 2. Train on Colab or Kaggle

Use the CUDA-enabled PyTorch already installed by the GPU runtime. Training uses Qwen3-VL-2B-Instruct revision `89644892e4d85e24eaac8bacfd4f463576704203`, 4-bit bitsandbytes loading, and rank-16, alpha-32 LoRA on language attention and MLP projections only. The vision encoder and aligner are frozen. One epoch uses batch size 1, eight accumulation steps, FP16, SDPA, learning rate 2e-4 and a 768-token limit. The run directory captures the actual package versions, GPU, split manifest, log and wall time.

```bash
nvidia-smi
python -m pip install 'ms-swift==4.6.1' 'transformers==4.57.6' 'qwen-vl-utils==0.0.14' decord bitsandbytes
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
MAX_STEPS=3 bash train.sh /content/crop-work/data /content/crop-work/smoke
bash train.sh /content/crop-work/data /content/crop-work/full
find /content/crop-work/full -name adapter_model.safetensors -print
```

Check the smoke log for finite loss and a saved adapter before starting the full run. If training on Kaggle, update `training_platform` in the manifest to `kaggle`. Use the full run's final checkpoint as the adapter and copy its complete folder to the laptop with the run log and split manifest. Training duration is measured by the script and depends on the GPU and session. Keep the test data out of tuning decisions.

## 3. Merge, convert, and quantize on the Dell

Use Linux or WSL2. Provide a CPU PyTorch install and set `WORK` to a directory with space for the downloaded base, merged BF16 weights, unquantized GGUF and Q4 files. Run these commands from the entry directory. Place the adapter checkpoint under `"$WORK/adapter"` and extract the held-out bundle to `"$WORK/held-out"`.

```bash
export WORK="$HOME/crop-work"
mkdir -p "$WORK" "$WORK/artifacts" "$WORK/held-out"
python3.11 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'transformers==4.57.6' peft accelerate safetensors huggingface_hub
python merge.py "$WORK/adapter" "$WORK/merged"

git clone https://github.com/ggml-org/llama.cpp "$WORK/llama.cpp"
git -C "$WORK/llama.cpp" checkout a866b4ade63921826c530307a368fbca3a334a0b
cmake -S "$WORK/llama.cpp" -B "$WORK/llama.cpp/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$WORK/llama.cpp/build" --config Release -j 4 \
  --target llama-server llama-quantize llama-mtmd-cli
python -m pip install -r "$WORK/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"

python "$WORK/llama.cpp/convert_hf_to_gguf.py" "$WORK/merged" \
  --outtype f16 --outfile "$WORK/tuned-f16.gguf"
"$WORK/llama.cpp/build/bin/llama-quantize" "$WORK/tuned-f16.gguf" \
  "$WORK/artifacts/CropTriage-Q4_K_M.gguf" Q4_K_M

hf download Qwen/Qwen3-VL-2B-Instruct-GGUF \
  Qwen3VL-2B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-2B-Instruct-F16.gguf \
  --revision 52d6c8ffea26cc873ac5ad116f8631268d7eb503 --local-dir "$WORK/base"
cp "$WORK/base/mmproj-Qwen3VL-2B-Instruct-F16.gguf" "$WORK/artifacts/"
```

`merge.py` rejects adapter tensors outside the language model, merges on a fresh FP32 base and saves BF16. The separate F16 projector comes from Qwen's original GGUF release and serves both runners. The converter exports the language GGUF in this invocation.

### Export directly from the official F16 GGUF

[merge_gguf.py](merge_gguf.py) is the lower-memory export used in the cloud workspace. It verifies the official F16 file checksum, then applies the LoRA update in FP32 to each language projection before storing the result in F16. It writes into the downloaded scratch file. The separate vision projector is unchanged. Use the llama.cpp b12800 source tree for its GGUF tensor mapping and the b12800 runtime for quantization.

```bash
git clone --depth 1 --branch b12800 https://github.com/ggml-org/llama.cpp "$WORK/llama-b12800-source"
python -m pip install numpy safetensors huggingface_hub
hf download Qwen/Qwen3-VL-2B-Instruct-GGUF Qwen3VL-2B-Instruct-F16.gguf \
  --revision 52d6c8ffea26cc873ac5ad116f8631268d7eb503 --local-dir "$WORK/export-scratch"
PYTHONPATH="$WORK/llama-b12800-source/gguf-py" python merge_gguf.py \
  "$WORK/adapter" "$WORK/export-scratch/Qwen3VL-2B-Instruct-F16.gguf"
llama-quantize "$WORK/export-scratch/Qwen3VL-2B-Instruct-F16.gguf" \
  "$WORK/artifacts/CropTriage-Q4_K_M.gguf" Q4_K_M 2
```

## 4. Run and benchmark offline

After downloading dependencies and artifacts, disconnect Wi-Fi and any other Internet connection. Use a local photo and local file paths. Run the server in its own terminal, then run the demo in another terminal with the same `WORK` value.

```bash
"$WORK/llama.cpp/build/bin/llama-server" \
  -m "$WORK/artifacts/CropTriage-Q4_K_M.gguf" \
  --mmproj "$WORK/artifacts/mmproj-Qwen3VL-2B-Instruct-F16.gguf" \
  --host 127.0.0.1 --port 8172 -ngl 0 --no-mmproj-offload -c 4096 -t 8 -np 1 --alias plant --jinja
```

```bash
python triage.py /absolute/path/to/leaf.jpg
```

Stop the demo server before evaluation. Run once with `--limit 9 --out "$WORK/eval-smoke"` in place of the final `--out` argument to verify both models load. Run the full comparison in a separate output directory. Budget for 700 timed CPU generations using the smoke-run latency.

```bash
python evaluate.py \
  --server "$WORK/llama.cpp/build/bin/llama-server" \
  --base "$WORK/base/Qwen3VL-2B-Instruct-Q4_K_M.gguf" \
  --tuned "$WORK/artifacts/CropTriage-Q4_K_M.gguf" \
  --projector "$WORK/artifacts/mmproj-Qwen3VL-2B-Instruct-F16.gguf" \
  --test "$WORK/held-out/test.jsonl" --out "$WORK/eval"
```

Both models use the same projector, QA rows, context and generation settings. `summary.json` gives accuracy for open crop and condition identification, separate binary crop verification accuracy, and token F1 / ROUGE-L F1 by category and level. Accuracy uses canonical names and aliases, so review negations, alternative diagnoses and atypical wording against raw outputs. Lexical overlap measures agreement with the reference text. Score a small paired sample by hand for visible symptom support and causal plausibility. Latency is seconds per QA request including its image, after one warmup per model. Peak RAM is the server process's Linux high-water RSS. Artifact size includes model and projector. Record the exact laptop CPU, RAM, OS, runtime revision and the offline check during the run.

## 5. Publish and submit

Copy [MODEL_CARD.md](MODEL_CARD.md) to `"$WORK/artifacts/README.md"` and fill in measured results, limitations and an inference example. Upload the two GGUF files and the card to Hugging Face and ModelScope using the [repository publishing guide](../../../guides/07-share-your-model.md). Link both hosts in this README and the primary model host in [model.yaml](model.yaml).

Use the generated `comparison.md` table and `summary.json` for the aggregate results. Pair examples from `base.jsonl` and `tuned.jsonl` by `row_id` and copy a real example into `before_after`, including the question and leaf ID. Document one failure and report whether each metric improved. The PlantVillage data largely show single leaves against controlled backgrounds, so performance on cluttered field photos or unseen crops is unknown. Use the output for triage and seek expert confirmation before acting on a diagnosis.

In the demo, show the context for farmers and extension workers, disconnect the laptop from the Internet, run a held-out photo through both models, and show the paired responses and measured tradeoffs. Post the demo publicly on LinkedIn or X, tag M Junaid and Qwen Pakistan, and use `#QwenLocalHackathon`. Set `demo_video` to that post URL. Add team names and GitHub handles, align the entry folder and slug with the participant's chosen name, and open a PR from the fork with only the entry folder. Submit the separate event form as well. The [event README](../README.md) lists the rules and deadline of Saturday, 28 November 2026, 11:59 PM PKT.
