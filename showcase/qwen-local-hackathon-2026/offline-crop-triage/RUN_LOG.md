
- Exit status 0. Cold-start wall time including model load was **28.970 seconds**. Peak process RSS was **3,135,760 KiB**. The language GGUF is 1,107,409,952 bytes and the projector is 819,394,848 bytes, for **1,926,804,800 bytes** total.
- Model output: “Based on the image provided, there are no clearly visible target-like ring patterns in the dead tissue areas. The leaf shows signs of damage, including yellowing and brown spots, but the patterns are not distinct enough to be identified as targets. The areas of discoloration appear more like irregular spots or patches, possibly indicating various types of plant disease or decay.”
- The runtime emitted its Qwen-VL grounding advisory about minimum image tokens. Keep the image-token configuration identical across the paired benchmark.
- This is one base-model result on the workspace CPU. Dell performance and fine-tuning effects still require measurement.

## 27 November 2026, benchmark endpoint smoke

The evaluator started the real b12800 `llama-server` on loopback in an isolated Linux network namespace and submitted the same leaf JPEG to `/v1/chat/completions`. It returned the same answer as the CLI with finish reason `stop`. The server used two CPU threads, a 4,096-token context and a 96-token response limit. This verifies the base model's image-request path. Paired scoring and Dell measurements are pending the trained adapter.

All 42 existing repository tests and the repository content checks passed after adding this log.

## 27 November 2026, Colab training smoke

The participant shared Colab output showing all three optimizer steps and two validation passes over 300 QA pairs. The trainer reported finite training loss (3.1893), validation loss (1.2366) and first-step gradient norm (25.7140). Trainer runtime was 124.9 seconds, including validation. The notebook found `checkpoint-3/adapter_model.safetensors`. PyTorch emitted a gradient-checkpointing warning for inputs without gradients. The run completed and reported a nonzero gradient norm. This verifies training startup and adapter saving. Held-out performance remains to be measured after the full run and export.

## 27 November 2026, full Colab training

The participant shared Colab output confirming one full epoch and 500/500 optimizer steps. The trainer saved `checkpoint-500` and reported 2,253.84 seconds of training runtime including validation, mean training loss 0.3733, dev loss 0.1016, dev token accuracy 94.94%, and 3.43 GiB in its memory metric. The run completed despite BufferError messages earlier in the output. The participant still needs to export the evidence bundle from Colab so the adapter can be inspected and the held-out benchmark run.

## 27 November 2026, evidence transfer

The participant attached `crop-training-evidence.zip`. The workspace transfer tool rejected the file because it exceeds 32 MiB. The archive contents and adapter are awaiting verification after transfer as smaller binary parts.

## 27 November 2026, verified evidence bundle

Reassembled all three binary parts in order. ZIP integrity passed. The archive SHA-256 is `60a97105fba89ee5f25bd6abac111864a1315e019141926f60f43793d87d2585` and the adapter SHA-256 is `805088c2b306fa8864fcf3f37a9dbdba12cfa201f75ac3712bfe6bcc9403739b`. The adapter config confirms rank 16, alpha 32 and the pinned Qwen3-VL base revision. The GPU record identifies a Tesla T4 with 15,360 MiB of GPU memory. The training wrapper recorded 2,282 seconds elapsed including startup. The bundled 350 held-out rows match the locally prepared test JSONL byte for byte.

The export uses the official F16 language GGUF, applies the trained LoRA update to its language projections, and runs llama-quantize Q4_K_M. See [the export commands](README.md#export-directly-from-the-official-f16-gguf).

## 27 November 2026, Q4_K_M export

Verified every bundled held-out image against the prepared dataset (309 byte-identical images), and checked that the training, dev and test image ID sets are pairwise disjoint. The export script checked the official F16 digest and merged 196 LoRA pairs across 28 language blocks. Quantization used llama.cpp b12800 with two threads and took 148.94 seconds. The tuned language file is 1,107,409,952 bytes with SHA-256 `6fd5c005b5975b3f926417480be5e7789eafa98e101957b8076abc33d0b060bf`. The unchanged official F16 projector is 819,394,848 bytes. Combined size is 1,926,804,800 bytes.

The nine-row pairing runs in a fresh Linux network namespace with only loopback up. The base run overlapped export work, so its latency is not a controlled performance comparison with the tuned run.

## 27 November 2026, offline paired smoke test

Both Q4_K_M models completed all nine held-out questions with the original F16 projector, using llama.cpp b12800 on the workspace CPU. Each question came from a different image and question category. Both models identified the grape leaf. The base used “Vitis vinifera” and “grapevine,” which exposed a missing alias in the scorer. The corrected crop score is 1/1 for both models. Both models missed the single specific-diagnosis item. The tuned model also answered “Leaf scorch” on a causal question whose reference agent was the late-blight organism. The base named the expected agent. This sample confirms offline operation and shows mixed model quality. The full 350-row evaluation and Dell measurements are still pending.
