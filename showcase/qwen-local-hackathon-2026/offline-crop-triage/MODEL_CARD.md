---
license: apache-2.0
base_model: Qwen/Qwen3-VL-2B-Instruct
datasets:
- SyedNazmusSakib/PlantVillageVQA
pipeline_tag: image-text-to-text
tags:
- gguf
- llama.cpp
- agriculture
---

# Offline Crop Triage

Model card draft for a Q4_K_M language model with Qwen's original F16 vision projector. The intended use is assisted triage from a leaf photograph, including crop recognition, a description of visible symptoms and a possible diagnosis. Field accuracy is unknown. Images with clutter, mixed symptoms, unseen crops or poor lighting differ from PlantVillage images.

## Data and method

The base is Qwen3-VL-2B-Instruct revision `89644892e4d85e24eaac8bacfd4f463576704203`. Training used language-only rank-16, alpha-32 QLoRA with the vision encoder and projector frozen. The train, dev and test subsets contain 4,000, 300 and 350 QA pairs with disjoint image IDs. The dataset is [PlantVillageVQA by Sakib et al.](https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA), CC BY 4.0, based on [PlantVillage by Hughes and Salathé](https://arxiv.org/abs/1511.08060).

## Usage

```bash
llama-mtmd-cli -m CropTriage-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3VL-2B-Instruct-F16.gguf \
  -ngl 0 --no-mmproj-offload -c 4096 -t 8 \
  --image leaf.jpg -p 'What symptoms are visible and what condition could explain them?'
```

## Evaluation

Training ran in Colab on a Tesla T4 for one epoch (500 optimizer steps). The trainer reported 37 minutes 34 seconds, 0.373 mean training loss, and 0.102 dev loss. Dev token accuracy was 94.94%. These metrics describe training and development data. Both Q4 models completed a nine-question offline smoke test with llama.cpp b12800. The tuned model named the correct crop on the single crop-identification item and missed the single specific-diagnosis item. It also confused late blight with leaf scorch on a separate causal question. Full held-out results and Dell measurements are pending.

Before publication, add the paired Q4 results, a paired response, an observed failure and the target laptop’s measured runtime, RAM and offline execution result. Link the showcase workflow and demo.
