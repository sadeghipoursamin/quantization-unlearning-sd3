# Quantization × Machine Unlearning in Stable Diffusion 3

Does post-training quantization (PTQ) bring back nudity in a nudity-unlearned text-to-image model?
This repository contains the code, per-prompt results and report for a study of a DUO-unlearned
**Stable Diffusion 3 Medium** under INT8 / INT4 quantization with two PTQ methods and three selective-quantization strategies.

**Full write-up: [`REPORT.md`](REPORT.md)** (methods, all result tables, analysis, limitations).

## Headline findings
* No quantization configuration produced a detectable return of nudity in the unlearned model (all 95% intervals for the change in Nudity Generation Rate span zero; a large return is excluded, a small one is not).
* The baseline's large NGR drop under quantization is mostly a benchmark artifact: the 184 prompts were selected because the fp16 baseline was positive on them. With no quantization, new seeds alone give 38.6% (baseline) and 21.2% (unlearned): a real unlearning effect of about −17 pp (−45% relative), not 100% → 28.8%.
* DUO changed only 96 attention layers by ≈0.15% of their norm (7–80× below the rounding error), but the change survives quantization on average (≈100% retained); only 0.37% (INT4) / 3.5% (INT8) of weights change their integer code.

## Repository layout
```
REPORT.md                      final report (all parts of the task)
code/quant_utils.py            fixed generation, NudeNet + CLIP evaluation, Q1/Q2/selective loaders, module audit, mechanism analysis
code/analyze_results.py        deltas, bootstrap intervals, controls, result table, plots
config/inference_config.json   frozen inference configuration (Part 3)
data/frozen_184_nudity_prompts.csv   frozen nudity benchmark (subset of Six-CD Nudity.csv)
results/*.json                 per-prompt results for every configuration (no images)
results/*_audit.txt            which modules were / were not quantized, per run
results/analysis_*.csv, results_table.md, delta_*.csv
results/figures/*.png          plots (Part 24)
results/results_env.txt        library versions
notebooks/                     Colab notebooks (outputs cleared): unlearning + baseline evaluation; quantization experiments
checkpoints/pytorch_lora_weights.safetensors   the DUO LoRA (if present)
docs/                          earlier development logs (superseded by REPORT.md)
```

## Data, models and what is not in this repository
* **Prompts.** Evaluation prompts come from the public Six-CD benchmark (https://github.com/Artanisax/Six-CD): `Datasets/SIX-CD/Nudity.csv` (candidate pool) and `Datasets/Dual-Version/Nudity/clean.csv` (100 benign prompts). Download:
  ```
  wget https://raw.githubusercontent.com/Artanisax/Six-CD/main/Datasets/SIX-CD/Nudity.csv
  wget https://raw.githubusercontent.com/Artanisax/Six-CD/main/Datasets/Dual-Version/Nudity/clean.csv
  ```
  The frozen 184-prompt subset used for all evaluations is in `data/`.
* **Base model.** `stabilityai/stable-diffusion-3-medium-diffusers` (gated on Hugging Face).
* **Unlearned model.** DUO (https://github.com/naver-ai/DUO), rank-32 LoRA, 1000 steps; training commands and the two small patches to the DUO scripts are documented in `docs/` and the notebook. The LoRA weights are in `checkpoints/`. The fused full checkpoint (`sd3-nudity-unlearned-fixed`, several GB) is **not** in git: `<< https://drive.google.com/drive/folders/1r-ciBHpkjsbXtC-zjpppKGhZH4oYXEvT?usp=drive_link >>`.
* **No images are stored.** Generated images and the DUO training images (which contain nudity) are deliberately excluded; results are per-prompt NudeNet scores, CLIP scores and CLIP image embeddings.

## Reproducing the experiments (Google Colab, A100)
1. `pip install -U diffusers transformers accelerate sentencepiece nudenet bitsandbytes`; log in to Hugging Face.
2. Mount Drive with the layout expected by `quant_utils.py` (`DUO_project/{config,datasets/SixCD,results,code}`), then `%run -i code/quant_utils.py`.
3. fp16 references: `run("B_FP16", lambda: load_fp16(BASE))`, `run("U_FP16", lambda: load_fp16(UNL))`.
4. Full-model quantization (names are `{B|U}_{Q1|Q2}_{INT8|INT4}`):
   `run(name, lambda: load_q1(path, bits))` and `run(name, lambda: load_q2_sim(path, bits))`.
5. Selective (Q2, INT4): `load_q2_sel(path, 4, ["transformer"])` (Sel-A), `load_q2_sel(path, 4, [text encoders])` (Sel-B), `load_q2_sel(path, 4, ALL_PARTS, protect=names_of_96_changed_layers)` (Sel-C).
6. Controls: `run(f"{m}_FP16_seed7777", ..., seed_offset=7777)` and the same for `Q1_INT4` / `Q2_INT4`.
7. `delta_vs_quant_error()`, `delta_survival()`, then `%run code/analyze_results.py`.

Every long loop is resume-safe (progress is saved to `results/<name>.json` every 10 prompts).

## Methods in one paragraph
Q1 = bitsandbytes (LLM.int8 for INT8, NF4 for INT4, fp16 compute). Q2 = uniform round-to-nearest, simulated in fp16 (per-channel INT8 weights + per-token dynamic INT8 activations; group-wise INT4 weights, group size 64). Both are calibration-free. Evaluation: NudeNet (5 classes, score > 0.5) on the frozen 184 prompts, and CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) on 100 benign prompts, with fixed seeds, 1024², 28 steps, guidance 7.0.

## Deviations from the task text
N = 184 instead of 200 (supervisor-approved); Q2 simulated because real torchao failed under the fixed fp16 config; FP16 rows equal the reference (the pipeline is already fp16); Q1 and Q2 differ in module coverage; extra controls (fresh-seed fp16 and INT4) and extra metrics (image drift, continuous NudeNet score). See `REPORT.md`.
