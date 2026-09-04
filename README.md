# Synthetic Dishonesty Probing

This repository contains code, lightweight metadata, configuration files, and JSON results for synthetic dishonesty probing experiments.

Large artifacts are intentionally excluded from the repository: `.npy` activation tensors, LoRA `adapter_model.safetensors` weights, zip archives, local virtual environments, and temporary system files. Upload large tensors and adapter weights to Hugging Face Hub, Hugging Face Datasets, or Zenodo, then add the links in the relevant README files.

## Reproduce

1. Create the environment with `conda env create -f environment.yml` or install `pip install -r requirements.txt`.
2. Add or download large activation tensors into `activation_extraction/activations`.
3. Add LoRA adapter weights from external storage into `finetuning/adapters`.
4. Run `python reproduce_tables.py` to rebuild paper tables from `configs/*.yaml` and `results/*.json`.

For gated Hugging Face models, set `HF_TOKEN` in your environment. Do not commit tokens or `.env` files.

## Layout

- `configs`: model and experiment configuration files.
- `data`: dataset extraction and paraphrase utilities plus lightweight JSON data found in the project.
- `finetuning`: LoRA training scripts and adapter metadata.
- `activation_extraction`: activation tensor documentation and placeholders.
- `probing`: probe training, transfer, leakage, and diagnostic scripts.
- `controls`: supplementary separability controls.
- `geometry_analysis`: geometric analyses and comprehensive activation analysis scripts.
- `paraphrase_robustness`: paraphrase transfer evaluation.
- `causal_steering`: activation steering scripts and generation placeholders.
- `results`: raw and derived lightweight JSON results used by the paper tables.
