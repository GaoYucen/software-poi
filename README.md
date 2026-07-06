
# POI Next Recommendation MVP

This repo implements a Path-LLM-inspired next POI recommendation prototype for the TSMC2014/Foursquare NYC and Tokyo check-in data.

The current pipeline uses schema v4 processed artifacts:

- weighted directed POI transition graph from training-only adjacent visits
- Node2Vec topology embeddings
- leakage-free POI text descriptions with optional pretrained Transformer embeddings
- right-padded histories with a dedicated PAD ID
- unique-POI topology/semantic contrastive alignment
- normalized query/candidate matching
- training-only user-POI preference priors for reranking
- closed-world full-ranking protocol for the main experiment

## Environment

Use the existing conda environment:

```bash
conda run -n py11 python -m pip install transformers safetensors tokenizers tqdm networkx
```

## Debug Smoke Test

```bash
conda run -n py11 python scripts/preprocess.py --config configs/debug.yaml
conda run -n py11 python scripts/pretrain_alignment.py --config configs/debug.yaml
conda run -n py11 python scripts/train.py --config configs/debug.yaml
conda run -n py11 python scripts/evaluate.py --checkpoint runs/debug/best.pt --split val
```

You can override config values from the CLI for quick experiments:

```bash
conda run -n py11 python scripts/train.py --config configs/debug.yaml --override epochs=5 --override run_dir=runs/debug_e5
conda run -n py11 python scripts/evaluate.py --checkpoint runs/debug/best.pt --split val --override transition_prior_weight=16
```

Score-prior sweeps reuse a fixed checkpoint and do not retrain the model:

```bash
conda run -n py11 python scripts/sweep_priors.py --checkpoint runs/debug/best.pt --split val --top 10
```

You can also evaluate training-only priors without a neural checkpoint:

```bash
conda run -n py11 python scripts/evaluate_priors.py --processed_dir processed/NYC_tfidf_v4 --split val --max_seq_len 20
```

## NYC MVP

```bash
conda run -n py11 python scripts/preprocess.py --config configs/nyc_mvp.yaml
conda run -n py11 python scripts/pretrain_alignment.py --config configs/nyc_mvp.yaml
conda run -n py11 python scripts/train.py --config configs/nyc_mvp.yaml
conda run -n py11 python scripts/evaluate.py --checkpoint runs/nyc_mvp/best.pt --split test
```

`configs/nyc_mvp.yaml` requires pretrained `gpt2` when `use_pretrained_gpt: true`. Set `fallback_to_random_gpt: true` only for debug or smoke-test runs.
The formal NYC config also uses `semantic_encoder: hf_transformer`, so the configured HuggingFace text model must be available locally or downloadable.

See `protocol.md` for the split, candidate, metric, and modality boundaries used for comparable experiments.

## Tests

```bash
conda run -n py11 python -m unittest discover -s tests
```
