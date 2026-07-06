# POI Next Recommendation MVP

This repo implements a Path-LLM-inspired next POI recommendation prototype for the TSMC2014/Foursquare NYC and Tokyo check-in data.

## Environment

Use the existing conda environment:

```bash
conda run -n py11 python -m pip install transformers safetensors tokenizers tqdm networkx
```

## Debug Smoke Test

```bash
conda run -n py11 python scripts/preprocess.py --city NYC --raw data/dataset_TSMC2014_NYC.txt --out processed/debug_nyc --limit_users 50 --max_seq_len 12
conda run -n py11 python scripts/train.py --config configs/debug.yaml
conda run -n py11 python scripts/evaluate.py --checkpoint runs/debug/best.pt --split val
```

## NYC MVP

```bash
conda run -n py11 python scripts/preprocess.py --city NYC --raw data/dataset_TSMC2014_NYC.txt --out processed/NYC
conda run -n py11 python scripts/train.py --config configs/nyc_mvp.yaml
conda run -n py11 python scripts/evaluate.py --checkpoint runs/nyc_mvp/best.pt --split test
```

`configs/nyc_mvp.yaml` tries to load pretrained `gpt2` through HuggingFace. If the weights are unavailable, the model falls back to a randomly initialized GPT-2 architecture so the training pipeline still runs.

## Tests

```bash
conda run -n py11 python -m unittest discover -s tests
```

