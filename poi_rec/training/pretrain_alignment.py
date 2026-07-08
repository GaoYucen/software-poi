from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from poi_rec.data.dataset import load_metadata, load_processed_arrays
from poi_rec.losses.alignment_loss import combined_alignment_loss
from poi_rec.models.poi_model import POIRecommendationModel
from poi_rec.training.logging_utils import setup_run_logger
from poi_rec.training.train import _amp_dtype, _configure_torch_runtime, _dataloader_kwargs, _ensure_processed
from poi_rec.utils.random import resolve_device, set_seed


def _alignment_state_dict(model: POIRecommendationModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if name.startswith("topology.") or name.startswith("semantic.") or name.startswith("alignment.")
    }


def _alignment_checkpoint_is_compatible(path: Path, metadata: dict[str, Any], device: torch.device) -> bool:
    if not path.exists():
        return False
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("preprocess_fingerprint") != metadata.get("preprocess_fingerprint"):
        return False
    for key in ("num_pois", "num_categories", "node2vec_dim", "text_embedding_dim"):
        if state.get(key) != metadata.get(key):
            return False
    return "alignment_state" in state


def pretrain_alignment_from_config(config: dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    _ensure_processed(config)
    processed_dir = Path(config["processed_dir"])
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger(
        "poi_rec.alignment_pretrain",
        Path(config.get("alignment_pretrain_log_path", run_dir / "alignment_pretrain.log")),
        log_to_console=bool(config.get("log_to_console", True)),
    )
    logger.info("alignment pretraining started; run_dir=%s", run_dir)

    metadata = load_metadata(processed_dir)
    device = resolve_device(str(config.get("device", "auto")))
    _configure_torch_runtime(config, device)
    checkpoint_path = Path(config.get("alignment_checkpoint") or run_dir / "alignment_pretrain.pt")
    if bool(config.get("reuse_alignment_checkpoint", True)) and _alignment_checkpoint_is_compatible(
        checkpoint_path,
        metadata,
        device,
    ):
        logger.info("reuse compatible alignment checkpoint: %s", checkpoint_path)
        return
    arrays = load_processed_arrays(processed_dir)
    train_seen = arrays["train_seen_poi"].bool()
    poi_ids = torch.nonzero(train_seen, as_tuple=False).flatten()
    if poi_ids.numel() == 0:
        raise ValueError("No train-seen POIs are available for alignment pretraining.")
    dataset = TensorDataset(poi_ids)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        **_dataloader_kwargs(config, device),
    )
    arrays = {key: value.to(device) for key, value in arrays.items()}
    pretrain_config = dict(config)
    pretrain_config["use_pretrained_gpt"] = False
    pretrain_config["fallback_to_random_gpt"] = True
    pretrain_config["gpt_freeze_policy"] = "random"
    model = POIRecommendationModel(metadata, arrays, pretrain_config).to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            name.startswith("topology.") or name.startswith("semantic.") or name.startswith("alignment.")
        )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config.get("alignment_pretrain_lr", config["lr"])),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    epochs = int(config.get("alignment_pretrain_epochs", 1))
    instance_weight = float(config.get("align_instance_weight", 1.0))
    feature_weight = float(config.get("align_feature_weight", 0.25))
    amp_enabled = bool(config.get("amp", False)) and device.type == "cuda"
    amp_dtype = _amp_dtype(config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            raw_poi = batch[0].to(device)
            shifted_poi = raw_poi + 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                topology = model.topology(shifted_poi)
                semantic = model.semantic(shifted_poi)
                aligned_topology, aligned_semantic = model.alignment(topology, semantic)
                mask = torch.ones((1, shifted_poi.shape[0]), dtype=torch.long, device=device)
                loss = combined_alignment_loss(
                    aligned_topology.unsqueeze(0),
                    aligned_semantic.unsqueeze(0),
                    mask,
                    shifted_poi.unsqueeze(0),
                    instance_weight=instance_weight,
                    feature_weight=feature_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        logger.info("alignment pretrain epoch %d/%d result: loss=%.6f", epoch, epochs, total / max(1, len(loader)))

    path = checkpoint_path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "alignment_state": _alignment_state_dict(model),
            "config": config,
            "metadata": metadata,
            "preprocess_fingerprint": metadata.get("preprocess_fingerprint"),
            "num_pois": metadata.get("num_pois"),
            "num_categories": metadata.get("num_categories"),
            "text_embedding_dim": metadata.get("text_embedding_dim"),
            "node2vec_dim": metadata.get("node2vec_dim"),
        },
        path,
    )
    logger.info("alignment checkpoint: %s", path)
