from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from poi_rec.data.candidates import DynamicCandidateGenerator
from poi_rec.data.dataset import POISequenceDataset, load_metadata, load_processed_arrays
from poi_rec.data.preprocess import build_preprocess_config, preprocess_fingerprint, preprocess_tsmc2014
from poi_rec.losses.alignment_loss import combined_alignment_loss
from poi_rec.losses.recommendation_loss import next_poi_loss_per_sample
from poi_rec.models.poi_model import POIRecommendationModel
from poi_rec.training.checkpoint import save_checkpoint
from poi_rec.training.logging_utils import setup_run_logger
from poi_rec.training.metrics import average_metric_dicts, ranking_metric_sums
from poi_rec.utils.random import resolve_device, set_seed


def _configure_torch_runtime(config: dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    allow_tf32 = bool(config.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", True))
    matmul_precision = config.get("float32_matmul_precision")
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(str(matmul_precision))


def _amp_dtype(config: dict[str, Any]) -> torch.dtype:
    name = str(config.get("amp_dtype", "bf16")).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError("amp_dtype must be one of: bf16, bfloat16, fp16, float16, half")


def _dataloader_kwargs(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    num_workers = int(config.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(config.get("pin_memory", device.type == "cuda")),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return kwargs


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _build_sampled_candidates(
    batch: dict[str, torch.Tensor],
    train_seen_ids: torch.Tensor,
    num_candidates: int,
    *,
    transition_top_ids: dict[int, list[int]] | None = None,
    user_poi_top_ids: dict[int, list[int]] | None = None,
    popularity_ids: torch.Tensor | None = None,
    hard_history_width: int = 20,
    hard_user_poi_top: int = 64,
    hard_transition_top: int = 64,
    hard_history_transition_top: int = 16,
    hard_popularity_top: int = 64,
) -> torch.Tensor:
    target = batch["target"]
    candidate_count = max(2, num_candidates)
    device = target.device
    target_cpu = target.detach().cpu().tolist()
    poi_cpu = batch["poi"].detach().cpu()
    mask_cpu = batch["attention_mask"].detach().cpu()
    user_cpu = batch["user_idx"].detach().cpu().tolist()
    train_seen_cpu = train_seen_ids.detach().cpu()
    popularity_cpu = popularity_ids.detach().cpu() if popularity_ids is not None else torch.empty(0, dtype=torch.long)

    rows: list[list[int]] = []
    for row_idx, target_id in enumerate(target_cpu):
        seen = {int(target_id)}
        row = [int(target_id)]

        def add_candidate(candidate_id: int) -> None:
            if len(row) >= candidate_count:
                return
            candidate_id = int(candidate_id)
            if candidate_id in seen:
                return
            seen.add(candidate_id)
            row.append(candidate_id)

        valid_positions = torch.nonzero(mask_cpu[row_idx].gt(0), as_tuple=False).flatten().tolist()
        history = [int(poi_cpu[row_idx, pos].item()) - 1 for pos in valid_positions]
        history = [poi for poi in history if poi >= 0]
        for poi_id in reversed(history[-max(0, hard_history_width) :]):
            add_candidate(poi_id)

        user_top = (user_poi_top_ids or {}).get(int(user_cpu[row_idx]), [])
        for poi_id in user_top[: max(0, hard_user_poi_top)]:
            add_candidate(poi_id)

        if history:
            last_poi = history[-1]
            for poi_id in (transition_top_ids or {}).get(last_poi, [])[: max(0, hard_transition_top)]:
                add_candidate(poi_id)
            if hard_history_transition_top > 0:
                for source_poi in reversed(history[:-1]):
                    for poi_id in (transition_top_ids or {}).get(source_poi, [])[:hard_history_transition_top]:
                        add_candidate(poi_id)
                        if len(row) >= candidate_count:
                            break
                    if len(row) >= candidate_count:
                        break

        for poi_id in popularity_cpu[: max(0, hard_popularity_top)].tolist():
            add_candidate(int(poi_id))

        while len(row) < candidate_count:
            random_idx = torch.randint(low=0, high=train_seen_cpu.shape[0], size=(1,)).item()
            add_candidate(int(train_seen_cpu[random_idx].item()))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long, device=device)


def _precompute_top_prior_ids(
    prior: dict[int, tuple[torch.Tensor, torch.Tensor]],
    topn: int,
) -> dict[int, list[int]]:
    if topn <= 0:
        return {}
    top_ids: dict[int, list[int]] = {}
    for key, (ids, values) in prior.items():
        if ids.numel() == 0:
            continue
        order = torch.argsort(values, descending=True)[:topn]
        top_ids[int(key)] = [int(ids[idx].item()) for idx in order]
    return top_ids


def _ensure_processed(config: dict[str, Any]) -> None:
    processed_dir = Path(config["processed_dir"])
    metadata_path = processed_dir / "metadata.json"
    preprocess_config = build_preprocess_config(
        raw_path=Path(config["raw_path"]),
        city=str(config.get("city", "NYC")),
        max_seq_len=int(config["max_seq_len"]),
        min_user_checkins=int(config.get("min_user_checkins", 2)),
        val_ratio=float(config.get("val_ratio", 0.1)),
        test_ratio=float(config.get("test_ratio", 0.1)),
        limit_users=config.get("limit_users"),
        node2vec_dim=int(config.get("node2vec_dim", 64)),
        node2vec_walk_length=int(config.get("node2vec_walk_length", 10)),
        node2vec_num_walks=int(config.get("node2vec_num_walks", 5)),
        node2vec_p=float(config.get("node2vec_p", 1.0)),
        node2vec_q=float(config.get("node2vec_q", 1.0)),
        text_encoder=str(config.get("semantic_encoder", config.get("text_encoder", "tfidf_svd"))),
        text_embedding_dim=int(config.get("text_embedding_dim", 64)),
        text_model_name=config.get("text_model_name"),
        seed=int(config.get("preprocess_seed", config.get("seed", 42))),
    )
    expected_fingerprint = preprocess_fingerprint(preprocess_config)
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if int(metadata.get("schema_version", 0)) >= 4 and metadata.get("preprocess_fingerprint") == expected_fingerprint:
            return
        if int(metadata.get("schema_version", 0)) >= 4:
            raise RuntimeError(
                f"Processed data in {processed_dir} does not match the current config fingerprint. "
                "Regenerate it with scripts/preprocess.py --config <config> or use a new processed_dir."
            )
    preprocess_tsmc2014(
        raw_path=Path(config["raw_path"]),
        out_dir=processed_dir,
        city=str(config.get("city", "NYC")),
        max_seq_len=int(config["max_seq_len"]),
        min_user_checkins=int(config.get("min_user_checkins", 2)),
        val_ratio=preprocess_config["val_ratio"],
        test_ratio=preprocess_config["test_ratio"],
        limit_users=preprocess_config["limit_users"],
        node2vec_dim=preprocess_config["node2vec_dim"],
        node2vec_walk_length=preprocess_config["node2vec_walk_length"],
        node2vec_num_walks=preprocess_config["node2vec_num_walks"],
        node2vec_p=preprocess_config["node2vec_p"],
        node2vec_q=preprocess_config["node2vec_q"],
        text_encoder=preprocess_config["text_encoder"],
        text_embedding_dim=preprocess_config["text_embedding_dim"],
        text_model_name=preprocess_config["text_model_name"],
        seed=preprocess_config["seed"],
    )


@torch.no_grad()
def evaluate_model(
    model: POIRecommendationModel,
    loader: DataLoader,
    device: torch.device,
    metrics_k: list[int],
    candidate_generator: DynamicCandidateGenerator | None = None,
) -> dict[str, float]:
    model.eval()
    metric_batches = []
    total_candidates = 0.0
    total_coverage = 0.0
    total_count = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        if candidate_generator is not None:
            candidate_ids, candidate_mask = candidate_generator.build(batch)
            if getattr(model, "use_candidate_reranker", False):
                candidate_bank = model.candidate_embeddings()
                candidate_scores = model.scores_for_candidates(
                    batch,
                    candidate_ids,
                    all_candidate_embeddings=candidate_bank,
                )
                constrained = torch.full(
                    (batch["target"].shape[0], model.num_pois),
                    -1e9,
                    dtype=candidate_scores.dtype,
                    device=device,
                )
                output = {"scores": constrained}
            else:
                output = model(batch)
                constrained = torch.full_like(output["scores"], -1e9)
                safe_ids = candidate_ids.clamp(min=0)
                candidate_scores = output["scores"].gather(1, safe_ids)
            safe_ids = candidate_ids.clamp(min=0)
            candidate_scores = candidate_scores.masked_fill(~candidate_mask, -1e9)
            constrained.scatter_(1, safe_ids, candidate_scores)
            output["scores"] = constrained
            target_in_candidates = candidate_ids.eq(batch["target"].unsqueeze(1)).logical_and(candidate_mask).any(dim=1)
            total_coverage += target_in_candidates.float().sum().item()
            total_candidates += candidate_mask.float().sum().item()
            total_count += int(batch["target"].shape[0])
        else:
            output = model(batch)
        metric_batches.append(ranking_metric_sums(output["scores"], batch["target"], metrics_k))
    metrics = average_metric_dicts(metric_batches)
    if candidate_generator is not None:
        metrics["candidate_target_coverage"] = total_coverage / max(1, total_count)
        metrics["candidate_avg_size"] = total_candidates / max(1, total_count)
    return metrics


def train_from_config(config: dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    _ensure_processed(config)
    processed_dir = Path(config["processed_dir"])
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger(
        "poi_rec.train",
        Path(config.get("train_log_path", run_dir / "train.log")),
        log_to_console=bool(config.get("log_to_console", True)),
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("training started; run_dir=%s", run_dir)

    metadata = load_metadata(processed_dir)
    arrays = load_processed_arrays(processed_dir)
    candidate_protocol = str(config.get("candidate_protocol", "all_poi"))
    train_ds = POISequenceDataset(
        processed_dir,
        "train",
        max_seq_len=int(config["max_seq_len"]),
        candidate_protocol=candidate_protocol,
    )
    val_ds = POISequenceDataset(
        processed_dir,
        "val",
        max_seq_len=int(config["max_seq_len"]),
        candidate_protocol=candidate_protocol,
    )
    if len(train_ds) == 0:
        raise ValueError("No training samples found. Try lowering min_user_checkins or checking the raw data.")

    device = resolve_device(str(config.get("device", "auto")))
    _configure_torch_runtime(config, device)
    loader_kwargs = _dataloader_kwargs(config, device)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        **loader_kwargs,
    )

    arrays = {key: value.to(device) for key, value in arrays.items()}
    model = POIRecommendationModel(metadata, arrays, config).to(device)
    alignment_checkpoint = config.get("alignment_checkpoint")
    if alignment_checkpoint is None:
        candidate = run_dir / "alignment_pretrain.pt"
        alignment_checkpoint = str(candidate) if candidate.exists() else None
    if alignment_checkpoint:
        state = torch.load(alignment_checkpoint, map_location=device, weights_only=True)
        expected_fingerprint = metadata.get("preprocess_fingerprint")
        if state.get("preprocess_fingerprint") != expected_fingerprint:
            raise RuntimeError(
                "Alignment checkpoint preprocess fingerprint does not match current processed data."
            )
        for key in ("num_pois", "num_categories", "node2vec_dim", "text_embedding_dim"):
            if state.get(key) != metadata.get(key):
                raise RuntimeError(f"Alignment checkpoint {key}={state.get(key)} does not match metadata {metadata.get(key)}")
        buffer_keys = [
            key
            for key in state["alignment_state"]
            if key.endswith(("transition_features", "node2vec_embeddings", "poi_category", "poi_coords", "text_embeddings", "user_poi_edges"))
            or ".transition_features" in key
            or ".node2vec_embeddings" in key
            or ".poi_category" in key
            or ".poi_coords" in key
            or ".text_embeddings" in key
            or ".user_poi_edges" in key
        ]
        if buffer_keys:
            raise RuntimeError(f"Alignment checkpoint contains data buffers: {buffer_keys}")
        missing, unexpected = model.load_state_dict(state["alignment_state"], strict=False)
        unexpected_relevant = [
            key for key in unexpected if key.startswith(("topology.", "semantic.", "alignment."))
        ]
        if unexpected_relevant:
            raise RuntimeError(f"Unexpected alignment checkpoint keys: {unexpected_relevant}")
        logger.info(
            "loaded alignment checkpoint: %s; missing non-alignment keys: %d",
            alignment_checkpoint,
            len(missing),
        )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    align_weight = float(config.get("align_loss_weight", 0.0))
    align_instance_weight = float(config.get("align_instance_weight", 1.0))
    align_feature_weight = float(config.get("align_feature_weight", 0.25))
    discrepancy_weight = float(config.get("curriculum", {}).get("discrepancy_weight", 0.0))
    curriculum_enabled = bool(config.get("curriculum", {}).get("enabled", False))
    train_with_priors = bool(config.get("apply_priors_during_training", False))
    sampled_cf = dict(config.get("sampled_cf", {}))
    sampled_cf_enabled = bool(sampled_cf.get("enabled", False))
    sampled_cf_candidates = int(sampled_cf.get("num_candidates", 256))
    train_seen_ids = torch.nonzero(arrays["train_seen_poi"].gt(0), as_tuple=False).flatten()
    if sampled_cf_enabled and train_seen_ids.numel() == 0:
        raise ValueError("sampled_cf requires at least one train-seen POI candidate.")
    popularity_ids = torch.argsort(arrays["train_visit_count"], descending=True)
    popularity_ids = popularity_ids[arrays["train_seen_poi"][popularity_ids].gt(0)]
    sampled_cf_include_priors = bool(sampled_cf.get("include_priors", True))
    sampled_hard_user_poi_top = int(sampled_cf.get("hard_user_poi_top", 64))
    sampled_hard_transition_top = int(sampled_cf.get("hard_transition_top", 64))
    sampled_hard_history_transition_top = int(sampled_cf.get("hard_history_transition_top", 16))
    transition_top_ids = _precompute_top_prior_ids(
        model.transition_prior,
        max(sampled_hard_transition_top, sampled_hard_history_transition_top),
    )
    user_poi_top_ids = _precompute_top_prior_ids(model.user_poi_prior, sampled_hard_user_poi_top)
    metrics_k = [int(k) for k in config.get("metrics_k", [5, 10, 20])]
    best_recall = float("-inf")
    best_metrics: dict[str, float] = {}
    monitor_key = str(config.get("monitor_metric", f"HR@{metrics_k[0]}"))
    monitor_mode = str(config.get("monitor_mode", "max"))
    best_score = float("-inf") if monitor_mode == "max" else float("inf")
    skip_val_during_training = bool(config.get("skip_val_during_training", False))
    eval_every_epochs = max(1, int(config.get("eval_every_epochs", 1)))
    amp_enabled = bool(config.get("amp", False)) and device.type == "cuda"
    amp_dtype = _amp_dtype(config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_next = 0.0
        total_align = 0.0
        candidate_bank = model.candidate_embeddings().detach() if getattr(model, "use_candidate_reranker", False) else None
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                if sampled_cf_enabled:
                    candidate_ids = _build_sampled_candidates(
                        batch,
                        train_seen_ids,
                        sampled_cf_candidates,
                        transition_top_ids=transition_top_ids,
                        user_poi_top_ids=user_poi_top_ids,
                        popularity_ids=popularity_ids,
                        hard_history_width=int(sampled_cf.get("hard_history_width", 20)),
                        hard_user_poi_top=sampled_hard_user_poi_top,
                        hard_transition_top=sampled_hard_transition_top,
                        hard_history_transition_top=sampled_hard_history_transition_top,
                        hard_popularity_top=int(sampled_cf.get("hard_popularity_top", 64)),
                    )
                    if getattr(model, "use_candidate_reranker", False):
                        sampled_scores = model.scores_for_candidates(
                            batch,
                            candidate_ids,
                            include_priors=sampled_cf_include_priors,
                            all_candidate_embeddings=candidate_bank,
                        )
                    else:
                        sampled_scores = model.collaborative_scores_for_candidates(batch, candidate_ids)
                        if sampled_cf_include_priors:
                            sampled_scores = sampled_scores + model.prior_scores_for_candidates(batch, candidate_ids)
                    rec_loss_items = next_poi_loss_per_sample(
                        sampled_scores,
                        torch.zeros(batch["target"].shape[0], dtype=torch.long, device=device),
                    )
                    output = {
                        "scores": sampled_scores,
                        "aligned_topology": torch.empty(0, device=device),
                        "aligned_semantic": torch.empty(0, device=device),
                        "discrepancy": torch.zeros_like(batch["attention_mask"], dtype=torch.float32),
                    }
                else:
                    output = model(batch, include_priors=train_with_priors, need_alignment_outputs=align_weight > 0)
                    rec_loss_items = next_poi_loss_per_sample(output["scores"], batch["target"])
                modal_difficulty = (
                    (output["discrepancy"] * batch["attention_mask"]).sum(dim=1)
                    / batch["attention_mask"].sum(dim=1).clamp(min=1)
                ).detach()
                if curriculum_enabled:
                    progress = epoch / max(1, int(config["epochs"]))
                    threshold = torch.quantile(modal_difficulty, min(1.0, 0.35 + 0.65 * progress))
                    keep = modal_difficulty.le(threshold)
                    if keep.any():
                        rec_loss_items = rec_loss_items[keep]
                        modal_difficulty = modal_difficulty[keep]
                if discrepancy_weight > 0:
                    weights = 1.0 + discrepancy_weight * modal_difficulty
                    rec_loss = (rec_loss_items * weights).mean()
                else:
                    rec_loss = rec_loss_items.mean()
                if align_weight > 0:
                    align_loss = combined_alignment_loss(
                        output["aligned_topology"],
                        output["aligned_semantic"],
                        batch["attention_mask"],
                        batch["poi"],
                        instance_weight=align_instance_weight,
                        feature_weight=align_feature_weight,
                    )
                else:
                    align_loss = output["scores"].new_tensor(0.0)
                loss = rec_loss + align_weight * align_loss
            scaler.scale(loss).backward()
            if float(config.get("gradient_clip", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            total_next += rec_loss.item()
            total_align += align_loss.item()

        train_summary = {
            "loss": total_loss / len(train_loader),
            "next_loss": total_next / len(train_loader),
            "align_loss": total_align / len(train_loader),
        }
        should_eval = (
            len(val_ds) > 0
            and not skip_val_during_training
            and (epoch % eval_every_epochs == 0 or epoch == int(config["epochs"]))
        )
        val_metrics = evaluate_model(model, val_loader, device, metrics_k) if should_eval else {}
        monitor = val_metrics.get(monitor_key, -train_summary["loss"])
        improved = monitor > best_score if monitor_mode == "max" else monitor < best_score
        if improved:
            best_score = monitor
            best_recall = monitor
            best_metrics = val_metrics
            save_checkpoint(run_dir / "best.pt", model, config, metadata, val_metrics)
        logger.info(
            "epoch %d/%d result: train=%s val=%s monitor=%s %.6f improved=%s",
            epoch,
            int(config["epochs"]),
            train_summary,
            val_metrics,
            monitor_key,
            float(monitor),
            improved,
        )

    if skip_val_during_training:
        save_checkpoint(run_dir / "best.pt", model, config, metadata, best_metrics)
    elif not (run_dir / "best.pt").exists():
        save_checkpoint(run_dir / "best.pt", model, config, metadata, best_metrics)
    logger.info("best checkpoint: %s", run_dir / "best.pt")
