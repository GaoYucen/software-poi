from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class DynamicCandidateGenerator:
    """Build per-sample candidate sets for constrained POI ranking.

    Candidate ids are raw POI indices in ``[0, num_pois)``. The generator uses
    training-only statistics already available in ``arrays.npz`` plus the
    training samples for temporal slot counts, avoiding validation/test leakage.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        arrays: dict[str, torch.Tensor],
        config: dict[str, Any],
    ) -> None:
        self.processed_dir = Path(processed_dir)
        cfg = dict(config.get("dynamic_candidates", {}))
        self.num_candidates = max(1, int(cfg.get("num_candidates", 2000)))
        self.transition_top = max(0, int(cfg.get("transition_top", 512)))
        self.history_transition_top = max(0, int(cfg.get("history_transition_top", 128)))
        self.user_history_top = max(0, int(cfg.get("user_history_top", 256)))
        self.spatial_top = max(0, int(cfg.get("spatial_top", 512)))
        self.temporal_top = max(0, int(cfg.get("temporal_top", 512)))
        self.popularity_top = max(0, int(cfg.get("popularity_top", 256)))
        self.include_target = bool(cfg.get("include_target", True))
        self.device = arrays["train_seen_poi"].device

        train_seen = arrays["train_seen_poi"].detach().bool().cpu()
        self.train_seen_ids = torch.nonzero(train_seen, as_tuple=False).flatten()
        self.num_pois = int(train_seen.numel())

        visit_count = arrays["train_visit_count"].detach().cpu()
        popularity = torch.argsort(visit_count, descending=True)
        self.popularity_ids = popularity[train_seen[popularity]].tolist()

        self.transition_top_ids = self._build_transition_top_ids(arrays["transition_edges"].detach().cpu())
        self.user_top_ids = self._build_user_top_ids(arrays["user_poi_edges"].detach().cpu())
        self.coords = arrays["poi_coords"].detach().cpu()
        self._spatial_cache: dict[int, list[int]] = {}
        self.temporal_top_ids = self._build_temporal_top_ids(train_seen)

    def _build_transition_top_ids(self, edges: torch.Tensor) -> dict[int, list[int]]:
        topn = max(self.transition_top, self.history_transition_top)
        if topn <= 0 or edges.numel() == 0:
            return {}
        grouped: dict[int, list[tuple[int, float]]] = {}
        for src, dst, weight in edges.tolist():
            grouped.setdefault(int(src), []).append((int(dst), float(weight)))
        return {
            src: [dst for dst, _ in sorted(items, key=lambda item: item[1], reverse=True)[:topn]]
            for src, items in grouped.items()
        }

    def _build_user_top_ids(self, edges: torch.Tensor) -> dict[int, list[int]]:
        if self.user_history_top <= 0 or edges.numel() == 0:
            return {}
        grouped: dict[int, list[tuple[int, float]]] = {}
        for user, poi, weight in edges.tolist():
            grouped.setdefault(int(user), []).append((int(poi), float(weight)))
        return {
            user: [poi for poi, _ in sorted(items, key=lambda item: item[1], reverse=True)[: self.user_history_top]]
            for user, items in grouped.items()
        }

    def _spatial_neighbors(self, poi_id: int) -> list[int]:
        if self.spatial_top <= 0 or self.train_seen_ids.numel() == 0:
            return []
        poi_id = int(poi_id)
        cached = self._spatial_cache.get(poi_id)
        if cached is not None:
            return cached
        topk = min(self.spatial_top + 1, int(self.train_seen_ids.numel()))
        seen_coords = self.coords[self.train_seen_ids]
        distances = ((seen_coords - self.coords[poi_id].unsqueeze(0)) ** 2).sum(dim=1)
        nearest = torch.topk(distances, k=topk, largest=False).indices
        ids = [int(self.train_seen_ids[idx].item()) for idx in nearest if int(self.train_seen_ids[idx].item()) != poi_id]
        self._spatial_cache[poi_id] = ids[: self.spatial_top]
        return self._spatial_cache[poi_id]

    def _build_temporal_top_ids(self, train_seen: torch.Tensor) -> dict[tuple[int, int], list[int]]:
        if self.temporal_top <= 0:
            return {}
        train_path = self.processed_dir / "train.json"
        if not train_path.exists():
            return {}
        with train_path.open("r", encoding="utf-8") as f:
            samples = json.load(f)
        counts: dict[tuple[int, int], dict[int, int]] = {}
        for sample in samples:
            target = int(sample["target_poi"])
            if target < 0 or target >= self.num_pois or not bool(train_seen[target]):
                continue
            hours = sample.get("history_hour", [])
            weekdays = sample.get("history_weekday", [])
            if not hours or not weekdays:
                continue
            key = (int(hours[-1]), int(weekdays[-1]))
            bucket = counts.setdefault(key, {})
            bucket[target] = bucket.get(target, 0) + 1
        return {
            key: [poi for poi, _ in sorted(bucket.items(), key=lambda item: item[1], reverse=True)[: self.temporal_top]]
            for key, bucket in counts.items()
        }

    def build(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(batch["target"].shape[0])
        candidates = torch.full((batch_size, self.num_candidates), -1, dtype=torch.long, device=self.device)
        valid_mask = torch.zeros((batch_size, self.num_candidates), dtype=torch.bool, device=self.device)

        poi_cpu = batch["poi"].detach().cpu()
        mask_cpu = batch["attention_mask"].detach().cpu()
        user_cpu = batch["user_idx"].detach().cpu().tolist()
        hour_cpu = batch["hour"].detach().cpu()
        weekday_cpu = batch["weekday"].detach().cpu()
        target_cpu = batch["target"].detach().cpu().tolist()

        for row_idx in range(batch_size):
            row: list[int] = []
            seen: set[int] = set()

            def add(candidate_id: int) -> None:
                if len(row) >= self.num_candidates:
                    return
                candidate_id = int(candidate_id)
                if candidate_id < 0 or candidate_id >= self.num_pois or candidate_id in seen:
                    return
                seen.add(candidate_id)
                row.append(candidate_id)

            if self.include_target:
                add(int(target_cpu[row_idx]))

            valid_positions = torch.nonzero(mask_cpu[row_idx].gt(0), as_tuple=False).flatten().tolist()
            history = [int(poi_cpu[row_idx, pos].item()) - 1 for pos in valid_positions]
            history = [poi for poi in history if poi >= 0]
            for poi_id in reversed(history):
                add(poi_id)

            for poi_id in self.user_top_ids.get(int(user_cpu[row_idx]), [])[: self.user_history_top]:
                add(poi_id)

            if history:
                last_poi = history[-1]
                for poi_id in self.transition_top_ids.get(last_poi, [])[: self.transition_top]:
                    add(poi_id)
                for poi_id in self._spatial_neighbors(last_poi)[: self.spatial_top]:
                    add(poi_id)
                for src in reversed(history[:-1]):
                    for poi_id in self.transition_top_ids.get(src, [])[: self.history_transition_top]:
                        add(poi_id)
                        if len(row) >= self.num_candidates:
                            break
                    if len(row) >= self.num_candidates:
                        break
                last_pos = valid_positions[-1]
                key = (int(hour_cpu[row_idx, last_pos].item()), int(weekday_cpu[row_idx, last_pos].item()))
                for poi_id in self.temporal_top_ids.get(key, [])[: self.temporal_top]:
                    add(poi_id)

            for poi_id in self.popularity_ids[: self.popularity_top]:
                add(poi_id)
            for poi_id in self.popularity_ids:
                add(poi_id)
                if len(row) >= self.num_candidates:
                    break

            if row:
                row_tensor = torch.tensor(row, dtype=torch.long, device=self.device)
                candidates[row_idx, : row_tensor.numel()] = row_tensor
                valid_mask[row_idx, : row_tensor.numel()] = True
        return candidates, valid_mask
