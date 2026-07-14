from __future__ import annotations

from typing import Any

import torch
from torch import nn

from poi_rec.models.alignment import AlignmentModule
from poi_rec.models.encoders import ProfileEncoder, SemanticEncoder, SpatialEncoder, TemporalEncoder, TopologyEncoder
from poi_rec.models.fusion import DynamicFusion
from poi_rec.models.gpt_backbone import GPTBackbone, last_valid_state
from poi_rec.models.priors import CandidateReranker, PriorEncoder


class POIRecommendationModel(nn.Module):
    def __init__(self, metadata: dict[str, Any], arrays: dict[str, torch.Tensor], config: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        self.hidden_dim = hidden_dim
        if str(config.get("topology_encoder", "node2vec")) != "node2vec":
            raise ValueError("Only topology_encoder='node2vec' is currently implemented.")
        if str(config.get("semantic_encoder", config.get("text_encoder", "tfidf_svd"))) not in {
            "tfidf_svd",
            "hf_transformer",
        }:
            raise ValueError("semantic_encoder must be 'tfidf_svd' or 'hf_transformer'.")
        self.num_pois = int(metadata["num_pois"])
        self.topology = TopologyEncoder(
            self.num_pois,
            arrays["transition_features"],
            arrays["node2vec_embeddings"],
            arrays["topology_available"],
            hidden_dim,
        )
        self.num_categories = int(metadata["num_categories"])
        self.num_users = int(metadata["num_users"])
        self.register_buffer("poi_category", arrays["poi_category"].long(), persistent=False)
        self.semantic = SemanticEncoder(
            self.num_categories,
            arrays["poi_category"],
            arrays["text_embeddings"],
            hidden_dim,
        )
        self.spatial = SpatialEncoder(arrays["poi_coords"], hidden_dim)
        self.temporal = TemporalEncoder(hidden_dim)
        self.candidate_protocol = str(config.get("candidate_protocol", "all_poi"))
        if self.candidate_protocol not in {"all_poi", "closed_world"}:
            raise ValueError("candidate_protocol must be 'all_poi' or 'closed_world'.")
        self.register_buffer("train_seen_poi", arrays["train_seen_poi"].float())
        popularity = torch.log1p(arrays["train_visit_count"].float())
        self.register_buffer("popularity_prior", popularity / popularity.max().clamp_min(1.0))
        self.transition_prior_weight = float(config.get("transition_prior_weight", 0.0))
        self.transition_prior_mode = str(config.get("transition_prior_mode", "prob"))
        self.history_transition_prior_weight = float(config.get("history_transition_prior_weight", 0.0))
        self.history_transition_decay = float(config.get("history_transition_decay", 0.8))
        self.history_transition_aggregation = str(config.get("history_transition_aggregation", "sum"))
        if self.history_transition_aggregation not in {"max", "sum"}:
            raise ValueError("history_transition_aggregation must be 'max' or 'sum'.")
        self.user_poi_prior_weight = float(config.get("user_poi_prior_weight", 0.0))
        self.user_poi_prior_mode = str(config.get("user_poi_prior_mode", "log_count"))
        self.co_visit_prior_weight = float(config.get("co_visit_prior_weight", 0.0))
        self.co_visit_prior_mode = str(config.get("co_visit_prior_mode", "log_count"))
        self.co_visit_max_user_pois = int(config.get("co_visit_max_user_pois", 100))
        self.category_transition_prior_weight = float(config.get("category_transition_prior_weight", 0.0))
        self.category_transition_prior_mode = str(config.get("category_transition_prior_mode", "prob"))
        self.user_category_prior_weight = float(config.get("user_category_prior_weight", 0.0))
        self.user_category_prior_mode = str(config.get("user_category_prior_mode", "log_count"))
        self.history_repeat_prior_weight = float(config.get("history_repeat_prior_weight", 0.0))
        self.history_repeat_decay = float(config.get("history_repeat_decay", 0.7))
        self.history_repeat_aggregation = str(config.get("history_repeat_aggregation", "max"))
        if self.history_repeat_aggregation not in {"max", "sum"}:
            raise ValueError("history_repeat_aggregation must be 'max' or 'sum'.")
        self.popularity_prior_weight = float(config.get("popularity_prior_weight", 0.0))
        self.spatial_prior_weight = float(config.get("spatial_prior_weight", 0.0))
        self.spatial_prior_temperature = float(config.get("spatial_prior_temperature", 1.0))
        self.use_prior_encoder = bool(config.get("use_prior_encoder", False))
        self.use_candidate_reranker = bool(config.get("use_candidate_reranker", False))
        if self.use_prior_encoder:
            self.prior_encoder = PriorEncoder(
                hidden_dim=hidden_dim,
                num_priors=4,  # popularity, category_transition, user_category, spatial
                dropout=float(config.get("dropout", 0.1)),
            )
        else:
            self.prior_encoder = None
        if self.use_candidate_reranker:
            self.candidate_reranker = CandidateReranker(
                num_features=10,
                hidden_dim=int(config.get("candidate_reranker_hidden_dim", 32)),
                dropout=float(config.get("dropout", 0.1)),
            )
            self.candidate_reranker_weight = float(config.get("candidate_reranker_weight", 1.0))
        else:
            self.candidate_reranker = None
            self.candidate_reranker_weight = 0.0
        self._transition_edges_cpu = arrays["transition_edges"].detach().cpu()
        self.transition_prior: dict[int, tuple[torch.Tensor, torch.Tensor]] = self._build_transition_prior(
            self._transition_edges_cpu,
            self.transition_prior_mode,
        )
        self._user_poi_edges_cpu = arrays["user_poi_edges"].detach().cpu()
        self.user_poi_prior: dict[int, tuple[torch.Tensor, torch.Tensor]] = self._build_user_poi_prior(
            self._user_poi_edges_cpu,
            self.user_poi_prior_mode,
        )
        self.co_visit_prior: dict[int, tuple[torch.Tensor, torch.Tensor]] = self._build_co_visit_prior(
            self._user_poi_edges_cpu,
            self.co_visit_prior_mode,
            self.co_visit_max_user_pois,
        )
        self.register_buffer(
            "category_transition_prior_matrix",
            self._build_category_transition_prior_matrix(
                self._transition_edges_cpu,
                arrays["poi_category"].detach().cpu(),
                self.num_categories,
                self.category_transition_prior_mode,
            ),
            persistent=False,
        )
        self.register_buffer(
            "user_category_prior_matrix",
            self._build_user_category_prior_matrix(
                self._user_poi_edges_cpu,
                arrays["poi_category"].detach().cpu(),
                self.num_users,
                self.num_categories,
                self.user_category_prior_mode,
            ),
            persistent=False,
        )
        self.use_user_id_embedding = bool(config.get("use_user_id_embedding", True))
        self.user_embedding = nn.Embedding(self.num_users, hidden_dim) if self.use_user_id_embedding else None
        self.profile_encoder = ProfileEncoder(self.user_category_prior_matrix, hidden_dim)
        self.alignment = AlignmentModule(hidden_dim)
        self.alignment_mode = str(config.get("alignment_mode", "projected"))
        if self.alignment_mode not in {"projected", "identity"}:
            raise ValueError("alignment_mode must be 'projected' or 'identity'.")
        self.use_topology_modality = bool(config.get("use_topology_modality", True))
        self.use_semantic_modality = bool(config.get("use_semantic_modality", True))
        self.use_spatial_modality = bool(config.get("use_spatial_modality", True))
        self.use_temporal_modality = bool(config.get("use_temporal_modality", True))
        self.fusion = DynamicFusion(
            hidden_dim,
            float(config.get("dropout", 0.1)),
            mode=str(config.get("fusion_mode", "context_gate")),
        )
        self.backbone = GPTBackbone(
            hidden_dim=hidden_dim,
            model_name=str(config.get("gpt_model_name", "gpt2")),
            use_pretrained=bool(config.get("use_pretrained_gpt", False)),
            freeze=bool(config.get("freeze_gpt", False)),
            unfreeze_last_n=int(config.get("unfreeze_last_n", 0)),
            freeze_policy=str(config.get("gpt_freeze_policy", "last_block")),
            fallback_to_random=bool(config.get("fallback_to_random_gpt", False)),
            layers=int(config.get("gpt_layers", 2)),
            heads=int(config.get("gpt_heads", 4)),
            max_seq_len=int(config["max_seq_len"]),
            dropout=float(config.get("dropout", 0.1)),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.matching_normalize = bool(config.get("matching_normalize", True))
        self.neural_score_weight = float(config.get("neural_score_weight", 1.0))
        self.collaborative_score_weight = float(config.get("collaborative_score_weight", 0.0))
        collaborative_enabled = bool(config.get("enable_collaborative", self.collaborative_score_weight > 0))
        collaborative_dim = int(config.get("collaborative_dim", hidden_dim))
        if collaborative_enabled:
            self.cf_user_embedding = nn.Embedding(self.num_users, collaborative_dim)
            self.cf_context_embedding = nn.Embedding(self.num_pois + 1, collaborative_dim, padding_idx=0)
            self.cf_candidate_embedding = nn.Embedding(self.num_pois + 1, collaborative_dim, padding_idx=0)
            self.cf_query_projection = nn.Sequential(
                nn.Linear(collaborative_dim, collaborative_dim),
                nn.GELU(),
                nn.LayerNorm(collaborative_dim),
            )
            self.cf_logit_scale = nn.Parameter(torch.log(torch.tensor(float(config.get("cf_initial_logit_scale", 1.0)))))
        else:
            self.cf_user_embedding = None
            self.cf_context_embedding = None
            self.cf_candidate_embedding = None
            self.cf_query_projection = None
            self.cf_logit_scale = None
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(float(config.get("initial_logit_scale", 1 / 0.07)))))

    @staticmethod
    def _build_transition_prior(edges: torch.Tensor, mode: str = "prob") -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        if mode not in {"prob", "binary", "count", "log_count"}:
            raise ValueError("transition_prior_mode must be one of: prob, binary, count, log_count.")
        grouped: dict[int, list[tuple[int, float]]] = {}
        if edges.numel() == 0:
            return {}
        for src, dst, weight in edges.cpu().tolist():
            grouped.setdefault(int(src), []).append((int(dst), float(weight)))
        priors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for src, items in grouped.items():
            dst = torch.tensor([item[0] for item in items], dtype=torch.long)
            weights = torch.tensor([item[1] for item in items], dtype=torch.float32)
            if mode == "prob":
                values = weights / weights.sum().clamp_min(1.0)
            elif mode == "binary":
                values = torch.ones_like(weights)
            elif mode == "count":
                values = weights / weights.max().clamp_min(1.0)
            else:
                values = torch.log1p(weights)
                values = values / values.max().clamp_min(1.0)
            priors[src] = (dst, values)
        return priors

    def configure_priors(
        self,
        *,
        transition_prior_weight: float | None = None,
        transition_prior_mode: str | None = None,
        history_transition_prior_weight: float | None = None,
        history_transition_decay: float | None = None,
        history_transition_aggregation: str | None = None,
        user_poi_prior_weight: float | None = None,
        user_poi_prior_mode: str | None = None,
        co_visit_prior_weight: float | None = None,
        co_visit_prior_mode: str | None = None,
        category_transition_prior_weight: float | None = None,
        category_transition_prior_mode: str | None = None,
        user_category_prior_weight: float | None = None,
        user_category_prior_mode: str | None = None,
        neural_score_weight: float | None = None,
        collaborative_score_weight: float | None = None,
        history_repeat_prior_weight: float | None = None,
        history_repeat_decay: float | None = None,
        history_repeat_aggregation: str | None = None,
        popularity_prior_weight: float | None = None,
        spatial_prior_weight: float | None = None,
        spatial_prior_temperature: float | None = None,
    ) -> None:
        if transition_prior_weight is not None:
            self.transition_prior_weight = float(transition_prior_weight)
        if transition_prior_mode is not None and transition_prior_mode != self.transition_prior_mode:
            self.transition_prior_mode = str(transition_prior_mode)
            self.transition_prior = self._build_transition_prior(self._transition_edges_cpu, self.transition_prior_mode)
        if history_transition_prior_weight is not None:
            self.history_transition_prior_weight = float(history_transition_prior_weight)
        if history_transition_decay is not None:
            self.history_transition_decay = float(history_transition_decay)
        if history_transition_aggregation is not None:
            if history_transition_aggregation not in {"max", "sum"}:
                raise ValueError("history_transition_aggregation must be 'max' or 'sum'.")
            self.history_transition_aggregation = str(history_transition_aggregation)
        if user_poi_prior_weight is not None:
            self.user_poi_prior_weight = float(user_poi_prior_weight)
        if user_poi_prior_mode is not None and user_poi_prior_mode != self.user_poi_prior_mode:
            self.user_poi_prior_mode = str(user_poi_prior_mode)
            self.user_poi_prior = self._build_user_poi_prior(self._user_poi_edges_cpu, self.user_poi_prior_mode)
        if co_visit_prior_weight is not None:
            self.co_visit_prior_weight = float(co_visit_prior_weight)
        if co_visit_prior_mode is not None and co_visit_prior_mode != self.co_visit_prior_mode:
            self.co_visit_prior_mode = str(co_visit_prior_mode)
            self.co_visit_prior = self._build_co_visit_prior(
                self._user_poi_edges_cpu,
                self.co_visit_prior_mode,
                self.co_visit_max_user_pois,
            )
        if category_transition_prior_weight is not None:
            self.category_transition_prior_weight = float(category_transition_prior_weight)
        if (
            category_transition_prior_mode is not None
            and category_transition_prior_mode != self.category_transition_prior_mode
        ):
            self.category_transition_prior_mode = str(category_transition_prior_mode)
            self.category_transition_prior_matrix = self._build_category_transition_prior_matrix(
                self._transition_edges_cpu,
                self.poi_category.detach().cpu(),
                self.num_categories,
                self.category_transition_prior_mode,
            ).to(self.poi_category.device)
        if user_category_prior_weight is not None:
            self.user_category_prior_weight = float(user_category_prior_weight)
        if user_category_prior_mode is not None and user_category_prior_mode != self.user_category_prior_mode:
            self.user_category_prior_mode = str(user_category_prior_mode)
            self.user_category_prior_matrix = self._build_user_category_prior_matrix(
                self._user_poi_edges_cpu,
                self.poi_category.detach().cpu(),
                self.num_users,
                self.num_categories,
                self.user_category_prior_mode,
            ).to(self.poi_category.device)
        if neural_score_weight is not None:
            self.neural_score_weight = float(neural_score_weight)
        if collaborative_score_weight is not None:
            self.collaborative_score_weight = float(collaborative_score_weight)
        if history_repeat_prior_weight is not None:
            self.history_repeat_prior_weight = float(history_repeat_prior_weight)
        if history_repeat_decay is not None:
            self.history_repeat_decay = float(history_repeat_decay)
        if history_repeat_aggregation is not None:
            if history_repeat_aggregation not in {"max", "sum"}:
                raise ValueError("history_repeat_aggregation must be 'max' or 'sum'.")
            self.history_repeat_aggregation = str(history_repeat_aggregation)
        if popularity_prior_weight is not None:
            self.popularity_prior_weight = float(popularity_prior_weight)
        if spatial_prior_weight is not None:
            self.spatial_prior_weight = float(spatial_prior_weight)
        if spatial_prior_temperature is not None:
            self.spatial_prior_temperature = float(spatial_prior_temperature)

    @classmethod
    def _build_user_poi_prior(cls, edges: torch.Tensor, mode: str = "log_count") -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        if mode not in {"prob", "binary", "count", "log_count"}:
            raise ValueError("user_poi_prior_mode must be one of: prob, binary, count, log_count.")
        grouped: dict[int, list[tuple[int, float]]] = {}
        if edges.numel() == 0:
            return {}
        for user, poi, weight in edges.cpu().tolist():
            grouped.setdefault(int(user), []).append((int(poi), float(weight)))
        priors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for user, items in grouped.items():
            poi = torch.tensor([item[0] for item in items], dtype=torch.long)
            weights = torch.tensor([item[1] for item in items], dtype=torch.float32)
            if mode == "prob":
                values = weights / weights.sum().clamp_min(1.0)
            elif mode == "binary":
                values = torch.ones_like(weights)
            elif mode == "count":
                values = weights / weights.max().clamp_min(1.0)
            else:
                values = torch.log1p(weights)
                values = values / values.max().clamp_min(1.0)
            priors[user] = (poi, values)
        return priors

    @classmethod
    def _build_co_visit_prior(
        cls,
        edges: torch.Tensor,
        mode: str = "log_count",
        max_user_pois: int = 100,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        if mode not in {"prob", "binary", "count", "log_count"}:
            raise ValueError("co_visit_prior_mode must be one of: prob, binary, count, log_count.")
        by_user: dict[int, list[tuple[int, float]]] = {}
        for user, poi, weight in edges.cpu().tolist():
            by_user.setdefault(int(user), []).append((int(poi), float(weight)))
        pair_counts: dict[tuple[int, int], float] = {}
        for items in by_user.values():
            selected = sorted(items, key=lambda item: item[1], reverse=True)[:max(1, max_user_pois)]
            for src, src_weight in selected:
                for dst, dst_weight in selected:
                    if src == dst:
                        continue
                    key = (src, dst)
                    pair_counts[key] = pair_counts.get(key, 0.0) + min(src_weight, dst_weight)
        grouped: dict[int, list[tuple[int, float]]] = {}
        for (src, dst), weight in pair_counts.items():
            grouped.setdefault(src, []).append((dst, weight))
        priors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for src, items in grouped.items():
            dst = torch.tensor([item[0] for item in items], dtype=torch.long)
            weights = torch.tensor([item[1] for item in items], dtype=torch.float32)
            if mode == "prob":
                values = weights / weights.sum().clamp_min(1.0)
            elif mode == "binary":
                values = torch.ones_like(weights)
            elif mode == "count":
                values = weights / weights.max().clamp_min(1.0)
            else:
                values = torch.log1p(weights)
                values = values / values.max().clamp_min(1.0)
            priors[src] = (dst, values)
        return priors

    @staticmethod
    def _normalize_prior_matrix(counts: torch.Tensor, mode: str) -> torch.Tensor:
        if mode not in {"prob", "binary", "count", "log_count"}:
            raise ValueError("prior mode must be one of: prob, binary, count, log_count.")
        if mode == "prob":
            return counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        if mode == "binary":
            return counts.gt(0).float()
        if mode == "count":
            return counts / counts.max(dim=1, keepdim=True).values.clamp_min(1.0)
        values = torch.log1p(counts)
        return values / values.max(dim=1, keepdim=True).values.clamp_min(1.0)

    @classmethod
    def _build_category_transition_prior_matrix(
        cls,
        edges: torch.Tensor,
        poi_category: torch.Tensor,
        num_categories: int,
        mode: str = "prob",
    ) -> torch.Tensor:
        counts = torch.zeros((num_categories, num_categories), dtype=torch.float32)
        if edges.numel() == 0:
            return counts
        categories = poi_category.long()
        for src, dst, weight in edges.cpu().tolist():
            src_cat = int(categories[int(src)].item())
            dst_cat = int(categories[int(dst)].item())
            counts[src_cat, dst_cat] += float(weight)
        return cls._normalize_prior_matrix(counts, mode)

    @classmethod
    def _build_user_category_prior_matrix(
        cls,
        edges: torch.Tensor,
        poi_category: torch.Tensor,
        num_users: int,
        num_categories: int,
        mode: str = "log_count",
    ) -> torch.Tensor:
        counts = torch.zeros((num_users, num_categories), dtype=torch.float32)
        if edges.numel() == 0:
            return counts
        categories = poi_category.long()
        for user, poi, weight in edges.cpu().tolist():
            counts[int(user), int(categories[int(poi)].item())] += float(weight)
        return cls._normalize_prior_matrix(counts, mode)

    def encode_sequence(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        poi = batch["poi"]
        topology = self.topology(poi)
        semantic = self.semantic(poi)
        spatial = self.spatial(poi)
        if not self.use_topology_modality:
            topology = torch.zeros_like(topology)
        if not self.use_semantic_modality:
            semantic = torch.zeros_like(semantic)
        if self.alignment_mode == "projected":
            aligned_topology, aligned_semantic = self.alignment(topology, semantic)
        else:
            aligned_topology, aligned_semantic = topology, semantic
        if not self.use_topology_modality:
            aligned_topology = torch.zeros_like(aligned_topology)
        if not self.use_semantic_modality:
            aligned_semantic = torch.zeros_like(aligned_semantic)
        temporal = self.temporal(batch["hour"], batch["weekday"], batch["delta_bucket"])
        if not self.use_spatial_modality:
            spatial = torch.zeros_like(spatial)
        if not self.use_temporal_modality:
            temporal = torch.zeros_like(temporal)
        profile = self.profile_encoder(batch["user_idx"], poi.shape[1])
        if self.user_embedding is None:
            user = torch.zeros_like(temporal)
        else:
            user = self.user_embedding(batch["user_idx"]).unsqueeze(1).expand_as(temporal)
        tokens = self.fusion(aligned_topology, aligned_semantic, spatial, temporal, user, profile)
        hidden = self.backbone(tokens, batch["attention_mask"])
        query = last_valid_state(hidden, batch["attention_mask"])
        discrepancy = self.alignment.discrepancy(aligned_topology, aligned_semantic)
        return query, aligned_topology, aligned_semantic, discrepancy

    def candidate_embeddings(self) -> torch.Tensor:
        topology = self.topology.all_embeddings()
        semantic = self.semantic.all_embeddings()
        spatial = self.spatial.all_embeddings()
        if not self.use_topology_modality:
            topology = torch.zeros_like(topology)
        if not self.use_semantic_modality:
            semantic = torch.zeros_like(semantic)
        if not self.use_spatial_modality:
            spatial = torch.zeros_like(spatial)
        if self.alignment_mode == "projected":
            aligned_topology, aligned_semantic = self.alignment(topology, semantic)
        else:
            aligned_topology, aligned_semantic = topology, semantic
        if not self.use_topology_modality:
            aligned_topology = torch.zeros_like(aligned_topology)
        if not self.use_semantic_modality:
            aligned_semantic = torch.zeros_like(aligned_semantic)
        static = torch.cat([aligned_topology, aligned_semantic, spatial], dim=-1)
        return self.candidate_projection(static)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        include_priors: bool = True,
        need_alignment_outputs: bool = True,
    ) -> dict[str, torch.Tensor]:
        if self.neural_score_weight > 0 or need_alignment_outputs:
            query, aligned_topology, aligned_semantic, discrepancy = self.encode_sequence(batch)
            query = self.query_projection(query)
        else:
            batch_size, seq_len = batch["poi"].shape
            device = batch["poi"].device
            aligned_topology = torch.zeros((batch_size, seq_len, self.hidden_dim), device=device)
            aligned_semantic = torch.zeros_like(aligned_topology)
            discrepancy = torch.zeros((batch_size, seq_len), device=device)

        if self.neural_score_weight > 0:
            candidates = self.candidate_embeddings()
            if self.matching_normalize:
                query = torch.nn.functional.normalize(query, dim=-1)
                candidates = torch.nn.functional.normalize(candidates, dim=-1)
            logit_scale = self.logit_scale.exp().clamp(max=100.0)
            scores = self.neural_score_weight * logit_scale * (query @ candidates.t())
        else:
            scores = torch.zeros((batch["poi"].shape[0], self.num_pois), device=batch["poi"].device)
        if self.collaborative_score_weight > 0:
            scores = scores + self.collaborative_score_weight * self._collaborative_scores(batch)
        if include_priors:
            scores = self._apply_candidate_priors(scores, batch)
        if self.candidate_protocol == "closed_world":
            scores = scores.masked_fill(self.train_seen_poi.unsqueeze(0).eq(0), -1e9)
        return {
            "scores": scores,
            "aligned_topology": aligned_topology,
            "aligned_semantic": aligned_semantic,
            "discrepancy": discrepancy,
        }

    def _collaborative_scores(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if (
            self.cf_user_embedding is None
            or self.cf_context_embedding is None
            or self.cf_candidate_embedding is None
            or self.cf_query_projection is None
            or self.cf_logit_scale is None
        ):
            raise RuntimeError("Collaborative scorer is not initialized.")
        poi = batch["poi"]
        mask = batch["attention_mask"].unsqueeze(-1).float()
        context_tokens = self.cf_context_embedding(poi)
        history = (context_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        lengths = batch["attention_mask"].sum(dim=1).clamp(min=1) - 1
        last_shifted = poi[torch.arange(poi.shape[0], device=poi.device), lengths]
        last = self.cf_context_embedding(last_shifted)
        user = self.cf_user_embedding(batch["user_idx"])
        query = self.cf_query_projection(user + last + history)
        candidate_ids = torch.arange(1, self.num_pois + 1, device=poi.device)
        candidates = self.cf_candidate_embedding(candidate_ids)
        query = torch.nn.functional.normalize(query, dim=-1)
        candidates = torch.nn.functional.normalize(candidates, dim=-1)
        return self.cf_logit_scale.exp().clamp(max=100.0) * (query @ candidates.t())

    def collaborative_scores_for_candidates(
        self,
        batch: dict[str, torch.Tensor],
        candidate_raw_ids: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.cf_user_embedding is None
            or self.cf_context_embedding is None
            or self.cf_candidate_embedding is None
            or self.cf_query_projection is None
            or self.cf_logit_scale is None
        ):
            raise RuntimeError("Collaborative scorer is not initialized.")
        poi = batch["poi"]
        mask = batch["attention_mask"].unsqueeze(-1).float()
        context_tokens = self.cf_context_embedding(poi)
        history = (context_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        lengths = batch["attention_mask"].sum(dim=1).clamp(min=1) - 1
        last_shifted = poi[torch.arange(poi.shape[0], device=poi.device), lengths]
        last = self.cf_context_embedding(last_shifted)
        user = self.cf_user_embedding(batch["user_idx"])
        query = self.cf_query_projection(user + last + history)
        candidates = self.cf_candidate_embedding(candidate_raw_ids + 1)
        query = torch.nn.functional.normalize(query, dim=-1)
        candidates = torch.nn.functional.normalize(candidates, dim=-1)
        return self.cf_logit_scale.exp().clamp(max=100.0) * torch.einsum("bd,bcd->bc", query, candidates)

    def neural_scores_for_candidates(
        self,
        batch: dict[str, torch.Tensor],
        candidate_raw_ids: torch.Tensor,
        all_candidate_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.neural_score_weight <= 0:
            return torch.zeros(candidate_raw_ids.shape, dtype=torch.float32, device=candidate_raw_ids.device)
        query, _, _, _ = self.encode_sequence(batch)
        query = self.query_projection(query)
        candidate_bank = all_candidate_embeddings if all_candidate_embeddings is not None else self.candidate_embeddings()
        candidates = candidate_bank[candidate_raw_ids]
        if self.matching_normalize:
            query = torch.nn.functional.normalize(query, dim=-1)
            candidates = torch.nn.functional.normalize(candidates, dim=-1)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        return self.neural_score_weight * logit_scale * torch.einsum("bd,bcd->bc", query, candidates)

    def reranker_scores_for_candidates(
        self,
        batch: dict[str, torch.Tensor],
        candidate_raw_ids: torch.Tensor,
        base_scores: torch.Tensor,
        prior_scores: torch.Tensor,
    ) -> torch.Tensor:
        if self.candidate_reranker is None or self.candidate_reranker_weight <= 0:
            return torch.zeros_like(base_scores)
        device = candidate_raw_ids.device
        batch_size = candidate_raw_ids.shape[0]
        num_candidates = candidate_raw_ids.shape[1]

        lengths = batch["attention_mask"].sum(dim=1).clamp(min=1) - 1
        last_shifted = batch["poi"][torch.arange(batch_size, device=device), lengths]
        last_raw = (last_shifted - 1).clamp(min=0)

        # --- Feature 0: base neural score ---
        base = base_scores.float()

        # --- Feature 1: prior score ---
        prior = prior_scores.float()

        # --- Feature 2: popularity ---
        popularity = self.popularity_prior[candidate_raw_ids].float()

        # --- Feature 3: category match with last POI ---
        candidate_category = self.poi_category[candidate_raw_ids]
        last_category = self.poi_category[last_raw]
        category_match = candidate_category.eq(last_category.unsqueeze(1)).float()

        # --- Feature 4: repeat flag (was this POI in user history?) ---
        poi_history = batch["poi"].detach().cpu()
        mask_history = batch["attention_mask"].detach().cpu()
        repeat = torch.zeros((batch_size, num_candidates), dtype=torch.float32, device=device)
        for row_idx in range(batch_size):
            valid_positions = torch.nonzero(mask_history[row_idx].gt(0), as_tuple=False).flatten().tolist()
            history_pois = set()
            for pos in valid_positions:
                raw = int(poi_history[row_idx, pos].item()) - 1
                if raw >= 0:
                    history_pois.add(raw)
            for cand_idx in range(num_candidates):
                if int(candidate_raw_ids[row_idx, cand_idx].item()) in history_pois:
                    repeat[row_idx, cand_idx] = 1.0

        # --- Feature 5: spatial distance to last POI ---
        coords = self.spatial.poi_coords
        last_coords = coords[last_raw]
        candidate_coords = coords[candidate_raw_ids]
        spatial_dist = torch.sqrt(
            ((candidate_coords - last_coords.unsqueeze(1)) ** 2).sum(dim=-1).clamp(min=1e-8)
        )
        spatial_distance = torch.exp(-spatial_dist * 10.0)

        # --- Feature 6: hour match score (visit frequency at this hour slot) ---
        last_positions = (lengths.detach().cpu().tolist() if lengths.numel() > 0 else [])
        hour_match = torch.zeros((batch_size, num_candidates), dtype=torch.float32, device=device)
        for row_idx in range(batch_size):
            if row_idx < len(last_positions):
                hour = int(batch["hour"][row_idx, last_positions[row_idx]].item())
                weekday = int(batch["weekday"][row_idx, last_positions[row_idx]].item())
                # Simple heuristic: prefer POIs visited at similar hours
                # Using a fixed preference: POIs with category_match get extra hour bonus
                hour_match[row_idx] = category_match[row_idx] * 0.5 + 0.5 * (1.0 - abs(float(hour) - 12.0) / 12.0)

        # --- Feature 7: last distance score (alternative spatial encoding) ---
        last_distance = torch.clamp(spatial_dist / max(self.spatial_prior_temperature, 1e-6), max=50.0)
        last_distance_score = torch.exp(-last_distance * 0.5)

        # --- Feature 8: category transition prior ---
        cat_trans = torch.zeros((batch_size, num_candidates), dtype=torch.float32, device=device)
        if self.category_transition_prior_weight > 0 or hasattr(self, "category_transition_prior_matrix"):
            cat_trans = self.category_transition_prior_matrix[
                last_category.unsqueeze(1), candidate_category
            ].float()

        # --- Feature 9: user category bias ---
        user_cat = torch.zeros((batch_size, num_candidates), dtype=torch.float32, device=device)
        if self.user_category_prior_weight > 0 or hasattr(self, "user_category_prior_matrix"):
            user_cat = self.user_category_prior_matrix[
                batch["user_idx"].unsqueeze(1), candidate_category
            ].float()

        features = torch.stack(
            [base, prior, popularity, category_match, repeat,
             spatial_distance, hour_match, last_distance_score, cat_trans, user_cat],
            dim=-1,
        )
        return self.candidate_reranker_weight * self.candidate_reranker(features)

    def scores_for_candidates(
        self,
        batch: dict[str, torch.Tensor],
        candidate_raw_ids: torch.Tensor,
        include_priors: bool = True,
        all_candidate_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.neural_scores_for_candidates(batch, candidate_raw_ids, all_candidate_embeddings=all_candidate_embeddings)
        if self.collaborative_score_weight > 0:
            scores = scores + self.collaborative_score_weight * self.collaborative_scores_for_candidates(
                batch, candidate_raw_ids
            )
        prior_scores = self.prior_scores_for_candidates(batch, candidate_raw_ids) if include_priors else torch.zeros_like(scores)
        scores = scores + prior_scores
        scores = scores + self.reranker_scores_for_candidates(batch, candidate_raw_ids, scores, prior_scores)
        return scores

    def prior_scores_for_candidates(
        self,
        batch: dict[str, torch.Tensor],
        candidate_raw_ids: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.zeros(candidate_raw_ids.shape, dtype=torch.float32, device=candidate_raw_ids.device)
        if (
            self.transition_prior_weight <= 0
            and self.history_transition_prior_weight <= 0
            and self.user_poi_prior_weight <= 0
            and self.co_visit_prior_weight <= 0
            and self.category_transition_prior_weight <= 0
            and self.user_category_prior_weight <= 0
            and self.history_repeat_prior_weight <= 0
            and self.popularity_prior_weight <= 0
            and self.spatial_prior_weight <= 0
        ):
            return scores

        if self.popularity_prior_weight > 0:
            scores = scores + self.popularity_prior_weight * self.popularity_prior[candidate_raw_ids]

        lengths = batch["attention_mask"].sum(dim=1).clamp(min=1) - 1
        last_shifted = batch["poi"][torch.arange(batch["poi"].shape[0], device=batch["poi"].device), lengths]
        last_raw = (last_shifted - 1).clamp(min=0)

        def add_sparse_prior(
            target_scores: torch.Tensor,
            row_idx: int,
            ids: torch.Tensor,
            values: torch.Tensor,
            weight: float,
            use_max: bool = False,
        ) -> None:
            if ids.numel() == 0:
                return
            row_candidates = candidate_raw_ids[row_idx]
            ids = ids.to(row_candidates.device)
            values = values.to(row_candidates.device)
            for prior_id, prior_value in zip(ids, values):
                match = row_candidates.eq(prior_id)
                if not bool(match.any()):
                    continue
                contribution = float(weight) * prior_value
                if use_max:
                    target_scores[row_idx, match] = torch.maximum(
                        target_scores[row_idx, match],
                        contribution.expand_as(target_scores[row_idx, match]),
                    )
                else:
                    target_scores[row_idx, match] += contribution

        if self.history_repeat_prior_weight > 0:
            repeat_scores = torch.zeros_like(scores)
            poi_history = batch["poi"].detach().cpu()
            mask_history = batch["attention_mask"].detach().cpu()
            decay = min(max(self.history_repeat_decay, 0.0), 1.0)
            for row_idx in range(poi_history.shape[0]):
                valid_positions = torch.nonzero(mask_history[row_idx].gt(0), as_tuple=False).flatten().tolist()
                for age, pos in enumerate(reversed(valid_positions)):
                    raw_poi = int(poi_history[row_idx, pos].item()) - 1
                    if raw_poi < 0:
                        continue
                    match = candidate_raw_ids[row_idx].eq(raw_poi)
                    if not bool(match.any()):
                        continue
                    contribution = self.history_repeat_prior_weight * (decay**age)
                    if self.history_repeat_aggregation == "sum":
                        repeat_scores[row_idx, match] += contribution
                    else:
                        repeat_scores[row_idx, match] = torch.maximum(
                            repeat_scores[row_idx, match],
                            repeat_scores[row_idx, match].new_full(repeat_scores[row_idx, match].shape, contribution),
                        )
            scores = scores + repeat_scores

        if self.transition_prior_weight > 0:
            for row_idx, src in enumerate(last_raw.detach().cpu().tolist()):
                prior = self.transition_prior.get(int(src))
                if prior is not None:
                    add_sparse_prior(scores, row_idx, prior[0], prior[1], self.transition_prior_weight)

        if self.history_transition_prior_weight > 0:
            history_transition_scores = torch.zeros_like(scores)
            poi_history = batch["poi"].detach().cpu()
            mask_history = batch["attention_mask"].detach().cpu()
            decay = min(max(self.history_transition_decay, 0.0), 1.0)
            for row_idx in range(poi_history.shape[0]):
                valid_positions = torch.nonzero(mask_history[row_idx].gt(0), as_tuple=False).flatten().tolist()
                for age, pos in enumerate(reversed(valid_positions)):
                    src = int(poi_history[row_idx, pos].item()) - 1
                    if src < 0:
                        continue
                    prior = self.transition_prior.get(src)
                    if prior is None:
                        continue
                    add_sparse_prior(
                        history_transition_scores,
                        row_idx,
                        prior[0],
                        (decay**age) * prior[1],
                        self.history_transition_prior_weight,
                        use_max=self.history_transition_aggregation == "max",
                    )
            scores = scores + history_transition_scores

        if self.co_visit_prior_weight > 0:
            for row_idx, src in enumerate(last_raw.detach().cpu().tolist()):
                prior = self.co_visit_prior.get(int(src))
                if prior is not None:
                    add_sparse_prior(scores, row_idx, prior[0], prior[1], self.co_visit_prior_weight)

        if self.category_transition_prior_weight > 0:
            last_category = self.poi_category[last_raw]
            candidate_category = self.poi_category[candidate_raw_ids]
            scores = scores + self.category_transition_prior_weight * self.category_transition_prior_matrix[
                last_category.unsqueeze(1), candidate_category
            ]

        if self.user_poi_prior_weight > 0:
            for row_idx, user in enumerate(batch["user_idx"].detach().cpu().tolist()):
                prior = self.user_poi_prior.get(int(user))
                if prior is not None:
                    add_sparse_prior(scores, row_idx, prior[0], prior[1], self.user_poi_prior_weight)

        if self.user_category_prior_weight > 0:
            candidate_category = self.poi_category[candidate_raw_ids]
            scores = scores + self.user_category_prior_weight * self.user_category_prior_matrix[
                batch["user_idx"].unsqueeze(1), candidate_category
            ]

        if self.spatial_prior_weight > 0:
            coords = self.spatial.poi_coords
            candidate_coords = coords[candidate_raw_ids]
            last_coords = coords[last_raw]
            distance_sq = ((candidate_coords - last_coords.unsqueeze(1)) ** 2).sum(dim=-1)
            scores = scores + self.spatial_prior_weight * torch.exp(
                -distance_sq / max(self.spatial_prior_temperature, 1e-6)
            )
        return scores

    def _collect_dense_prior_features(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Collect dense (non-sparse) prior scores: popularity, category_transition, user_category, spatial.

        Returns:
            shape (batch_size, num_pois, 4) tensor of raw prior scores.
        """
        device = batch["poi"].device
        batch_size = batch["poi"].shape[0]
        num_pois = self.num_pois
        prior_feats = torch.zeros((batch_size, num_pois, 4), device=device)

        # 0: popularity prior
        prior_feats[:, :, 0] = self.popularity_prior.unsqueeze(0)

        lengths = batch["attention_mask"].sum(dim=1).clamp(min=1) - 1
        last_shifted = batch["poi"][torch.arange(batch_size, device=device), lengths]
        last_raw = (last_shifted - 1).clamp(min=0)

        # 1: category_transition prior
        last_category = self.poi_category[last_raw]
        candidate_category = self.poi_category
        prior_feats[:, :, 1] = self.category_transition_prior_matrix[last_category][:, candidate_category]

        # 2: user_category prior
        prior_feats[:, :, 2] = self.user_category_prior_matrix[batch["user_idx"]][:, candidate_category]

        # 3: spatial prior
        coords = self.spatial.poi_coords
        last_coords = coords[last_raw]
        distance_sq = ((last_coords.unsqueeze(1) - coords.unsqueeze(0)) ** 2).sum(dim=-1)
        prior_feats[:, :, 3] = torch.exp(-distance_sq / max(self.spatial_prior_temperature, 1e-6))

        return prior_feats

    def _apply_candidate_priors(self, scores: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.use_prior_encoder and (
            self.transition_prior_weight <= 0
            and self.history_transition_prior_weight <= 0
            and self.user_poi_prior_weight <= 0
            and self.co_visit_prior_weight <= 0
            and self.category_transition_prior_weight <= 0
            and self.user_category_prior_weight <= 0
            and self.history_repeat_prior_weight <= 0
            and self.popularity_prior_weight <= 0
            and self.spatial_prior_weight <= 0
        ):
            return scores
        # Work in float32 to avoid dtype mismatches when index_put_ with float32 prior values
        adjusted = scores.float()

        lengths = batch["attention_mask"].sum(dim=1).clamp(min=1) - 1
        last_shifted = batch["poi"][torch.arange(batch["poi"].shape[0], device=batch["poi"].device), lengths]
        last_raw = (last_shifted - 1).clamp(min=0)

        # When PriorEncoder is enabled, replace the 4 dense priors (popularity, category_transition,
        # user_category, spatial) with a learned non-linear combination.
        if self.use_prior_encoder and self.prior_encoder is not None:
            dense_feats = self._collect_dense_prior_features(batch)
            # Under autocast, linear layers output bf16, but we need float32 for adjusted
            learned_prior_scores = self.prior_encoder(dense_feats).float()
            adjusted = adjusted + learned_prior_scores
        else:
            if self.popularity_prior_weight > 0:
                adjusted = adjusted + self.popularity_prior_weight * self.popularity_prior.unsqueeze(0)
            if self.category_transition_prior_weight > 0:
                last_category = self.poi_category[last_raw]
                candidate_category = self.poi_category
                category_scores = self.category_transition_prior_matrix[last_category][:, candidate_category]
                adjusted = adjusted + self.category_transition_prior_weight * category_scores
            if self.user_category_prior_weight > 0:
                candidate_category = self.poi_category
                category_scores = self.user_category_prior_matrix[batch["user_idx"]][:, candidate_category]
                adjusted = adjusted + self.user_category_prior_weight * category_scores
            if self.spatial_prior_weight > 0:
                coords = self.spatial.poi_coords
                last_coords = coords[last_raw]
                distance_sq = ((last_coords.unsqueeze(1) - coords.unsqueeze(0)) ** 2).sum(dim=-1)
                spatial_prior = torch.exp(-distance_sq / max(self.spatial_prior_temperature, 1e-6))
                adjusted = adjusted + self.spatial_prior_weight * spatial_prior

        if self.history_repeat_prior_weight > 0:
            repeat_scores = torch.zeros_like(adjusted)
            poi_history = batch["poi"].detach().cpu()
            mask_history = batch["attention_mask"].detach().cpu()
            decay = min(max(self.history_repeat_decay, 0.0), 1.0)
            for row_idx in range(poi_history.shape[0]):
                valid_positions = torch.nonzero(mask_history[row_idx].gt(0), as_tuple=False).flatten().tolist()
                for age, pos in enumerate(reversed(valid_positions)):
                    raw_poi = int(poi_history[row_idx, pos].item()) - 1
                    if raw_poi < 0:
                        continue
                    score = decay**age
                    if self.history_repeat_aggregation == "sum":
                        repeat_scores[row_idx, raw_poi] += score
                    else:
                        repeat_scores[row_idx, raw_poi] = max(float(repeat_scores[row_idx, raw_poi].item()), score)
            adjusted = adjusted + self.history_repeat_prior_weight * repeat_scores

        if self.transition_prior_weight > 0:
            prior_scores = torch.zeros_like(adjusted)
            for row_idx, src in enumerate(last_raw.detach().cpu().tolist()):
                prior = self.transition_prior.get(int(src))
                if prior is None:
                    continue
                dst, values = prior
                prior_scores[row_idx, dst.to(adjusted.device)] = values.to(adjusted.device)
            adjusted = adjusted + self.transition_prior_weight * prior_scores

        if self.history_transition_prior_weight > 0:
            prior_scores = torch.zeros_like(adjusted)
            poi_history = batch["poi"].detach().cpu()
            mask_history = batch["attention_mask"].detach().cpu()
            decay = min(max(self.history_transition_decay, 0.0), 1.0)
            for row_idx in range(poi_history.shape[0]):
                valid_positions = torch.nonzero(mask_history[row_idx].gt(0), as_tuple=False).flatten().tolist()
                for age, pos in enumerate(reversed(valid_positions)):
                    src = int(poi_history[row_idx, pos].item()) - 1
                    if src < 0:
                        continue
                    prior = self.transition_prior.get(src)
                    if prior is None:
                        continue
                    dst, values = prior
                    weighted = (decay**age) * values.to(adjusted.device)
                    dst = dst.to(adjusted.device)
                    if self.history_transition_aggregation == "sum":
                        prior_scores[row_idx, dst] += weighted
                    else:
                        prior_scores[row_idx, dst] = torch.maximum(prior_scores[row_idx, dst], weighted)
            adjusted = adjusted + self.history_transition_prior_weight * prior_scores

        if self.co_visit_prior_weight > 0:
            prior_scores = torch.zeros_like(adjusted)
            for row_idx, src in enumerate(last_raw.detach().cpu().tolist()):
                prior = self.co_visit_prior.get(int(src))
                if prior is None:
                    continue
                dst, values = prior
                prior_scores[row_idx, dst.to(adjusted.device)] = values.to(adjusted.device)
            adjusted = adjusted + self.co_visit_prior_weight * prior_scores

        if self.user_poi_prior_weight > 0:
            prior_scores = torch.zeros_like(adjusted)
            for row_idx, user in enumerate(batch["user_idx"].detach().cpu().tolist()):
                prior = self.user_poi_prior.get(int(user))
                if prior is None:
                    continue
                poi, values = prior
                prior_scores[row_idx, poi.to(adjusted.device)] = values.to(adjusted.device)
            adjusted = adjusted + self.user_poi_prior_weight * prior_scores

        # Dense priors (category_transition, user_category, spatial) are handled above:
        # - PriorEncoder path: learned non-linear combination
        # - Else path: individual weighted sum
        # Both paths already accounted for these 3 dense priors, so we skip them here
        # to avoid double-counting.
        return adjusted
