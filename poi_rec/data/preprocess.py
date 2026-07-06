from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from node2vec import Node2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


COLUMNS = [
    "user_id",
    "venue_id",
    "category_id",
    "category_name",
    "latitude",
    "longitude",
    "timezone_offset",
    "utc_time",
]

SCHEMA_VERSION = 4


@dataclass(frozen=True)
class Sample:
    user_idx: int
    history_poi: list[int]
    history_hour: list[int]
    history_weekday: list[int]
    history_delta_bucket: list[int]
    target_poi: int


def read_tsmc2014(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, sep="\t", header=None, names=COLUMNS, encoding="latin-1")
    df["utc_time"] = pd.to_datetime(
        df["utc_time"],
        format="%a %b %d %H:%M:%S %z %Y",
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["utc_time", "venue_id", "user_id"]).copy()
    df["local_time"] = df["utc_time"] + pd.to_timedelta(df["timezone_offset"], unit="m")
    df["hour"] = df["local_time"].dt.hour.astype(int)
    df["weekday"] = df["local_time"].dt.weekday.astype(int)
    return df


def _make_mapping(values: pd.Series) -> dict[str, int]:
    return {str(value): idx for idx, value in enumerate(sorted(values.astype(str).unique()))}


def _delta_bucket(minutes: float) -> int:
    if minutes < 0:
        return 0
    if minutes < 30:
        return 1
    if minutes < 60:
        return 2
    if minutes < 180:
        return 3
    if minutes < 360:
        return 4
    if minutes < 720:
        return 5
    if minutes < 1440:
        return 6
    return 7


def _split_indices(n: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    test_count = max(1, int(round(n * test_ratio))) if n >= 3 else 0
    val_count = max(1, int(round(n * val_ratio))) if n - test_count >= 3 else 0
    train_end = max(1, n - val_count - test_count)
    val_end = max(train_end, n - test_count)
    return train_end, val_end


def _build_samples(
    df: pd.DataFrame,
    max_seq_len: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[tuple[int, int, float]],
    dict[int, tuple[int, int]],
    dict[int, int],
    list[tuple[int, int, float]],
]:
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    edge_counts: dict[tuple[int, int], float] = {}
    split_bounds: dict[int, tuple[int, int]] = {}
    train_visit_counts: dict[int, int] = {}
    user_poi_counts: dict[tuple[int, int], float] = {}
    for _, group in df.groupby("user_idx", sort=False):
        group = group.sort_values("utc_time")
        if len(group) < 2:
            continue
        poi = group["poi_idx"].to_numpy(dtype=np.int64)
        hours = group["hour"].to_numpy(dtype=np.int64)
        weekdays = group["weekday"].to_numpy(dtype=np.int64)
        times = group["utc_time"].astype("int64").to_numpy()
        deltas = np.zeros(len(group), dtype=np.int64)
        for i in range(1, len(group)):
            minutes = (times[i] - times[i - 1]) / 1e9 / 60.0
            deltas[i] = _delta_bucket(minutes)

        train_end, val_end = _split_indices(len(group), val_ratio, test_ratio)
        split_bounds[int(group["user_idx"].iloc[0])] = (train_end, val_end)
        train_seq = poi[:train_end]
        user_idx = int(group["user_idx"].iloc[0])
        for poi_idx in train_seq:
            train_visit_counts[int(poi_idx)] = train_visit_counts.get(int(poi_idx), 0) + 1
            user_poi_key = (user_idx, int(poi_idx))
            user_poi_counts[user_poi_key] = user_poi_counts.get(user_poi_key, 0.0) + 1.0
        for src, dst in zip(train_seq[:-1], train_seq[1:]):
            key = (int(src), int(dst))
            edge_counts[key] = edge_counts.get(key, 0.0) + 1.0
        for target_pos in range(1, len(group)):
            start = max(0, target_pos - max_seq_len)
            sample = Sample(
                user_idx=int(group["user_idx"].iloc[0]),
                history_poi=poi[start:target_pos].astype(int).tolist(),
                history_hour=hours[start:target_pos].astype(int).tolist(),
                history_weekday=weekdays[start:target_pos].astype(int).tolist(),
                history_delta_bucket=deltas[start:target_pos].astype(int).tolist(),
                target_poi=int(poi[target_pos]),
            )
            if target_pos < train_end:
                split = "train"
            elif target_pos < val_end:
                split = "val"
            else:
                split = "test"
            splits[split].append(asdict(sample))
    edges = [(src, dst, weight) for (src, dst), weight in sorted(edge_counts.items())]
    user_poi_edges = [(user, poi, weight) for (user, poi), weight in sorted(user_poi_counts.items())]
    return splits, edges, split_bounds, train_visit_counts, user_poi_edges


def _build_transition_counts(edges: list[tuple[int, int, float]], num_pois: int) -> np.ndarray:
    counts = np.zeros((num_pois, 4), dtype=np.float32)
    for src, dst, weight in edges:
        counts[src, 0] += weight
        counts[dst, 1] += weight
        if src != dst:
            counts[src, 2] += weight
            counts[dst, 3] += weight
    return np.log1p(counts)


def _build_node2vec_embeddings(
    edges: list[tuple[int, int, float]],
    num_pois: int,
    dimensions: int,
    walk_length: int,
    num_walks: int,
    p: float,
    q: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_pois == 0:
        return np.zeros((0, dimensions), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    graph = nx.DiGraph()
    for src, dst, weight in edges:
        graph.add_edge(str(src), str(dst), weight=float(weight))
    available = np.zeros((num_pois,), dtype=np.float32)
    for src, dst, _ in edges:
        available[src] = 1.0
        available[dst] = 1.0
    if not edges:
        return np.zeros((num_pois, dimensions), dtype=np.float32), available
    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        weight_key="weight",
        workers=1,
        quiet=True,
        seed=seed,
    )
    model = node2vec.fit(window=min(10, max(2, walk_length)), min_count=1, batch_words=128, seed=seed)
    embeddings = np.zeros((num_pois, dimensions), dtype=np.float32)
    for idx in range(num_pois):
        key = str(idx)
        if key in model.wv:
            embeddings[idx] = model.wv[key]
    embeddings[available == 0.0] = 0.0
    return embeddings, available


def _poi_text_descriptions(poi_meta: pd.DataFrame, city: str) -> list[str]:
    descriptions = []
    for _, row in poi_meta.iterrows():
        descriptions.append(
            " ".join(
                [
                    f"This point of interest is a {row['category_name']}.",
                    f"It belongs to the {row['category_name']} venue category.",
                    f"The city context is {city}.",
                ]
            )
        )
    return descriptions


def _build_tfidf_svd_embeddings(descriptions: list[str], fit_mask: np.ndarray, dimensions: int) -> np.ndarray:
    if not descriptions:
        return np.zeros((0, dimensions), dtype=np.float32)
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    fit_descriptions = [text for text, keep in zip(descriptions, fit_mask.tolist()) if keep]
    if not fit_descriptions:
        fit_descriptions = descriptions
    train_tfidf = vectorizer.fit_transform(fit_descriptions)
    all_tfidf = vectorizer.transform(descriptions)
    max_components = max(1, min(dimensions, train_tfidf.shape[0] - 1, train_tfidf.shape[1] - 1))
    if max_components < 2:
        dense = all_tfidf.toarray().astype(np.float32)
    else:
        svd = TruncatedSVD(n_components=max_components, random_state=42)
        svd.fit(train_tfidf)
        dense = svd.transform(all_tfidf).astype(np.float32)
    dense = normalize(dense, norm="l2", axis=1)
    if dense.shape[1] < dimensions:
        padded = np.zeros((dense.shape[0], dimensions), dtype=np.float32)
        padded[:, : dense.shape[1]] = dense
        return padded
    return dense[:, :dimensions].astype(np.float32)


def _build_hf_text_embeddings(
    descriptions: list[str],
    model_name: str,
    batch_size: int = 64,
    max_length: int = 64,
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    if not descriptions:
        return np.zeros((0, 0), dtype=np.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(descriptions), batch_size):
            batch = descriptions[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            outputs.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_preprocess_config(
    *,
    raw_path: Path,
    city: str,
    max_seq_len: int,
    min_user_checkins: int,
    val_ratio: float,
    test_ratio: float,
    limit_users: int | None,
    node2vec_dim: int,
    node2vec_walk_length: int,
    node2vec_num_walks: int,
    node2vec_p: float,
    node2vec_q: float,
    text_encoder: str,
    text_embedding_dim: int,
    text_model_name: str | None,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "raw_path": str(raw_path),
        "raw_sha256": _file_sha256(raw_path) if raw_path.exists() else None,
        "city": city,
        "max_seq_len": max_seq_len,
        "min_user_checkins": min_user_checkins,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "limit_users": limit_users,
        "node2vec_dim": node2vec_dim,
        "node2vec_walk_length": node2vec_walk_length,
        "node2vec_num_walks": node2vec_num_walks,
        "node2vec_p": node2vec_p,
        "node2vec_q": node2vec_q,
        "text_encoder": text_encoder,
        "text_embedding_dim": text_embedding_dim,
        "text_model_name": text_model_name,
        "seed": seed,
    }


def preprocess_fingerprint(preprocess_config: dict[str, Any]) -> str:
    payload = json.dumps(preprocess_config, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def preprocess_tsmc2014(
    raw_path: Path,
    out_dir: Path,
    city: str,
    max_seq_len: int = 20,
    min_user_checkins: int = 2,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    limit_users: int | None = None,
    node2vec_dim: int = 64,
    node2vec_walk_length: int = 10,
    node2vec_num_walks: int = 5,
    node2vec_p: float = 1.0,
    node2vec_q: float = 1.0,
    text_encoder: str = "tfidf_svd",
    text_embedding_dim: int = 64,
    text_model_name: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocess_config = build_preprocess_config(
        raw_path=raw_path,
        city=city,
        max_seq_len=max_seq_len,
        min_user_checkins=min_user_checkins,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        limit_users=limit_users,
        node2vec_dim=node2vec_dim,
        node2vec_walk_length=node2vec_walk_length,
        node2vec_num_walks=node2vec_num_walks,
        node2vec_p=node2vec_p,
        node2vec_q=node2vec_q,
        text_encoder=text_encoder,
        text_embedding_dim=text_embedding_dim,
        text_model_name=text_model_name,
        seed=seed,
    )
    fingerprint = preprocess_fingerprint(preprocess_config)
    df = read_tsmc2014(raw_path)
    user_counts = df["user_id"].value_counts()
    keep_users = user_counts[user_counts >= min_user_checkins].index.astype(str).tolist()
    keep_users = sorted(keep_users)
    if limit_users is not None:
        keep_users = keep_users[:limit_users]
    df = df[df["user_id"].astype(str).isin(keep_users)].copy()

    user_to_idx = _make_mapping(df["user_id"])
    poi_to_idx = _make_mapping(df["venue_id"])
    category_to_idx = _make_mapping(df["category_id"])
    df["user_idx"] = df["user_id"].astype(str).map(user_to_idx).astype(int)
    df["poi_idx"] = df["venue_id"].astype(str).map(poi_to_idx).astype(int)
    df["category_idx"] = df["category_id"].astype(str).map(category_to_idx).astype(int)
    df = df.sort_values(["user_idx", "utc_time"]).reset_index(drop=True)

    poi_meta = (
        df.groupby("poi_idx")
        .agg(
            venue_id=("venue_id", "first"),
            category_id=("category_id", "first"),
            category_name=("category_name", "first"),
            category_idx=("category_idx", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            visit_count=("poi_idx", "size"),
        )
        .sort_index()
    )
    coords = poi_meta[["latitude", "longitude"]].to_numpy(dtype=np.float32)
    coord_mean = coords.mean(axis=0)
    coord_std = coords.std(axis=0)
    coord_std[coord_std == 0.0] = 1.0
    coords_norm = (coords - coord_mean) / coord_std

    splits, transition_edges, split_bounds, train_visit_counts, user_poi_edges = _build_samples(
        df,
        max_seq_len=max_seq_len,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    transition_features = _build_transition_counts(transition_edges, num_pois=len(poi_to_idx))
    node2vec_embeddings, topology_available = _build_node2vec_embeddings(
        transition_edges,
        num_pois=len(poi_to_idx),
        dimensions=node2vec_dim,
        walk_length=node2vec_walk_length,
        num_walks=node2vec_num_walks,
        p=node2vec_p,
        q=node2vec_q,
        seed=seed,
    )
    poi_text = _poi_text_descriptions(poi_meta, city)
    train_seen_mask = np.zeros((len(poi_to_idx),), dtype=bool)
    train_visit_array = np.zeros((len(poi_to_idx),), dtype=np.float32)
    for poi_idx, count in train_visit_counts.items():
        train_seen_mask[poi_idx] = True
        train_visit_array[poi_idx] = float(count)
    if text_encoder == "tfidf_svd":
        text_embeddings = _build_tfidf_svd_embeddings(poi_text, train_seen_mask, text_embedding_dim)
    elif text_encoder == "hf_transformer":
        if not text_model_name:
            raise ValueError("text_model_name is required when text_encoder='hf_transformer'")
        text_embeddings = _build_hf_text_embeddings(poi_text, text_model_name)
    else:
        raise ValueError(f"Unknown text_encoder: {text_encoder}")
    edge_array = np.array(transition_edges, dtype=np.float32).reshape(-1, 3)
    user_poi_edge_array = np.array(user_poi_edges, dtype=np.float32).reshape(-1, 3)

    np.savez_compressed(
        out_dir / "arrays.npz",
        poi_category=poi_meta["category_idx"].to_numpy(dtype=np.int64),
        poi_coords=coords_norm.astype(np.float32),
        transition_features=transition_features,
        transition_edges=edge_array,
        node2vec_embeddings=node2vec_embeddings,
        topology_available=topology_available,
        train_seen_poi=train_seen_mask.astype(np.float32),
        train_visit_count=train_visit_array,
        user_poi_edges=user_poi_edge_array,
        text_embeddings=text_embeddings,
    )
    for split, samples in splits.items():
        _write_json(out_dir / f"{split}.json", samples)
    _write_json(
        out_dir / "metadata.json",
        {
            "city": city,
            "schema_version": SCHEMA_VERSION,
            "preprocess_fingerprint": fingerprint,
            "preprocess_config": preprocess_config,
            "raw_path": str(raw_path),
            "max_seq_len": max_seq_len,
            "num_users": len(user_to_idx),
            "num_pois": len(poi_to_idx),
            "num_categories": len(category_to_idx),
            "node2vec_dim": node2vec_dim,
            "text_encoder": text_encoder,
            "text_embedding_dim": int(text_embeddings.shape[1]),
            "text_model_name": text_model_name,
            "num_transition_edges": len(transition_edges),
            "train_transition_weight_sum": float(sum(weight for _, _, weight in transition_edges)),
            "train_adjacent_transition_count": int(sum(len(samples) for split, samples in splits.items() if split == "train")),
            "train_seen_pois": int(train_seen_mask.sum()),
            "num_user_poi_edges": len(user_poi_edges),
            "train_user_poi_weight_sum": float(sum(weight for _, _, weight in user_poi_edges)),
            "topology_available_pois": int(topology_available.sum()),
            "coord_mean": coord_mean.astype(float).tolist(),
            "coord_std": coord_std.astype(float).tolist(),
            "split_sizes": {split: len(samples) for split, samples in splits.items()},
            "split_protocol": "user_chronological_ratio",
        },
    )
    _write_json(out_dir / "poi_text.json", {str(i): text for i, text in enumerate(poi_text)})
    _write_json(out_dir / "user_split_bounds.json", {str(k): list(v) for k, v in split_bounds.items()})
    _write_json(out_dir / "user_to_idx.json", user_to_idx)
    _write_json(out_dir / "poi_to_idx.json", poi_to_idx)
    _write_json(out_dir / "category_to_idx.json", category_to_idx)
    poi_meta.to_json(out_dir / "poi_metadata.json", orient="index", force_ascii=False, indent=2)
    return {
        "city": city,
        "num_users": len(user_to_idx),
        "num_pois": len(poi_to_idx),
        "num_categories": len(category_to_idx),
        "num_transition_edges": len(transition_edges),
        "train_transition_weight_sum": float(sum(weight for _, _, weight in transition_edges)),
        "preprocess_fingerprint": fingerprint,
        **{f"{split}_samples": len(samples) for split, samples in splits.items()},
    }
