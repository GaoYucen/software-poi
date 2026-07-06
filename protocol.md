# software-poi Experiment Protocol

## Dataset

- Source: TSMC2014/Foursquare NYC and Tokyo check-ins.
- Raw columns: user ID, venue ID, category ID, category name, latitude, longitude, timezone offset, UTC time.
- Time is sorted per user by UTC timestamp. Local hour and weekday are computed with the timezone offset.

## Filtering

- Default minimum user history: `min_user_checkins: 2`.
- POI mapping is built from the filtered city dataset so inductive analysis can identify train-unseen targets.
- Behavioral statistics used for topology or popularity are computed from the training prefix only.

## Split

- Default split: per-user chronological ratio split.
- Train: earliest `1 - val_ratio - test_ratio` portion.
- Validation: next `val_ratio` portion.
- Test: final `test_ratio` portion.
- Each target position creates one next-POI sample with up to `max_seq_len` previous visits.

## Candidate Protocol

- Main reported protocol: closed-world full ranking.
- Closed-world means candidates and evaluated targets are restricted to train-seen POIs.
- Inductive all-POI ranking must be reported separately because train-unseen POIs do not have training positives.

## Modalities

- Topology: training-only directed weighted POI transition graph, Node2Vec embedding, and graph degree statistics.
- Semantics: POI static text only, using category/city descriptions; no future visit count, category ID hash, latitude, or longitude in text.
- Spatial: normalized latitude and longitude, separate from semantic text.
- Temporal: local hour, weekday, and inter-check-in time bucket.

## Metrics

- Full-ranking HR@K, NDCG@K, MRR.
- Recall@K is emitted as an alias of HR@K for single-ground-truth compatibility.
- Primary model selection metric: `NDCG@10`.

## Reproducibility

- Processed artifacts include a preprocessing fingerprint over raw checksum and preprocessing/model-input parameters.
- Alignment checkpoints must match the processed fingerprint and must not contain data buffers.
- Baselines must be rerun under this same split, candidate, and metric protocol before comparison.

