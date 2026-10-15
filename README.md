# Availability-Aware E-Commerce Recommendation

This project uses the Retailrocket e-commerce dataset to compare popularity and item-based recommendations, then measures how filtering products with a known unavailable state affects recommendation quality. The data has historical availability states, not physical inventory quantities.

## Data

The four Kaggle CSV files belong in `data/raw/retailrocket/`:

- `events.csv`
- `item_properties_part1.csv`
- `item_properties_part2.csv`
- `category_tree.csv`

Raw and processed data are ignored by Git. To download the public dataset with KaggleHub:

```bash
.venv/bin/python download_dataset.py
```

The `available` property is 0 for unavailable and 1 for available. Missing states remain unknown.

## Reproduce

Run these commands from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python audit_data.py data/raw/retailrocket
.venv/bin/python prepare_data.py data/raw/retailrocket
.venv/bin/python recommendation_models.py data/processed
.venv/bin/python -m unittest discover -s tests -v
```

`prepare_data.py` splits events chronologically (default 80% train) and creates weighted implicit interactions (view=1, add-to-cart=3, transaction=5), binary future targets, and an availability snapshot using the latest property timestamp at or before the training cutoff.

`recommendation_models.py` compares global popularity with item-based cosine similarity. It reports Recall@10, NDCG@10, catalog coverage, unavailable-item exposure, unknown-state exposure, and runtime before and after filtering. Known unavailable products are removed; unknown products remain eligible. Global popularity fills any remaining Top 10 slots.

## Outputs

- `data/processed/metadata.json`: split counts, cutoff, and availability-state coverage.
- `outputs/popularity/metrics.json`: popularity results.
- `outputs/item_based_cosine/metrics.json`: item-based results.
- `outputs/evaluation_summary.json`: shared evaluation context.
- `AI_USAGE.md`: AI-tool usage log.

Availability is evaluated at the fixed training cutoff. Results are offline and do not estimate physical inventory or causal sales impact.

## Current run

The chronological split contains 2,204,881 training events, 551,220 test events, and 24,735 evaluated visitors with both history and future targets. Historical availability can be matched at event time for 80.7% of all events; at the training cutoff it covers 77.7% of catalog items and 86.4% of test target pairs. Among test target pairs with a known state, 48.9% are marked unavailable.

Item-based cosine reached Recall@10 0.1501 and NDCG@10 0.1397 before filtering. After removing known-unavailable items, Recall@10 was 0.0837 and NDCG@10 was 0.0791; known-unavailable recommendation exposure fell from 47.7% to 0%. These figures show a relevance-availability trade-off for this fixed cutoff, not a real-world sales effect.
