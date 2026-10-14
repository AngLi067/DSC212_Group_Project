# Inventory-Aware Recommendation

This project compares grocery product recommendation methods and studies how simulated product availability changes recommendation quality. Inventory availability is simulated because the Instacart Market Basket Analysis data does not include inventory snapshots.

## Research question

Can inventory-aware ranking reduce recommendations for simulated unavailable products while retaining useful recommendation quality? How does that trade-off change under different availability scenarios?

## Planned methods

- Popularity baseline trained from historical (`prior`) orders.
- Implicit-feedback ALS trained from historical user-product interactions.
- ALS recommendation lists reranked with simulated inventory risk, including hard filtering and risk-penalized ranking.

## Planned evaluation

Evaluate against held-out `train` orders using Recall@10, NDCG@10, catalog coverage, simulated unavailable-item recommendation rate, and a documented availability-adjusted Recall@10. Compare high-, medium-, and low-availability scenarios with fixed random seeds. No held-out order labels may be used to fit recommendation scores or generate inventory states.

## Data

Expected CSV files:

- `orders.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`
- `order_products__prior.csv`
- `order_products__train.csv`

Keep source data out of version control unless the team has explicitly decided otherwise. The repository currently contains no dataset files; add or point the pipeline at the files before beginning the data audit.

## Initial implementation sequence

1. Validate the input files, columns, keys, and user overlap between historical and held-out orders.
2. Build a sparse historical user-product interaction matrix and a held-out target table.
3. Implement and evaluate the popularity baseline.
4. Implement ALS and generate comparable top-10 candidate lists.
5. Add seeded availability scenarios and inventory-aware ranking policies.
6. Compute metrics, plot trade-offs, and document assumptions and reproduction steps.

## Project status

Setup in progress. The first code and data-audit files will be added incrementally, one at a time.
