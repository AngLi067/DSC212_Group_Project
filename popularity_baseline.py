"""Fit a historical popularity baseline and evaluate held-out top-k lists."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz


def evaluate(recommendations: np.ndarray, targets, k: int) -> dict[str, float | int]:
    recalls: list[float] = []
    ndcgs: list[float] = []
    evaluated_users = 0

    for user_index in range(targets.shape[0]):
        start, end = targets.indptr[user_index : user_index + 2]
        target_items = targets.indices[start:end]
        if target_items.size == 0:
            continue
        evaluated_users += 1
        rec_items = recommendations[user_index, :k]
        relevant = np.isin(rec_items, target_items, assume_unique=False)
        recalls.append(float(relevant.sum() / target_items.size))

        hit_ranks = np.flatnonzero(relevant)
        dcg = float(np.sum(1.0 / np.log2(hit_ranks + 2)))
        ideal_hits = min(target_items.size, k)
        idcg = float(np.sum(1.0 / np.log2(np.arange(ideal_hits) + 2)))
        ndcgs.append(dcg / idcg)

    active_recommendations = recommendations[targets.getnnz(axis=1) > 0, :k]
    coverage = np.unique(active_recommendations).size / targets.shape[1]
    return {
        "users_evaluated": evaluated_users,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "catalog_coverage": float(coverage),
        "catalog_items_recommended": int(np.unique(active_recommendations).size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_dir", type=Path, nargs="?", default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/popularity"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")

    processed_dir = args.processed_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    interactions = load_npz(processed_dir / "interactions.npz").tocsr()
    targets = load_npz(processed_dir / "targets.npz").tocsr()
    product_ids = np.load(processed_dir / "product_ids.npy")

    popularity = np.asarray(interactions.sum(axis=0)).ravel()
    item_indices = np.arange(popularity.size)
    ranked_items = np.lexsort((item_indices, -popularity))
    top_items = ranked_items[: args.k]
    recommendations = np.broadcast_to(top_items, (interactions.shape[0], top_items.size)).copy()

    metrics = evaluate(recommendations, targets, args.k)
    metrics["baseline"] = "global prior-order popularity"
    metrics["popularity_definition"] = "sum of historical user-product order counts"
    metrics["k"] = args.k
    np.save(output_dir / "recommendations.npy", recommendations.astype(np.int32))
    np.save(output_dir / "recommended_product_ids.npy", product_ids[top_items])
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Baseline outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
