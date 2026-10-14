"""Train implicit ALS on prior interactions and evaluate held-out train orders."""

import argparse
import json
from pathlib import Path

import implicit
import numpy as np
from scipy.sparse import load_npz

from popularity_baseline import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_dir", type=Path, nargs="?", default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/als"))
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.factors, args.iterations, args.k, args.threads) <= 0:
        parser.error("factors, iterations, k, and threads must be positive")

    processed_dir = args.processed_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    interactions = load_npz(processed_dir / "interactions.npz").tocsr().astype(np.float32)
    targets = load_npz(processed_dir / "targets.npz").tocsr()
    product_ids = np.load(processed_dir / "product_ids.npy")

    # Model purchase as implicit positive feedback. Repeated prior orders are
    # collapsed to 1 so one frequent buyer cannot create extreme confidence.
    model_interactions = interactions.copy()
    model_interactions.data.fill(1.0)

    model = implicit.als.AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.regularization,
        alpha=args.alpha,
        iterations=args.iterations,
        num_threads=args.threads,
        random_state=args.seed,
    )
    model.fit(model_interactions, show_progress=True)

    evaluation_users = np.flatnonzero(np.asarray(targets.getnnz(axis=1)).ravel() > 0)
    recommended_indices = np.empty((evaluation_users.size, args.k), dtype=np.int32)
    recommendation_scores = np.empty((evaluation_users.size, args.k), dtype=np.float32)
    batch_size = 2048
    for start in range(0, evaluation_users.size, batch_size):
        batch_users = evaluation_users[start : start + batch_size]
        item_indices, scores = model.recommend(
            batch_users,
            model_interactions[batch_users],
            N=args.k,
            filter_already_liked_items=False,
        )
        recommended_indices[start : start + batch_users.size] = item_indices
        recommendation_scores[start : start + batch_users.size] = scores

    metrics = evaluate_for_users(recommended_indices, evaluation_users, targets, args.k)
    metrics.update(
        {
            "baseline": "implicit ALS",
            "factors": args.factors,
            "iterations": args.iterations,
            "regularization": args.regularization,
            "alpha": args.alpha,
            "seed": args.seed,
            "k": args.k,
            "training_feedback": "binary prior user-product interactions",
        }
    )
    np.save(output_dir / "user_indices.npy", evaluation_users.astype(np.int32))
    np.save(output_dir / "recommendations.npy", recommended_indices)
    np.save(output_dir / "recommendation_scores.npy", recommendation_scores)
    np.save(output_dir / "recommended_product_ids.npy", product_ids[recommended_indices])
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"ALS outputs written to: {output_dir}")


def evaluate_for_users(recommendations: np.ndarray, user_indices: np.ndarray, targets, k: int) -> dict[str, float | int]:
    """Evaluate batched recommendations whose rows map through user_indices."""
    recalls: list[float] = []
    ndcgs: list[float] = []
    for output_row, user_index in enumerate(user_indices):
        start, end = targets.indptr[user_index : user_index + 2]
        target_items = targets.indices[start:end]
        relevant = np.isin(recommendations[output_row, :k], target_items)
        recalls.append(float(relevant.sum() / target_items.size))
        hit_ranks = np.flatnonzero(relevant)
        dcg = float(np.sum(1.0 / np.log2(hit_ranks + 2)))
        ideal_hits = min(target_items.size, k)
        idcg = float(np.sum(1.0 / np.log2(np.arange(ideal_hits) + 2)))
        ndcgs.append(dcg / idcg)

    covered_items = np.unique(recommendations[:, :k]).size
    return {
        "users_evaluated": int(user_indices.size),
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "catalog_coverage": float(covered_items / targets.shape[1]),
        "catalog_items_recommended": int(covered_items),
    }


if __name__ == "__main__":
    main()
