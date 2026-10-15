"""Compare popularity and item-based top-K recommendations with availability filtering."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import implicit.nearest_neighbours as nearest_neighbours
import numpy as np
from scipy.sparse import load_npz


def evaluate(recommendations: np.ndarray, user_indices: np.ndarray, targets, availability: np.ndarray, k: int) -> dict:
    recalls = []
    ndcgs = []
    valid_counts = []
    unavailable_count = 0
    unknown_count = 0
    covered_items: set[int] = set()
    discounts = 1.0 / np.log2(np.arange(k) + 2)

    for output_row, user_index in enumerate(user_indices):
        start, end = targets.indptr[user_index : user_index + 2]
        target_items = targets.indices[start:end]
        row = recommendations[output_row, :k]
        valid = row >= 0
        rec_items = row[valid]
        relevant = np.isin(rec_items, target_items)
        recalls.append(float(relevant.sum() / target_items.size) if target_items.size else 0.0)
        dcg = float(discounts[np.flatnonzero(relevant)].sum())
        ideal_hits = min(target_items.size, k)
        idcg = float(discounts[:ideal_hits].sum())
        ndcgs.append(dcg / idcg if idcg else 0.0)

        states = availability[rec_items]
        valid_counts.append(rec_items.size)
        unavailable_count += int(np.count_nonzero(states == 0))
        unknown_count += int(np.count_nonzero(states < 0))
        covered_items.update(map(int, rec_items))

    total_recommendations = sum(valid_counts)
    return {
        "users_evaluated": int(user_indices.size),
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "catalog_items_recommended": len(covered_items),
        "catalog_coverage": len(covered_items) / availability.size,
        "recommendation_slots_filled": int(total_recommendations),
        "unavailable_recommendation_rate": unavailable_count / total_recommendations if total_recommendations else 0.0,
        "unknown_availability_recommendation_rate": unknown_count / total_recommendations if total_recommendations else 0.0,
    }


def filter_known_unavailable(
    candidates: np.ndarray,
    availability: np.ndarray,
    popularity_order: np.ndarray,
    k: int,
) -> np.ndarray:
    filtered = np.full((candidates.shape[0], k), -1, dtype=np.int32)
    for row_index, row in enumerate(candidates):
        valid = row >= 0
        valid[valid] &= availability[row[valid]] != 0
        eligible = row[valid]
        take = min(k, eligible.size)
        filtered[row_index, :take] = eligible[:take]
        if take < k:
            seen = set(map(int, eligible[:take]))
            slot = take
            for item in popularity_order:
                item = int(item)
                if item not in seen and availability[item] != 0:
                    filtered[row_index, slot] = item
                    seen.add(item)
                    slot += 1
                    if slot == k:
                        break
    return filtered


def recommend_item_based(
    interactions,
    user_indices: np.ndarray,
    candidate_k: int,
    neighbors: int,
    threads: int,
    popularity_order: np.ndarray,
) -> np.ndarray:
    model = nearest_neighbours.CosineRecommender(K=neighbors, num_threads=threads)
    model.fit(interactions, show_progress=True)
    candidates = np.empty((user_indices.size, candidate_k), dtype=np.int32)
    batch_size = 2048
    for start in range(0, user_indices.size, batch_size):
        users = user_indices[start : start + batch_size]
        item_indices, _ = model.recommend(
            users,
            interactions[users].tocsr(),
            N=candidate_k,
            filter_already_liked_items=False,
        )
        candidates[start : start + users.size] = item_indices

    # Sparse visitors can have fewer than candidate_k usable neighbours.
    # Fill only those remaining slots with the popularity ranking.
    for row in candidates:
        valid = row[row >= 0]
        if valid.size < candidate_k:
            seen = set(map(int, valid))
            fill = []
            needed = candidate_k - valid.size
            for item in popularity_order:
                item = int(item)
                if item not in seen:
                    fill.append(item)
                    if len(fill) == needed:
                        break
            row[:] = -1
            row[: valid.size] = valid
            row[valid.size : valid.size + len(fill)] = fill
    return candidates


def save_model_results(name: str, candidates: np.ndarray, user_indices: np.ndarray, targets, availability: np.ndarray, item_ids: np.ndarray, popularity_order: np.ndarray, k: int, output_dir: Path, model_runtime: float) -> None:
    evaluation_started = perf_counter()
    top_k = candidates[:, :k].copy()
    filtered = filter_known_unavailable(candidates, availability, popularity_order, k)
    metrics = {
        "model": name,
        "k": k,
        "raw": evaluate(top_k, user_indices, targets, availability, k),
        "known_unavailable_filtered": evaluate(filtered, user_indices, targets, availability, k),
        "filter_rule": "remove items whose latest availability state at the train cutoff is 0; retain unknown states",
        "filtered_list_fallback": "global popularity fills any remaining slots with available or unknown items",
        "runtime_seconds_including_evaluation": round(model_runtime + perf_counter() - evaluation_started, 3),
    }
    if name == "item_based_cosine":
        metrics["candidate_fallback"] = "global popularity fills candidate slots when a user has too few item-neighbour results"
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(model_dir / "user_indices.npy", user_indices.astype(np.int32))
    raw_ids = np.where(top_k >= 0, item_ids[np.maximum(top_k, 0)], -1)
    np.save(model_dir / "raw_recommendations.npy", raw_ids)
    np.save(model_dir / "filtered_recommendations.npy", np.where(filtered >= 0, item_ids[np.maximum(filtered, 0)], -1))
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_dir", type=Path, nargs="?", default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.k, args.candidate_k, args.neighbors, args.threads) <= 0 or args.candidate_k < args.k:
        parser.error("k, candidate-k, neighbors, and threads must be positive; candidate-k must be >= k")

    processed_dir = args.processed_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    interactions = load_npz(processed_dir / "train_interactions.npz").tocsr()
    targets = load_npz(processed_dir / "test_targets.npz").tocsr()
    item_ids = np.load(processed_dir / "item_ids.npy")
    availability = np.load(processed_dir / "availability_at_cutoff.npy")
    active = np.asarray(interactions.getnnz(axis=1)).ravel() > 0
    has_target = np.asarray(targets.getnnz(axis=1)).ravel() > 0
    user_indices = np.flatnonzero(active & has_target)
    if user_indices.size == 0:
        raise ValueError("No test users have both pre-cutoff interactions and post-cutoff targets")

    popularity_started = perf_counter()
    popularity = np.asarray(interactions.sum(axis=0)).ravel()
    item_order = np.lexsort((np.arange(item_ids.size), -popularity))
    pop_candidates = np.broadcast_to(item_order[: args.candidate_k], (user_indices.size, min(args.candidate_k, item_ids.size)))
    save_model_results(
        "popularity", pop_candidates, user_indices, targets, availability, item_ids, item_order, args.k,
        output_dir, perf_counter() - popularity_started
    )

    item_started = perf_counter()
    item_candidates = recommend_item_based(
        interactions, user_indices, args.candidate_k, args.neighbors, args.threads, item_order
    )
    save_model_results(
        "item_based_cosine", item_candidates, user_indices, targets, availability, item_ids, item_order, args.k,
        output_dir, perf_counter() - item_started
    )

    metadata = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = {
        "split_cutoff_timestamp_ms": metadata["cutoff_timestamp_ms"],
        "users_evaluated": int(user_indices.size),
        "catalog_availability_coverage_at_cutoff": metadata["catalog_availability_asof_coverage_at_cutoff"],
        "test_target_availability_coverage_at_cutoff": metadata["test_target_availability_asof_coverage_at_cutoff"],
        "test_target_known_unavailable_rate_at_cutoff": metadata["test_target_known_unavailable_rate_at_cutoff"],
        "unknown_items_remain_eligible": True,
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
