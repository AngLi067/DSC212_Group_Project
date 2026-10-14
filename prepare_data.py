"""Build leakage-safe sparse interactions and held-out targets from Instacart CSVs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, save_npz


def csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def data_row_count(path: Path) -> int:
    """Count CSV data lines quickly; the source files have one record per line."""
    newline_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            newline_count += chunk.count(b"\n")
    return newline_count - 1  # header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing the Instacart CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use IDs only to define a shared matrix index. Held-out product labels are
    # written to the target matrix below and never enter the interaction matrix.
    user_id_set: set[int] = set()
    max_order_id = 0
    for row in csv_rows(data_dir / "orders.csv"):
        user_id_set.add(int(row["user_id"]))
        max_order_id = max(max_order_id, int(row["order_id"]))
    user_ids = np.array(sorted(user_id_set), dtype=np.int32)
    user_index = {int(user_id): index for index, user_id in enumerate(user_ids)}

    product_ids = np.array(
        sorted(int(row["product_id"]) for row in csv_rows(data_dir / "products.csv")),
        dtype=np.int32,
    )
    product_index = {int(product_id): index for index, product_id in enumerate(product_ids)}

    order_to_user = np.full(max_order_id + 1, -1, dtype=np.int32)
    train_order_mask = np.zeros(max_order_id + 1, dtype=np.bool_)
    for row in csv_rows(data_dir / "orders.csv"):
        order_id = int(row["order_id"])
        order_to_user[order_id] = user_index[int(row["user_id"])]
        train_order_mask[order_id] = row["eval_set"] == "train"

    prior_path = data_dir / "order_products__prior.csv"
    train_path = data_dir / "order_products__train.csv"
    prior_count = data_row_count(prior_path)
    train_count = data_row_count(train_path)

    prior_rows = np.empty(prior_count, dtype=np.int32)
    prior_cols = np.empty(prior_count, dtype=np.int32)
    prior_values = np.ones(prior_count, dtype=np.float32)
    for offset, row in enumerate(csv_rows(prior_path)):
        order_id = int(row["order_id"])
        user_row = order_to_user[order_id]
        if user_row < 0 or train_order_mask[order_id]:
            raise ValueError(f"Prior label points to an unknown or held-out order: {order_id}")
        prior_rows[offset] = user_row
        prior_cols[offset] = product_index[int(row["product_id"])]

    train_rows = np.empty(train_count, dtype=np.int32)
    train_cols = np.empty(train_count, dtype=np.int32)
    train_values = np.ones(train_count, dtype=np.float32)
    for offset, row in enumerate(csv_rows(train_path)):
        order_id = int(row["order_id"])
        user_row = order_to_user[order_id]
        if user_row < 0 or not train_order_mask[order_id]:
            raise ValueError(f"Train label points to an unknown or non-train order: {order_id}")
        train_rows[offset] = user_row
        train_cols[offset] = product_index[int(row["product_id"])]

    shape = (len(user_ids), len(product_ids))
    interactions = coo_matrix((prior_values, (prior_rows, prior_cols)), shape=shape).tocsr()
    targets = coo_matrix((train_values, (train_rows, train_cols)), shape=shape).tocsr()
    interactions.sum_duplicates()
    targets.sum_duplicates()
    interactions.sort_indices()
    targets.sort_indices()

    np.save(output_dir / "user_ids.npy", user_ids)
    np.save(output_dir / "product_ids.npy", product_ids)
    save_npz(output_dir / "interactions.npz", interactions, compressed=True)
    save_npz(output_dir / "targets.npz", targets, compressed=True)
    metadata = {
        "n_users": len(user_ids),
        "n_products": len(product_ids),
        "interaction_shape": list(interactions.shape),
        "historical_user_product_pairs": int(interactions.nnz),
        "heldout_user_product_pairs": int(targets.nnz),
        "interaction_value": "number of prior orders containing the user-product pair",
        "target_value": "binary membership in the held-out train order",
        "target_labels_used_for_fitting": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Processed data written to: {output_dir}")


if __name__ == "__main__":
    main()
