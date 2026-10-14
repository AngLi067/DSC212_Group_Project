"""Stream-check the downloaded Instacart CSV files and split relationships."""

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "orders.csv": {"order_id", "user_id", "eval_set"},
    "products.csv": {"product_id"},
    "aisles.csv": {"aisle_id"},
    "departments.csv": {"department_id"},
    "order_products__prior.csv": {"order_id", "product_id"},
    "order_products__train.csv": {"order_id", "product_id"},
}

SPLIT_CODES = {"prior": 1, "train": 2, "test": 3}


def rows(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def require_columns(path: Path, required: set[str]) -> list[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
    missing = required - set(columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing the six Instacart CSV files")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()

    for filename, required in REQUIRED_COLUMNS.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
        require_columns(path, required)

    # A compact lookup table (one byte per order ID) avoids a large Python
    # dictionary for the multi-million-row orders.csv file.
    max_order_id = max(int(row["order_id"]) for row in rows(data_dir / "orders.csv"))
    order_split = bytearray(max_order_id + 1)
    historical_users: set[int] = set()
    train_order_users: dict[int, int] = {}
    order_rows = 0
    duplicate_order_ids = 0
    unknown_splits: set[str] = set()

    for row in rows(data_dir / "orders.csv"):
        order_id = int(row["order_id"])
        user_id = int(row["user_id"])
        code = SPLIT_CODES.get(row["eval_set"], 0)
        if order_split[order_id]:
            duplicate_order_ids += 1
        order_split[order_id] = code
        if code == SPLIT_CODES["prior"]:
            historical_users.add(user_id)
        elif code == SPLIT_CODES["train"]:
            train_order_users[order_id] = user_id
        elif code == 0:
            unknown_splits.add(row["eval_set"])
        order_rows += 1

    product_ids = {int(row["product_id"]) for row in rows(data_dir / "products.csv")}
    prior_stats = scan_order_products(data_dir / "order_products__prior.csv", 1, order_split, product_ids)
    train_stats = scan_order_products(data_dir / "order_products__train.csv", 2, order_split, product_ids)
    train_users = set(train_order_users.values())
    overlap = train_users & historical_users

    print(f"Data directory: {data_dir}")
    print(f"Orders: {order_rows:,}; products: {len(product_ids):,}")
    print(f"Order ID duplicates: {duplicate_order_ids:,}; unknown eval_set values: {sorted(unknown_splits)}")
    print(f"Prior order-product rows: {prior_stats['rows']:,}; wrong-split order IDs: {prior_stats['wrong_split']:,}; unknown product IDs: {prior_stats['unknown_product']:,}")
    print(f"Train order-product rows: {train_stats['rows']:,}; wrong-split order IDs: {train_stats['wrong_split']:,}; unknown product IDs: {train_stats['unknown_product']:,}")
    print(f"Train users: {len(train_users):,}; also present in prior: {len(overlap):,}; without prior history: {len(train_users - overlap):,}")


def scan_order_products(path: Path, expected_split: int, order_split: bytearray, product_ids: set[int]) -> dict[str, int]:
    stats = {"rows": 0, "wrong_split": 0, "unknown_product": 0}
    for row in rows(path):
        order_id = int(row["order_id"])
        product_id = int(row["product_id"])
        stats["rows"] += 1
        if order_id >= len(order_split) or order_split[order_id] != expected_split:
            stats["wrong_split"] += 1
        if product_id not in product_ids:
            stats["unknown_product"] += 1
    return stats


if __name__ == "__main__":
    main()
