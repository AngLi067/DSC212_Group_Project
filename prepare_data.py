"""Create chronological Retailrocket interactions and availability-at-cutoff data."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix, save_npz


EVENT_WEIGHTS = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}
AVAILABILITY_FILES = ("item_properties_part1.csv", "item_properties_part2.csv")
AVAILABILITY_DTYPE = np.dtype([("item", "<i4"), ("timestamp", "<i8"), ("value", "i1")])


def csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def count_data_lines(path: Path) -> int:
    newlines = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            newlines += chunk.count(b"\n")
    return max(0, newlines - 1)


def load_events(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    count = count_data_lines(path)
    dtype = np.dtype([("timestamp", "<i8"), ("visitor", "<i4"), ("item", "<i4"), ("weight", "<f4")])
    events = np.empty(count, dtype=dtype)
    event_counts: dict[str, int] = {}
    for index, row in enumerate(csv_rows(path)):
        event_name = row["event"]
        if event_name not in EVENT_WEIGHTS:
            raise ValueError(f"Unknown event type: {event_name}")
        events[index] = (int(row["timestamp"]), int(row["visitorid"]), int(row["itemid"]), EVENT_WEIGHTS[event_name])
        event_counts[event_name] = event_counts.get(event_name, 0) + 1
    if len(events) != count:
        raise ValueError(f"Expected {count} event rows but parsed {len(events)}")
    return events, event_counts


def load_availability(data_dir: Path) -> np.ndarray:
    chunks: list[np.ndarray] = []
    chunk_rows: list[tuple[int, int, int]] = []
    chunk_size = 250_000
    for filename in AVAILABILITY_FILES:
        for row in csv_rows(data_dir / filename):
            if row["property"] != "available":
                continue
            value = row["value"]
            if value not in {"0", "1"}:
                raise ValueError(f"Unexpected availability value {value!r} in {filename}")
            chunk_rows.append((int(row["itemid"]), int(row["timestamp"]), int(value)))
            if len(chunk_rows) == chunk_size:
                chunks.append(np.array(chunk_rows, dtype=AVAILABILITY_DTYPE))
                chunk_rows.clear()
    if chunk_rows:
        chunks.append(np.array(chunk_rows, dtype=AVAILABILITY_DTYPE))
    if not chunks:
        raise ValueError("No 'available' property rows found")
    availability = np.concatenate(chunks)
    availability.sort(order=("item", "timestamp"), kind="stable")
    return availability


def asof_indices(
    property_keys: np.ndarray,
    property_items: np.ndarray,
    item_ids: np.ndarray,
    timestamps: np.ndarray,
    multiplier: int,
) -> np.ndarray:
    """Return the most recent same-item property row at or before each time."""
    query_keys = item_ids.astype(np.int64) * multiplier + timestamps.astype(np.int64)
    indices = np.searchsorted(property_keys, query_keys, side="right") - 1
    valid = indices >= 0
    valid[valid] &= property_items[indices[valid]] == item_ids[valid]
    indices[~valid] = -1
    return indices


def main() -> None:
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("data/raw/retailrocket"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()
    if not 0.5 <= args.train_fraction < 1.0:
        parser.error("--train-fraction must be >= 0.5 and < 1.0")

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    events, event_counts = load_events(data_dir / "events.csv")
    availability = load_availability(data_dir)

    cutoff = int(np.quantile(events["timestamp"], args.train_fraction))
    is_train = events["timestamp"] <= cutoff
    is_test = ~is_train
    user_ids = np.unique(events["visitor"])
    item_ids = np.unique(events["item"])
    event_user_rows = np.searchsorted(user_ids, events["visitor"])
    event_item_cols = np.searchsorted(item_ids, events["item"])

    shape = (user_ids.size, item_ids.size)
    interactions = coo_matrix(
        (events["weight"][is_train], (event_user_rows[is_train], event_item_cols[is_train])),
        shape=shape,
        dtype=np.float32,
    ).tocsr()
    interactions.sum_duplicates()
    interactions.sort_indices()

    targets = coo_matrix(
        (np.ones(is_test.sum(), dtype=np.float32), (event_user_rows[is_test], event_item_cols[is_test])),
        shape=shape,
        dtype=np.float32,
    ).tocsr()
    targets.sum_duplicates()
    targets.data.fill(1.0)
    targets.sort_indices()

    multiplier = int(max(events["timestamp"].max(), availability["timestamp"].max())) + 1
    property_keys = availability["item"].astype(np.int64) * multiplier + availability["timestamp"]
    cutoff_indices = asof_indices(
        property_keys,
        availability["item"],
        item_ids,
        np.full(item_ids.size, cutoff, dtype=np.int64),
        multiplier,
    )
    availability_at_cutoff = np.full(item_ids.size, -1, dtype=np.int8)
    known_cutoff = cutoff_indices >= 0
    availability_at_cutoff[known_cutoff] = availability["value"][cutoff_indices[known_cutoff]]

    event_indices = asof_indices(
        property_keys,
        availability["item"],
        events["item"],
        events["timestamp"],
        multiplier,
    )
    target_counts_by_item = np.asarray(targets.sum(axis=0)).ravel()
    target_pairs = int(target_counts_by_item.sum())
    target_known_pairs = int(target_counts_by_item[known_cutoff].sum())
    target_unavailable_pairs = int(target_counts_by_item[availability_at_cutoff == 0].sum())
    metadata = {
        "dataset": "Retailrocket e-commerce dataset",
        "n_events": int(events.size),
        "event_counts": event_counts,
        "n_users": int(user_ids.size),
        "n_items": int(item_ids.size),
        "train_fraction_requested": args.train_fraction,
        "train_events": int(is_train.sum()),
        "test_events": int(is_test.sum()),
        "cutoff_timestamp_ms": cutoff,
        "train_interaction_pairs": int(interactions.nnz),
        "test_target_pairs": int(targets.nnz),
        "event_level_availability_asof_coverage": float(np.mean(event_indices >= 0)),
        "catalog_availability_asof_coverage_at_cutoff": float(np.mean(known_cutoff)),
        "test_target_availability_asof_coverage_at_cutoff": target_known_pairs / target_pairs if target_pairs else 0.0,
        "test_target_known_unavailable_rate_at_cutoff": target_unavailable_pairs / target_known_pairs if target_known_pairs else 0.0,
        "availability_values": {"0": "unavailable", "1": "available", "-1": "unknown"},
        "event_weights": EVENT_WEIGHTS,
        "availability_cutoff_rule": "latest available-property timestamp <= global chronological train cutoff",
        "preparation_runtime_seconds": round(perf_counter() - started, 3),
    }

    np.save(output_dir / "user_ids.npy", user_ids.astype(np.int32))
    np.save(output_dir / "item_ids.npy", item_ids.astype(np.int32))
    np.save(output_dir / "availability_at_cutoff.npy", availability_at_cutoff)
    save_npz(output_dir / "train_interactions.npz", interactions, compressed=True)
    save_npz(output_dir / "test_targets.npz", targets, compressed=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Processed data written to: {output_dir}")


if __name__ == "__main__":
    main()
