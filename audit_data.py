"""Validate Retailrocket event, item-property, and category files."""

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = {
    "events.csv": {"timestamp", "visitorid", "event", "itemid", "transactionid"},
    "item_properties_part1.csv": {"timestamp", "itemid", "property", "value"},
    "item_properties_part2.csv": {"timestamp", "itemid", "property", "value"},
    "category_tree.csv": {"categoryid", "parentid"},
}


def rows(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def require_columns(path: Path, required: set[str]) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        columns = set(csv.DictReader(stream).fieldnames or [])
    missing = required - columns
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("data/raw/retailrocket"))
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()

    for filename, required in REQUIRED_COLUMNS.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
        require_columns(path, required)

    visitors: set[int] = set()
    event_items: set[int] = set()
    event_counts: Counter[str] = Counter()
    min_timestamp: int | None = None
    max_timestamp: int | None = None
    event_rows = 0
    for row in rows(data_dir / "events.csv"):
        timestamp = int(row["timestamp"])
        visitors.add(int(row["visitorid"]))
        event_items.add(int(row["itemid"]))
        event_counts[row["event"]] += 1
        min_timestamp = timestamp if min_timestamp is None else min(min_timestamp, timestamp)
        max_timestamp = timestamp if max_timestamp is None else max(max_timestamp, timestamp)
        event_rows += 1

    category_ids: set[int] = set()
    for row in rows(data_dir / "category_tree.csv"):
        category_ids.add(int(row["categoryid"]))

    property_rows = 0
    availability_rows = 0
    availability_items: set[int] = set()
    invalid_availability_values = 0
    invalid_category_values = 0
    for filename in ("item_properties_part1.csv", "item_properties_part2.csv"):
        for row in rows(data_dir / filename):
            property_rows += 1
            if row["property"] == "available":
                availability_rows += 1
                availability_items.add(int(row["itemid"]))
                if row["value"] not in {"0", "1"}:
                    invalid_availability_values += 1
            elif row["property"] == "categoryid":
                try:
                    category_value = int(row["value"])
                except ValueError:
                    invalid_category_values += 1
                else:
                    if category_value not in category_ids:
                        invalid_category_values += 1

    if event_rows == 0 or not visitors or not event_items:
        raise ValueError("events.csv contains no usable events")
    if invalid_availability_values:
        raise ValueError(f"Found {invalid_availability_values} availability values outside 0/1")

    iso_min = datetime.fromtimestamp(min_timestamp / 1000, tz=timezone.utc).isoformat()
    iso_max = datetime.fromtimestamp(max_timestamp / 1000, tz=timezone.utc).isoformat()
    print(f"Data directory: {data_dir}")
    print(f"Events: {event_rows:,}; visitors: {len(visitors):,}; event items: {len(event_items):,}")
    print(f"Event types: {dict(event_counts)}")
    print(f"Event time range (UTC): {iso_min} to {iso_max}")
    print(f"Property rows: {property_rows:,}; availability records: {availability_rows:,}; items with availability history: {len(availability_items):,}")
    print(f"Invalid availability values: {invalid_availability_values}; category values absent from tree: {invalid_category_values}")


if __name__ == "__main__":
    main()
