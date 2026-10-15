"""Download the Retailrocket e-commerce dataset with KaggleHub."""

from pathlib import Path

import kagglehub


DATASET_HANDLE = "retailrocket/ecommerce-dataset"
EXPECTED_FILES = {
    "events.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
    "category_tree.csv",
}


def main() -> None:
    dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE)).expanduser().resolve()
    print(f"Dataset cache path: {dataset_path}")

    found = {path.name for path in dataset_path.rglob("*.csv")}
    missing = sorted(EXPECTED_FILES - found)
    if missing:
        print("Expected CSV files not found:")
        for name in missing:
            print(f"  - {name}")
    else:
        print("All expected dataset CSV files are present.")


if __name__ == "__main__":
    main()
