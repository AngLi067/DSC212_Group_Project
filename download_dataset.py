"""Download the Instacart Market Basket Analysis dataset with KaggleHub."""

from pathlib import Path

import kagglehub


DATASET_HANDLE = "psparks/instacart-market-basket-analysis"
EXPECTED_FILES = {
    "orders.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
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
        print("All six expected CSV files are present.")


if __name__ == "__main__":
    main()
