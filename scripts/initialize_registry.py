"""Create valid empty placeholders only; never overwrite local registries."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rat_usv.data.inventory import INVENTORY_SCHEMA
from rat_usv.schemas.registry import initialize_empty


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "registry"
    initialize_empty(root)
    source = root / "source_inventory.parquet"
    if not source.exists():
        with source.open("xb") as stream:
            pq.write_table(pa.Table.from_pylist([], schema=INVENTORY_SCHEMA), stream)
    checksums = root / "checksums.sha256"
    if not checksums.exists():
        with checksums.open("x"):
            pass


if __name__ == "__main__":
    main()
