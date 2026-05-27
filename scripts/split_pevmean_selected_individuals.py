"""Split PEVmean-GA selected-individual CSVs into one file per train size.

The merged ``pevmean_ga_selected_individuals.csv`` files are convenient for
batch processing but awkward to inspect because each one can be hundreds of MB.
This script streams each CSV and writes smaller files grouped by ``n_train_size``
inside a ``selected_individuals`` directory under the trait output directory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TextIO


DEFAULT_ROOT = Path("outputs/final_results/e3_pevmean_ga")
DEFAULT_INPUT_NAME = "pevmean_ga_selected_individuals.csv"
DEFAULT_GROUP_COLUMN = "n_train_size"


def _close_all(handles: dict[str, TextIO]) -> None:
    for handle in handles.values():
        handle.close()


def split_selected_individuals(
    input_path: Path,
    output_dir: Path,
    group_column: str = DEFAULT_GROUP_COLUMN,
    overwrite: bool = True,
) -> dict[str, int]:
    """Split ``input_path`` into one CSV per distinct ``group_column`` value."""

    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, csv.DictWriter[str]] = {}
    handles: dict[str, TextIO] = {}
    row_counts: dict[str, int] = {}

    try:
        with input_path.open("r", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError(f"{input_path} has no header row")
            if group_column not in reader.fieldnames:
                raise ValueError(
                    f"{input_path} does not contain grouping column {group_column!r}"
                )

            for row in reader:
                group_value = row[group_column]
                if group_value == "":
                    raise ValueError(f"{input_path} contains an empty {group_column!r} value")

                writer = writers.get(group_value)
                if writer is None:
                    mode = "w" if overwrite else "x"
                    out_path = output_dir / f"k_{group_value}.csv"
                    handle = out_path.open(mode, newline="", encoding="utf-8")
                    handles[group_value] = handle
                    writer = csv.DictWriter(handle, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    writers[group_value] = writer
                    row_counts[group_value] = 0

                writer.writerow(row)
                row_counts[group_value] += 1
    finally:
        _close_all(handles)

    return row_counts


def discover_inputs(root: Path, input_name: str) -> list[Path]:
    return sorted(path for path in root.glob(f"*/{input_name}") if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split PEVmean-GA selected-individual CSVs by n_train_size."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root containing one directory per trait (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--input-name",
        default=DEFAULT_INPUT_NAME,
        help=f"Selected-individual filename to split (default: {DEFAULT_INPUT_NAME})",
    )
    parser.add_argument(
        "--output-dir-name",
        default="selected_individuals",
        help="Directory to create inside each trait directory.",
    )
    parser.add_argument(
        "--group-column",
        default=DEFAULT_GROUP_COLUMN,
        help=f"Column to split by (default: {DEFAULT_GROUP_COLUMN})",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if any split output file already exists.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Optional explicit selected-individual CSV paths. Defaults to all traits under --root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = args.inputs or discover_inputs(args.root, args.input_name)
    if not inputs:
        raise SystemExit(f"No input files found under {args.root} matching */{args.input_name}")

    for input_path in inputs:
        trait_dir = input_path.parent
        output_dir = trait_dir / args.output_dir_name
        counts = split_selected_individuals(
            input_path=input_path,
            output_dir=output_dir,
            group_column=args.group_column,
            overwrite=not args.no_overwrite,
        )
        parts = ", ".join(f"k={k}: {v}" for k, v in sorted(counts.items(), key=lambda x: int(x[0])))
        print(f"{input_path} -> {output_dir} ({parts})")


if __name__ == "__main__":
    main()
