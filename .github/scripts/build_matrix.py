#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import re
import sys


def version_key(value: str) -> tuple:
    parts = re.findall(r"\d+", value)
    return tuple(int(p) for p in parts)


def read_grid(path: pathlib.Path, row_header: str) -> set[tuple[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc
        header = [cell.strip() for cell in header]
        if not header or header[0] != row_header:
            raise ValueError(f"{path} must have header '{row_header},...'")
        col_headers = header[1:]
        if not col_headers or any(not col for col in col_headers):
            raise ValueError(f"{path} must define column headers")
        pairs = set()
        rows_seen = set()
        row_index = 1
        for row in reader:
            row_index += 1
            if not row or all(not cell.strip() for cell in row):
                continue
            row = [cell.strip() for cell in row]
            if len(row) > len(header):
                raise ValueError(f"{path} row {row_index} has too many columns")
            if len(row) < len(header):
                row += [""] * (len(header) - len(row))
            row_key = row[0]
            if not row_key:
                continue
            if row_key in rows_seen:
                raise ValueError(f"{path} has duplicate row '{row_key}'")
            rows_seen.add(row_key)
            for col_header, value in zip(col_headers, row[1:]):
                if value == "1":
                    pairs.add((row_key, col_header))
                elif value in ("0", ""):
                    continue
                else:
                    raise ValueError(
                        f"{path} row '{row_key}' col '{col_header}' has invalid value '{value}'"
                    )
        return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python-torch",
        default=".github/matrices/python_torch.csv",
        help="CSV grid with header: python,<torch versions> and 1/0 values",
    )
    parser.add_argument(
        "--torch-lightning",
        default=".github/matrices/torch_lightning.csv",
        help="CSV grid with header: torch,<lightning versions> and 1/0 values",
    )
    parser.add_argument(
        "--python-lightning",
        default=".github/matrices/python_lightning.csv",
        help="CSV grid with header: python,<lightning versions> and 1/0 values",
    )
    args = parser.parse_args()

    python_torch = read_grid(pathlib.Path(args.python_torch), "python")
    torch_lightning = read_grid(pathlib.Path(args.torch_lightning), "torch")
    python_lightning = read_grid(pathlib.Path(args.python_lightning), "python")

    torch_to_lightnings: dict[str, set[str]] = {}
    for torch_version, lightning_version in torch_lightning:
        torch_to_lightnings.setdefault(torch_version, set()).add(lightning_version)

    include = []
    seen = set()
    for python_version, torch_version in python_torch:
        for lightning_version in torch_to_lightnings.get(torch_version, set()):
            if (python_version, lightning_version) not in python_lightning:
                continue
            key = (python_version, torch_version, lightning_version)
            if key in seen:
                continue
            seen.add(key)
            include.append(
                {
                    "python-version": python_version,
                    "torch-version": torch_version,
                    "lightning-version": lightning_version,
                }
            )

    include.sort(
        key=lambda item: (
            version_key(item["python-version"]),
            version_key(item["torch-version"]),
            version_key(item["lightning-version"]),
        )
    )

    matrix = {"include": include}
    sys.stdout.write(json.dumps(matrix, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
