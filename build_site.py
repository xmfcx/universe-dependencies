#!/usr/bin/env python3
"""Build the static candidate browser and a reproducible data snapshot."""

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TIERS = {"ready", "review", "used_elsewhere", "used_by_launch"}
NUMERIC_FIELDS = {
    "Rank", "Score", "Internal_Recursive_Dependent_Count",
    "Internal_Direct_Dependent_Count", "Internal_Prerequisite_Count",
    "External_Manifest_Dependent_Count", "External_High_Reference_Count",
    "External_Review_Reference_Count",
}
BOOLEAN_FIELDS = {"Launch_Reachable", "Standalone"}
REQUIRED_FIELDS = NUMERIC_FIELDS | BOOLEAN_FIELDS | {
    "Package", "Tier", "Launch_Path", "Internal_Prerequisites", "Source_Path"
}


def load_candidates(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Candidate CSV is empty")
    for expected_rank, row in enumerate(rows, 1):
        for key in NUMERIC_FIELDS:
            row[key] = int(row[key])
        for key in BOOLEAN_FIELDS:
            if row[key] not in {"True", "False"}:
                raise ValueError(f"Invalid {key} for {row['Package']}: {row[key]}")
            row[key] = row[key] == "True"
        if row["Rank"] != expected_rank or row["Tier"] not in TIERS:
            raise ValueError(f"Invalid rank or tier for {row['Package']}")
        if row["Tier"] == "ready" and row["Launch_Reachable"]:
            raise ValueError(f"Launch reachable package marked ready: {row['Package']}")
        if row["Launch_Reachable"] and not row["Launch_Path"]:
            raise ValueError(f"Missing launch path for {row['Package']}")
        if expected_rank > 1 and rows[expected_rank - 2]["Score"] < row["Score"]:
            raise ValueError("Candidate scores are not descending")
    return rows


def build(csv_path, output_dir, commits, generated_at=None):
    rows = load_candidates(csv_path)
    metadata = {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_packages": len(rows),
        "tier_counts": dict(Counter(row["Tier"] for row in rows)),
        "commits": commits,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    site_dir = Path(__file__).resolve().parent / "site"
    for filename in ("index.html", "style.css", "app.js"):
        shutil.copyfile(site_dir / filename, output_dir / filename)
    shutil.copyfile(csv_path, output_dir / "candidates.csv")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "data.json").write_text(json.dumps({
        "metadata": metadata, "candidates": rows
    }, separators=(",", ":")) + "\n", encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/index_candidates.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("public"))
    parser.add_argument("--autoware-commit", default="")
    parser.add_argument("--universe-commit", default="")
    parser.add_argument("--launch-commit", default="")
    args = parser.parse_args(argv)
    metadata = build(args.csv, args.output_dir, {
        "autoware": args.autoware_commit,
        "universe": args.universe_commit,
        "launch": args.launch_commit,
    })
    print(f"Built {args.output_dir} with {metadata['total_packages']} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
