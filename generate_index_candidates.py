#!/usr/bin/env python3
"""Rank Universe packages for potential registration in Autoware Index."""

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import analyze_package_dependencies as analyzer


DEFAULT_UNIVERSE = Path("src/universe/autoware_universe")
DEFAULT_LAUNCH = Path("src/launcher/autoware_launch")


def launch_paths(result, launch_path):
    """Return a shortest known dependency/reference path from launch to each package."""
    workspace = result["workspace"]
    launch = (workspace / launch_path).resolve()
    if not launch.is_relative_to(workspace) or not launch.is_dir():
        raise ValueError(f"Launch path must be a directory inside workspace: {launch}")

    packages = result["packages"]
    roots = sorted(name for name, info in packages.items() if info["path"].is_relative_to(launch))
    if not roots:
        raise ValueError(f"No packages found under launch path: {launch}")

    direct = result["direct"]
    package_paths = {info["path"]: name for name, info in packages.items()}
    external, internal = analyzer.scan_references(
        workspace, result["universe"], set(packages), package_paths, direct
    )
    graph = {name: set(dependencies) for name, dependencies in direct.items()}
    unowned_launch_refs = []
    for group in (external, internal):
        for target, refs in group.items():
            for ref in refs:
                if ref["confidence"] != "high":
                    continue
                source = ref["package"]
                if source in graph and source != target:
                    graph[source].add(target)
                elif source is None and (workspace / ref["path"]).is_relative_to(launch):
                    unowned_launch_refs.append((target, f"{ref['path']}:{ref['line']}"))

    paths = {name: name for name in roots}
    pending = deque(roots)
    for target, source in sorted(unowned_launch_refs):
        if target not in paths:
            paths[target] = f"{source} -> {target}"
            pending.append(target)
    while pending:
        source = pending.popleft()
        for target in sorted(graph[source]):
            if target not in paths:
                paths[target] = f"{paths[source]} -> {target}"
                pending.append(target)
    return paths


def rank_candidates(result, paths):
    """Give each package a score in a separate band for each eligibility tier."""
    bases = {"ready": 100, "review": 70, "used_elsewhere": 40, "used_by_launch": 10}
    records = []
    for item in result["candidates"]:
        name = item["name"]
        if name in paths:
            tier = "used_by_launch"
        elif item["status"] == "eligible":
            tier = "ready"
        elif item["status"] == "review":
            tier = "review"
        else:
            tier = "used_elsewhere"
        dependents = len(item["internal_recursive_dependents"])
        prerequisites = len(item["internal_prerequisites"])
        penalty = 8 * dependents + 3 * prerequisites
        score = bases[tier] - min(9 if tier == "used_by_launch" else 29, penalty)
        records.append({
            "Package": name,
            "Score": score,
            "Tier": tier,
            "Launch_Reachable": name in paths,
            "Launch_Path": paths.get(name, ""),
            "Standalone": item["standalone"],
            "Internal_Recursive_Dependent_Count": dependents,
            "Internal_Direct_Dependent_Count": len(item["internal_direct_dependents"]),
            "Internal_Prerequisite_Count": prerequisites,
            "Internal_Prerequisites": ";".join(item["internal_prerequisites"]),
            "External_Manifest_Dependent_Count": len(item["external_manifest_dependents"]),
            "External_High_Reference_Count": sum(
                ref["confidence"] == "high" for ref in item["external_references"]
            ),
            "External_Review_Reference_Count": sum(
                ref["confidence"] == "review" for ref in item["external_references"]
            ),
            "Source_Path": item["path"],
        })
    records.sort(key=lambda row: (
        -row["Score"], row["Internal_Recursive_Dependent_Count"],
        row["Internal_Prerequisite_Count"], row["Package"],
    ))
    for rank, record in enumerate(records, 1):
        record["Rank"] = rank
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Autoware workspace root")
    parser.add_argument("--universe-path", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--launch-path", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--output", type=Path, default=Path("results/index_candidates.csv"))
    args = parser.parse_args(argv)
    try:
        result = analyzer.analyze(args.workspace, args.universe_path)
        paths = launch_paths(result, args.launch_path)
        records = rank_candidates(result, paths)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["Rank", *(key for key in records[0] if key != "Rank")],
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote {len(records)} ranked packages to {args.output}")
    print(f"Ready: {sum(row['Tier'] == 'ready' for row in records)}; "
          f"reachable from launch: {sum(row['Launch_Reachable'] for row in records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
