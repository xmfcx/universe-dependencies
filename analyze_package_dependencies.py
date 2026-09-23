#!/usr/bin/env python3
"""Find Autoware Universe packages with no consumers elsewhere in a workspace."""

import argparse
import csv
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


SKIP_DIRS = {
    ".git", ".claude", ".cache", ".tox", ".venv", "__pycache__",
    "build", "install", "log", "node_modules", "venv",
}
TEXT_SUFFIXES = {
    ".bash", ".c", ".cc", ".cfg", ".cmake", ".cpp", ".h", ".hpp",
    ".ini", ".j2", ".jinja", ".json", ".launch", ".md", ".py",
    ".repos", ".rst", ".rviz", ".sh", ".toml", ".txt", ".urdf",
    ".xacro", ".xml", ".yaml", ".yml",
}
ACTIVE_SUFFIXES = TEXT_SUFFIXES - {".md", ".rst", ".txt", ".repos"}
MAX_TEXT_BYTES = 2_000_000
GENERATED_REPORT_NAMES = {"candidates.json", "dependencies.json", "statistics.json"}


def iter_workspace_files(root):
    """Yield source files while skipping generated and ignored trees."""
    def on_error(error):
        raise OSError(f"Cannot inspect workspace directory: {error}") from error

    for current, dirs, files in os.walk(root, onerror=on_error):
        current = Path(current)
        dirs[:] = sorted(
            name for name in dirs
            if name not in SKIP_DIRS
            and not (current / name / "COLCON_IGNORE").exists()
            and not (current / name / "AMENT_IGNORE").exists()
        )
        for name in sorted(files):
            yield current / name


def discover_packages(workspace):
    """Read all active package manifests under the workspace source tree."""
    source = workspace / "src"
    if not source.is_dir():
        raise ValueError(f"Workspace source directory does not exist: {source}")
    packages, package_paths = {}, {}
    for path in iter_workspace_files(source):
        if path.name != "package.xml":
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as error:
            raise ValueError(f"Invalid manifest {path}: {error}") from error
        name = (root.findtext("name") or "").strip()
        if not name:
            raise ValueError(f"Manifest has no package name: {path}")
        if name in packages:
            raise ValueError(f"Duplicate package {name}: {packages[name]['path']} and {path}")
        dependencies = defaultdict(list)
        for element in root:
            if element.tag == "depend" or element.tag.endswith("_depend"):
                dependency = (element.text or "").strip()
                if dependency:
                    dependencies[dependency].append({
                        "type": element.tag, "condition": element.get("condition")
                    })
        packages[name] = {"path": path.parent, "dependencies": dict(dependencies)}
        package_paths[path.parent] = name
    return packages, package_paths


def build_graph(packages):
    """Return direct manifest edges and reverse adjacency."""
    direct = {name: set() for name in packages}
    dependents = {name: set() for name in packages}
    for name, info in packages.items():
        for dependency in info["dependencies"]:
            if dependency in packages and dependency != name:
                direct[name].add(dependency)
                dependents[dependency].add(name)
    return direct, dependents


def recursive_dependents(name, dependents):
    found, pending = set(), list(dependents[name])
    while pending:
        dependent = pending.pop()
        if dependent == name or dependent in found:
            continue
        found.add(dependent)
        pending.extend(dependents[dependent] - found)
    return found


def reference_kind(path, line, position):
    """Comments, URLs, and documentation are evidence for review only."""
    stripped = line.lstrip()
    before = line[:position]
    if (
        (path.suffix.lower() in {".md", ".rst", ".txt", ".repos"} and path.name != "CMakeLists.txt")
        or stripped.startswith(("#", "//", "<!--", "*"))
        or (path.suffix.lower() in {".yaml", ".yml", ".py", ".sh", ".bash", ".cmake"} and "#" in before)
        or (path.suffix.lower() in {".c", ".cc", ".cpp", ".h", ".hpp"} and "//" in before)
        or "github.com/" in line
    ):
        return "mention", "review"
    if path.suffix.lower() not in ACTIVE_SUFFIXES and path.name not in {"CMakeLists.txt", "Dockerfile"}:
        return "mention", "review"
    if ".launch." in path.name or path.suffix.lower() == ".launch":
        return "launch", "high"
    if re.search(r"(?:find-pkg-share|find-pkg-prefix|find\s+|FindPackageShare|get_package_share_directory|\bpkg\s*=|\bpackage\s*=)", line):
        return "runtime", "high"
    if path.suffix.lower() in {".yaml", ".yml", ".json", ".rviz", ".xml", ".urdf", ".xacro"}:
        return "config", "high"
    return "source", "high"


def scan_references(workspace, universe, universe_names, package_paths, direct):
    """Find external mentions and undeclared internal package references."""
    names = sorted(universe_names, key=lambda name: (-len(name), name))
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?:" + "|".join(map(re.escape, names)) + r")(?![A-Za-z0-9_])"
    )
    external, internal = defaultdict(list), defaultdict(list)
    for path in iter_workspace_files(workspace):
        if path.name == "package.xml" or path.name in GENERATED_REPORT_NAMES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"CMakeLists.txt", "Dockerfile"}:
            continue
        if path.stat().st_size > MAX_TEXT_BYTES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        owner = path.parent
        while owner != workspace and owner not in package_paths:
            owner = owner.parent
        owner_name = package_paths.get(owner)
        is_internal_file = path.is_relative_to(universe)
        if is_internal_file and owner_name not in universe_names:
            continue
        relative = path.relative_to(workspace).as_posix()
        for number, line in enumerate(content.splitlines(), 1):
            matches = {}
            for match in pattern.finditer(line):
                kind, confidence = reference_kind(path, line, match.start())
                name = match.group()
                if name not in matches or confidence == "high":
                    matches[name] = (kind, confidence)
            for name, (kind, confidence) in matches.items():
                if is_internal_file and (
                    confidence != "high" or name == owner_name or name in direct[owner_name]
                ):
                    continue
                target = internal if is_internal_file else external
                target[name].append({
                    "path": relative, "line": number, "package": owner_name,
                    "kind": kind, "confidence": confidence,
                    "excerpt": line.strip()[:200],
                })
    for group in (external, internal):
        for items in group.values():
            items.sort(key=lambda item: (item["path"], item["line"]))
    return external, internal


def analyze(workspace, universe_path):
    workspace = Path(workspace).resolve()
    universe = (workspace / universe_path).resolve()
    if not universe.is_relative_to(workspace) or not universe.is_dir():
        raise ValueError(f"Universe path must be a directory inside workspace: {universe}")
    packages, package_paths = discover_packages(workspace)
    universe_names = {name for name, info in packages.items() if info["path"].is_relative_to(universe)}
    if not universe_names:
        raise ValueError(f"No packages found under Universe path: {universe}")
    direct, dependents = build_graph(packages)
    external_references, internal_references = scan_references(
        workspace, universe, universe_names, package_paths, direct
    )
    # Preserve the manifest graph for exact edge exports. File references add
    # conservative edges only to candidate coupling/ranking.
    usage_direct = {name: set(targets) for name, targets in direct.items()}
    usage_dependents = {name: set(sources) for name, sources in dependents.items()}
    for target, refs in internal_references.items():
        for ref in refs:
            source = ref["package"]
            usage_direct[source].add(target)
            usage_dependents[target].add(source)
    candidates = []
    for name in sorted(universe_names):
        package = packages[name]
        internal_recursive = sorted(recursive_dependents(name, usage_dependents) & universe_names)
        external_manifest = [{
            "package": dependent,
            "path": packages[dependent]["path"].relative_to(workspace).as_posix(),
            "declarations": packages[dependent]["dependencies"][name],
        } for dependent in sorted(dependents[name] - universe_names)]
        external_references_for_name = external_references.get(name, [])
        if external_manifest or any(ref["confidence"] == "high" for ref in external_references_for_name):
            status = "blocked_external"
        elif external_references_for_name:
            status = "review"
        else:
            status = "eligible"
        candidates.append({
            "name": name,
            "path": package["path"].relative_to(workspace).as_posix(),
            "status": status, "rank": None,
            "internal_prerequisites": sorted(usage_direct[name] & universe_names),
            "internal_manifest_prerequisites": sorted(direct[name] & universe_names),
            "internal_reference_prerequisites": sorted((usage_direct[name] - direct[name]) & universe_names),
            "internal_direct_dependents": sorted(usage_dependents[name] & universe_names),
            "internal_manifest_dependents": sorted(dependents[name] & universe_names),
            "internal_recursive_dependents": internal_recursive,
            "internal_nonmanifest_references": internal_references.get(name, []),
            "external_manifest_dependents": external_manifest,
            "external_references": external_references_for_name,
        })
        candidates[-1]["standalone"] = (
            status == "eligible"
            and not candidates[-1]["internal_recursive_dependents"]
            and not candidates[-1]["internal_prerequisites"]
        )
    eligible = sorted(
        (item for item in candidates if item["status"] == "eligible"),
        key=lambda item: (len(item["internal_recursive_dependents"]),
                          len(item["internal_prerequisites"]), item["name"]),
    )
    for rank, item in enumerate(eligible, 1):
        item["rank"] = rank
    order = {"eligible": 0, "review": 1, "blocked_external": 2}
    candidates.sort(key=lambda item: (order[item["status"]], item["rank"] or 0, item["name"]))
    counts = {status: sum(item["status"] == status for item in candidates) for status in order}
    return {
        "workspace": workspace, "universe": universe, "packages": packages,
        "universe_names": universe_names, "direct": direct,
        "dependents": dependents, "candidates": candidates, "counts": counts,
        "standalone_eligible": sum(item["standalone"] for item in candidates),
    }


def export_json(result, output_dir):
    universe_names, dependents = result["universe_names"], result["dependents"]
    records = []
    for name in sorted(universe_names):
        recursive = sorted(recursive_dependents(name, dependents) & universe_names)
        records.append({
            "name": name, "dependent_count": len(recursive), "dependents": recursive,
            "direct_dependents": sorted(dependents[name] & universe_names),
            "direct_dependencies": sorted(result["direct"][name] & universe_names),
        })
    records.sort(key=lambda item: (-item["dependent_count"], item["name"]))
    direct_count = sum(len(result["direct"][name] & universe_names) for name in universe_names)
    (output_dir / "dependencies.json").write_text(json.dumps({
        "metadata": {"total_packages": len(universe_names),
                     "total_direct_dependency_edges": direct_count,
                     "dependent_count_semantics": "recursive Universe dependents"},
        "packages": records,
    }, indent=2) + "\n")
    (output_dir / "candidates.json").write_text(json.dumps({
        "metadata": {
            "workspace": str(result["workspace"]),
            "universe_path": result["universe"].relative_to(result["workspace"]).as_posix(),
            "workspace_packages": len(result["packages"]),
            "universe_packages": len(universe_names), "status_counts": result["counts"],
            "standalone_eligible": result["standalone_eligible"],
            "ranking": "Fewest recursive Universe dependents, then fewest direct Universe prerequisites, then name; includes strong undeclared internal file references",
        }, "candidates": result["candidates"],
    }, indent=2) + "\n")
    counts = sorted(len(recursive_dependents(name, dependents) & universe_names) for name in universe_names)
    n = len(counts)
    (output_dir / "statistics.json").write_text(json.dumps({
        "total_packages": n, "total_direct_dependency_edges": direct_count,
        "leaf_packages": sum(count == 0 for count in counts),
        "max_dependents": counts[-1], "min_dependents": counts[0],
        "mean_dependents": sum(counts) / n, "median_dependents": counts[n // 2],
        "percentile_90": counts[int(n * 0.9)], "percentile_95": counts[int(n * 0.95)],
        "percentile_99": counts[int(n * 0.99)], "candidate_status_counts": result["counts"],
        "standalone_eligible": result["standalone_eligible"],
    }, indent=2) + "\n")


def export_csv(result, output_dir):
    universe_names, dependents = result["universe_names"], result["dependents"]
    with (output_dir / "dependencies.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["Package", "Dependent_Count", "Dependents", "Direct_Dependent_Count", "Direct_Dependents"])
        ordered = sorted(universe_names, key=lambda name: (-len(recursive_dependents(name, dependents) & universe_names), name))
        for name in ordered:
            recursive = sorted(recursive_dependents(name, dependents) & universe_names)
            direct = sorted(dependents[name] & universe_names)
            writer.writerow([name, len(recursive), ";".join(recursive), len(direct), ";".join(direct)])
    with (output_dir / "candidates.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["Package", "Status", "Rank", "Standalone", "Internal_Dependent_Count", "Internal_Prerequisites", "Internal_Nonmanifest_References", "External_Manifest_Dependents", "External_High_References", "External_Review_References"])
        for item in result["candidates"]:
            references = item["external_references"]
            writer.writerow([
                item["name"], item["status"], item["rank"] or "", item["standalone"],
                len(item["internal_recursive_dependents"]),
                ";".join(item["internal_prerequisites"]),
                ";".join(f"{ref['path']}:{ref['line']}" for ref in item["internal_nonmanifest_references"]),
                ";".join(dep["package"] for dep in item["external_manifest_dependents"]),
                ";".join(f"{ref['path']}:{ref['line']}" for ref in references if ref["confidence"] == "high"),
                ";".join(f"{ref['path']}:{ref['line']}" for ref in references if ref["confidence"] == "review"),
            ])


def direct_edges(result):
    universe_names = result["universe_names"]
    for source in sorted(result["packages"]):
        for target in sorted(result["direct"][source] & universe_names):
            yield source, target, "internal" if source in universe_names else "external"


def export_edges(result, output_dir):
    with (output_dir / "dependency_edges.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["Source", "Target", "Relationship", "Scope"])
        for source, target, scope in direct_edges(result):
            writer.writerow([source, target, "depends_on", scope])


def export_graphml(result, output_dir):
    edges = list(direct_edges(result))
    names = result["universe_names"] | {source for source, _, _ in edges}
    namespace = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace("", namespace)
    graphml = ET.Element(f"{{{namespace}}}graphml")
    ET.SubElement(graphml, f"{{{namespace}}}key", {
        "id": "scope", "for": "node", "attr.name": "scope", "attr.type": "string"
    })
    graph = ET.SubElement(graphml, f"{{{namespace}}}graph", {"id": "G", "edgedefault": "directed"})
    for name in sorted(names):
        node = ET.SubElement(graph, f"{{{namespace}}}node", {"id": name})
        ET.SubElement(node, f"{{{namespace}}}data", {"key": "scope"}).text = (
            "universe" if name in result["universe_names"] else "external"
        )
    for number, (source, target, _) in enumerate(edges):
        ET.SubElement(graph, f"{{{namespace}}}edge", {
            "id": f"e{number}", "source": source, "target": target
        })
    ET.indent(graphml, space="  ")
    ET.ElementTree(graphml).write(output_dir / "dependencies.graphml", encoding="utf-8", xml_declaration=True)


def print_report(result):
    counts = result["counts"]
    print(f"Workspace packages: {len(result['packages'])}; Universe packages: {len(result['universe_names'])}")
    print(f"Candidates: {counts['eligible']} eligible, {counts['review']} review, {counts['blocked_external']} blocked by external usage")
    print(f"Standalone eligible packages: {result['standalone_eligible']}")
    print("\nMost promising packages (fewest Universe dependents and prerequisites):")
    print(f"{'Rank':>4}  {'Package':<58} {'Dependents':>10} {'Prereqs':>7}")
    for item in (item for item in result["candidates"] if item["status"] == "eligible"):
        if item["rank"] > 30:
            break
        print(f"{item['rank']:>4}  {item['name']:<58} {len(item['internal_recursive_dependents']):>10} {len(item['internal_prerequisites']):>7}")
    print("\nExternal use and supporting paths appear in candidates.json/csv when exported.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Autoware workspace root (default: current directory)")
    parser.add_argument("--universe-path", type=Path, default=Path("src/universe/autoware_universe"), help="Universe path relative to workspace")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("."), help="Export directory (default: current directory)")
    parser.add_argument("-j", "--jobs", type=int, help="Accepted for compatibility; analysis runs in one process")
    parser.add_argument("--export-json", action="store_true", help="Write dependencies, statistics and candidates JSON")
    parser.add_argument("--export-csv", action="store_true", help="Write dependencies and candidates CSV")
    parser.add_argument("--export-edges", action="store_true", help="Write direct dependency edges CSV")
    parser.add_argument("--export-graphml", action="store_true", help="Write direct dependency graph GraphML")
    parser.add_argument("--export-all", action="store_true", help="Write all formats")
    parser.add_argument("--no-interactive", action="store_true", help="Accepted for compatibility; analysis is non-interactive")
    args = parser.parse_args(argv)
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        result = analyze(args.workspace, args.universe_path)
        print_report(result)
        if args.export_all or args.export_json or args.export_csv or args.export_edges or args.export_graphml:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.export_all or args.export_json:
            export_json(result, args.output_dir)
        if args.export_all or args.export_csv:
            export_csv(result, args.output_dir)
        if args.export_all or args.export_edges:
            export_edges(result, args.output_dir)
        if args.export_all or args.export_graphml:
            export_graphml(result, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
