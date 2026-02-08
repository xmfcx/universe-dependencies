#!/usr/bin/env python3
"""
Analyze colcon package dependencies.
For each package, count how many packages depend on it and list packages topologically from leaf to root.
"""

import subprocess
import sys
import json
import csv
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path


def get_all_packages() -> List[str]:
    """Get list of all package names in the workspace."""
    result = subprocess.run(
        ['colcon', 'list', '--base-paths', 'src/universe/autoware_universe', '--names-only'],
        capture_output=True,
        text=True,
        check=True
    )
    return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]


def get_packages_above(package_name: str) -> Tuple[str, List[str]]:
    """Get all packages that depend on the given package."""
    try:
        result = subprocess.run(
            ['colcon', 'list', '--base-paths', 'src/universe/autoware_universe', '--packages-above', package_name, '--names-only'],
            capture_output=True,
            text=True,
            check=True
        )
        packages = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        # Remove the package itself from the list
        dependents = [pkg for pkg in packages if pkg != package_name]
        return (package_name, dependents)
    except subprocess.CalledProcessError:
        return (package_name, [])


def build_dependency_graph(packages: List[str], num_workers: int = None) -> Dict[str, Set[str]]:
    """Build a dependency graph showing which packages depend on each package."""
    if num_workers is None:
        num_workers = cpu_count()

    print(f"Building dependency graph using {num_workers} workers...")
    dependency_graph = defaultdict(set)

    total = len(packages)
    completed = 0

    # Use multiprocessing pool to process packages in parallel
    with Pool(processes=num_workers) as pool:
        # Process packages in chunks to show progress
        chunk_size = max(1, total // (num_workers * 4))

        for package_name, dependents in pool.imap_unordered(
                get_packages_above,
                packages,
                chunksize=chunk_size
        ):
            dependency_graph[package_name] = set(dependents)
            completed += 1
            print(f"  Progress: {completed}/{total} packages analyzed ({100 * completed // total}%)", end='\r')

    print("\n")
    return dependency_graph


def topological_sort(packages: List[str], dependency_graph: Dict[str, Set[str]]) -> List[str]:
    """
    Sort packages topologically from leaf to root.
    Leaf packages (those with no dependents) come first.
    Root packages (those with many dependents) come last.
    """
    # Count dependents for each package
    dependent_counts = {pkg: len(dependency_graph.get(pkg, set())) for pkg in packages}

    # Sort by dependent count (ascending) - leaves first, roots last
    sorted_packages = sorted(packages, key=lambda p: (dependent_counts[p], p))

    return sorted_packages


def export_to_json(packages: List[str], dependency_graph: Dict[str, Set[str]], output_file: str):
    """Export dependency data to JSON format."""
    data = {
        "metadata": {
            "total_packages": len(packages),
            "total_dependencies": sum(len(deps) for deps in dependency_graph.values())
        },
        "packages": []
    }

    for package in packages:
        dependents = sorted(dependency_graph[package])
        data["packages"].append({
            "name": package,
            "dependent_count": len(dependents),
            "dependents": dependents
        })

    # Sort by dependent count for easier reading
    data["packages"].sort(key=lambda x: x["dependent_count"], reverse=True)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"JSON export saved to: {output_file}")


def export_to_csv(packages: List[str], dependency_graph: Dict[str, Set[str]], output_file: str):
    """Export dependency summary to CSV format."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Package', 'Dependent_Count', 'Dependents'])

        # Sort by dependent count (descending)
        sorted_packages = sorted(packages, key=lambda p: len(dependency_graph[p]), reverse=True)

        for package in sorted_packages:
            dependents = sorted(dependency_graph[package])
            dependents_str = ';'.join(dependents) if dependents else ''
            writer.writerow([package, len(dependents), dependents_str])

    print(f"CSV export saved to: {output_file}")


def export_to_edges_csv(packages: List[str], dependency_graph: Dict[str, Set[str]], output_file: str):
    """Export dependency edges to CSV format for graph visualization."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Source', 'Target', 'Relationship'])

        for package in packages:
            for dependent in sorted(dependency_graph[package]):
                # dependent depends on package
                writer.writerow([dependent, package, 'depends_on'])

    print(f"Edges CSV export saved to: {output_file}")


def export_to_graphml(packages: List[str], dependency_graph: Dict[str, Set[str]], output_file: str):
    """Export dependency graph to GraphML format for visualization tools like Gephi, Cytoscape."""
    with open(output_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n')
        f.write('    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
        f.write('    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n')
        f.write('    http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n')

        # Define attributes
        f.write('  <key id="label" for="node" attr.name="label" attr.type="string"/>\n')
        f.write('  <key id="dependent_count" for="node" attr.name="dependent_count" attr.type="int"/>\n')

        f.write('  <graph id="G" edgedefault="directed">\n')

        # Write nodes
        for package in packages:
            pkg_id = package.replace('-', '_').replace('.', '_')
            f.write(f'    <node id="{pkg_id}">\n')
            f.write(f'      <data key="label">{package}</data>\n')
            f.write(f'      <data key="dependent_count">{len(dependency_graph[package])}</data>\n')
            f.write(f'    </node>\n')

        # Write edges
        edge_id = 0
        for package in packages:
            pkg_id = package.replace('-', '_').replace('.', '_')
            for dependent in dependency_graph[package]:
                dep_id = dependent.replace('-', '_').replace('.', '_')
                # dependent depends on package
                f.write(f'    <edge id="e{edge_id}" source="{dep_id}" target="{pkg_id}"/>\n')
                edge_id += 1

        f.write('  </graph>\n')
        f.write('</graphml>\n')

    print(f"GraphML export saved to: {output_file}")


def export_statistics(packages: List[str], dependency_graph: Dict[str, Set[str]], output_file: str):
    """Export statistical summary to JSON."""
    dependent_counts = [len(dependency_graph[pkg]) for pkg in packages]

    # Calculate percentiles
    sorted_counts = sorted(dependent_counts)
    n = len(sorted_counts)

    stats = {
        "total_packages": len(packages),
        "total_dependency_edges": sum(dependent_counts),
        "leaf_packages": sum(1 for c in dependent_counts if c == 0),
        "max_dependents": max(dependent_counts),
        "min_dependents": min(dependent_counts),
        "mean_dependents": sum(dependent_counts) / len(dependent_counts),
        "median_dependents": sorted_counts[n // 2],
        "percentile_90": sorted_counts[int(n * 0.9)],
        "percentile_95": sorted_counts[int(n * 0.95)],
        "percentile_99": sorted_counts[int(n * 0.99)],
        "top_10_packages": []
    }

    # Top 10 most depended-upon packages
    top_packages = sorted(packages, key=lambda p: len(dependency_graph[p]), reverse=True)[:10]
    for package in top_packages:
        stats["top_10_packages"].append({
            "name": package,
            "dependent_count": len(dependency_graph[package])
        })

    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics export saved to: {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Analyze package dependencies in Autoware workspace'
    )
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=cpu_count(),
        help=f'Number of parallel jobs (default: {cpu_count()})'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='.',
        help='Output directory for export files (default: current directory)'
    )
    parser.add_argument(
        '--export-json',
        action='store_true',
        help='Export data to JSON format'
    )
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Export data to CSV format'
    )
    parser.add_argument(
        '--export-edges',
        action='store_true',
        help='Export dependency edges to CSV format'
    )
    parser.add_argument(
        '--export-graphml',
        action='store_true',
        help='Export graph to GraphML format (for Gephi, Cytoscape, etc.)'
    )
    parser.add_argument(
        '--export-all',
        action='store_true',
        help='Export to all formats'
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Skip interactive prompts'
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Analyzing package dependencies in Autoware workspace...\n")

    # Get all packages
    print("Discovering packages...")
    packages = get_all_packages()
    print(f"Found {len(packages)} packages\n")

    # Build dependency graph
    dependency_graph = build_dependency_graph(packages, num_workers=args.jobs)

    # Sort topologically
    sorted_packages = topological_sort(packages, dependency_graph)

    # Print results
    print("=" * 80)
    print("PACKAGE DEPENDENCY ANALYSIS (Leaf to Root)")
    print("=" * 80)
    print(f"{'Package Name':<60} {'Dependent Count':>15}")
    print("-" * 80)

    for package in sorted_packages:
        dependent_count = len(dependency_graph[package])
        print(f"{package:<60} {dependent_count:>15}")

    print("=" * 80)

    # Print statistics
    print("\nSTATISTICS:")
    dependent_counts = [len(dependency_graph[pkg]) for pkg in packages]
    print(f"  Total packages: {len(packages)}")
    print(f"  Total dependency edges: {sum(dependent_counts)}")
    print(f"  Leaf packages (0 dependents): {sum(1 for c in dependent_counts if c == 0)}")
    print(f"  Max dependents: {max(dependent_counts)}")
    print(f"  Average dependents: {sum(dependent_counts) / len(dependent_counts):.2f}")

    # Show top 10 most depended-upon packages
    print("\nTOP 10 MOST DEPENDED-UPON PACKAGES:")
    top_packages = sorted(
        packages,
        key=lambda p: len(dependency_graph[p]),
        reverse=True
    )[:10]

    for i, package in enumerate(top_packages, 1):
        count = len(dependency_graph[package])
        print(f"  {i:2}. {package:<55} ({count} dependents)")

    # Handle exports
    print("\n" + "=" * 80)

    if args.export_all or args.export_json:
        export_to_json(packages, dependency_graph, output_dir / "dependencies.json")
        export_statistics(packages, dependency_graph, output_dir / "statistics.json")

    if args.export_all or args.export_csv:
        export_to_csv(packages, dependency_graph, output_dir / "dependencies.csv")

    if args.export_all or args.export_edges:
        export_to_edges_csv(packages, dependency_graph, output_dir / "dependency_edges.csv")

    if args.export_all or args.export_graphml:
        export_to_graphml(packages, dependency_graph, output_dir / "dependencies.graphml")

    # Interactive export if no export flags were provided
    if not (
            args.export_all or args.export_json or args.export_csv or args.export_edges or args.export_graphml) and not args.no_interactive:
        response = input("\nExport data? (all/json/csv/edges/graphml/none): ").strip().lower()

        if response == 'all':
            export_to_json(packages, dependency_graph, output_dir / "dependencies.json")
            export_statistics(packages, dependency_graph, output_dir / "statistics.json")
            export_to_csv(packages, dependency_graph, output_dir / "dependencies.csv")
            export_to_edges_csv(packages, dependency_graph, output_dir / "dependency_edges.csv")
            export_to_graphml(packages, dependency_graph, output_dir / "dependencies.graphml")
        elif response == 'json':
            export_to_json(packages, dependency_graph, output_dir / "dependencies.json")
            export_statistics(packages, dependency_graph, output_dir / "statistics.json")
        elif response == 'csv':
            export_to_csv(packages, dependency_graph, output_dir / "dependencies.csv")
        elif response == 'edges':
            export_to_edges_csv(packages, dependency_graph, output_dir / "dependency_edges.csv")
        elif response == 'graphml':
            export_to_graphml(packages, dependency_graph, output_dir / "dependencies.graphml")
        elif response != 'none':
            print("Invalid choice. No export performed.")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
