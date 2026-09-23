# Autoware Universe dependency analysis

Analyze an Autoware workspace to find Universe packages that have no known users
outside `autoware_universe`. The script reads `package.xml` throughout the
workspace and scans launch, configuration, and source files for literal package
names. It does not require colcon or a built workspace.

```bash
python3 analyze_package_dependencies.py \
  --workspace /path/to/autoware \
  --universe-path src/universe/autoware_universe \
  --output-dir results \
  --export-all
```

The default workspace is the current directory, and the default Universe path
is `src/universe/autoware_universe`. With no export flag the script prints a
ranked summary. `--export-json`, `--export-csv`, `--export-edges`, and
`--export-graphml` select individual formats. `--export-all` writes every
format. The old `-j/--jobs` and `--no-interactive` arguments remain accepted for
existing commands; the analysis is now single-process and non-interactive.

## Candidate report

`candidates.json` and `candidates.csv` classify each Universe package:

- `blocked_external`: another workspace package declares a manifest dependency,
  or a launch/configuration/source file outside Universe contains a non-comment
  package-name reference.
- `review`: only comments, documentation, or other lower-confidence mentions
  occur outside Universe.
- `eligible`: no external manifest or literal file reference was found.

Each result includes external package names, paths and line numbers, direct
internal prerequisites, and direct and recursive internal dependents. Strong
references from another Universe package's launch/configuration/source files
are recorded as `internal_nonmanifest_references` when its manifest omits the
dependency. They contribute to internal prerequisites and dependents used in
ranking, but do not change external eligibility. An eligible package is marked
`standalone` when it has no internal prerequisites or dependents from either
manifests or these file references. Eligible packages are ranked first by fewer
recursive internal dependents, then by fewer direct internal prerequisites.
This is a triage list, not proof that a package
is safe to move: names built dynamically, external repositories, and runtime
configuration outside the workspace cannot be detected by literal scanning.
Conditional manifest dependencies are included conservatively and retain their
conditions in JSON.

`dependencies.json` and `dependencies.csv` retain the recursive Universe
dependent counts from the earlier report and add direct dependents. By
contrast, `dependency_edges.csv` and `dependencies.graphml` contain **direct**
manifest edges only. An edge points from the dependent package to its
dependency; edges from outside Universe into Universe are included and marked
`external` in CSV.

The scanner ignores generated `build`, `install`, and `log` directories, common
cache directories, and trees with `COLCON_IGNORE` or `AMENT_IGNORE`. Manifests
are discovered under `workspace/src`; file references are scanned throughout
the workspace outside the Universe tree. Large or non-UTF-8 files and
unrecognized file types are not inspected.

Run the fixture tests with:

```bash
python3 -m unittest discover -s tests -v
```

The [earlier analysis document](https://drive.google.com/file/d/1kSYKON-CYDl9wPkpwuQOxSXPhbQegUiK/view?usp=sharing)
is retained for comparison.
