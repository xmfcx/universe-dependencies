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

## Index candidate ranking

Generate a ranked CSV for Autoware Index from a checked-out workspace:

```bash
python3 generate_index_candidates.py \
  --workspace /path/to/autoware \
  --output results/index_candidates.csv
```

The generator re-runs the workspace analysis, then starts at every package
under `src/launcher/autoware_launch`. It follows manifest dependencies and
strong literal package references in active files through all discovered
workspace packages. It also includes strong references in unowned files under
the launch tree. `Launch_Path` shows one shortest known path for each reachable
Universe package. Use `--launch-path` and `--universe-path` if the checkout has
different locations. Active YAML under the launch tree, including `.github`
configuration, counts as a reference.

The CSV contains every Universe package, ranked best to worst. `ready` means
no external manifest or active-file usage was found and the package is outside
the launch closure. `review` means only lower-confidence external mentions
were found. `used_elsewhere` means active use outside launch was found.
`used_by_launch` is last regardless of its other references. Each tier gets
its own score range: 71–100, 41–70, 11–40, and 1–10 respectively. Within a
tier, the score loses 8 points per recursive Universe dependent and 3 per
direct Universe prerequisite, capped at the tier's lower bound. Ties use
fewer dependents, fewer prerequisites, then package name. A `ready` package
with no known users or Universe prerequisites scores 100. The score is a
triage aid, not a build or registration guarantee. Dynamic package names,
packages outside the workspace, and skipped files can leave references
undetected.

## Daily report and Pages site

The [daily workflow](.github/workflows/daily-candidates.yml) runs at 03:17 UTC,
on pushes to `main`, and on manual dispatch. It checks out the Autoware meta
repository, imports `autoware.repos` with the overrides in
`autoware-nightly.repos`, runs the candidate generator, and builds a static
browser. The workflow writes `candidates.csv` and `metadata.json` to the
separate `data` branch, then deploys the same snapshot to GitHub Pages. The
metadata records the run time and the exact Autoware, Universe, and launch
commits used. No ROS build is needed.

For the first deployment, set **Settings → Pages → Build and deployment →
Source** to **GitHub Actions**. The workflow creates the `data` branch on its
first successful run. It needs the repository's `GITHUB_TOKEN` to have
`contents: write` for that branch and Pages deployment enabled for the
repository. Scheduled workflows can be disabled by GitHub after 60 days of
repository inactivity; run the workflow manually or re-enable its schedule if
that happens.

To preview the site locally after generating the CSV:

```bash
python3 build_site.py --csv results/index_candidates.csv --output-dir public
python3 -m http.server --directory public 8000
```

Open `http://localhost:8000/`. The published site offers status filters,
package search, launch paths, source revision links, and a CSV download.

The [earlier analysis document](https://drive.google.com/file/d/1kSYKON-CYDl9wPkpwuQOxSXPhbQegUiK/view?usp=sharing)
is retained for comparison.
