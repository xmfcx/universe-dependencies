import csv
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_package_dependencies as analyzer


class AnalysisFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.universe = self.workspace / "src/universe/autoware_universe"
        self.launch = self.workspace / "src/launcher/autoware_launch"

    def package(self, directory, name, dependencies=()):
        directory.mkdir(parents=True, exist_ok=True)
        lines = ["<package>", f"<name>{name}</name>"]
        for dependency in dependencies:
            lines.append(f"<exec_depend>{dependency}</exec_depend>")
        lines.append("</package>")
        (directory / "package.xml").write_text("\n".join(lines))

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_workspace_usage_and_direct_edges(self):
        for name, dependencies in (
            ("base", ()), ("middle", ("base",)), ("top", ("middle",)),
            ("manifest_blocked", ()), ("launch_blocked", ()),
            ("python_blocked", ()), ("config_blocked", ()), ("cmake_blocked", ()),
            ("review_only", ()), ("free", ()),
        ):
            self.package(self.universe / name, name, dependencies)
        self.package(self.launch / "launcher", "launcher", ("manifest_blocked",))
        self.write(self.launch / "launcher/launch/run.launch.xml", '<node pkg="launch_blocked" exec="node"/>\n')
        self.write(self.launch / "launcher/launch/run.launch.py", 'Node(package="python_blocked", executable="node")\n')
        self.write(self.launch / "launcher/config/nodes.yaml", 'package: config_blocked\n')
        self.write(self.launch / "launcher/CMakeLists.txt", 'find_package(cmake_blocked REQUIRED)\n')
        self.write(self.workspace / "README.md", "Mention review_only for context.\n")
        self.write(self.workspace / "scripts/example.py", 'name = "free_extra"\n')
        result = analyzer.analyze(self.workspace, Path("src/universe/autoware_universe"))
        candidates = {item["name"]: item for item in result["candidates"]}

        for name in ("manifest_blocked", "launch_blocked", "python_blocked", "config_blocked", "cmake_blocked"):
            self.assertEqual(candidates[name]["status"], "blocked_external")
        self.assertEqual(candidates["review_only"]["status"], "review")
        self.assertEqual(candidates["free"]["status"], "eligible")
        self.assertEqual(candidates["base"]["internal_direct_dependents"], ["middle"])
        self.assertEqual(candidates["base"]["internal_recursive_dependents"], ["middle", "top"])
        self.assertEqual(candidates["top"]["internal_prerequisites"], ["middle"])
        self.assertEqual(candidates["manifest_blocked"]["external_manifest_dependents"][0]["package"], "launcher")
        self.assertEqual(candidates["launch_blocked"]["external_references"][0]["line"], 1)
        self.assertEqual(candidates["launch_blocked"]["external_references"][0]["confidence"], "high")

        output = self.workspace / "exports"
        output.mkdir()
        analyzer.export_json(result, output)
        analyzer.export_csv(result, output)
        analyzer.export_edges(result, output)
        analyzer.export_graphml(result, output)
        with (output / "dependency_edges.csv").open(newline="") as file:
            edges = {(row["Source"], row["Target"], row["Scope"]) for row in csv.DictReader(file)}
        self.assertIn(("top", "middle", "internal"), edges)
        self.assertIn(("middle", "base", "internal"), edges)
        self.assertIn(("launcher", "manifest_blocked", "external"), edges)
        self.assertNotIn(("top", "base", "internal"), edges)
        graph = ET.parse(output / "dependencies.graphml").getroot()
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        graph_edges = {(node.get("source"), node.get("target")) for node in graph.findall(".//g:edge", ns)}
        self.assertIn(("middle", "base"), graph_edges)
        self.assertNotIn(("top", "base"), graph_edges)
        report = json.loads((output / "candidates.json").read_text())
        self.assertEqual(report["metadata"]["workspace_packages"], 11)
        self.assertEqual(report["metadata"]["status_counts"]["blocked_external"], 5)

    def test_ignored_tree_does_not_create_package_or_usage(self):
        self.package(self.universe / "free", "free")
        ignored = self.workspace / "src/unused"
        self.package(ignored / "hidden", "hidden", ("free",))
        (ignored / "COLCON_IGNORE").write_text("")
        self.write(self.workspace / "build/generated.launch.xml", '<node pkg="free"/>\n')
        self.write(self.workspace / "results/candidates.json", '{"package": "free"}\n')
        result = analyzer.analyze(self.workspace, Path("src/universe/autoware_universe"))
        self.assertEqual(len(result["packages"]), 1)
        self.assertEqual(result["candidates"][0]["status"], "eligible")

    def test_internal_launch_reference_changes_standalone_ranking(self):
        self.package(self.universe / "target", "target")
        self.package(self.universe / "consumer", "consumer")
        self.package(self.universe / "free", "free")
        self.write(self.universe / "consumer/launch/run.launch.xml", '<node pkg="target" exec="node"/>\n')
        self.write(self.universe / "target/config/self.yaml", 'package: target\n')
        self.write(self.universe / "consumer/config/comments.yaml", '# package: free\n')
        result = analyzer.analyze(self.workspace, Path("src/universe/autoware_universe"))
        candidates = {item["name"]: item for item in result["candidates"]}

        self.assertEqual(candidates["target"]["status"], "eligible")
        self.assertFalse(candidates["target"]["standalone"])
        self.assertEqual(candidates["target"]["internal_direct_dependents"], ["consumer"])
        self.assertEqual(candidates["target"]["internal_manifest_dependents"], [])
        self.assertEqual(candidates["target"]["internal_nonmanifest_references"][0]["package"], "consumer")
        self.assertEqual(candidates["consumer"]["internal_reference_prerequisites"], ["target"])
        self.assertTrue(candidates["free"]["standalone"])
        self.assertLess(candidates["free"]["rank"], candidates["target"]["rank"])
        self.assertNotIn(("consumer", "target", "internal"), set(analyzer.direct_edges(result)))


if __name__ == "__main__":
    unittest.main()
