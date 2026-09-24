import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_package_dependencies as analyzer
import generate_index_candidates as generator


class GeneratorFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.universe = self.workspace / generator.DEFAULT_UNIVERSE
        self.launch = self.workspace / generator.DEFAULT_LAUNCH

    def package(self, path, name, dependencies=()):
        path.mkdir(parents=True)
        (path / "package.xml").write_text(
            "<package><name>" + name + "</name>" + "".join(
                f"<exec_depend>{dependency}</exec_depend>" for dependency in dependencies
            ) + "</package>"
        )

    def test_launch_closure_and_ranked_csv(self):
        for name, deps in (
            ("free", ()), ("leaf_with_prereq", ("free",)),
            ("manifest_target", ("nested_target",)), ("nested_target", ()),
            ("direct_target", ()), ("runtime_target", ()),
            ("unowned_target", ()), ("review_target", ()),
        ):
            self.package(self.universe / name, name, deps)
        self.package(self.launch / "launch", "launch", ("intermediate",))
        self.package(self.workspace / "src/other/intermediate", "intermediate", ("manifest_target",))
        (self.workspace / "src/other/intermediate/config.yaml").write_text(
            "package: runtime_target\n"
        )
        (self.launch / "launch/config.yaml").write_text("package: direct_target\n")
        (self.launch / "launch/comments.yaml").write_text("# package: review_target\n")
        (self.launch / "extra.yaml").write_text("package: unowned_target\n")
        (self.workspace / "README.md").write_text("review_target is mentioned here.\n")

        result = analyzer.analyze(self.workspace, generator.DEFAULT_UNIVERSE)
        paths = generator.launch_paths(result, generator.DEFAULT_LAUNCH)
        records = generator.rank_candidates(result, paths)
        by_name = {row["Package"]: row for row in records}

        self.assertEqual(paths["manifest_target"], "launch -> intermediate -> manifest_target")
        self.assertEqual(paths["nested_target"], "launch -> intermediate -> manifest_target -> nested_target")
        self.assertEqual(paths["runtime_target"], "launch -> intermediate -> runtime_target")
        self.assertEqual(paths["direct_target"], "launch -> direct_target")
        self.assertIn("extra.yaml:1 -> unowned_target", paths["unowned_target"])
        self.assertNotIn("free", paths)
        self.assertNotIn("review_target", paths)
        self.assertEqual(by_name["free"]["Score"], 92)  # One internal dependent.
        self.assertEqual(by_name["leaf_with_prereq"]["Score"], 97)
        self.assertEqual(by_name["review_target"]["Tier"], "review")
        self.assertEqual(by_name["manifest_target"]["Tier"], "used_by_launch")
        self.assertLess(by_name["leaf_with_prereq"]["Rank"], by_name["free"]["Rank"])
        self.assertLess(by_name["free"]["Rank"], by_name["review_target"]["Rank"])
        self.assertLess(by_name["review_target"]["Rank"], by_name["manifest_target"]["Rank"])

        output = self.workspace / "index_candidates.csv"
        self.assertEqual(generator.main([
            "--workspace", str(self.workspace), "--output", str(output)
        ]), 0)
        with output.open(newline="") as file:
            exported = list(csv.DictReader(file))
        self.assertEqual(len(exported), 8)
        self.assertEqual(exported[0]["Rank"], "1")
        self.assertEqual(exported[-1]["Tier"], "used_by_launch")


if __name__ == "__main__":
    unittest.main()
