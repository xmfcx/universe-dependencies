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
        self.assertEqual(by_name["leaf_with_prereq"]["Score"], 80)
        self.assertEqual(by_name["review_target"]["Tier"], "review")
        self.assertEqual(by_name["manifest_target"]["Tier"], "used_by_launch")
        self.assertLess(by_name["free"]["Rank"], by_name["leaf_with_prereq"]["Rank"])
        self.assertLess(by_name["leaf_with_prereq"]["Rank"], by_name["review_target"]["Rank"])
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

    def test_other_non_universe_dependencies_do_not_lower_score(self):
        self.package(self.workspace / "src/core/core_helper", "core_helper")
        self.package(self.universe / "non_universe_deps", "non_universe_deps", ("core_helper", "rosdep_key"))
        self.package(self.universe / "universe_base", "universe_base")
        self.package(self.universe / "universe_consumer", "universe_consumer", ("universe_base",))

        result = analyzer.analyze(self.workspace, generator.DEFAULT_UNIVERSE)
        by_name = {row["Package"]: row for row in generator.rank_candidates(result, {})}

        self.assertEqual(by_name["non_universe_deps"]["Score"], 100)
        self.assertEqual(by_name["non_universe_deps"]["Internal_Prerequisite_Count"], 0)
        self.assertEqual(by_name["universe_consumer"]["Score"], 80)
        self.assertEqual(by_name["universe_consumer"]["Internal_Prerequisites"], "universe_base")
        self.assertLess(by_name["non_universe_deps"]["Rank"], by_name["universe_consumer"]["Rank"])
        self.assertTrue(all(
            row["Internal_Prerequisite_Count"] == 0
            for row in by_name.values() if row["Score"] > 80
        ))

    def test_tier4_repository_dependencies_receive_one_penalty(self):
        tier4_root = self.workspace / generator.DEFAULT_TIER4_MSGS
        for name in ("tier4_planning_msgs", "tier4_debug_msgs", "tier4_auto_msgs_converter"):
            self.package(tier4_root / name, name)
        self.package(self.workspace / "src/universe/external/tier4_autoware_msgs_extra",
                     "tier4_unrelated_msgs")
        self.package(self.workspace / "src/core/core_helper", "core_helper", ("tier4_debug_msgs",))
        for name, dependencies in (
            ("free", ()),
            ("single", ("tier4_planning_msgs",)),
            ("multiple", ("tier4_planning_msgs", "tier4_debug_msgs", "tier4_planning_msgs")),
            ("converter", ("tier4_auto_msgs_converter",)),
            ("indirect", ("core_helper",)),
            ("unrelated", ("tier4_unrelated_msgs", "rosdep_key")),
        ):
            self.package(self.universe / name, name, dependencies)
        self.package(self.launch / "launch", "launch")

        result = analyzer.analyze(self.workspace, generator.DEFAULT_UNIVERSE)
        by_name = {row["Package"]: row for row in generator.rank_candidates(result, {})}

        for name in ("single", "multiple", "converter"):
            with self.subTest(package=name):
                self.assertEqual(by_name[name]["Score"], 95)
                self.assertEqual(by_name[name]["Tier4_Message_Penalty"], 5)
                self.assertEqual(by_name[name]["Tier"], "ready")
                self.assertEqual(by_name[name]["Internal_Prerequisite_Count"], 0)
                self.assertTrue(by_name[name]["Standalone"])
                self.assertLess(by_name["free"]["Rank"], by_name[name]["Rank"])
        self.assertEqual(by_name["single"]["Tier4_Message_Dependencies"], "tier4_planning_msgs")
        self.assertEqual(by_name["multiple"]["Tier4_Message_Dependencies"],
                         "tier4_debug_msgs;tier4_planning_msgs")
        self.assertEqual(by_name["converter"]["Tier4_Message_Dependencies"], "tier4_auto_msgs_converter")
        for name in ("free", "indirect", "unrelated"):
            with self.subTest(package=name):
                self.assertEqual(by_name[name]["Score"], 100)
                self.assertEqual(by_name[name]["Tier4_Message_Penalty"], 0)
                self.assertEqual(by_name[name]["Tier4_Message_Dependencies"], "")

        output = self.workspace / "index_candidates.csv"
        generator.main(["--workspace", str(self.workspace), "--output", str(output)])
        with output.open(newline="") as file:
            exported = {row["Package"]: row for row in csv.DictReader(file)}
        self.assertEqual(exported["multiple"]["Score"], "95")
        self.assertEqual(exported["multiple"]["Tier4_Message_Penalty"], "5")
        self.assertEqual(exported["multiple"]["Tier4_Message_Dependencies"],
                         "tier4_debug_msgs;tier4_planning_msgs")

    def test_tier4_penalty_respects_each_tier_minimum(self):
        for tier, maximum, minimum in (
            ("ready", 100, 71), ("review", 70, 41),
            ("used_elsewhere", 40, 11), ("used_by_launch", 10, 1),
        ):
            with self.subTest(tier=tier):
                self.assertEqual(generator.candidate_score(tier, 0, 0, True), maximum - 5)
                self.assertEqual(generator.candidate_score(tier, 0, 1, True), max(minimum, maximum - 25))
                self.assertEqual(generator.candidate_score(tier, 1, 1, True), minimum)
                self.assertEqual(generator.candidate_score(tier, 100, 0, True), minimum)
                self.assertEqual(generator.candidate_score(tier, 0, 100, True), minimum)

    def test_exported_tier4_penalty_is_actual_score_reduction(self):
        self.package(self.workspace / generator.DEFAULT_TIER4_MSGS / "tier4_planning_msgs",
                     "tier4_planning_msgs")
        self.package(self.universe / "base", "base")
        self.package(self.universe / "consumer", "consumer", ("base", "tier4_planning_msgs"))
        self.package(self.universe / "user", "user", ("consumer",))
        result = analyzer.analyze(self.workspace, generator.DEFAULT_UNIVERSE)

        for paths, score, penalty in (({}, 71, 1), ({"consumer": "launch -> consumer"}, 1, 0)):
            with self.subTest(paths=paths):
                by_name = {row["Package"]: row for row in generator.rank_candidates(result, paths)}
                self.assertEqual(by_name["consumer"]["Score"], score)
                self.assertEqual(by_name["consumer"]["Tier4_Message_Penalty"], penalty)
                self.assertEqual(by_name["consumer"]["Tier4_Message_Dependencies"], "tier4_planning_msgs")

    def test_custom_tier4_checkout_path(self):
        custom_path = Path("src/custom/messages")
        self.package(self.workspace / custom_path / "tier4_planning_msgs", "tier4_planning_msgs")
        self.package(self.universe / "consumer", "consumer", ("tier4_planning_msgs",))
        self.package(self.launch / "launch", "launch")
        result = analyzer.analyze(self.workspace, generator.DEFAULT_UNIVERSE)
        self.assertEqual(generator.rank_candidates(result, {})[0]["Score"], 100)

        output = self.workspace / "index_candidates.csv"
        generator.main([
            "--workspace", str(self.workspace), "--tier4-msgs-path", str(custom_path),
            "--output", str(output),
        ])
        with output.open(newline="") as file:
            row = next(csv.DictReader(file))
        self.assertEqual(row["Score"], "95")
        self.assertEqual(row["Tier4_Message_Dependencies"], "tier4_planning_msgs")


if __name__ == "__main__":
    unittest.main()
