import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_site


class BuildSiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv_path = self.root / "candidates.csv"
        fields = ["Rank", "Package", "Score", "Tier", "Launch_Reachable", "Launch_Path",
                  "Standalone", "Internal_Recursive_Dependent_Count", "Internal_Direct_Dependent_Count",
                  "Internal_Prerequisite_Count", "Internal_Prerequisites",
                  "Tier4_Message_Dependencies", "Tier4_Message_Penalty",
                  "External_Manifest_Dependent_Count", "External_High_Reference_Count",
                  "External_Review_Reference_Count", "Source_Path"]
        with self.csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict.fromkeys(fields, "") | {
                "Rank": 1, "Package": "free", "Score": 100, "Tier": "ready",
                "Launch_Reachable": "False", "Standalone": "True",
                "Internal_Recursive_Dependent_Count": 0, "Internal_Direct_Dependent_Count": 0,
                "Internal_Prerequisite_Count": 0, "External_Manifest_Dependent_Count": 0,
                "Tier4_Message_Penalty": 0,
                "External_High_Reference_Count": 0, "External_Review_Reference_Count": 0,
                "Source_Path": "src/universe/autoware_universe/free",
            })

    def test_builds_static_site_and_snapshot(self):
        output = self.root / "public"
        metadata = build_site.build(self.csv_path, output, {
            "autoware": "abc", "universe": "def", "launch": "ghi"
        }, generated_at="2026-09-24T00:00:00+00:00")
        self.assertEqual(metadata["tier_counts"], {"ready": 1})
        self.assertEqual(metadata, json.loads((output / "metadata.json").read_text()))
        data = json.loads((output / "data.json").read_text())
        self.assertEqual(data["candidates"][0]["Score"], 100)
        self.assertFalse(data["candidates"][0]["Launch_Reachable"])
        self.assertEqual((output / "candidates.csv").read_bytes(), self.csv_path.read_bytes())
        for name in ("index.html", "style.css", "app.js"):
            self.assertTrue((output / name).is_file())

    def test_preserves_tier4_dependencies_and_numeric_penalty_in_snapshot(self):
        with self.csv_path.open(newline="") as file:
            row = next(csv.DictReader(file))
        with self.csv_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writerow(row | {
                "Rank": 2, "Package": "tier4_consumer", "Score": 95,
                "Tier4_Message_Dependencies": "tier4_debug_msgs;tier4_planning_msgs",
                "Tier4_Message_Penalty": 5,
                "Source_Path": "src/universe/autoware_universe/tier4_consumer",
            })

        output = self.root / "public"
        build_site.build(self.csv_path, output, {})
        rows = json.loads((output / "data.json").read_text())["candidates"]
        self.assertEqual(rows[0]["Tier4_Message_Penalty"], 0)
        self.assertEqual(rows[0]["Tier4_Message_Dependencies"], "")
        self.assertEqual(rows[1]["Score"], 95)
        self.assertEqual(rows[1]["Tier4_Message_Penalty"], 5)
        self.assertEqual(rows[1]["Tier4_Message_Dependencies"], "tier4_debug_msgs;tier4_planning_msgs")

    def test_rejects_wrong_rank(self):
        text = self.csv_path.read_text().replace("1,free,", "2,free,")
        self.csv_path.write_text(text)
        with self.assertRaisesRegex(ValueError, "Invalid rank"):
            build_site.load_candidates(self.csv_path)


if __name__ == "__main__":
    unittest.main()
