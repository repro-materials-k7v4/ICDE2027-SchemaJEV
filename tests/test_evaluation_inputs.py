import copy
import json
from pathlib import Path
import tempfile
import unittest

from schemajev.evaluation_inputs import (
    canonical_sha256,
    load_evaluation_conditions,
    load_evaluation_views,
)


ROOT = Path(__file__).resolve().parents[1]


class FinalEvaluationInputsTest(unittest.TestCase):
    def test_final_protocol_is_complete(self):
        panel = load_evaluation_views()
        self.assertEqual(len(panel), 18)
        self.assertEqual(sum(view["rows"] for view in panel), 30046)
        self.assertEqual([view["view"] for view in panel[:4]], [
            "scienceqa__choice", "cosmosqa__choice", "social_iqa__choice", "sick__choice"
        ])
        conditions = load_evaluation_conditions()
        self.assertEqual(canonical_sha256(conditions),
                         "45f3aadc5b5ad7424d13125313aeac6c1b5463573f5e7b83e9cb3d28758c6403")
        self.assertTrue(all(set(condition) == {"context", "instruction"}
                            for condition in conditions.values()))

    def check_tampering_rejected(self, mutate):
        panel = copy.deepcopy(load_evaluation_views())
        mutate(panel)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval_views.json"
            path.write_text(json.dumps(panel), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_evaluation_views(path)

    def test_changed_prompt_is_rejected(self):
        self.check_tampering_rejected(
            lambda panel: panel[0]["description"].update(context="Different prompt")
        )

    def test_rehashed_changed_prompt_still_breaks_final_map(self):
        def mutate(panel):
            panel[0]["description"]["context"] = "Different prompt"
            panel[0]["description_sha256"] = canonical_sha256(panel[0]["description"])
        self.check_tampering_rejected(mutate)

    def test_model_condition_cannot_include_view_id(self):
        self.check_tampering_rejected(
            lambda panel: panel[0]["description"].update(dataset_id=panel[0]["view"])
        )

    def test_duplicate_view_is_rejected(self):
        self.check_tampering_rejected(lambda panel: panel.__setitem__(1, panel[0]))

    def test_wrong_cohort_is_rejected(self):
        self.check_tampering_rejected(lambda panel: panel[0].update(cohort="OOD6"))

    def test_missing_view_is_rejected(self):
        self.check_tampering_rejected(lambda panel: panel.pop())

    def test_changed_mask_is_rejected(self):
        self.check_tampering_rejected(lambda panel: panel[0].update(mask_sha256="0" * 64))

    def test_reordered_views_are_rejected(self):
        def mutate(panel):
            panel[0], panel[1] = panel[1], panel[0]
        self.check_tampering_rejected(mutate)

    def test_training_conditions_are_separate(self):
        training = json.loads((ROOT / "data/train_conditions.json").read_text())
        evaluation = load_evaluation_conditions()
        self.assertEqual(len(training), 60)
        for key in set(training).intersection(evaluation):
            self.assertNotEqual(training[key], evaluation[key])


if __name__ == "__main__":
    unittest.main()
