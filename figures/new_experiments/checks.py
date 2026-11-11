"""CPU-only synthetic checks for the native factor interventions."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from schemajev.mechanisms import (
    COMPONENTS, DECISION_TARGETS, ENCODER_TARGETS, STRENGTHS,
    transform_adapters, validate_adapters,
)
from schemajev.runtime_benchmark import measure, summarize, tensor_storage


class Interventions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.adapters = {
            layer: {name: (torch.ones(a), torch.ones(b)) for name, (a, b) in
                    (ENCODER_TARGETS if layer < 22 else DECISION_TARGETS).items()}
            for layer in range(24)
        }

    def test_strength(self):
        original = self.adapters
        self.assertIs(transform_adapters(torch, original), original)
        for strength in STRENGTHS:
            result = transform_adapters(torch, original, strength=strength)
            self.assertEqual(validate_adapters(torch, result), 96)
            for layer, row in result.items():
                for name, (a, b) in row.items():
                    old_a, old_b = original[layer][name]
                    self.assertIs(a, old_a)
                    self.assertTrue(torch.equal(a[:2] @ b[:, :3], strength * (old_a[:2] @ old_b[:, :3])))
                    self.assertTrue(torch.equal(old_b, torch.ones_like(old_b)))

    def test_components(self):
        for mode in COMPONENTS:
            result = transform_adapters(torch, self.adapters, components=mode)
            enabled = 0
            for layer, row in result.items():
                for name, (a, b) in row.items():
                    on = mode == "full" or (mode == "encoder_only" and layer < 22) or (mode == "head_only" and layer >= 22)
                    self.assertIs(a, self.adapters[layer][name][0])
                    self.assertEqual(bool(torch.count_nonzero(b)), on)
                    if on:
                        self.assertIs(b, self.adapters[layer][name][1])
                        enabled += 1
            self.assertEqual(enabled, {"backbone": 0, "encoder_only": 88, "head_only": 8, "full": 96}[mode])

    def test_invalid_inputs(self):
        for strength in (True, -1, 0.1, float("nan")):
            with self.assertRaises(ValueError):
                transform_adapters(torch, self.adapters, strength=strength)
        invalid = dict(self.adapters)
        invalid[0] = dict(invalid[0])
        invalid[0].pop("attn.Wqkv")
        with self.assertRaises(ValueError):
            transform_adapters(torch, invalid)
        invalid[0] = dict(self.adapters[0])
        a, b = invalid[0]["attn.Wqkv"]
        bad = b.clone()
        bad[0, 0] = float("nan")
        invalid[0]["attn.Wqkv"] = (a, bad)
        with self.assertRaises(ValueError):
            transform_adapters(torch, invalid, strength=0)

    def test_storage_and_summary(self):
        tensor = torch.arange(4, dtype=torch.float32)
        result = tensor_storage(torch, {"whole": tensor, "view": tensor[:2]})
        self.assertEqual(result["logical_tensor_bytes"], 24)
        self.assertEqual(result["unique_tensor_storage_bytes"], 16)
        self.assertEqual(result["persisted_task_artifact_bytes"], 0)
        self.assertEqual(summarize([1, 2, 3])["median"], 2)
        self.assertAlmostEqual(summarize([1, 2, 3])["p90"], 2.8)
        self.assertIsNone(summarize([1])["sample_sd"])
        with self.assertRaises(ValueError):
            measure(torch, lambda: None, idle_check=lambda: None, warmup=0)


if __name__ == "__main__":
    unittest.main()
