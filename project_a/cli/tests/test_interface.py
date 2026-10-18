import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from common import VERSION, read_json
from baseline import DemoBaseline, load_baseline
from make_demo import generate
from make_stress import generate as generate_stress, corrupt, rng_for
from predict import predict_tensor, main as predict_main
from evaluate import evaluate
from score import combine
from submission.model import build_refiner


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(523)
        self.model = build_refiner(5).eval()
        self.base = DemoBaseline().eval()
        self.image = torch.rand(1, 3, 64, 64)

    def test_exact_k_and_prefix(self):
        a, calls_a = predict_tensor(self.base, self.model, self.image, 2)
        b, calls_b = predict_tensor(self.base, self.model, self.image, 4)
        self.assertEqual((len(a), calls_a, len(b), calls_b), (3, 2, 5, 4))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_zero_is_untouched_baseline(self):
        trace, calls = predict_tensor(self.base, self.model, self.image, 0)
        expected = self.base(self.image)[1].sigmoid().numpy()
        np.testing.assert_array_equal(trace[0], expected)
        self.assertEqual(calls, 0)

    def test_updates_change_prediction(self):
        trace, _ = predict_tensor(self.base, self.model, self.image, 4)
        self.assertGreater(np.abs(trace[-1] - trace[0]).max(), 0)

    def test_extrapolation_counts(self):
        for k in (6, 8):
            trace, calls = predict_tensor(self.base, self.model, self.image, k)
            self.assertEqual(calls, k)
            self.assertEqual(len(trace), k + 1)
            self.assertTrue(np.isfinite(trace[-1]).all())

    def test_feedback_interventions_change_demo(self):
        normal, _ = predict_tensor(self.base, self.model, self.image, 4)
        for intervention in ("reset-state", "no-mask-feedback"):
            changed, _ = predict_tensor(self.base, self.model, self.image, 4, intervention)
            self.assertGreater(np.abs(normal[-1] - changed[-1]).max(), 0)

    def test_no_op_core_is_caught(self):
        def bypass(features, state, logits):
            return state, logits
        self.model.step = bypass
        with self.assertRaisesRegex(ValueError, "exactly once"):
            predict_tensor(self.base, self.model, self.image, 1)

    def test_missing_official_baseline_fails(self):
        with self.assertRaises(ValueError):
            load_baseline(None, "cpu")
        with self.assertRaises(RuntimeError):
            load_baseline(Path("not_released.safetensors"), "cpu")

    def test_cli_and_evaluator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            demo = generate(root / "data")
            predict_main(["--demo", "--checkpoint", str(demo / "model.safetensors"),
                          "--input", str(demo / "images"), "--output", str(root / "pred"),
                          "--steps", "4", "--save-trace"])
            report = evaluate(demo / "manifest.json", root / "pred")
            self.assertEqual(report["steps"], 4)
            self.assertEqual(report["number_of_images"], 2)
            self.assertTrue(report["demo"])
            self.assertEqual(set(report["by_domain"]), {"urban", "rural"})
            # Output overwrite guard.
            with self.assertRaises(ValueError):
                predict_main(["--demo", "--checkpoint", str(demo / "model.safetensors"),
                              "--input", str(demo / "images"), "--output", str(root / "pred"),
                              "--steps", "1"])
            # Extra predictions are not silently ignored.
            np.save(root / "pred/probabilities/extra.npy", np.zeros((96, 96), np.float32))
            with self.assertRaises(ValueError):
                evaluate(demo / "manifest.json", root / "pred")

    def test_stress_is_reproducible_and_keeps_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            demo = generate(root / "data")
            a = generate_stress(demo / "manifest.json", root / "stress1", 523)
            b = generate_stress(demo / "manifest.json", root / "stress2", 523)
            ra, rb = read_json(a), read_json(b)
            for x, y in zip(ra, rb):
                self.assertEqual(x["corruption"], y["corruption"])
                self.assertEqual(x["label"], y["label"])
                self.assertEqual((a.parent / x["image"]).read_bytes(),
                                 (b.parent / y["image"]).read_bytes())

    def test_each_stress_family(self):
        array = np.random.default_rng(523).integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
        for kind in ("shadow", "occlusion", "blur"):
            result, params = corrupt(Image.fromarray(array), kind, rng_for(523, kind))
            changed = np.asarray(result)
            self.assertEqual(changed.shape, array.shape)
            self.assertEqual(changed.dtype, np.uint8)
            self.assertTrue(np.any(changed != array))
            self.assertEqual(params["kind"], kind)
            if kind != "blur":
                x, y, w, h = params["rectangle_xywh"]
                self.assertTrue(0 <= x < x + w <= 96)
                self.assertTrue(0 <= y < y + h <= 96)

    def test_combiner_guards(self):
        reports = [{"condition": condition, "steps": k, "protocol": VERSION,
                    "demo": True, "ablation": "none", "checkpoint_sha256": "x",
                    "baseline_sha256": "b", "submission_source_sha256": "s",
                    "source_set_sha256": "i", "overall": {"quality": quality}}
                   for condition, k, quality in [("ordinary", 1, .6), ("ordinary", 2, .64),
                                                  ("ordinary", 4, .68), ("stress", 4, .55)]]
        with self.assertRaises(ValueError):
            combine(reports)
        self.assertAlmostEqual(combine(reports, allow_demo=True)["score"], 63)
        reports[-1]["checkpoint_sha256"] = "different"
        with self.assertRaises(ValueError):
            combine(reports, allow_demo=True)


if __name__ == "__main__":
    unittest.main()
