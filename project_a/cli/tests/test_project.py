"""Our tests (the staff ones are test_interface.py / test_metrics.py).

    python -m unittest discover -s tests -v        # from the cli folder

1. Every architecture preset passes the staff runner's checks and fits the parameter caps.
2. Config overrides and the checkpoint round trip; image packing and 2D RoPE.
3. CV folds are stratified; OOF / halves never give the refiner y0 from an H0 that saw the image.
4. End to end on synthetic tiles: train_h0 -> train -> export -> infer from an empty folder.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from baseline import DemoBaseline
from predict import predict_tensor
from splits import fold_plan, make_folds, read_folds, scan, write_folds
from submission.config import (PRESETS, apply_overrides, config_from_json, config_to_json,
                               get_config)
from submission.image import ImagePackedBase, pack_image, packed_channels, unpack_image
from submission.model import build_refiner, load_refiner
from submission.pos import apply_rope, rope2d

CORE_LIMIT, TOTAL_LIMIT = 5_000_000, 10_000_000


def make_data(root, per_group=6, size=64):
    """Synthetic LoveDA layout: {Train,Val}/{Urban,Rural}/{images_png,masks_png}/<n>.png."""
    rng = np.random.default_rng(0)
    n = 0
    for source in ("Train", "Val"):
        for domain in ("Urban", "Rural"):
            for sub in ("images_png", "masks_png"):
                (root / source / domain / sub).mkdir(parents=True)
            for _ in range(per_group):
                Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(
                    root / source / domain / "images_png" / f"{n}.png")
                Image.fromarray(rng.integers(0, 8, (size, size), dtype=np.uint8)).save(
                    root / source / domain / "masks_png" / f"{n}.png")
                n += 1


class PresetTests(unittest.TestCase):
    def test_every_preset_passes_runner_checks(self):
        torch.manual_seed(0)
        image = torch.rand(1, 3, 64, 64)
        for name, cfg in PRESETS.items():
            with self.subTest(preset=name):
                base = DemoBaseline().eval()
                if cfg.image.use:                                       # image packed into f
                    base = ImagePackedBase(base, cfg.image.stride)
                model = build_refiner(base.feature_channels, cfg).eval()
                trace4, calls = predict_tensor(base, model, image, 4)   # core-call + parameter checks
                trace2, _ = predict_tensor(base, model, image, 2)
                self.assertEqual(calls, 4)
                for a, b in zip(trace2, trace4[:3]):                    # prefix consistency
                    np.testing.assert_allclose(a, b, atol=1e-6)
                self.assertFalse(np.array_equal(trace4[1], trace4[0]))  # steps change the answer

    def test_presets_fit_caps(self):
        for name, cfg in PRESETS.items():
            with self.subTest(preset=name):
                model = build_refiner(384 + packed_channels(cfg), cfg)
                self.assertLessEqual(sum(p.numel() for p in model.core.parameters()), CORE_LIMIT)
                self.assertLessEqual(sum(p.numel() for p in model.parameters()), TOTAL_LIMIT)


class ConfigTests(unittest.TestCase):
    def test_overrides(self):
        cfg = apply_overrides(get_config("attempt7"), ["core.layers=2", "tokenizer.fuse=concat",
                                                       "head.gate=false", "head.dilations=1,2", "d=128"])
        self.assertEqual((cfg.core.layers, cfg.tokenizer.fuse, cfg.head.gate, cfg.head.dilations, cfg.d),
                         (2, "concat", False, (1, 2), 128))
        with self.assertRaises(KeyError):
            apply_overrides(cfg, ["core.nope=1"])
        with self.assertRaises(ValueError):
            apply_overrides(cfg, ["core.layers"])

    def test_json_round_trip_and_checkpoint(self):
        cfg = apply_overrides(get_config("attempt6"), ["core.layers=2"])
        self.assertEqual(config_from_json(config_to_json(cfg)), cfg)
        from safetensors.torch import save_file
        model = build_refiner(5, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.safetensors"
            save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path),
                      metadata={"refiner_cfg": config_to_json(cfg)})
            self.assertEqual(load_refiner(path, 5).cfg, cfg)               # rebuilt from metadata


class PositionAndImageTests(unittest.TestCase):
    def test_pack_unpack_round_trip(self):
        features, image = torch.randn(2, 5, 4, 3), torch.rand(2, 3, 64, 48)
        x, img = unpack_image(pack_image(features, image, 16), 16)
        self.assertTrue(torch.equal(x, features))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        torch.testing.assert_close(img * std + mean, image)

    def test_rope_scores_depend_on_relative_position_only(self):
        q, k = torch.randn(1, 2, 12, 32), torch.randn(1, 2, 12, 32)
        def scores(offset):
            cos, sin = rope2d(3, 4, 32, offset=offset)
            return apply_rope(q, cos, sin) @ apply_rope(k, cos, sin).transpose(-1, -2)
        torch.testing.assert_close(scores(None), scores(torch.tensor([[5, 7]])), atol=1e-4, rtol=1e-4)


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "data"
        make_data(cls.root, per_group=10, size=16)
        write_folds(make_folds(scan(cls.root), k=5), Path(cls.tmp.name) / "folds.csv")
        cls.rows = read_folds(Path(cls.tmp.name) / "folds.csv", cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_folds_stratified_and_complete(self):
        self.assertEqual(len(self.rows), 40)
        for f in range(5):
            for d in ("urban", "rural"):
                self.assertEqual(sum(r["fold"] == f and r["domain"] == d for r in self.rows), 4)
        self.assertTrue(all(Path(r["image"]).is_file() and Path(r["label"]).is_file() for r in self.rows))
        self.assertEqual(make_folds(scan(self.root), k=5), make_folds(scan(self.root), k=5))  # deterministic

    def test_oof_never_routes_to_an_h0_that_saw_the_image(self):
        for v in range(5):
            plan = fold_plan(self.rows, v, "oof")
            val_ids = {r["id"] for r in plan["val"]}
            self.assertEqual(len(plan["refiner"]) + len(plan["val"]), len(self.rows))   # refiner uses all
            for name, rows in plan["h0"].items():
                self.assertFalse(val_ids & {r["id"] for r in rows}, f"{name} trains on the val fold")
            h0_ids = {n: {r["id"] for r in rows} for n, rows in plan["h0"].items()}
            for r in plan["refiner"]:
                self.assertNotIn(r["id"], h0_ids[plan["route"][r["id"]]])
            self.assertEqual(plan["eval_h0"], "all")

    def test_halves_disjoint(self):
        plan = fold_plan(self.rows, 2, "halves")
        a = {r["id"] for r in plan["h0"]["a"]}
        b = {r["id"] for r in plan["refiner"]}
        self.assertFalse(a & b)
        self.assertEqual(len(a) + len(b), 32)
        self.assertFalse((a | b) & {r["id"] for r in plan["val"]})


class PipelineTests(unittest.TestCase):
    def test_train_export_infer(self):
        for arch in ("attempt7", "attempt10"):
            with self.subTest(arch=arch):
                self.run_pipeline(arch)

    def run_pipeline(self, arch):
        import export_submission
        import infer
        import train
        import train_h0
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_data(tmp / "data")
            folds = tmp / "folds.csv"
            write_folds(make_folds(scan(tmp / "data")), folds)
            common = ["--data-root", str(tmp / "data"), "--folds", str(folds), "--val-fold", "0",
                      "--epochs", "1", "--batch-size", "4", "--workers", "0", "--device", "cpu"]
            train_h0.main(common + ["--encoder", "pool", "--out-dir", str(tmp / "h0")])
            run_args = common + ["--h0-dir", str(tmp / "h0"), "--crop", "64", "--steps", "2", "--arch", arch,
                                 "--output", str(tmp / "run" / "refiner.safetensors")]
            train.main(run_args)
            with self.assertRaises(SystemExit):                    # same run again without --resume
                train.main(run_args)
            with self.assertRaises(SystemExit):                    # someone else's settings, same folder
                train.main(run_args + ["--lr", "1e-4", "--resume", str(tmp / "run" / "refiner.resume.pt")])
            train.main(run_args + ["--epochs", "1", "--resume", str(tmp / "run" / "refiner.resume.pt")])
            with open(tmp / "run" / "history.csv", newline="") as fh:    # 0 = before training, 1 = epoch 1
                history = list(csv.DictReader(fh))
            self.assertEqual([r["epoch"] for r in history], ["0", "1"])
            self.assertTrue(history[1]["train_y2"] and history[1]["val_loss_k4"] and history[0]["val_quality_k0"])
            for col in ("grad_core", "clip_rate", "diag_dy_t8", "diag_loss_t0", "diag_quality_k4_reset"):
                self.assertTrue(history[1][col], col)
            samples = tmp / "samples" / "nested"
            samples.mkdir(parents=True)
            for i in (0, 1):
                Image.open(tmp / "data" / "Val" / "Urban" / "images_png" / f"{12 + i}.png").save(samples / f"{i}.png")
            export_submission.main(["--refiner", str(tmp / "run" / "refiner_best.safetensors"),
                                    "--out", str(tmp / "dist"), "--check", str(tmp / "samples"),
                                    "--steps", "0,1,2"])
            out = tmp / "pred"
            infer.main(["--input", str(tmp / "samples"), "--output", str(out), "--steps", "0,2",
                        "--weights", str(tmp / "dist" / "weights" / "submission.safetensors"),
                        "--device", "cpu"])
            mask = np.asarray(Image.open(out / "k2" / "nested" / "0.png"))
            self.assertEqual(mask.shape, (64, 64))
            self.assertTrue(set(np.unique(mask)) <= {0, 1})
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["README.txt", "k0", "k2"])


if __name__ == "__main__":
    unittest.main()
