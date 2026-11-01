"""Behavioral checks for data handling, freezing, resume, and offline export."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

from project_a.cli.submission.model import build_refiner
from road_refinement.baseline import TemporaryBaseline
from road_refinement.cli import main
from road_refinement.data import RoadCrops, discover, masked_bce, read_tile
from road_refinement.inference import load_bundle, predict_image
from src.tokenizer.input.Tx import Tx


def make_data(root):
    """Small course-layout fixtures with image/mask alignment encoded in red."""
    for split in ("Train", "Val"):
        for domain in ("Urban", "Rural"):
            for index in range(2):
                h, w = (64, 64) if split == "Train" else (67, 79)
                labels = np.ones((h, w), dtype=np.uint8)
                labels[25:32, 4:w-4] = 3
                labels[4:h-4, 20+index:25+index] = 3
                labels[:2] = 0
                rgb = np.full((h, w, 3), 30, dtype=np.uint8)
                rgb[labels == 3] = 220
                for folder, values in (("images_png", rgb), ("masks_png", labels)):
                    target = Path(root) / split / domain / folder / f"{index:03}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(values).save(target)


class MidwayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_tokenizer_shape_and_amp(self):
        tokenizer = Tx(256)
        image = torch.randn(2, 256, 7, 9, requires_grad=True)
        output = tokenizer(image)
        self.assertEqual(output.shape, (2, 63, 256))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertEqual(tokenizer(image.detach()).dtype, torch.bfloat16)

    def test_pairing_crops_and_ignore_loss(self):
        with tempfile.TemporaryDirectory() as temp:
            make_data(temp)
            tiles = discover(temp, "Train", 1)
            self.assertEqual([t.id for t in tiles], ["urban/000", "rural/000"])
            images, labels = RoadCrops(tiles, 96, 2, 523)[0]
            np.testing.assert_array_equal(images[0].numpy() > 0.5, labels[0].numpy() == 3)
            self.assertTrue((labels[:, 64:] == 0).all())
            logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
            target = torch.tensor([[[[3, 1], [0, 0]]]])
            loss = masked_bce(logits, target)
            loss.backward()
            self.assertTrue((logits.grad[0, 0, 1] == 0).all())
            changed = logits.detach().clone()
            changed[0, 0, 1] = 100
            self.assertEqual(loss.item(), masked_bce(changed, target).item())
            with self.assertRaisesRegex(ValueError, "All-ignore"):
                masked_bce(logits, torch.zeros_like(target))
            Path(tiles[0].label).unlink()
            with self.assertRaisesRegex(ValueError, "Missing paired"):
                discover(temp, "Train", 1)

    def test_all_ignore_sampling_fails_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            make_data(temp)
            tiles = discover(temp, "Train", 1)
            for tile in tiles:
                Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(tile.label)
            with self.assertRaisesRegex(ValueError, "20 attempts"):
                RoadCrops(tiles, 64, 1, 523)[0]

    def test_encoder_and_baseline_freezing(self):
        baseline = TemporaryBaseline()
        baseline.freeze_encoder()
        baseline.train()
        encoder = baseline.segmenter.encoder
        buffers = {name: value.clone() for name, value in encoder.named_buffers()}
        before = baseline.segmenter.segmentation_head[0].weight.detach().clone()
        optimizer = torch.optim.AdamW((p for p in baseline.parameters() if p.requires_grad), lr=1e-3)
        _, logits = baseline(torch.rand(2, 3, 64, 64))
        masked_bce(logits, torch.ones(2, 1, 64, 64, dtype=torch.long)).backward()
        optimizer.step()
        self.assertFalse(encoder.training)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        for name, value in encoder.named_buffers():
            self.assertTrue(torch.equal(buffers[name], value), name)
        self.assertFalse(torch.equal(before, baseline.segmenter.segmentation_head[0].weight))
        baseline.eval().requires_grad_(False)
        baseline.zero_grad(set_to_none=True)
        with torch.no_grad():
            features, logits = baseline(torch.rand(2, 3, 64, 64))
        refiner = build_refiner(256)
        state = refiner.initialize(features, logits)
        state, logits = refiner.step(features, state, logits)
        masked_bce(logits, torch.ones(2, 1, 64, 64, dtype=torch.long)).backward()
        self.assertTrue(all(p.grad is None for p in baseline.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in refiner.parameters()))

    def test_training_resume_export_and_offline_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, run = root / "data", root / "run"
            make_data(data)
            options = ["--cpu-threads", "1", "train", "--data-root", str(data),
                       "--run-dir", str(run), "--device", "cpu", "--synthetic", "--workers", "0",
                       "--baseline-updates", "2", "--refiner-updates", "2", "--save-every", "1",
                       "--train-per-domain", "1", "--val-per-domain", "1", "--crop-size", "64"]
            main(options)
            main(options + ["--resume", "--refiner-updates", "3"])
            resumed = load_file(str(run / "checkpoints/refiner_latest.safetensors"))
            # Compare continuation with an uninterrupted run at identical updates.
            uninterrupted = options.copy()
            uninterrupted[uninterrupted.index(str(run))] = str(root / "uninterrupted")
            main(uninterrupted + ["--refiner-updates", "3"])
            expected = load_file(str(root / "uninterrupted/checkpoints/refiner_latest.safetensors"))
            for name in resumed:
                self.assertTrue(torch.equal(resumed[name], expected[name]), name)
            bundle = root / "bundle"
            main(["export", "--run-dir", str(run), "--output", str(bundle)])
            baseline, refiner, manifest = load_bundle(bundle, torch.device("cpu"))
            self.assertTrue(manifest["synthetic"])
            self.assertFalse(manifest["official_baseline"])
            self.assertFalse(list(bundle.rglob("*_trainer.pt")))
            image = torch.rand(1, 3, 67, 79)
            _, short, calls = predict_image(baseline, refiner, image, 2, True)
            logits, long, long_calls = predict_image(baseline, refiner, image, 4, True)
            self.assertEqual((calls, long_calls, tuple(logits.shape)), (2, 4, (1, 1, 67, 79)))
            for a, b in zip(short, long):
                np.testing.assert_array_equal(a, b)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(bundle / "code") + os.pathsep + env.get("PYTHONPATH", "")
            env["OMP_NUM_THREADS"] = "1"
            env["HF_HUB_OFFLINE"] = "1"
            # Only exported code is on the working directory; block network access.
            script = "from unittest.mock import patch; from road_refinement.cli import main; " \
                     "guard=patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')); " \
                     "guard.start(); main()"
            single = data / "Val/Urban/images_png/000.png"
            command = [sys.executable, "-c", script, "--cpu-threads", "1", "predict", "--bundle", str(bundle),
                       "--input", str(single), "--output", str(root / "single"), "--steps", "4", "--device", "cpu"]
            subprocess.run(command, cwd=root, env=env, check=True, capture_output=True, text=True)
            self.assertEqual(np.load(root / "single/probabilities/000.npy").shape, (67, 79))
            images = root / "images"
            (images / "urban").mkdir(parents=True)
            import shutil
            shutil.copyfile(single, images / "urban/000.png")
            main(["predict", "--bundle", str(bundle), "--input", str(images),
                  "--output", str(root / "folder"), "--device", "cpu"])
            self.assertTrue((root / "folder/probabilities/urban/000.npy").is_file())
            shutil.copyfile(single, images / "urban/000.jpg")
            with self.assertRaises(SystemExit):
                main(["predict", "--bundle", str(bundle), "--input", str(images),
                      "--output", str(root / "duplicate"), "--device", "cpu"])
            report = root / "evaluation.json"
            main(["evaluate", "--bundle", str(bundle), "--data-root", str(data),
                  "--val-per-domain", "1", "--output", str(report), "--device", "cpu"])
            result = json.loads(report.read_text())
            self.assertEqual([r["steps"] for r in result["ordinary"]], [0, 1, 2, 4])
            self.assertEqual(result["stress"]["steps"], 4)
            self.assertTrue(0 <= result["local_score"] <= 100)


if __name__ == "__main__":
    unittest.main()
