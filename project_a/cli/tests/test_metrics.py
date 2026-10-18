import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from common import check_probability, safe_id
from metrics import Counts, image_counts, summarize, leaderboard_score


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.ones((32, 32), np.uint8)
        self.labels[8:24, 12:20] = 3
        self.perfect = (self.labels == 3).astype(np.float32)

    def test_perfect(self):
        result = summarize(image_counts(self.perfect, self.labels))
        for name in ("road_iou", "cldice", "boundary_iou", "quality"):
            self.assertEqual(result[name], 1.0)

    def test_both_empty(self):
        labels = np.ones((32, 32), np.uint8)
        result = summarize(image_counts(np.zeros((32, 32), np.float32), labels))
        self.assertEqual(result["quality"], 1.0)

    def test_only_prediction_empty(self):
        result = summarize(image_counts(np.zeros((32, 32), np.float32), self.labels))
        self.assertEqual(result["quality"], 0.0)

    def test_only_truth_empty(self):
        result = summarize(image_counts(self.perfect, np.ones((32, 32), np.uint8)))
        self.assertEqual(result["quality"], 0.0)

    def test_ignore_has_no_effect(self):
        labels = self.labels.copy()
        labels[:4] = 0
        first = self.perfect.copy()
        second = self.perfect.copy()
        second[:4] = 1
        self.assertEqual(image_counts(first, labels), image_counts(second, labels))

    def test_all_ignore_is_rejected_as_dataset(self):
        c = image_counts(self.perfect, np.zeros_like(self.labels))
        self.assertEqual(c.valid, 0)
        with self.assertRaises(ValueError):
            summarize(c)

    def test_threshold_includes_half(self):
        p = self.perfect * 0.5
        self.assertEqual(summarize(image_counts(p, self.labels))["road_iou"], 1.0)

    def test_known_overlap(self):
        p = self.perfect.copy()
        p[8:16, 12:20] = 0
        result = summarize(image_counts(p, self.labels))
        self.assertEqual(result["road_iou"], 0.5)
        self.assertLess(result["cldice"], 1)

    def test_counts_pool_not_image_average(self):
        a = Counts(valid=100, intersection=1, union=1)
        b = Counts(valid=100, intersection=0, union=9)
        self.assertEqual(summarize(a + b)["road_iou"], 0.1)

    def test_invalid_probability(self):
        for bad in (np.full((32, 32), np.nan, np.float32),
                    np.full((32, 32), 1.1, np.float32),
                    np.zeros((31, 32), np.float32),
                    np.zeros((32, 32), np.float64)):
            with self.assertRaises(ValueError):
                image_counts(bad, self.labels)

    def test_wrong_label_encoding(self):
        for labels in (np.zeros((32, 32, 3), np.uint8),
                       np.full((32, 32), 255, np.uint8),
                       self.labels.astype(np.float32)):
            with self.assertRaises(ValueError):
                image_counts(self.perfect, labels)

    def test_score_example(self):
        self.assertAlmostEqual(leaderboard_score(.60, .64, .68, .55), 63.0)
        self.assertEqual(leaderboard_score(1, 1, 1, 1), 100.0)

    def test_bad_score(self):
        with self.assertRaises(ValueError):
            leaderboard_score(1, 1, np.nan, 1)

    def test_safe_ids(self):
        self.assertEqual(safe_id("urban/001"), "urban/001")
        for bad in ("../secret", "/absolute", "", "a/../b", "a\\b", "a//b"):
            with self.assertRaises(ValueError):
                safe_id(bad)

    def test_three_pixel_boundary_band(self):
        # Ground truth is a 16x8 rectangle, so eroded core is 10x2.
        counts = image_counts(self.perfect, self.labels)
        self.assertEqual(counts.boundary_union, 16 * 8 - 10 * 2)

    def test_frame_excluded_from_boundary(self):
        labels = np.full((32, 32), 3, np.uint8)
        counts = image_counts(np.ones((32, 32), np.float32), labels)
        self.assertEqual(counts.boundary_union, 0)


if __name__ == "__main__":
    unittest.main()
