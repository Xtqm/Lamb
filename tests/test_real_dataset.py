"""
Unit tests for JED-M2 Real Web Trajectory Dataset, Generalization Splits, and Disjointness.
Verifies:
- Partitioning into cross-task, cross-website, and cross-domain generalization splits.
- Programmatic site disjointness assertions and leakage detection.

STRICT CONVENTION: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

import os
from pathlib import Path
import unittest

from jed_m2.real_dataset import assert_website_disjointness, get_generalization_splits


class TestRealDatasetSplits(unittest.TestCase):
    """Test suite for generalization split partitioning and site disjointness."""

    def setUp(self):
        self.real_manifest_path = Path("data/manifests/real_mind2web_v1/manifest.json")

    def test_real_manifest_disjointness(self):
        """Verifies that the authentic Mind2Web v1 manifest has 0% site overlap across train, val, and test."""
        if not self.real_manifest_path.is_file():
            self.skipTest("Manifest file not found.")
        # Must not raise AssertionError
        assert_website_disjointness(self.real_manifest_path)
        print("PASS: test_real_manifest_disjointness - Authenticated zero site overlap")

    def test_disjointness_leakage_detection(self):
        """Verifies that assert_website_disjointness raises AssertionError when leakage exists."""
        leaking_manifest = {
            "tasks": [
                {"annotation_id": "1", "website": "amazon.com", "split": "train", "domain": "shopping"},
                {"annotation_id": "2", "website": "ebay.com", "split": "val", "domain": "shopping"},
                {"annotation_id": "3", "website": "amazon.com", "split": "test", "domain": "shopping"},
            ]
        }
        with self.assertRaises(AssertionError) as ctx:
            assert_website_disjointness(leaking_manifest)
        self.assertIn("amazon.com", str(ctx.exception))
        print("PASS: test_disjointness_leakage_detection - Successfully caught website leakage")

    def test_generalization_splits_categorization(self):
        """Verifies accurate categorization into cross_task, cross_website, and cross_domain."""
        mock_manifest = {
            "tasks": [
                # Train tasks
                {"annotation_id": "tr1", "website": "site_a", "split": "train", "domain": "shopping"},
                {"annotation_id": "tr2", "website": "site_b", "split": "train", "domain": "travel"},
                # Test tasks:
                # 1. cross_task: seen website (site_a)
                {"annotation_id": "te1", "website": "site_a", "split": "test", "domain": "shopping"},
                # 2. cross_website: unseen website (site_c), seen domain (shopping)
                {"annotation_id": "te2", "website": "site_c", "split": "test", "domain": "shopping"},
                # 3. cross_domain: unseen website (site_d), unseen domain (finance)
                {"annotation_id": "te3", "website": "site_d", "split": "test", "domain": "finance"},
            ]
        }
        splits = get_generalization_splits(mock_manifest)
        self.assertEqual(len(splits["cross_task"]), 1)
        self.assertEqual(splits["cross_task"][0]["annotation_id"], "te1")

        self.assertEqual(len(splits["cross_website"]), 1)
        self.assertEqual(splits["cross_website"][0]["annotation_id"], "te2")

        self.assertEqual(len(splits["cross_domain"]), 1)
        self.assertEqual(splits["cross_domain"][0]["annotation_id"], "te3")
        print("PASS: test_generalization_splits_categorization - All 3 splits correctly categorized")

    def test_explicit_generalization_tag(self):
        """Verifies that explicit generalization_split tags are honored."""
        tagged_manifest = {
            "tasks": [
                {"annotation_id": "t1", "generalization_split": "cross_task"},
                {"annotation_id": "t2", "generalization_split": "cross_website"},
                {"annotation_id": "t3", "generalization_split": "cross_domain"},
            ]
        }
        splits = get_generalization_splits(tagged_manifest)
        self.assertEqual(len(splits["cross_task"]), 1)
        self.assertEqual(len(splits["cross_website"]), 1)
        self.assertEqual(len(splits["cross_domain"]), 1)
        print("PASS: test_explicit_generalization_tag - Explicit tags honored")


if __name__ == "__main__":
    unittest.main()
