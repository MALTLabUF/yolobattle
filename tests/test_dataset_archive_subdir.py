"""Explicit archive layouts survive fresh downloads and reuse of old caches."""
import ast
from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

import yolobattle.model_training as training_package
from yolobattle.model_training.benchmarks.arabic_handwriting import ARABIC_HANDWRITING_V1
from yolobattle.model_training.coco_gt_dispatch import build_coco_gt_for_dataset
from yolobattle.model_training.dataset_setup import IMG_EXTS, make_split
from yolobattle.model_training.datasets import MARKER_NAME, _looks_ready, ensure_download_once


def build_training_split(spec, output):
    # Run the actual split helper without importing the CLI's GPU dependencies.
    source = Path(training_package.__file__).with_name("train.py")
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "build_split_for")
    namespace = dict(Path=Path, make_split=make_split, IMG_EXTS=IMG_EXTS)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["build_split_for"](0.20, spec, out_dir=output)


class ArchiveSubdirTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "ArabicHandwriting"
        self.spec = ARABIC_HANDWRITING_V1.dataset_at(str(self.root))
        self.archive = self.base / "fixture.zip"
        png = BytesIO()
        Image.new("RGB", (32, 32)).save(png, format="PNG")
        with zipfile.ZipFile(self.archive, "w") as archive:
            prefix = self.spec.archive_subdir
            archive.writestr(f"{prefix}/handWM6.names", "Alif\nTa\nRa\nSin\nHea\n")
            archive.writestr(f"{prefix}/handWM6.cfg", "original config\n")
            for i in range(10):
                stem = f"{prefix}/{self.spec.flat_dir}/{i:08d}"
                archive.writestr(f"{stem}.png", png.getvalue())
                archive.writestr(f"{stem}.txt", f"{i % 5} 0.5 0.5 0.25 0.25\n")
        self.spec = replace(self.spec, sha256=hashlib.sha256(self.archive.read_bytes()).hexdigest())

    def download(self, url, destination):
        shutil.copy2(self.archive, destination)

    def test_fresh_download_split_and_evaluation_preserve_archive_layout(self):
        with patch.dict("os.environ", {"YOLOBATTLE_OFFLINE": "0"}), \
                patch("yolobattle.model_training.datasets._download", side_effect=self.download) as download:
            content = ensure_download_once(self.spec)
            self.assertEqual(content, self.root / self.spec.archive_subdir)
            download.assert_called_once()
        marker = json.loads((self.root / MARKER_NAME).read_text())
        self.assertEqual(marker["sha256"], self.spec.sha256)
        self.assertEqual(marker["profile_dataset"]["archive_subdir"], self.spec.archive_subdir)
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        output = self.base / "splits"
        data_file, _ = build_training_split(self.spec, output)
        data = dict(line.split(" = ", 1) for line in Path(data_file).read_text().splitlines())
        self.assertEqual(Path(data["names"]), content / "handWM6.names")
        train = Path(data["train"]).read_text().splitlines()
        valid = Path(data["valid"]).read_text().splitlines()
        self.assertEqual((len(train), len(valid)), (8, 2))
        self.assertFalse(set(train) & set(valid))
        self.assertTrue(all(Path(p).is_file() and Path(p).is_relative_to(content) for p in train + valid))
        gt = output / "gt.json"
        build_coco_gt_for_dataset(dataset=self.spec, valid_list=Path(data["valid"]), out_json=gt)
        document = json.loads(gt.read_text())
        self.assertEqual(len(document["images"]), 2)
        self.assertEqual(len(document["annotations"]), 2)
        self.assertEqual([c["name"] for c in document["categories"]], list(self.spec.class_names))
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        with patch("yolobattle.model_training.datasets._download") as download:
            self.assertEqual(ensure_download_once(self.spec), content)
            download.assert_not_called()

    def test_existing_extraction_with_old_marker_is_reused_offline(self):
        with zipfile.ZipFile(self.archive) as archive:
            archive.extractall(self.root)
        old_marker = json.dumps({"sha256": self.spec.sha256, "profile_dataset": {"root": str(self.root)}})
        (self.root / MARKER_NAME).write_text(old_marker)
        with patch.dict("os.environ", {"YOLOBATTLE_OFFLINE": "1"}), \
                patch("yolobattle.model_training.datasets._download") as download, \
                patch("yolobattle.model_training.datasets._extract") as extract:
            self.assertEqual(ensure_download_once(self.spec), self.spec.content_root)
            download.assert_not_called()
            extract.assert_not_called()
        self.assertEqual((self.root / MARKER_NAME).read_text(), old_marker)

    def test_wrong_explicit_path_keeps_existing_cache_without_redownload(self):
        self.test_existing_extraction_with_old_marker_is_reused_offline()
        with patch("yolobattle.model_training.datasets._download") as download:
            with self.assertRaisesRegex(FileNotFoundError, "check archive_subdir"):
                ensure_download_once(replace(self.spec, archive_subdir="wrong"))
            download.assert_not_called()
        self.assertTrue((self.spec.content_root / "handWM6.names").is_file())

    def test_wrong_fresh_layout_does_not_write_completion_marker(self):
        with patch.dict("os.environ", {"YOLOBATTLE_OFFLINE": "0"}), \
                patch("yolobattle.model_training.datasets._download", side_effect=self.download):
            with self.assertRaisesRegex(FileNotFoundError, "check archive_subdir"):
                ensure_download_once(replace(self.spec, archive_subdir="wrong"))
        self.assertFalse((self.root / MARKER_NAME).exists())
        self.assertTrue((self.spec.content_root / "handWM6.names").is_file())

    def test_missing_flat_directory_is_not_ready_with_empty_sets(self):
        self.root.mkdir()
        self.assertFalse(_looks_ready(self.root, self.spec))

    def test_subdirectory_cannot_escape_extraction_root(self):
        for subdir in ("../outside", "/absolute"):
            with self.subTest(subdir=subdir), self.assertRaises(ValueError):
                ensure_download_once(replace(self.spec, archive_subdir=subdir))

    def test_no_subdirectory_retains_flat_dataset_behavior(self):
        spec = replace(self.spec, archive_subdir=None)
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("handWM6.names", "Alif\nTa\nRa\nSin\nHea\n")
            archive.writestr("darkmark_image_cache/resize/one.txt", "")
        spec = replace(spec, sha256=hashlib.sha256(self.archive.read_bytes()).hexdigest())
        with patch.dict("os.environ", {"YOLOBATTLE_OFFLINE": "0"}), \
                patch("yolobattle.model_training.datasets._download", side_effect=self.download):
            self.assertEqual(ensure_download_once(spec), self.root)
        self.assertTrue((self.root / spec.flat_dir).is_dir())


if __name__ == "__main__":
    unittest.main()
