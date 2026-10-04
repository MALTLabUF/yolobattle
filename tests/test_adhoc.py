"""Existing Darknet projects must retain their architecture and split."""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import ast
from dataclasses import replace
import importlib.util
import json
import os
import shlex
import types
import unittest
import zipfile
from unittest.mock import Mock, patch

from yolobattle.model_training.adhoc import load_adhoc_profile, prepare_adhoc


CFG = """# architecture and optimizer settings must survive verbatim
[net]
width=224
height=160
batch=8
subdivisions=4
max_batches=37
learning_rate=.002
letter_box=1
policy=steps
steps=20,30
[convolutional]
filters=18
activation=leaky
[yolo]
classes=1
anchors=10,14,23,27,37,58
[yolo]
classes=1
"""


class AdhocTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "hand WM6"
        self.root.mkdir()
        (self.root / "hand.cfg").write_text(CFG, encoding="utf-8")
        (self.root / "hand.names").write_text("Alif\n", encoding="utf-8")
        (self.root / "hand.data").write_text(
            "classes=1\nnames=hand.names\ntrain=hand_train.txt\nvalid=hand_valid.txt\nbackup=weights\n",
            encoding="utf-8",
        )
        images = self.root / "Set 01"
        images.mkdir()
        for name in ("one", "two", "three"):
            (images / f"{name}.jpg").touch()
        (self.root / "hand_train.txt").write_text("Set 01/two.jpg\nSet 01/one.jpg\n", encoding="utf-8")
        (self.root / "hand_valid.txt").write_text("Set 01/three.jpg\n", encoding="utf-8")
        self.out = Path(self.temporary.name) / "output"

    def load(self, **kwargs):
        return load_adhoc_profile(str(self.root), **kwargs)

    def test_existing_cfg_and_split_are_preserved_and_inputs_untouched(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        profile = self.load()
        self.assertEqual((profile.width, profile.height, profile.iterations), (224, 160, 37))
        self.assertEqual(profile.val_fracs, (1 / 3,))
        staged = prepare_adhoc(profile, self.out)
        self.assertEqual(Path(staged.cfg_out).read_bytes(), before[self.root / "hand.cfg"])
        self.assertEqual((self.out / "train.txt").read_text().splitlines(),
                         [str((self.root / "Set 01" / f"{name}.jpg").resolve()) for name in ("two", "one")])
        self.assertIn(f"backup = {self.out.resolve()}", Path(staged.data_path).read_text())
        self.assertEqual(json.loads((self.out / "dataset_split.json").read_text())["counts"],
                         {"train_total": 2, "valid_total": 1})
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        self.assertTrue(profile.darknet_project.letter_box)
        self.assertIsNone(profile.darknet_project.weights)

    def test_ambiguous_cfg_requires_selection(self):
        (self.root / "second.cfg").write_text(CFG)
        with self.assertRaisesRegex(ValueError, "Expected one .cfg"):
            self.load()
        self.assertEqual(self.load(cfg="hand.cfg").template, "hand")

    def test_path_map_relocates_metadata_and_images_from_windows(self):
        (self.root / "hand.data").write_text(
            "classes=1\nnames=C:\\old\\hand.names\ntrain=C:\\old\\hand_train.txt\nvalid=C:\\old\\hand_valid.txt\n")
        (self.root / "hand_train.txt").write_text("C:\\old\\Set 01\\one.jpg\n")
        profile = self.load(path_maps=[f"C:/old={self.root}"])
        self.assertEqual(profile.darknet_project.train, (str((self.root / "Set 01/one.jpg").resolve()),))

    def test_missing_image_fails_before_training(self):
        (self.root / "hand_valid.txt").write_text("/unmounted/missing.jpg\n")
        with self.assertRaisesRegex(ValueError, "path-map"):
            self.load()

    def test_migrated_hpc_paths_are_rewritten_in_data_and_both_lists(self):
        old = "/home/nisreen/nn/handWM6"
        host = "/mnt/lustre/users/nalaas/nn/handWM6"
        # self.root represents the directory visible inside the container.
        # Exercise untouched input files and files partly edited on the host.
        for image_prefix in (old, host):
            with self.subTest(image_prefix=image_prefix):
                (self.root / "hand.data").write_text(
                    f"classes=1\nnames={old}/hand.names\ntrain={old}/hand_train.txt\n"
                    f"valid={old}/hand_valid.txt\nbackup={old}/weights\n", encoding="utf-8")
                (self.root / "hand_train.txt").write_text(
                    f"{image_prefix}/Set 01/two.jpg\n{image_prefix}/Set 01/one.jpg\n", encoding="utf-8")
                (self.root / "hand_valid.txt").write_text(
                    f"{image_prefix}/Set 01/three.jpg\n", encoding="utf-8")
                before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
                profile = self.load()
                explicit = self.load(path_maps=[f"{old}={self.root}", f"{host}={self.root}"])
                self.assertEqual(profile.darknet_project.train, explicit.darknet_project.train)
                self.assertEqual(profile.darknet_project.valid, explicit.darknet_project.valid)
                staged = prepare_adhoc(profile, self.out)
                contents = Path(staged.data_path).read_text()
                self.assertIn(f"train = {self.out.resolve() / 'train.txt'}", contents)
                self.assertIn(f"valid = {self.out.resolve() / 'valid.txt'}", contents)
                self.assertIn(f"names = {self.out.resolve() / 'classes.names'}", contents)
                self.assertIn(f"backup = {self.out.resolve()}", contents)
                for key, names in (("train", ("two", "one")), ("valid", ("three",))):
                    contents += (self.out / f"{key}.txt").read_text()
                    self.assertEqual((self.out / f"{key}.txt").read_text().splitlines(),
                                     [str((self.root / "Set 01" / f"{name}.jpg").resolve()) for name in names])
                self.assertNotIn(old, contents)
                self.assertNotIn(host, contents)
                self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_auto_relocation_from_windows_to_a_renamed_container_folder(self):
        (self.root / "hand.data").write_text(
            "classes=1\nnames=C:\\old\\handWM6\\hand.names\n"
            "train=hand_train.txt\nvalid=hand_valid.txt\n")
        (self.root / "hand_train.txt").write_text("C:\\old\\handWM6\\Set 01\\one.jpg\n")
        self.assertEqual(self.load().darknet_project.train,
                         (str((self.root / "Set 01/one.jpg").resolve()),))

    def test_relative_metadata_can_relocate_images_using_the_project_directory(self):
        (self.root / "hand_valid.txt").write_text(f"/previous/machine/{self.root.name}/Set 01/three.jpg\n")
        self.assertEqual(self.load().darknet_project.valid,
                         (str((self.root / "Set 01/three.jpg").resolve()),))

    def test_auto_relocation_rejects_ambiguous_metadata(self):
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "hand.names").write_text("DifferentClass\n")
        data = self.root / "hand.data"
        data.write_text(data.read_text().replace("names=hand.names", "names=/old/nested/hand.names"))
        with self.assertRaisesRegex(ValueError, "Ambiguous relocated path"):
            self.load()
        profile = self.load(path_maps=[f"/old/nested={self.root}"])
        self.assertEqual(profile.darknet_project.names, ("Alif",))

    def test_explicit_mapping_is_authoritative_even_when_automatic_match_exists(self):
        data = self.root / "hand.data"
        data.write_text(data.read_text().replace("names=hand.names", "names=/old/hand.names"))
        self.assertEqual(self.load().darknet_project.names, ("Alif",))
        with self.assertRaisesRegex(ValueError, "Cannot find"):
            self.load(path_maps=[f"/old={self.root / 'does-not-exist'}"])

    def test_relocations_are_recorded_in_run_provenance(self):
        data = self.root / "hand.data"
        data.write_text(data.read_text().replace("names=hand.names", "names=/old/hand.names"))
        prepare_adhoc(self.load(), self.out)
        provenance = json.loads((self.out / "adhoc.json").read_text())
        self.assertIn(dict(source="/old", destination=self.root.resolve().as_posix(), method="automatic"),
                      provenance["path_relocations"])

    def test_repeated_image_paths_are_validated_once_without_changing_the_split(self):
        (self.root / "hand_train.txt").write_text("Set 01/one.jpg\n" * 3)
        (self.root / "hand_valid.txt").write_text("Set 01/one.jpg\n")
        checked = []
        original = Path.is_file

        def is_file(path):
            if path.name == "one.jpg":
                checked.append(path)
            return original(path)

        with patch.object(Path, "is_file", is_file):
            profile = self.load()
        self.assertEqual(len(checked), 1)
        self.assertEqual(len(profile.darknet_project.train), 3)
        self.assertEqual(len(profile.darknet_project.valid), 1)

    def test_auto_relocation_does_not_search_for_images_by_basename(self):
        (self.root / "hand_valid.txt").write_text("/unrelated/project/three.jpg\n")
        with self.assertRaisesRegex(ValueError, "Cannot find"):
            self.load()

    def test_path_map_does_not_match_a_different_directory_with_the_same_prefix(self):
        old = "/home/nisreen/nn/handWM6"
        (self.root / "hand_valid.txt").write_text(f"{old}_other/Set 01/three.jpg\n")
        with self.assertRaisesRegex(ValueError, "Cannot find"):
            self.load(path_maps=[f"{old}={self.root}"])

    def test_every_yolo_head_is_checked(self):
        (self.root / "hand.cfg").write_text(CFG + "[yolo]\nclasses=2\n")
        with self.assertRaisesRegex(ValueError, "Class counts"):
            self.load()

    def test_empty_lists_and_invalid_net_settings_are_rejected(self):
        for old, new in (("width=224", "width=0"), ("learning_rate=.002", "learning_rate=nan"),
                         ("subdivisions=4", "subdivisions=3")):
            with self.subTest(setting=new):
                (self.root / "hand.cfg").write_text(CFG.replace(old, new))
                with self.assertRaises(ValueError):
                    self.load()
        (self.root / "hand.cfg").write_text(CFG)
        (self.root / "hand_train.txt").write_text("")
        with self.assertRaisesRegex(ValueError, "empty"):
            self.load()

    def test_initial_weights_must_be_explicit_and_exist(self):
        weights = self.root / "initial.weights"
        weights.touch()
        self.assertIsNone(self.load().darknet_project.weights)
        self.assertEqual(self.load(weights=weights.name).darknet_project.weights, weights.resolve())
        with self.assertRaisesRegex(ValueError, "Cannot find"):
            self.load(weights="missing.weights")

    def test_backend_uses_supplied_cfg_and_only_this_runs_checkpoints(self):
        from yolobattle.model_training.backends import DarknetBackend, _darknet_checkpoints
        weights = self.root / "initial.weights"
        weights.touch()
        backend = DarknetBackend()
        with patch("yolobattle.model_training.backends.generate_cfg_file") as generate:
            staged, command = backend.prepare(self.load(weights=weights.name), template="hand",
                                               output_dir=self.out, gpu_indices=[0], gpus_str="0")
        generate.assert_not_called()
        argv = shlex.split(command.partition(" 2>&1")[0])
        self.assertEqual(argv[-3:], [staged.data_path, staged.cfg_out, str(weights.resolve())])
        self.assertEqual(backend.counts(staged, self.out), (2, 1, 148))
        self.assertTrue(all(p.parent == self.out.resolve() for p in _darknet_checkpoints(staged, "best")))
        final = self.out / "hand_final.weights"
        final.touch()
        best = self.out / "hand_best.weights"
        best.touch()
        with patch("yolobattle.model_training.backends.export_darknet_detections") as export:
            backend.export_coco(staged, output_dir=self.out, gt_json="gt.json", det_json="det.json",
                                valid_list=str(self.out / "valid.txt"), threshold=.01, gpu_indices=[0])
        self.assertEqual(export.call_args.kwargs["weights_path"], str(best.resolve()))
        self.assertTrue(export.call_args.kwargs["letter_box"])
        (self.out / "hand_last.weights").touch()
        backend.finalize(staged, self.out)  # Must not copy a checkpoint onto itself.

    def test_staged_split_supports_shared_coco_ground_truth(self):
        from PIL import Image
        from yolobattle.model_training.coco_build_gt import build_coco_gt_from_yolo_lists
        valid_image = self.root / "Set 01/three.jpg"
        Image.new("RGB", (20, 10)).save(valid_image)
        valid_image.with_suffix(".txt").write_text("0 0.5 0.5 0.4 0.2\n")
        prepare_adhoc(self.load(), self.out)
        gt = self.out / "gt.json"
        build_coco_gt_from_yolo_lists(list_file=str(self.out / "valid.txt"), out_json=str(gt),
                                      names_path=str(self.out / "classes.names"))
        document = json.loads(gt.read_text())
        self.assertEqual(len(document["images"]), 1)
        self.assertEqual(document["categories"][0]["name"], "Alif")
        self.assertEqual(document["annotations"][0]["bbox"], [6.0, 4.0, 8.0, 2.0])

    def test_separate_label_directories_use_native_views_without_changing_eval_paths(self):
        from yolobattle.model_training.backends import DarknetBackend, _data_values
        for split in ("train", "valid"):
            image = self.root / "images" / split / "same.jpg"
            label = self.root / "labels" / split / "same.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"image contents")
            label.write_text("0 0.5 0.5 0.4 0.2\n")
            (self.root / f"hand_{split}.txt").write_text(image.relative_to(self.root).as_posix() + "\n")
        staged, _ = DarknetBackend().prepare(self.load(), template="hand", output_dir=self.out,
                                            gpu_indices=[], gpus_str="")
        data = _data_values(staged.data_path)
        for split in ("train", "valid"):
            original = (self.root / "images" / split / "same.jpg").resolve()
            self.assertEqual((self.out / f"{split}.txt").read_text().strip(), str(original))
            native = Path(Path(data[split]).read_text().strip())
            self.assertTrue(native.is_relative_to(self.out.resolve() / "darknet_inputs"))
            self.assertEqual(native.read_bytes(), original.read_bytes())
            self.assertEqual(native.with_suffix(".txt").read_text(), "0 0.5 0.5 0.4 0.2\n")
        # Exercise the actual bundle block without the GPU training stages.
        import yolobattle.model_training as package
        tree = ast.parse(Path(package.__file__).with_name("train.py").read_text(encoding="utf-8"))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_once")
        bundle_block = next(node for node in run.body if isinstance(node, ast.With)
                            and isinstance(node.items[0].context_expr, ast.Call)
                            and isinstance(node.items[0].context_expr.func, ast.Attribute)
                            and node.items[0].context_expr.func.attr == "ZipFile")
        bundle_path = self.out / "bundle.zip"
        exec(compile(ast.Module(body=[bundle_block], type_ignores=[]), "bundle", "exec"),
             dict(Path=Path, os=os, zipfile=zipfile, output_dir=str(self.out), p=staged,
                  bundle=bundle_path.name, bundle_path=bundle_path))
        with zipfile.ZipFile(bundle_path) as archive:
            self.assertIn("train.txt", archive.namelist())
            self.assertFalse(any(name.startswith("darknet_inputs/") for name in archive.namelist()))

    def run_cli(self, *args):
        # Execute the actual CLI block with the GPU run boundary replaced. This
        # tests argument handling and routing without importing GPU drivers.
        import yolobattle.model_training as package
        source = Path(package.__file__).with_name("train.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        main = tree.body[-1]
        self.assertIsInstance(main, ast.If)
        run = Mock()
        namespace = dict(__name__="__main__", argparse=argparse, os=os, Path=Path,
                         replace=replace, run_once=run, get_profile=Mock(),
                         ensure_download_once=Mock(), build_split_for=Mock(), equalize_for_split=Mock())
        with patch("sys.argv", ["train", *args]), patch.dict(os.environ, {"YOLOBATTLE_CONTAINER": "1"}), \
                patch("os.makedirs"), patch("sys.stdout"), patch("sys.stderr"):
            with self.assertRaises(SystemExit) as raised:
                exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), "exec"), namespace)
        return raised.exception.code, namespace

    def test_cli_routes_single_run_without_download_split_or_budget_changes(self):
        code, namespace = self.run_cli("--adhoc", str(self.root), "--num-gpus", "2")
        self.assertEqual(code, 0)
        run = namespace["run_once"]
        run.assert_called_once()
        profile = run.call_args.kwargs["p"]
        self.assertEqual((profile.num_gpus, profile.iterations), (2, 37))
        self.assertTrue(run.call_args.kwargs["out_root"].endswith(os.path.join("outputs", "adhoc_hand_WM6")))
        for helper in ("get_profile", "ensure_download_once", "build_split_for", "equalize_for_split"):
            namespace[helper].assert_not_called()

    def test_cli_rejects_ambiguous_sources_and_cfg_overrides(self):
        for extra in (("--profile", "LegoGearsDarknetBenchmark"),
                      ("--custom-profile", "custom"),
                      ("--iterations", "100"), ("--val-frac", ".2")):
            with self.subTest(extra=extra):
                code, namespace = self.run_cli("--adhoc", str(self.root), *extra)
                self.assertEqual(code, 2)
                namespace["run_once"].assert_not_called()

    def test_apptainer_wrapper_binds_folder_and_passes_training_options(self):
        import yolobattle
        source = Path(yolobattle.__file__).with_name("apptainer.py")
        spec = importlib.util.spec_from_file_location("adhoc_apptainer_test", source)
        wrapper = importlib.util.module_from_spec(spec)
        # The experiment executor is unrelated to a direct container run.
        ee = types.ModuleType("cloudmesh.ee.experimentexecutor")
        ee.ExperimentExecutor = Mock()
        with patch.dict("sys.modules", {"cloudmesh.ee.experimentexecutor": ee}):
            spec.loader.exec_module(wrapper)
        client = Mock()
        image = Path(self.temporary.name) / "offline.sif"
        image.touch()
        with patch.object(wrapper, "_client", return_value=client), \
                patch.object(wrapper, "_repo_root", return_value=Path(self.temporary.name)), \
                patch.object(wrapper, "_clean_darknet_workspace"), \
                patch.object(wrapper, "_build_image") as build:
            wrapper.main(["run", "--adhoc", str(self.root), "--image", str(image), "--offline",
                          "--no-stream", "--", "--cfg-path", "hand.cfg", "--num-gpus", "2"])
        build.assert_not_called()
        self.assertIn(f"{self.root.resolve()}:{self.root.resolve()}:ro", client.run.call_args.kwargs["bind"])
        self.assertEqual(client.run.call_args.kwargs["args"],
                         ["--adhoc", str(self.root.resolve()), "--cfg-path", "hand.cfg", "--num-gpus", "2"])


if __name__ == "__main__":
    unittest.main()
