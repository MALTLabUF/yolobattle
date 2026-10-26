# yolobattle

YOLO training and benchmarking tools.

## Setup

### General
``` 
make pip
make run
```

### HPC Environments

#### HiPerGator

```bash 
module load python/3.12
python3 -m venv venv
source venv/bin/activate
make pip
make slurm
```

#### Rivanna

```bash
module load miniforge/24.11.3-py3.12
python3 -m venv venv
source venv/bin/activate
make pip
make slurm
```

## CLI

- `yolobattle -m train --profile <PROFILE>`
- `yolobattle train --profile <PROFILE>`

### Where to configure a benchmark

Each dataset has one configuration module under
[`src/yolobattle/model_training/benchmarks`](src/yolobattle/model_training/benchmarks).
Its dataset recipe, comparison policy, and training profiles live together:

| Dataset | Configuration file |
| --- | --- |
| Arabic handwriting | [arabic_handwriting.py](src/yolobattle/model_training/benchmarks/arabic_handwriting.py) |
| Leather | [leather.py](src/yolobattle/model_training/benchmarks/leather.py) |
| Lego Gears | [lego_gears.py](src/yolobattle/model_training/benchmarks/lego_gears.py) |
| Cubes | [cubes.py](src/yolobattle/model_training/benchmarks/cubes.py) |
| Playing cards | [cards.py](src/yolobattle/model_training/benchmarks/cards.py) |
| Local fisheye traffic | [fisheye_traffic.py](src/yolobattle/model_training/benchmarks/fisheye_traffic.py) |
| FishEye8K | [fisheye8k.py](src/yolobattle/model_training/benchmarks/fisheye8k.py) |

Within that file:

- `DatasetRecipe` defines the download URL, checksum, classes, and image/label layout.
- `BenchmarkPolicy` defines dimensions, split seed/fraction, training iterations,
  and shared evaluation rules.
- `BENCHMARK_PROFILES` defines the backend, model choices, and training parameters.
  `LEGACY_SWEEP_PROFILES` contains existing experimental variants of those profiles.

For example, Arabic handwriting's `templates=("yolov4",)` selects Darknet YOLOv4.
With `"templates"` in `sweep_keys`, changing it to `("yolov4", "yolov7")` runs both.
Ultralytics profiles select models using `ultra_model`, or
`sweep_values["ultra_model"]` when `"ultra_model"` is in `sweep_keys`.
The PyTorch YOLOv4 profile uses `pytorch_cfg`.

For archives with nested directories, set `DatasetRecipe.archive_subdir` to
the dataset's relative path inside the ZIP. Arabic handwriting uses
`"mnt/lustre/users/nalaas/nn/handWM6"`. `root` remains the extraction directory;
`names`, `flat_dir`, and supplied split directories are relative to
`root / archive_subdir` (`DatasetSpec.content_root`). Explicit layouts are
preserved without moving files. The loader reuses a completed extraction when
that directory is ready, including downloads made before this setting existed.
The URL and archive checksum remain unchanged. With no `archive_subdir`, existing
dataset normalization continues to apply.

To add a dataset, create a module following this structure, then import it and
add it to `BENCHMARK_MODULES` in
[`profile_registry.py`](src/yolobattle/model_training/profile_registry.py).
Provide both profile dictionaries (use `{}` for unused categories). Registry
keys must match each profile's `name`; duplicate names are rejected.
Reusable classes and helpers stay in `benchmark_definitions.py`,
`benchmark_policy.py`, and `profile_models.py`. These files need no changes
when adding a dataset. Existing imports of moved constants remain supported
for compatibility; new code should import settings from the dataset module.

### Ad hoc Darknet project folders

Use an existing Darknet project without adding a named profile:

```bash
yolobattle train --adhoc /path/to/handWM6
```

The folder must contain one `.data` file specifying `classes`, `names`,
`train`, and `valid`. The names file, image lists, images, and YOLO `.txt`
labels must be accessible inside the container. Without `--template`, the
folder must also contain a `.cfg`; YoloBattle trains that cfg unchanged.

To train a different Darknet model, select its cfg template, for example
`--template yolov4-custom`. This treats `yolov4-custom` as a model choice like
`yolov4` or `yolov7`. The folder's cfg is ignored and may be absent. The
selected template supplies the network, image size, batch settings, learning
rate, and update budget. YoloBattle sets the dataset class count and detection
filters, recalculates anchors from the training labels, updates the learning
rate schedule, and writes the resulting cfg to the run output. Use
`--iterations` and `--learning-rate` to override template defaults.

The existing train/validation lists are preserved, with their paths normalized
in the output folder. There is no dataset download, re-splitting, or budget
equalization.
Source files are not modified. Checkpoints and the usual benchmark reports are
written under `/outputs/adhoc_<folder>/...` in
the container, or `artifacts/outputs/adhoc_<folder>/...` when run directly.

Adjacent `.txt` annotations and separate `images/` / `labels/` directories
use the same label-resolution helper as dataset-backed profiles. Where needed,
Darknet receives a run-local adjacent-label view under `darknet_inputs/`, while
evaluation retains the original image paths. That input view is excluded from
the benchmark ZIP. `adhoc.json` records the selected template or source cfg
hash and any path-prefix relocations.

For the Singularity command, after updating the SIF's training code:

```bash
DATASET="$HOME/handWM6"  # change to your dataset folder
SINGULARITYENV_APPTAINER_ENVIRONMENT=1 singularity run --nv \
  --bind "$HOME/yolobattle-workspace:/workspace" \
  --bind "$HOME/yolobattle-outputs:/outputs" \
  --bind "$DATASET:$DATASET:ro" \
  "$HOME/yolobattle-darknet-legogears-offline.sif" \
  --adhoc "$DATASET"
```

An already-built SIF needs updated YoloBattle code to accept `--template` with
`--adhoc`. Rebuild it from the updated checkout. For an existing ad hoc cfg,
you can instead copy the updated `yolobattle/src` directory to the HPC host
and add `--bind "$HOME/yolobattle/src:/opt/app/src:ro"` before the SIF filename
(adjust the host path to your checkout). The `yolov4-custom` offline template
also requires the newly built image, because the old image lacks that asset.

The host-side wrapper also supports this option and mounts the folder at its
original absolute path:

```bash
yolobattle apptainer run --image "$HOME/yolobattle-darknet-legogears-offline.sif" \
  --adhoc "$HOME/handWM6"
```

If there are multiple data files, select one with `--data-path handWM6.data`.
Without `--template`, select among multiple cfg files with
`--cfg-path handWM6.cfg`. Paths may be relative to the selected folder or
absolute. Training starts from scratch unless you explicitly supply compatible
initial weights with `--weights`.
Without `--template`, Darknet training overrides such as `--iterations` are
rejected; edit a copy of the cfg to change its training settings. With
`--template`, `--iterations`, `--learning-rate`, and `--color-preset` can change
the generated cfg. `--num-gpus` is supported in either mode.

To train the mounted `handWM6` dataset with the bundled `yolov4-custom` model:

```bash
DATASET="$HOME/handWM6"  # change to your dataset folder
SINGULARITYENV_APPTAINER_ENVIRONMENT=1 singularity run --nv \
  --bind "$HOME/yolobattle-workspace:/workspace" \
  --bind "$HOME/yolobattle-outputs:/outputs" \
  --bind "$DATASET:/datasets/handWM6:ro" \
  "$HOME/yolobattle-darknet-legogears-offline.sif" \
  --adhoc /datasets/handWM6 \
  --template yolov4-custom \
  --iterations 30000
```

Set `--iterations` to the update budget you want. Without it, the selected
template's `max_batches` is used.

To train the same dataset with Ultralytics instead, choose its backend:

```bash
yolobattle train --adhoc /path/to/handWM6 \
  --adhoc-backend ultralytics \
  --ultra-model yolo11n.yaml
```

This reuses the `.data` class names and train/validation image lists, and
supports adjacent YOLO labels or the standard `images/` and `labels/` layout.
YoloBattle writes an Ultralytics dataset YAML and label-compatible run-local
input view; it does not modify the source dataset. The Darknet `.cfg` supplies
image size and batch; its learning rate is used as Ultralytics `lr0`, and
`max_batches` is translated into an epoch budget. Its network architecture,
anchors, augmentation settings, and learning-rate schedule are not translated.
`yolo11n.yaml` selects the Ultralytics architecture and trains it from scratch.
The Ultralytics SIF bundles `yolo26n.pt` only for Ultralytics' AMP self-check;
it is not used to initialize this from-scratch training run.
Ultralytics ad hoc runs also accept `--learning-rate`,
`--iterations` (translated to epochs), `--training-seed`, `--ultra-model`,
and `--imgsz 1280`. Ultralytics training uses one square target size; when
letterboxed, a 1280x960 frame fits the 1280 target without being shrunk.
Training augmentations can still transform it. With an explicit `--imgsz`,
the reported input size and evaluation target match the training target.
For a Slurm batch, set `imgsz: [1280]` and a single model such as
`ultra_model: ["yolo11l.pt"]` in `slurm/ultralytics/config.batch.yaml`.
Use `--fliplr 0 --flipud 0` to disable horizontal and vertical flips for
letter datasets where mirroring changes the class. Both options accept a
probability from 0 to 1; omitting one keeps the Ultralytics default. They also
work with named Ultralytics profiles and Slurm batch experiments. A color
preset does not change either flip probability.
Rebuild the Ultralytics SIF from this source to use these arguments with a
direct `singularity run` command.

For a cluster run, use the Ultralytics SIF and bind the dataset at a path
visible inside the container:

```bash
DATASET="$HOME/handWM6"
SINGULARITYENV_APPTAINER_ENVIRONMENT=1 singularity run --nv \
  --bind "$HOME/yolobattle-workspace:/workspace" \
  --bind "$HOME/yolobattle-outputs:/outputs" \
  --bind "$DATASET:/datasets/handWM6:ro" \
  "$HOME/yolobattle-ultralytics.sif" \
  --adhoc /datasets/handWM6 \
  --adhoc-backend ultralytics \
  --ultra-model yolo11n.yaml \
  --fliplr 0 --flipud 0
```

Relative paths resolve beside the referring file, then relative to the project
folder. For a self-contained project moved from another machine, ad hoc mode
automatically relocates absolute paths using the local names/split files and
the project directory name, preserving the paths below the project root.
An optional explicit mapping overrides automatic resolution, for example
`--path-map /old/location/handWM6=/datasets/handWM6`.
Mappings also support Windows prefixes such as `--path-map 'C:/old/handWM6=/datasets/handWM6'`
and may be repeated. With `yolobattle apptainer run`, put these training options
after `--`. Use `--profile` or `--adhoc`.

#### Moving a dataset between machines and containers

`--adhoc` discovers the project files and automatically resolves relocated
paths for a self-contained dataset. It infers old roots by matching the names
and split files to files under the selected folder. Image paths are resolved
using those roots or an exact project-directory component, keeping the nested
directory structure. This applies to `.data` and every image entry in both
train/validation lists, replacing the three manual `sed` edits.
The originals remain unchanged; corrected copies are written in the run's
output directory. The `.data` backup directory is redirected there as well.

For example, if the files still reference `/home/nisreen/nn/handWM6` but now
live at `/mnt/lustre/users/nalaas/nn/handWM6`, mount that host folder at
`/datasets/handWM6` and supply that **container path** to `--adhoc`:

```bash
SINGULARITYENV_APPTAINER_ENVIRONMENT=1 singularity run --nv \
  --bind "$HOME/yolobattle-workspace:/workspace" \
  --bind "$HOME/yolobattle-outputs:/outputs" \
  --bind "$HOME/yolobattle/src:/opt/app/src:ro" \
  --bind "/mnt/lustre/users/nalaas/nn/handWM6:/datasets/handWM6:ro" \
  "$HOME/yolobattle-darknet-legogears-offline.sif" \
  --adhoc /datasets/handWM6
```

The source bind assumes the updated checkout was copied to
`$HOME/yolobattle`; omit it if the SIF already contains the updated code.
Paths partly edited to the new host location also resolve automatically when
they retain the same project-directory component (`handWM6` in this example).
Automatic relocation prefers matching files inside the selected project,
even if an older copy is still accessible elsewhere. It does not recursively
search for images by basename. If multiple local metadata paths match, it
reports ambiguity instead of choosing one.

Use `--path-map` for an ambiguous layout or files stored outside the project.
For example, `--path-map /home/nisreen/nn/handWM6=/datasets/handWM6` explicitly
selects that destination. Explicit mappings take priority and must point to
existing files. Multiple old prefixes can map to the same directory. Mappings replace
directory prefixes, not arbitrary substrings, and the longest matching prefix
wins. They do not create mounts: the mapped destination must be accessible
inside the container. Missing files produce an error before training starts.

With the `yolobattle apptainer run` wrapper, the automatic bind preserves the
host's absolute path. The same automatic relocation works inside that mount:

```bash
yolobattle apptainer run \
  --image "$HOME/yolobattle-darknet-legogears-offline.sif" \
  --adhoc /mnt/lustre/users/nalaas/nn/handWM6
```

## Docker

- `yolobattle docker build`
- `yolobattle docker run --profile <PROFILE>`

### PyTorch YOLOv4

`pytorch_yolov4` is a first-class backend built from the repaired
[jpfleischer/pytorch-YOLOv4](https://github.com/jpfleischer/pytorch-YOLOv4)
fork. It uses the same `DatasetSpec` and split-generation path as Darknet and
Ultralytics, then generates Tianxiaomo-format labels and a class-correct,
rectangular cfg.

```bash
yolobattle docker build --backend pytorch_yolov4
yolobattle docker run --profile LegoGearsPyTorchYOLOv4 --gpus 0
```

The image pins the fork revision and runs a 224×160 YOLOv4-tiny smoke test while
building. Training outputs, including the generated split, cfg, logs, and
resumable checkpoints, are written to `artifacts/outputs`. Darknet, Ultralytics,
and this backend use the same 224×160 LegoGears geometry.

### Benchmark backends

Framework-specific logic lives in `model_training/backends.py`. A backend owns
only preparation, its training command, native-log parsing, artifact lookup,
and COCO detection export. The shared runner owns COCO ground truth, COCOeval,
confusion matrices, benchmark CSV/YAML, and bundles. Add a backend by
implementing that adapter contract and registering it; do not add framework
branches to the shared benchmark stages.

The benchmark CSV/YAML includes `Peak GPU VRAM Used (MiB)`, the highest
device-wide VRAM reading for the selected training GPUs in `mylogfile.log`.
With multiple training GPUs, this is the largest reading on any one GPU, not
their sum. The value is blank when GPU mapping or usable samples are missing.
`GPU VRAM` is the device's total capacity.

Canonical profiles share immutable benchmark policies: geometry, split rule,
iteration budget, export confidence, NMS IoU, checkpoint selection, COCO IoUs,
and confusion-matrix thresholds. The policy fingerprint is written to each
benchmark CSV/YAML. Current framework-comparison pairs are:

| Dataset | Policy | Profiles | Geometry / budget |
| --- | --- | --- | --- |
| LegoGears | `legogears_224x160_v1` | Darknet, Ultralytics, PyTorch YOLOv4 | 224×160 / 7000 iterations |
| Leather | `leather_256x256_v1` | Darknet, Ultralytics | 256×256 / 7000 iterations |
| Fisheye Traffic (local) | `fisheye_traffic_960x736_v1` | Darknet, Ultralytics | 960×736 / 8000 iterations |
| FishEye8K | `fisheye8k_official_1280x1280_v1` | Darknet, Ultralytics | 1280×1280 / 8000 iterations; official train/test split |
| Cubes | `cubes_224x160_v1` | Darknet, Ultralytics | 224×160 / 7000 iterations |
| Cards | `cards_768x576_v1` | Darknet, Ultralytics | 768×576 / 6000 iterations |

Canonical profiles use `iterations` as the common training budget; backends
that require epochs derive them from the generated split and batch
configuration. Existing non-`Benchmark` profiles remain available for legacy
runs and parameter sweeps. LegoGears' legacy sweep fractions (10%, 15%, 20%,
and 80%) are declared by `legogears_224x160_v1`; its canonical comparison
fraction remains 20%.

Each canonical policy is paired with its framework-independent dataset recipe
in `model_training/benchmark_definitions.py`. Framework profiles provide only
the runtime mount path plus framework-specific training settings.

## Apptainer

- `yolobattle apptainer build --backend darknet`
- `yolobattle apptainer run --profile <PROFILE>`
- `yolobattle apptainer slurm --backend darknet`
- `yolobattle apptainer slurm --backend ultralytics --batch`

### Download the Darknet SIF from GitHub Actions

The manual `Build Darknet Apptainer image` workflow builds
`apptainer/darknet/apptainer.def` on an Ubuntu GitHub-hosted runner. Open the
repository's **Actions** tab, select the workflow, choose **Run workflow**, and
download the artifact from the completed run. Leave the
`include_legogears_offline` input set to `false` for the existing online image;
its artifact contains `yolobattle-darknet.sif` and its SHA-256 checksum.

On an HPC system, verify and use the downloaded image with:

```bash
sha256sum --check yolobattle-darknet.sif.sha256
yolobattle apptainer run --image "$PWD/yolobattle-darknet.sif" --profile <PROFILE>
```

To build one self-contained offline image for the Lego Gears Darknet profiles,
set the `include_legogears_offline` workflow input to `true`. The workflow
downloads and packages the Lego Gears archive, the `yolov4-tiny`,
`yolov4-custom`, and `yolov7-tiny` cfg templates, and a Git bundle for the
selected Darknet ref and commit. The `yolov4-custom` template can also generate
a cfg for a dataset handled by a Darknet profile when selected with
`--template yolov4-custom`. The source is still compiled when the container
starts, so the build can use the GPU visible on the HPC node. The resulting
artifact contains `yolobattle-darknet-legogears-offline.sif` and its SHA-256
checksum.

Verify and run that image with:

```bash
sha256sum --check yolobattle-darknet-legogears-offline.sif.sha256
yolobattle apptainer run \
  --image "$PWD/yolobattle-darknet-legogears-offline.sif" \
  --profile LegoGearsDarknetBenchmark \
  --offline
```

The SIF also contains the training code and enables offline mode when bundled
assets are present, so it can be launched directly with Apptainer without the
host-side `yolobattle` Python wrapper:

```bash
mkdir -p offline-workspace artifacts/outputs
apptainer run --nv \
  --bind "$PWD/offline-workspace:/workspace,$PWD/artifacts/outputs:/outputs" \
  "$PWD/yolobattle-darknet-legogears-offline.sif" \
  --profile LegoGearsDarknetBenchmark
```

The first run extracts the bundled archive into the writable workspace. The
dataset and generated split files stay there; training outputs go to
`artifacts/outputs`.

The offline SIF uses only its bundled assets and errors if one is missing. The
wrapper does not auto-build a missing offline image. The bundled dataset supports
`LegoGearsDarknetBenchmark` and `LegoGearsDarknet`. With updated training code,
`--adhoc` can also use this image with your own mounted cfg and dataset, and
`--template yolov4-custom` generates a cfg in the run output as described
above. The regular SIF keeps the existing online behavior for other profiles.

## Slurm Batch (cloudmesh-ee API)

- Requires `cloudmesh-ee` and `cloudmesh-rivanna` installed in the active Python environment.
- Default batch template/config:
  - `slurm/<backend>/script.in.slurm`
  - `slurm/<backend>/config.batch.yaml`
- Generate and submit a batch:
  - `yolobattle apptainer slurm --backend darknet --batch`
- Generate only (no submit):
  - `yolobattle apptainer slurm --backend ultralytics --batch --batch-no-submit`
- Override config/source/output/name:
  - `yolobattle apptainer slurm --backend ultralytics --batch --batch-config path/to/config.yaml --batch-source path/to/script.in.slurm --batch-output-dir project --batch-name chocolatechip_runs`
 
## Profiles 
- LegoGearsDarknetBenchmark
- LegoGearsUltraBenchmark
- LegoGearsPyTorchYOLOv4
- LegoGearsDarknet (legacy validation-fraction sweep)
- LegoGearsUltra (legacy validation-fraction sweep)
- LeatherDarknetBenchmark
- LeatherUltraBenchmark
- LeatherDarknet
- LeatherUltra
- FisheyeTrafficDarknetBenchmark
- FisheyeTrafficUltraBenchmark
- FisheyeTrafficDarknetLocal
- FisheyeTrafficDarknetLocalJPG
- FisheyeTrafficUltralyticsLocal
- FishEye8KDarknetBenchmark
- FishEye8KUltraBenchmark
- FishEye8KDarknet
- FishEye8KUltralytics
- CubesDarknetBenchmark
- CubesUltraBenchmark
- CubesDarknet
- CubesUltra
- CardsDarknet
- CardsUltra
