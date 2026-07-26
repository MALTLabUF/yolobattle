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
