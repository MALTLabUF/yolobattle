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
- LegoGearsDarknet
- LegoGearsUltra
- LeatherDarknet
- LeatherUltra
- FisheyeTrafficDarknetLocal
- FisheyeTrafficDarknetLocalJPG
- FisheyeTrafficUltralyticsLocal
- CubesDarknet
- CubesUltra
- CardsDarknet
- CardsUltra
