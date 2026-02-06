# yolobattle

YOLO training and benchmarking tools.

## Setup

### General
``` 
make pip
make run
```

### HPC Environments
``` 
module load python/3.12
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
