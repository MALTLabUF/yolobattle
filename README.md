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
