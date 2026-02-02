# yolobattle

YOLO training and benchmarking tools.

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
