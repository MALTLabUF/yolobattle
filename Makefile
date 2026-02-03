.PHONY: run down logs slurm-darknet slurm-ultra

run:
	yolobattle docker run --profile LegoGearsDarknet

ultra:
	yolobattle docker run --profile LegoGearsUltra

down:
	yolobattle docker stop

logs:
	yolobattle docker logs --follow

slurm-darknet:
	yolobattle apptainer slurm --backend darknet

slurm-ultra:
	yolobattle apptainer slurm --backend ultralytics

slurm:
	yolobattle apptainer run --profile LegoGearsDarknet


pip:
	pip install -e .
