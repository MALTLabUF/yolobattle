.PHONY: run down logs slurm-darknet slurm-ultra slurm move

run:
	yolobattle docker run --profile LegoGearsDarknet

ultra:
	yolobattle docker run --profile LegoGearsUltra


build:
	yolobattle docker build

down:
	yolobattle docker stop

logs:
	yolobattle docker logs --follow

slurm:
	yolobattle apptainer slurm --backend darknet --batch

slurm-ultra:
	yolobattle apptainer slurm --backend ultralytics --batch

#slurm:
#	yolobattle apptainer run --profile LegoGearsDarknet


pip:
	pip install -e .

# Move local run artifacts into the runs-yolobattle repo
move:
	@if [ ! -d "artifacts/outputs" ]; then \
		echo "No artifacts/outputs directory found."; \
		exit 1; \
	fi
	@mkdir -p ../runs-yolobattle/outputs
	@python tools/move_runs.py
