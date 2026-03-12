.PHONY: run down logs slurm-darknet slurm-ultra slurm move repeat

PY ?= python3

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
	@$(PY) tools/move_runs.py

# Repeat sequential runs with nohup: default N=10, override with `make repeat N=5`
#make repeat N=10 LOG=mylog.txt
repeat:
	@N=$${N:-10}; \
	LOG=$${LOG:-repeat.log}; \
	nohup sh -c 'i=1; while [ $$i -le '$$N' ]; do echo "Run $$i / '$$N'"; make run; i=$$((i+1)); done' \
		> "$$LOG" 2>&1 &
	@echo "Started in background with nohup. Log: $$LOG"
