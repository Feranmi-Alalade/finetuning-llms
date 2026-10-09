.PHONY: help install data train eval clean

help:
	@echo "Available commands:"
	@echo "  make install                - Install dependencies with uv"
	@echo "  make data                   - Generate all training data"
	@echo "  make train LINEAGE=m0_ocr  - Train a specific lineage"
	@echo "  make eval LINEAGE=m0_ocr   - Evaluate a specific lineage"
	@echo "  make clean                  - Remove caches and temp files"

install:
	uv pip install -e ".[dev]"

data:
	uv run python pipelines/data/generate_teacher_labels.py
	uv run python pipelines/data/generate_stability_data.py
	uv run python pipelines/data/build_datasets.py

train:
	uv run python pipelines/training/train_lineage.py --lineage $(LINEAGE)

eval:
	uv run python pipelines/evaluation/eval_lineage.py --lineage $(LINEAGE)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +