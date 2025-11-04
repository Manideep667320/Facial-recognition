# Reproducibility & Environment Notes

Project: Voyex Facial Recognition v1 - Submission by <Your Name / Handle>

## 1 — Environment
- OS: Ubuntu 22.04 LTS (recommended)
- GPU: NVIDIA (e.g., RTX 3080 with 10-12 GB RAM) — report exact GPU in your submission.
- Python: 3.10
- PyTorch & torchvision: versions from `requirements.txt` (see sha256 below)
- Install:
  - Preferred: create a virtualenv or conda env then `pip install -r requirements.txt`
  - If using a specific CUDA build, install matching PyTorch wheel (e.g. from pytorch.org).

## 2 — Repository snapshot
- Commit / snapshot: include the git commit hash in `01_readme/commit.txt` and a zip of the repo in the Drive deliverable.
- Checksum for main checkpoint:
  - `sha256sum checkpoints/best.pth` saved to `10_verification/best.sha256`

## 3 — Data
- I used the provided dataset only: `data/train`, `data/val`, `data/test_public`, `data/test_private` (no external face-ID datasets).
- A data manifest with sha256 for every file is included in `04_data_manifest/dataset_manifest.json`.

## 4 — Exact commands to reproduce (assumes data present)
### 4.1 Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
### 4.2 Training
```bash
python src/train.py --config config.yaml
```
### 4.3 Validation
```bash
python src/validate.py --config config.yaml
```
### 4.4 Submission
```bash
python tools/validate_submission.py submission.json data/test_public class_mapping.json
```
