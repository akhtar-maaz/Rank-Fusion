# A3 — Rank Fusion and Learning to Rank

Python implementation of hand-written RRF, CombSUM, CombMNZ, pointwise regression,
and RankNet, plus LightGBM LambdaMART training and a transfer recipe.

## Current status

The hand-written components and training pipeline pass all 51 tests on macOS
(Python 3.12) and on an IIT Delhi HPC compute node (Python 3.9, four CPUs, 8 GB).
The real Fold1 TRAIN-only cross-validation sweep is running. The checked-in
model artifacts are still the starter's toy-trained artifacts until the real
training run finishes. The report will be written after results are frozen.

## Setup and checks

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/smoke_test.sh
```

On macOS, install OpenMP with `brew install libomp`. The smoke script makes
LightGBM share PyTorch's bundled OpenMP runtime, and tests a temporary copy:
it cannot overwrite the actual submitted models.

## Data

Use only MSLR-WEB10K Fold1 TRAIN and DEV. The official archive is linked from
[Microsoft Research](https://www.microsoft.com/en-us/research/project/mslr/).
Its published MD5 is `97c5d4e7c171e475c91d7031e4fd8e79`.

```bash
python scripts/prepare_mslr.py --archive /path/to/MSLR-WEB10K.zip --out data
# Alternatively, use an already prepared Fold1 directory:
python scripts/prepare_mslr.py --fold-dir /path/to/Fold1 --out data
```

Archive preparation opens only `Fold1/train.txt` and `Fold1/vali.txt`;
it never opens the held-out file or another fold. Real data, downloads,
environments, and intermediate runs are ignored by Git.

## Reproducible training

```bash
python scripts/cv_lambdamart.py --train data/train --out runs/lambdamart_cv.json
python scripts/train_model.py --train data/train --config runs/lambdamart_cv.json
python scripts/train_neural.py --train data/train --kind both --hidden 128 64 \
    --epochs 30 --patience 5 --history-out results/neural_history.json
python scripts/select_fusion.py --train data/train/train.txt --out results/fusion_selection.json
# Run only after configurations and models are frozen:
python scripts/evaluate_models.py --features data/dev/vali.txt
python scripts/run_recipe.py
```

CV groups documents by query and reports validation mean, standard deviation,
training accuracy, and the validation/training ratio at fixed tree counts.
No reported CV fold is used for early stopping. The neural script selects
epochs on an internal TRAIN fold, then refits on all TRAIN for that fixed
number of epochs. Both neural models use the same scoring architecture.
DEV is used only for final evaluation.

Fusion configuration lives in `submission/fusion_config.json`. The textbook
fusion functions remain independently testable. The transfer recipe handles
arbitrary feature widths, uses the supplied validation file for early
stopping, and runs on four threads. Training uses a two-pass float32 loader
to avoid retaining Python objects for every feature of the full dataset.

For HPC submission and offline dependency installation, see [docs/HPC.md](docs/HPC.md).
The grading harness is unchanged; the ranking interface preserves its exact
signatures and loads artifacts without training.

## Training-data declaration

| Artifact | Currently trained on | Selection / early stopping |
|---|---|---|
| `model.txt` | Starter toy data; real TRAIN fit pending | TRAIN-only grouped CV running |
| `pointwise_model.pt` | Starter toy data; real TRAIN fit pending | Internal TRAIN fold planned |
| `pairwise_model.pt` | Starter toy data; real TRAIN fit pending | Internal TRAIN fold planned |
| `recipe.py` | Staff-supplied benchmark TRAIN when invoked | Its supplied validation split |

## AI-use disclosure

OpenAI Codex assisted with reading the specification, implementing and reviewing
the algorithms, writing tests and training scripts, and configuring HPC execution.

## Code provenance

The original starter supplies the unchanged harness, MLP, and standardizer.
Fusion and neural-loss formulas follow the assignment PDF and starter primer;
no external fusion or ranking-training implementation was copied. The
[Microsoft dataset page](https://www.microsoft.com/en-us/research/project/mslr/)
and [IIT Delhi HPC guide](https://supercomputing.iitd.ac.in/) were consulted
for dataset access and scheduled execution.
