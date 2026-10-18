# IIT Delhi HPC training

The course allocation is `col764.course`. The supplied PBS scripts request the
`standard` queue with four CPUs and 8 GB RAM on a Genoa node. Verify access with
the smoke job before starting the sweep. Training belongs on scheduled compute
nodes; use the login node for setup, file transfer, and job submission.

Use an assignment directory in your own HPC home, with the repository layout
and an isolated `.venv`. Python 3.9 is available on the current login node;
the code also runs on the grader's Python 3.12. Pinning the four top-level
dependencies in `requirements.txt` keeps both environments compatible.

The login node may have no outbound internet. In that case download Linux
x86_64 CPython 3.9 wheels on an internet-connected computer, transfer them,
and install with `pip install --no-index --find-links <wheel-directory>`.
Use PyTorch's official CPU wheel index; no GPU is required. Cross-platform
`pip download` may omit `tomli` and `exceptiongroup` because it evaluates
environment markers using the downloading interpreter; include these for 3.9.

From a login shell, in the assignment directory:

```bash
qsub scripts/hpc_smoke.pbs
qstat -u "$USER"
# After smoke tests pass:
qsub scripts/hpc_cv.pbs
# After all eight CV subjobs finish successfully:
qsub scripts/hpc_train.pbs
```

The CV array permits at most four simultaneous four-CPU jobs. Each job fits
five grouped folds for one regularization configuration and evaluates fixed
tree-count checkpoints, without early stopping on the reported folds.
The final job collects the results, refits the selected LambdaMART model,
selects neural stopping epochs on an internal TRAIN fold, refits both neural
models on full TRAIN, selects fusion on TRAIN, and finally evaluates DEV.

Copy back `submission/model.txt`, both `submission/*_model.pt` files,
`submission/fusion_config.json`, and `results/`. Save the PBS output logs
locally. Real TRAIN/DEV files and downloaded packages stay outside Git.

The submitted transfer recipe runs independently, without PBS, network access,
or any dependency on these HPC scripts. Its limit remains 20 minutes on four
threads per benchmark.
