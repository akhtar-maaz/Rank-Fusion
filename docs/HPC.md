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
.venv/bin/python scripts/collect_cv.py runs/cv/part_{0..7}.json \
    --out runs/lambdamart_cv_initial.json
# The initial sweep's best result was at its 400-tree upper boundary.
# Extend that configuration on the same TRAIN folds, without consulting DEV:
qsub scripts/hpc_extend.pbs
# After the extension completes successfully:
qsub scripts/hpc_train.pbs
```

The CV array permits at most four simultaneous four-CPU jobs. Each job fits
five grouped folds for one regularization configuration and evaluates fixed
tree-count checkpoints, without early stopping on the reported folds.
The bounded extension checks 800 and 1200 trees for the initial winner.
The final job collects the results, refits the selected LambdaMART model,
selects neural stopping epochs on an internal TRAIN fold, refits both neural
models on full TRAIN, selects fusion on TRAIN, and finally evaluates DEV.

Copy back `submission/model.txt`, both `submission/*_model.pt` files,
`submission/fusion_config.json`, and `results/`. Save the PBS output logs
locally. Real TRAIN/DEV files and downloaded packages stay outside Git.

After copying back and committing the trained artifacts and results, create
the upload archive from tracked repository files:

```bash
python scripts/package_submission.py
```

The ZIP retains the original repository structure and toy datasets, and the
packager refuses tracked real TRAIN/DEV files, downloads, or environments.

The submitted transfer recipe runs independently, without PBS, network access,
or any dependency on these HPC scripts. Its limit remains 20 minutes on four
threads per benchmark.

## Second-round experiments and frozen artifact reproduction

The root README provides direct commands for reproducing the selected final
artifacts from the committed configuration. The batch scripts below continue
the original workspace, retaining its first-round CV parts and baseline
artifacts. For a fresh search, first generate those original parts and retain
the first-submission neural artifacts (commit `6533171`) as the baseline.

```bash
qsub scripts/hpc_cv_v2.pbs          # candidates 0-2
qsub scripts/hpc_cv_rank_large.pbs  # candidate 3, bounded at 800 trees
qsub scripts/hpc_cv_focus.pbs       # reported depth-limited candidate 4 only
qsub scripts/hpc_neural_v2.pbs      # candidates 0-2
qsub scripts/hpc_neural_warmup.pbs  # candidates 3-4
```

Run the weighted fusion comparison locally or inside a compute allocation,
then place its two JSON outputs in the HPC work directory's `runs/`:

```bash
python scripts/select_fusion.py --train data/train/train.txt --weighted \
    --out runs/fusion_selection_v2.json --config-out runs/fusion_config_v2.json
```

After candidate 3's CV completes, `hpc_prefit_v2.pbs` optionally fits that
current winner on full TRAIN while other CV jobs finish. The final job reuses
it only when its parameters, feature mode, dataset counts, and SHA-256 match
the selected configuration. After all five reported tree parts and all five
neural candidates complete:

```bash
qsub scripts/hpc_train_v2.pbs
qsub scripts/hpc_verify.pbs  # final frozen-model and conformance checks
```

The final job retains the first-round records, selects from 49 complete tree
configurations, promotes the TRAIN-selected neural artifact, evaluates DEV,
and verifies an independent full-TRAIN tree refit. `hpc_verify.pbs` rechecks
that proof, evaluates the frozen models, runs synthetic transfer checks, and
tests a temporary copy. It performs no real-data training. An optional
query-relative depth-limited CV variant was cancelled for runtime and is not
part of the reported results.

## Third-round TRAIN robustness comparison

V3 retains the v2 fusion and both neural artifacts exactly. The selected
tree configuration and adaptive recipe are described in the root README.
The reported search can be replayed with:

```bash
qsub scripts/hpc_cv_v3.pbs  # control / normalized, all five TRAIN folds
# After both subjobs finish, including writing their OOF .npz files:
qsub scripts/hpc_blends_v3.pbs
# After the blend report is complete:
qsub scripts/hpc_train_v3.pbs
```

The final job applies the fixed mixture/clean-accuracy guard, refits the
800+200-tree blend, evaluates frozen DEV, independently refits from TRAIN,
checks identical native-model hashes, runs synthetic transfer checks, checks
protected v2 hashes, and tests a temporary copy. A fresh workspace needs
the committed `results/lambdamart_cv_v2.json` and `results/artifacts_v2.json`.
For artifact reproduction alone, use the root README's `train_model.py`
command with the committed final configuration; CV need not be repeated.

`hpc_recipe_v3.pbs` compares the adaptive C1 policy with v2 on three
TRAIN-derived nested holdouts. `hpc_recipe_budget_bounded_v3.pbs` checks
the final row/width-dependent tree ceiling on a near-maximum derived input.
The initial wide budget run and slow regularization probes are retained in
the execution record. The optional bounded regularization array was also
cancelled for runtime; its incomplete results are excluded, and the final
selector requires candidates 2/3 to be explicitly marked excluded.

Copy back the new `model.txt`, final metrics/reproduction/manifest records,
recipe smoke records and PBS logs. Verify model hashes across HPC and local
copies with `scripts/record_artifacts.py`. The v2 neural and fusion files
must retain their hashes. The final upload archive is
`dist/rank-fusion-submission-v3.zip`.

## Fourth-round workflow

`hpc_cv_v4.pbs` runs four fixed-count five-fold TRAIN experiments. The
dependent `hpc_select_v4.pbs` evaluates the frozen 36-setting comparison.
After promotion, `hpc_train_v4.pbs` fits the artifact and evaluates frozen
DEV, while `hpc_reproduce_v4.pbs` independently refits from the same config.
`hpc_verify_v4.pbs` depends on both, checks byte identity, protected v3 hashes,
native/staff loading and the temporary-copy smoke tests. Each job receives
four CPU threads and 8 GB. Heavy fitting runs only in scheduled compute jobs.
Actual job IDs, outcomes and resource use are in `results/execution.json`.
The earlier held sequential job was replaced before execution by these
parallel refits; it did not consume training data or produce an artifact.

## Fifth-round workflow

`hpc_cv_v5.pbs` uses `configs/v5_candidates.json` for four fixed-count
five-fold TRAIN experiments. The dependent `hpc_select_v5.pbs` compares the
fixed 54-setting plan with v4 and verifies native export of the winner.
Only after promotion, `hpc_train_v5.pbs` fits the final artifact and evaluates
frozen DEV while `hpc_reproduce_v5.pbs` independently refits it.
`hpc_verify_v5.pbs` depends on both and checks byte identity, protected hashes,
native/staff loading and smoke tests. Every job receives four threads and
8 GB. Actual IDs, runtime exclusions and resource use are in
`results/execution.json`. No heavy fitting runs on the login node.
`hpc_seed_check_v5.pbs` measures both frozen configurations under perturbation
seed 271828 on TRAIN validation folds; it never changes model selection.

## Sixth-round workflow

`hpc_cv_v6.pbs` and `hpc_neural_v6.pbs` run the frozen four-candidate tree
and pairwise TRAIN experiments. Dependent selection jobs validate all
complete folds, reproduce v5 controls and apply frozen promotion guards.
`hpc_train_v6.pbs` and `hpc_reproduce_v6.pbs` independently refit promoted
choices; retained files are copied byte-for-byte from the v5 checkpoint.
The primary job evaluates DEV once after both choices are fixed.
`hpc_seed_check_v6.pbs` measures a promoted tree choice under a second seed,
and `hpc_verify_v6.pbs` checks reproduction, staff loading, protected hashes
and the full smoke suite. Actual IDs, resources and outcomes are in
`results/execution.json`. Heavy fitting runs only on scheduled compute nodes.

Reproduce the current pairwise artifact from the frozen five-fold selection:

```bash
python scripts/fit_pairwise_v6.py --train data/train \
  --selection results/neural_selection_v6.json \
  --out submission/pairwise_model.pt --history-out runs/pairwise_refit.json
```

This uses ten supervised MSE warm-up epochs followed by two weighted
RankNet epochs; it does not repeat selection or load DEV. Earlier neural
commands above describe the archived v2 checkpoint.

## Seventh-round workflow

`hpc_cv_v7.pbs` runs four scheduled tree candidates; `hpc_select_v7.pbs`
checks complete folds, exact v6 control and all promotion gates.
One `hpc_neural_v7.pbs` job fits the three seeds within each of five TRAIN
folds and selects among the three frozen neural candidates. This is not
a neural array or a separate neural-selection job. Primary/independent jobs
fresh-fit promoted configurations and preserve exact v6 bytes for either
fallback. Only the primary job evaluates DEV after both choices are frozen.
The seed job is descriptive and the verify job checks reproduction,
protected hashes, pristine loading and the full smoke suite.
`results/execution.json` records ten actual jobs (four array children and
six single jobs), saved PBS requests and actual CPU/memory/walltime per job.
Tree fitting requests 12 GB; neural CV, selection and verification use their
saved per-workflow requests. Heavy fits run only on scheduled compute nodes.

## Eighth-round workflow

`hpc_cv_v8.pbs` runs four scheduled tree candidates; `hpc_select_v8.pbs`
checks complete folds, exact v7 control and all promotion gates.
One `hpc_neural_v8.pbs` job fits nine distinct complete member specifications
within each TRAIN fold (45 fits total) and selects among five frozen
neural candidates, including compiled averages and a direct dense singleton. This is not
a neural array or a separate neural-selection job. Primary/independent jobs
fresh-fit promoted configurations and preserve exact v7 bytes for either
fallback. Only the primary job evaluates DEV after both choices are frozen.
The seed job is descriptive and the verify job checks reproduction,
protected hashes, pristine loading and the full smoke suite.
`results/execution.json` records ten actual jobs (four array children and
six single jobs), saved PBS requests and actual CPU/memory/walltime per job.
Tree fitting requests 12 GB; neural CV, selection and verification use their
saved per-workflow requests. Heavy fits run only on scheduled compute nodes.

## Ninth-round workflow

`hpc_cv_v9.pbs` schedules four planned candidates. Selection validates all
completed folds from 49 planned comparisons and the exact V8 control before
frozen gates. The actual completed count and any explicit whole-candidate
runtime exclusions appear in the selection and execution records.
Primary/independent jobs fresh-fit a promoted tree or copy exact V8 bytes.
Only the primary job evaluates DEV after the TRAIN-only choice is frozen.
The descriptive seed job uses 902617 if promoted and otherwise records a skip.
Verification checks byte identity, protected neural/core/fusion/recipe hashes,
staff loading and the full smoke suite. No V9 neural fitting or selection job
exists: four CV children plus selection, primary, independent, seed and verify
make nine actual jobs. Requests and observed CPU/memory/walltime are recorded
per job in `results/execution.json`. Heavy fits use scheduled compute nodes.
