# Assignment 3 report for submission #597

The final PDF is `../../2023BB10156.pdf` inside Rank-Fusion. Its layout follows
the user-supplied Assignment 2 report at `/Users/maazakhtar/COL/2023BB10156.pdf`.
That reference is left unchanged.

The report describes the evaluated V12 archive, commit
`f28b61880a56cac51566890671405db013bb3c94`, rather than the newer scorer
in the repository working tree. Reported staff outcomes come from the
user-provided #597 checkpoint saved in `results/leaderboard_v12.json`;
no final class rank or C1 result is inferred.

## Reproduce

From the Rank-Fusion repository:

1. Run `.venv/bin/python reports/submission_597/evaluate_sweep.py` to evaluate
   fixed 200/400/800-tree prefixes of the first query-rank component in the
   unchanged V12 artifact. This uses only the existing released DEV file.
   It verifies the archive, model, source and data hashes, and checks that
   the full scorer reproduces its archived DEV metric. It does not fit,
   select, or replace any model.
2. Run `build_report.py` with a Python environment containing ReportLab and
   the macOS Arial fonts. The bundled Codex Python runtime was used here.
3. Review the resulting three-page PDF visually. `report_evidence.json`
   records the exact archived metrics, CV summary, grader checkpoint,
   post-submission diagnostic and PDF hash.

The DEV sweep is explicitly dated 24 October 2026 and labeled retrospective
and in-sample: the tree used released TRAIN+DEV for fitting. Original CV
results are drawn from the V12 ZIP, not invented or rerun on DEV. The neural
fits used TRAIN only; pointwise stopping used an internal TRAIN holdout.

The report, generators and evidence do not alter any competition artifact,
prior ZIP, graded source or stored experiment result.
