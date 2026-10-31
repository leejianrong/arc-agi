# ARC-AGI-2 baseline notebook (Kaggle)

Stdlib-only baseline (geometry + color bijection, identity/flip fallbacks) that
writes the competition's `submission.json`. Licensed MIT-0 (notice at the top of
each file here), as the competition requires for released code.

- `solve.py` — the whole notebook body; finds `arc-agi_test_challenges.json`
  under `/kaggle/input` and writes `/kaggle/working/submission.json`.
- `score_local.py` — scores a submission against the public evaluation
  solutions (`kaggle competitions download arc-prize-2026-arc-agi-2`).
- `kernel-metadata.json` — private, internet-off script kernel;
  `kaggle kernels push -p kaggle/arc2_baseline`, then
  `kaggle competitions submit arc-prize-2026-arc-agi-2 -k <kernel> -v <n> -f submission.json -m <msg>`.

First submission (2026-10-30): local public-eval score 0.00%, expected for this
baseline; its job was proving the pipeline end to end.
