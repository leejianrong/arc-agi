# Scientific validity controls

## Dataset boundary

`arc_env/dataset_splits.json` is the reviewable source of truth for ARC-AGI-1 data use. It pins the
development and locked-evaluation directories by task count and a canonical SHA-256 over every
relative filename and file body. Any content change makes run creation fail until the manifest is
deliberately reviewed and updated.

- `development` is the official ARC training set. Search, training, and checkpoint selection may use
  its training examples.
- `locked_evaluation` is scoring-only. `load_search_task(..., split="locked_evaluation")` raises
  `SplitPolicyError` before reading a task.
- Search receives `SearchTask`, which contains development train pairs and test inputs, but has no
  test-output field. Trusted scorer, audit, and visualization code may use the complete `Task` view.

The current 91-task search subset is an explicit ID list in the same manifest. It is separate from
`arc_env/task_loader.py`, which remains a compatibility module containing known-correct solver
fixtures for regression tests and audits. Production trainers and blind search do not import that
module. Object-space fixture-skeleton search is available only through the explicitly non-blind
`object_env.search --fixture-arg-search` diagnostic.

This is an in-process API boundary, not a hostile-code sandbox. Competition evaluation must also run
solver code in a filesystem/container boundary that cannot open hidden answer files.

## Run provenance

Every newly written `run_meta.json` uses schema version 2 and includes:

- dataset name, source, version, split, task IDs, task count, canonical hash, and lock status;
- the hash and exact source-file set of the executable action library;
- RNG seed;
- an algorithm-specific, up-front compute budget;
- task-specific macro/demonstration provenance;
- the checkpoint-selection partition and an explicit assertion that locked-evaluation outputs were
  not used.

PPO's historical `eval_*` metric keys remain for compatibility, but they measure a fixed
development-train selection pair, not a held-out or locked-evaluation pair. LLM and human action
sequences are marked task-specific and ineligible for blind evaluation. A PPO warm start records the
SHA-256 of the exact demonstration episode used.

## Updating pinned inputs

Changing the vendored ARC data or executable action-library files is intentionally a manifest-breaking
event. Recompute hashes with `arc_env.provenance.canonical_files_sha256`, inspect the content diff,
update the relevant pin, and include that review in the same PR. Never update a pin merely to silence
`ProvenanceMismatch`.
