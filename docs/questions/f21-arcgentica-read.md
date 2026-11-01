# F21 - What arcgentica does that we don't (read 2026-11-01, ARC-178)

Source: `github.com/symbolica-ai/arcgentica`, MIT, one commit, read locally. `arc_agent/{agent,prompts,types}.py` is about 340
lines; the model runs on their closed Agentica server, so the loop itself is not in the repo. 85.3% (Opus 4.6) at about
$6.94 per task and 128k tokens, 2 attempts, up to 10 sub-agents. Not comparable to our ~35k-token Qwen budget.

## Differences that may transfer

1. **Generalisation instructions in the system prompt.** Ours has one line, in the refine reply. Theirs repeats it in steps 2, 4 and
   5: cover symmetric orientations and shapes, avoid constants tuned to the train examples (thresholds, offsets, dimensions,
   gaps, flags), and compare against the `challenges` (the test inputs) before finishing. Costs no tokens at run time. Cheap
   to try in direct and agent prompts, and a plausible fix for the "verified on train, wrong on test" failures in F20.
2. **Prescribed tooling.** Tells the model to use `scipy.ndimage.label` for objects and offers `skimage`, `scipy`, `sympy`. Check which
   libraries our sandbox has before adding this.
3. **Helper functions in the REPL:** `accuracy`, `soft_accuracy` (cell-level), `example_to_diagram` (ASCII grid). We expose
   wrong-cell counts in submit replies but not these as callable helpers.
4. **The agent sees test inputs from the start** and must reason about them. We show them too; the difference is the explicit
   "will it generalise to the challenges" checkpoint.
5. **Attempts instead of rounds:** `num_attempts=2` independent runs, `num_retries=3` only for transient errors. No plateau logic
   anywhere in the repo; it relies on the model finishing, with a large budget.

## Not transferable here

Sub-agent spawning (`call_agent`) multiplies tokens; at ~35k per task there is no room. The 5 s per-program timeout is
already similar to ours. No budget or give-up logic exists to copy, so ARC-166 stays our own design.

## Action

Add the generalisation checklist to the direct and agent prompts behind a flag, and test it with the scripted player for
mechanics and the Qwen bench for effect (no free proxy for the effect).
