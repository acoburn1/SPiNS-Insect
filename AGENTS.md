## Operating Rules (Read First)

This repo should be changed carefully and minimally.
Default assumption: preserve behavior unless the task explicitly asks for a behavior change.

### 1) Confirm understanding and report clearly
- Before coding, make sure the request is understood.
- If scope/intent is ambiguous, stop and ask.
- After changes, report exactly:
  - what changed
  - why
  - what was validated
  - any remaining risk/uncertainty

### 2) Refactors must prove logical consistency
- For refactors/reorg/dedup, identify current behavior first.
- Validate before/after with relevant checks when available.
- Treat unexpected behavior changes as failure unless explicitly requested.

### 3) Stay in scope; never delete without explicit instruction
- Do only what the task asks.
- No unrelated cleanup, renames, moves, or architecture changes.
- Never delete files, directories, branches, tags, or other assets unless the task explicitly says to.
- Never perform dangerous branch operations outside the requested scope.

### 4) Gather full context before editing
- Inspect all directly related files and call sites.
- Understand connected logic, schemas/configs, and shared utilities.
- Reuse existing shared logic when it fits.
- If repeated logic is introduced, add/extend shared utility logic where appropriate.

### 5) Keep code simple; avoid boilerplate defensive noise (very important)
- Write normal, readable code focused on the real logic path.
- Do not add backup code for every hypothetical scenario.
- Add validation/guards only where truly needed.
- Raise exceptions when required inputs are missing or malformed.
- No placeholder implementations.

## Working style
- Prefer the smallest correct patch.
- Preserve existing interfaces/data formats unless explicitly asked to change them.
- Do not add dependencies unless clearly necessary.
