## Purpose
This repository must be changed carefully. Preserve behavior unless the user explicitly requests a behavior change.

## Core working principles
- Follow the user's request exactly. Do not broaden scope.
- Never assume unclear requirements, intent, architecture, or edge-case behavior.
- If anything important is ambiguous, stop and ask a clarifying question before making changes.
- Always gather full context before editing:
  - inspect all directly related files
  - inspect shared helpers, schemas, configs, and call sites
  - inspect how the target code is used elsewhere
- Prefer the smallest correct change over a broad refactor.
- Do not make unrelated cleanup edits.
- Do not rename, move, or reorganize files unless the task requires it.
- Preserve existing style and structure unless the user explicitly asks for a redesign.

## Git and branch safety
- Never commit directly to `main`.
- Never push directly to `main`.
- Never merge into `main`.
- Always work on a feature branch.
- Open a pull request for all changes intended for `main`.
- Do not change branch protection or repository settings.
- Never change the default branch.

## Deletion rules
- Never delete `main`.
- Never delete any protected branch.
- Never delete files, directories, branches, tags, or other assets without explicit approval from the user.
- If deletion appears helpful, propose it first and ask for approval.
- After a pull request is merged, you may suggest deleting the associated feature branch, but ask for approval before doing so.
- Do not delete a branch if it is not fully merged, is still in active use, or is clearly intended to be kept.

## Refactors and reorganizations
- If the task involves refactoring, reorganizing, moving, renaming, or deduplicating code, first identify the existing behavior that must be preserved.
- Before making refactor/reorganization changes, run the relevant tests or validation commands if available to establish a baseline.
- After making changes, run the same tests or validation commands again.
- Treat a refactor as failed if behavior changes unintentionally, tests regress, or outputs differ unexpectedly.
- When refactoring, preserve public interfaces, expected file outputs, and existing semantics unless the user explicitly requests otherwise.

## Validation and testing
- If AGENTS.md or repository docs specify tests/checks, run them.
- If the task changes executable logic, run the most relevant tests possible.
- If no formal tests exist, perform the best available validation and explain what was checked.
- Do not claim something is fixed or preserved unless you validated it.
- If you could not run validation, say so clearly and explain why.

## Code change standards
- Preserve behavior unless explicitly asked to change it.
- Preserve data formats, file names, external interfaces, and output contracts unless explicitly asked to change them.
- Avoid introducing hidden side effects.
- Avoid speculative optimizations.
- Avoid placeholder implementations unless explicitly requested.
- Avoid comments that restate obvious code.
- Do not add dependencies unless necessary and justified by the task.

## Handling uncertainty
- If you are confused, stop and ask.
- If requirements conflict, stop and ask.
- If the codebase behavior is unclear, inspect more files before editing.
- If there are multiple reasonable interpretations, present them and ask which one is intended.
- Do not silently choose a risky interpretation.

## Change discipline
- Keep patches focused and minimal.
- Prefer modifying existing code over creating parallel code paths unless separation is required.
- Avoid changing formatting in untouched areas.
- Avoid broad search-and-replace operations unless the task explicitly calls for them and you have checked impact.

## Communication in PRs or summaries
- Clearly state:
  - what changed
  - why it changed
  - what was validated
  - any uncertainty or follow-up risk
- If validation was partial, say exactly what was and was not checked.

## Repository-specific preferences
- Strictly adhere to the requested scope.
- Prioritize correctness over speed.
- Be explicit when unsure rather than guessing.
- For reorganizing or refactoring tasks, verify before/after behavior to ensure logic was not changed.
- Look for full context before starting, including connected helpers and downstream usage.
- Ask clarifying questions before proceeding when requirements or behavior are not fully clear.
