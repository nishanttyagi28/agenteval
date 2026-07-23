# Coding agent evaluation starter

This template focuses on observable engineering behavior rather than benchmark
puzzles: inspecting relevant files, making bounded changes, running tests,
preserving APIs, handling unsafe input, and avoiding unjustified dependencies.

Before running it:

1. Replace the disabled adapter with your coding-agent adapter.
2. Map the illustrative tool names (`read_file`, `edit_file`, `run_tests`, and
   `search_code`) to the names emitted by your adapter.
3. Point the prompts at small, disposable fixture repositories containing the
   described defects.
4. Replace rubric text with repository-specific acceptance criteria.
5. Review traces and generated patches manually before accepting a baseline.

Run this suite only in an isolated fixture checkout. A coding agent can execute
commands and change files with the permissions granted to its process.
