# AgentEval for VS Code

A minimal VS Code extension that adds one command to the command palette —
**AgentEval: Run Suite** — which shells out to `python -m agenteval run` in
the current workspace folder and streams the output into an **AgentEval**
output channel.

This is intentionally small: no webview, no tree view, no status bar widget.
It exists to let you trigger an AgentEval run without leaving the editor.

## What it does

1. You run **AgentEval: Run Suite** from the command palette (`Ctrl+Shift+P` /
   `Cmd+Shift+P`).
2. The extension spawns `python -m agenteval run [extra args]` with `cwd` set
   to your first workspace folder.
3. stdout/stderr are streamed live into the **AgentEval** output channel
   (View → Output → select "AgentEval" from the dropdown).
4. On completion you get a toast notification — success if the process exited
   0, an error otherwise (with a pointer back to the output channel for
   details).

If `python` can't be found, or `agenteval` isn't installed in that
interpreter, you'll get a clear error message instead of a silent failure —
see [Settings](#settings) for how to point at the right interpreter.

## Prerequisites (for building/testing this extension locally)

- [Node.js](https://nodejs.org/) 18+ and npm
- VS Code 1.85+
- Separately: a Python environment with `agenteval` installed
  (`pip install nishanttyagi-agenteval`, or `pip install -e .` from the repo
  root) if you actually want **Run Suite** to succeed when you invoke it.

## Build and run locally

```bash
cd vscode-extension
npm install
npm run compile
```

Then, in VS Code:

1. Open the `vscode-extension/` folder (this folder specifically, not the
   repo root) as its own VS Code window.
2. Press `F5` (or Run → Start Debugging). This launches a second VS Code
   window — the **Extension Development Host** — with the extension loaded.
3. In that new window, open a folder that contains (or is) an `agenteval`
   project — e.g. open the AgentEval repo root itself.
4. Open the command palette and run **AgentEval: Run Suite**.
5. Watch the **AgentEval** output channel for live progress, and a toast at
   the end for pass/fail.

`npm run watch` recompiles on save if you're iterating on `src/extension.ts`;
just re-launch the Extension Development Host (or use the debugger's restart
button) to pick up changes.

## Settings

| Setting                  | Default    | Description                                                                                          |
| ------------------------ | ---------- | ------------------------------------------------------------------------------------------------------ |
| `agenteval.pythonPath`   | `"python"` | Python executable used to run `-m agenteval run`. Set an absolute path if `agenteval` lives in a specific virtualenv/conda env. |
| `agenteval.extraArgs`    | `[]`       | Extra CLI args appended after `run`, e.g. `["--agent", "my_agent"]` or `["--all"]`.                   |

Example `settings.json`:

```json
{
  "agenteval.pythonPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "agenteval.extraArgs": ["--agent", "agentic_data_analyst", "--quiet"]
}
```

## Known limitations (by design, for a v0 scaffold)

- Single command, no tree view of individual cases, no inline decorations.
- No automated test suite for the extension itself (the Python side —
  `core/report.py`, `core/history.py`, the CLI — has full pytest coverage;
  this extension is a thin, manually-verified shell around the CLI).
- Only the first workspace folder is used in multi-root workspaces.

## Publishing (not done here)

This scaffold is deliberately unpublished. When you're ready:

```bash
npm install -g @vscode/vsce
cd vscode-extension
vsce package   # produces agenteval-vscode-0.0.1.vsix
```

`vsce package` needs a real `publisher` you've registered on the
[VS Code Marketplace](https://marketplace.visualstudio.com/manage) and,
typically, an icon and a bumped version — none of which are set up here.
`vsce publish` (or uploading the `.vsix` manually) is the manual step
mentioned above; this scaffold stops at "builds and runs locally."
