import * as cp from "child_process";
import * as vscode from "vscode";

let outputChannel: vscode.OutputChannel | undefined;

export function activate(context: vscode.ExtensionContext): void {
  const disposable = vscode.commands.registerCommand("agenteval.runSuite", runSuite);
  context.subscriptions.push(disposable);
}

export function deactivate(): void {
  outputChannel?.dispose();
  outputChannel = undefined;
}

function getOutputChannel(): vscode.OutputChannel {
  if (!outputChannel) {
    outputChannel = vscode.window.createOutputChannel("AgentEval");
  }
  return outputChannel;
}

function runSuite(): void {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    vscode.window.showErrorMessage(
      "AgentEval: open a folder or workspace before running a suite."
    );
    return;
  }
  const cwd = folders[0].uri.fsPath;

  const config = vscode.workspace.getConfiguration("agenteval");
  const pythonPath = config.get<string>("pythonPath", "python");
  const extraArgs = config.get<string[]>("extraArgs", []);
  const args = ["-m", "agenteval", "run", ...extraArgs];

  const channel = getOutputChannel();
  channel.clear();
  channel.show(true);
  channel.appendLine(`$ ${pythonPath} ${args.join(" ")}`);
  channel.appendLine(`(cwd: ${cwd})`);
  channel.appendLine("");

  let child: cp.ChildProcessWithoutNullStreams;
  try {
    child = cp.spawn(pythonPath, args, { cwd });
  } catch (err) {
    reportSpawnFailure(channel, pythonPath, err);
    return;
  }

  // On some platforms a bad executable only surfaces asynchronously via this
  // event rather than a synchronous throw from spawn() above.
  child.on("error", (err: NodeJS.ErrnoException) => {
    reportSpawnFailure(channel, pythonPath, err);
  });

  child.stdout.on("data", (data: Buffer) => channel.append(data.toString()));
  child.stderr.on("data", (data: Buffer) => channel.append(data.toString()));

  child.on("close", (code: number | null) => {
    channel.appendLine("");
    if (code === 0) {
      channel.appendLine("AgentEval: suite completed successfully (exit code 0).");
      vscode.window.showInformationMessage("AgentEval: suite completed successfully.");
    } else {
      channel.appendLine(`AgentEval: suite exited with code ${code}.`);
      vscode.window.showErrorMessage(
        `AgentEval: suite failed (exit code ${code}). See the "AgentEval" output channel for details.`
      );
    }
  });
}

function reportSpawnFailure(
  channel: vscode.OutputChannel,
  pythonPath: string,
  err: unknown
): void {
  const isMissingExecutable =
    err instanceof Error && (err as NodeJS.ErrnoException).code === "ENOENT";
  const message = isMissingExecutable
    ? `AgentEval: could not find "${pythonPath}". Install Python and the agenteval package ` +
      '(pip install nishanttyagi-agenteval), or set "agenteval.pythonPath" in settings to the ' +
      "interpreter that has it installed."
    : `AgentEval: failed to start the suite (${err instanceof Error ? err.message : String(err)}).`;
  channel.appendLine(message);
  vscode.window.showErrorMessage(message);
}
