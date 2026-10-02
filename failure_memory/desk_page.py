"""Self-contained Release Desk page. No frontend build step."""

from __future__ import annotations

PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Release Desk — AgentEval</title>
<style>
  :root {
    --ink: #1c1612;
    --muted: #6d6256;
    --paper: #f3ecdf;
    --panel: #fffaf3;
    --line: #e4d8c6;
    --teal: #1d5c4f;
    --teal-ink: #f4fff9;
    --wait: #9a3412;
    --wait-bg: #fff0e6;
    --ship: #3f6212;
    --ship-bg: #f1f6e4;
    --bad: #9f1239;
    --shadow: 0 18px 40px rgba(48, 32, 16, 0.08);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    min-height: 100vh;
    background:
      radial-gradient(900px 280px at 0% -10%, rgba(29, 92, 79, 0.14), transparent 60%),
      var(--paper);
    color: var(--ink);
    font-family: "Avenir Next", "Segoe UI", sans-serif;
    letter-spacing: 0.01em;
  }
  header {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: flex-end;
    padding: 28px 32px 8px;
  }
  .mark { font-family: Palatino, "Iowan Old Style", Georgia, serif; }
  .mark small {
    display: block;
    font-family: "Avenir Next", "Segoe UI", sans-serif;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-size: 11px;
    color: var(--muted);
  }
  h1 { margin: 4px 0 0; font-size: 40px; font-weight: 500; letter-spacing: -0.03em; }
  .local-pill {
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: 999px;
    padding: 8px 12px;
    color: var(--muted);
    font-size: 12px;
  }
  .headline {
    margin: 8px 32px 0;
    font-size: 18px;
    color: var(--ink);
  }
  .metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    padding: 18px 32px 8px;
  }
  .metric {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: var(--shadow);
  }
  .metric span { display: block; color: var(--muted); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; }
  .metric strong { display: block; margin-top: 6px; font-size: 28px; font-family: Palatino, Georgia, serif; font-weight: 500; }
  .metric.wait strong { color: var(--wait); }
  .metric.ship strong { color: var(--ship); }
  main {
    display: grid;
    grid-template-columns: minmax(280px, 380px) 1fr;
    gap: 16px;
    padding: 12px 32px 32px;
  }
  .queue, .detail, .ingest {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
  }
  .queue { overflow: auto; max-height: calc(100vh - 250px); }
  .queue-head, .detail-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 16px;
    border-bottom: 1px solid var(--line);
  }
  button, select, input, textarea {
    font: inherit;
    color: inherit;
  }
  button {
    border: 0;
    border-radius: 999px;
    padding: 10px 14px;
    cursor: pointer;
  }
  button.primary { background: var(--teal); color: var(--teal-ink); }
  button.ghost { background: transparent; border: 1px solid var(--line); }
  button.danger { background: transparent; color: var(--bad); border: 1px solid #fecdd3; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .item {
    width: 100%;
    text-align: left;
    border-radius: 0;
    background: transparent;
    border-bottom: 1px solid var(--line);
    padding: 14px 16px;
  }
  .item.active { background: #f3faf6; }
  .item b { display: block; font-weight: 600; }
  .item em { font-style: normal; color: var(--muted); font-size: 13px; }
  .badge {
    display: inline-block;
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    background: var(--wait-bg);
    color: var(--wait);
  }
  .badge.exported, .badge.approved { background: var(--ship-bg); color: var(--ship); }
  .badge.rejected { background: #fff1f2; color: var(--bad); }
  .detail { padding: 0 0 18px; }
  .detail-body { padding: 16px 18px 0; }
  .block {
    background: #f7f1e6;
    border-radius: 12px;
    padding: 12px 14px;
    margin: 0 0 12px;
    white-space: pre-wrap;
    line-height: 1.45;
  }
  label { display: block; font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); margin: 12px 0 6px; }
  input, select, textarea {
    width: 100%;
    border: 1px solid var(--line);
    background: white;
    border-radius: 10px;
    padding: 10px 12px;
  }
  textarea { min-height: 74px; resize: vertical; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
  .check { display: flex; gap: 8px; align-items: center; text-transform: none; letter-spacing: 0; font-size: 14px; color: var(--ink); }
  .check input { width: auto; }
  .banner {
    margin: 12px 32px 0;
    padding: 10px 12px;
    border-radius: 12px;
    background: #fff7ed;
    color: var(--wait);
    display: none;
  }
  .banner.show { display: block; }
  .empty { padding: 28px 16px; color: var(--muted); }
  .ci {
    margin-top: 14px;
    font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
    font-size: 12px;
    background: var(--ink);
    color: #f6f1e7;
    border-radius: 10px;
    padding: 10px 12px;
  }
  .ingest { margin: 0 32px 28px; padding: 14px 16px 16px; }
  .ingest h2 { margin: 0 0 8px; font-size: 16px; font-family: Palatino, Georgia, serif; font-weight: 500; }
  .split { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: start; }
  @media (max-width: 900px) {
    header, .headline, .banner, .ingest { padding-left: 16px; padding-right: 16px; margin-left: 0; margin-right: 0; }
    .metrics, main { padding-left: 16px; padding-right: 16px; }
    .metrics, main, .split { grid-template-columns: 1fr; }
    h1 { font-size: 32px; }
    .queue { max-height: none; }
  }
</style>
</head>
<body>
<header>
  <div class="mark">
    <small>AgentEval __VERSION__</small>
    <h1>Release Desk</h1>
  </div>
  <div class="local-pill">Local review · no login · secrets redacted before they are stored</div>
</header>
<p class="headline" id="headline">Loading the queue…</p>
<div class="banner" id="banner"></div>
<section class="metrics" id="metrics"></section>
<main>
  <section class="queue">
    <div class="queue-head">
      <strong>Incidents</strong>
      <button class="ghost" id="refresh" type="button">Refresh</button>
    </div>
    <div id="list"></div>
  </section>
  <section class="detail">
    <div class="detail-head"><strong id="detail-title">Select an incident</strong><span id="detail-badge"></span></div>
    <div class="detail-body" id="detail">
      <p class="empty">A shipped incident becomes a golden case. The next agent run that still does this fails CI.</p>
    </div>
  </section>
</main>
<section class="ingest">
  <h2>Ingest a production incident</h2>
  <p style="margin:0 0 10px; color: var(--muted);">Paste a webhook body, or load the refund sample. The same trace id is stored once.</p>
  <div class="split">
    <textarea id="payload" spellcheck="false">{
  "incident": {
    "agent": "refund-desk",
    "prompt": "Cancel order 4821 and refund the customer.",
    "output": "Cancelled order 4821. No refund was issued.",
    "tools_called": ["cancel_order"],
    "expected_tools": ["lookup_order", "issue_refund"]
  }
}</textarea>
    <div class="row" style="margin-top:0; flex-direction: column;">
      <button class="primary" id="ingest" type="button">Ingest</button>
      <button class="ghost" id="sample" type="button">Load refund sample</button>
    </div>
  </div>
</section>
<script>
const $ = (id) => document.getElementById(id);
let selected = null;
let summary = null;

function showError(err) {
  const banner = $("banner");
  banner.textContent = err && err.message ? err.message : String(err);
  banner.classList.add("show");
}
function clearError() {
  $("banner").classList.remove("show");
}
async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}
function badge(state) {
  const span = document.createElement("span");
  span.className = "badge " + (state || "");
  span.textContent = (state || "unknown").replaceAll("_", " ");
  return span;
}
function metric(label, value, kind) {
  const card = document.createElement("article");
  card.className = "metric" + (kind ? " " + kind : "");
  const span = document.createElement("span");
  span.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = String(value);
  card.append(span, strong);
  return card;
}
function renderMetrics(data) {
  summary = data;
  $("headline").textContent = data.headline;
  const root = $("metrics");
  root.replaceChildren(
    metric("Waiting review", data.waiting_review, "wait"),
    metric("Repeat incidents", data.repeat_incidents, "wait"),
    metric("Ready to ship", data.ready_to_ship, ""),
    metric("Already in CI", data.in_ci, "ship")
  );
}
function renderList(items) {
  const root = $("list");
  root.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No incidents yet. Ingest the refund sample to see a failure that has not become a test.";
    root.append(empty);
    return;
  }
  for (const item of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.id = item.candidate_id;
    button.className = "item" + (selected === item.candidate_id ? " active" : "");
    const title = document.createElement("b");
    title.textContent = item.agent_name || item.title || item.candidate_id;
    const meta = document.createElement("em");
    const times = item.occurrence_count === 1 ? "seen once" : "seen " + item.occurrence_count + " times";
    meta.textContent = (item.failure_category || "unclassified") + " · " + times;
    button.append(title, document.createElement("br"), meta, document.createElement("br"), badge(item.state));
    button.addEventListener("click", () => openDetail(item.candidate_id));
    root.append(button);
  }
}
function field(labelText, control) {
  const label = document.createElement("label");
  label.textContent = labelText;
  return [label, control];
}
function renderDetail(data) {
  selected = data.candidate_id;
  $("detail-title").textContent = (data.agent_name || "Incident") + " · " + (data.failure_category || "");
  $("detail-badge").replaceChildren(badge(data.state));
  const root = $("detail");
  root.replaceChildren();
  const prompt = document.createElement("div");
  prompt.className = "block";
  prompt.textContent = data.prompt || "(prompt was not captured — this incident cannot ship)";
  const output = document.createElement("div");
  output.className = "block";
  output.textContent = data.output || "(no output captured)";
  const tools = document.createElement("p");
  const called = (data.tools_called || []).join(", ") || "none";
  const expected = (data.must_call_tools || []).join(", ") || "not recorded";
  tools.textContent = "Called: " + called + ". Expected: " + expected + ".";
  root.append(...field("What the customer asked", prompt), ...field("What the agent did", output), tools);
  if (data.secrets_redacted) {
    const note = document.createElement("p");
    note.textContent = "A secret-shaped string was redacted before this incident was stored.";
    root.append(note);
  }
  if (data.state === "exported") {
    const done = document.createElement("p");
    done.textContent = "This failure is already a CI case" + (data.stable_case_id ? " (" + data.stable_case_id + ")." : ".");
    const cmd = document.createElement("div");
    cmd.className = "ci";
    cmd.textContent = summary ? summary.ci_command : "";
    root.append(done, cmd);
    return;
  }
  const type = document.createElement("select");
  for (const value of ["contains", "exact", "llm_judge"]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    type.append(option);
  }
  const truth = document.createElement("textarea");
  truth.placeholder = "refund issued";
  truth.value = (data.expected_behaviour && data.expected_behaviour.ground_truth) || "";
  const must = document.createElement("input");
  must.value = (data.must_call_tools || []).join(", ");
  const hallu = document.createElement("input");
  hallu.type = "checkbox";
  hallu.checked = true;
  const halluLabel = document.createElement("label");
  halluLabel.className = "check";
  halluLabel.append(hallu, document.createTextNode("Fail the case if the answer invents facts"));
  const note = document.createElement("textarea");
  note.placeholder = "Why this should block the next release";
  root.append(
    ...field("Correctness check", type),
    ...field("Ground truth", truth),
    ...field("Tools the agent must call", must),
    halluLabel,
    ...field("Reviewer note", note)
  );
  const row = document.createElement("div");
  row.className = "row";
  if (data.state === "rejected") {
    const reopen = document.createElement("button");
    reopen.className = "primary";
    reopen.type = "button";
    reopen.textContent = "Reopen";
    reopen.addEventListener("click", () => act("/reopen", { note: note.value }));
    row.append(reopen);
  } else {
    const ship = document.createElement("button");
    ship.className = "primary";
    ship.type = "button";
    ship.textContent = "Ship to CI";
    ship.addEventListener("click", () => {
      const tools = must.value.split(",").map((part) => part.trim()).filter(Boolean);
      act("/ship", {
        note: note.value,
        expected_behaviour: {
          correctness_type: type.value,
          ground_truth: truth.value,
          must_call_tools: tools,
          must_not_hallucinate: hallu.checked
        }
      });
    });
    const reject = document.createElement("button");
    reject.className = "danger";
    reject.type = "button";
    reject.textContent = "Reject";
    reject.addEventListener("click", () => act("/reject", { note: note.value }));
    row.append(ship, reject);
  }
  root.append(row);
}
async function refresh(keep) {
  clearError();
  const [sum, queue] = await Promise.all([
    api("/api/gate/summary"),
    api("/api/gate/queue")
  ]);
  renderMetrics(sum);
  renderList(queue.items || []);
  if (keep) await openDetail(keep);
}
async function openDetail(id) {
  clearError();
  const data = await api("/api/gate/candidates/" + encodeURIComponent(id));
  renderDetail(data);
  document.querySelectorAll(".item").forEach((node) => {
    node.classList.toggle("active", node.dataset.id === data.candidate_id);
  });
}
async function act(suffix, body) {
  clearError();
  try {
    await api("/api/gate/candidates/" + encodeURIComponent(selected) + suffix, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    await refresh(selected);
  } catch (err) {
    showError(err);
  }
}
$("refresh").addEventListener("click", () => refresh(selected).catch(showError));
$("sample").addEventListener("click", async () => {
  clearError();
  try {
    const result = await api("/api/gate/sample", { method: "POST" });
    const id = result.results && result.results[0] && result.results[0].candidate_id;
    await refresh(id || selected);
  } catch (err) {
    showError(err);
  }
});
$("ingest").addEventListener("click", async () => {
  clearError();
  let body;
  try {
    body = JSON.parse($("payload").value);
  } catch (err) {
    showError(new Error("That payload is not JSON"));
    return;
  }
  try {
    const result = await api("/api/gate/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const id = result.results && result.results[0] && result.results[0].candidate_id;
    await refresh(id || selected);
  } catch (err) {
    showError(err);
  }
});
refresh(null).catch(showError);
</script>
</body>
</html>
"""


def render_page(version: str) -> str:
    return PAGE_HTML.replace("__VERSION__", version)
