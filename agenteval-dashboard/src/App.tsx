import { useCallback, useEffect, useMemo, useState } from 'react'
import { checkHealth, getApiUrl, runScan } from './api'
import { SAMPLE_JSONL } from './sample'
import type { Finding, QueryResult, ScanReport, Severity } from './types'

function severityStyles(sev: Severity): string {
  const s = sev.toLowerCase()
  if (s === 'block') return 'bg-red-500/15 text-red-300 ring-red-500/30'
  if (s === 'review') return 'bg-amber-500/15 text-amber-200 ring-amber-500/30'
  if (s === 'advisory') return 'bg-sky-500/15 text-sky-200 ring-sky-500/30'
  return 'bg-zinc-500/15 text-zinc-300 ring-zinc-500/30'
}

function Badge({
  label,
  value,
  tone,
}: {
  label: string
  value: number
  tone: 'block' | 'review' | 'pass'
}) {
  const tones = {
    block: 'from-red-500/20 to-red-950/40 ring-red-500/40 text-red-200',
    review: 'from-amber-500/20 to-amber-950/40 ring-amber-500/40 text-amber-100',
    pass: 'from-emerald-500/20 to-emerald-950/40 ring-emerald-500/40 text-emerald-200',
  } as const
  return (
    <div
      className={`flex min-w-[7.5rem] flex-1 flex-col rounded-xl bg-gradient-to-br px-4 py-3 ring-1 ${tones[tone]}`}
    >
      <span className="text-[0.65rem] font-semibold tracking-[0.14em] uppercase opacity-80">
        {label}
      </span>
      <span className="mt-1 font-mono text-2xl font-semibold tabular-nums">{value}</span>
    </div>
  )
}

function Spinner() {
  return (
    <div className="flex items-center gap-3 text-sm text-zinc-400">
      <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-emerald-400/30 border-t-emerald-400" />
      Scanning corpus…
    </div>
  )
}

function SkeletonResults() {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="skeleton h-20 rounded-xl" />
        ))}
      </div>
      {[0, 1, 2].map((i) => (
        <div key={i} className="skeleton h-28 rounded-xl" />
      ))}
    </div>
  )
}

function FindingCard({
  finding,
  sql,
}: {
  finding: Finding
  sql?: string
}) {
  return (
    <article className="rounded-xl border border-zinc-800/80 bg-zinc-900/60 p-4 shadow-lg shadow-black/20 transition hover:border-zinc-700">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-md bg-zinc-800 px-2 py-0.5 font-mono text-xs font-medium text-emerald-300 ring-1 ring-zinc-700">
            {finding.rule_id}
          </span>
          <span
            className={`rounded-md px-2 py-0.5 text-[0.65rem] font-semibold tracking-wide uppercase ring-1 ${severityStyles(finding.severity)}`}
          >
            {finding.severity}
          </span>
          {finding.query_id && (
            <span className="font-mono text-xs text-zinc-500">{finding.query_id}</span>
          )}
        </div>
      </div>
      <p className="mt-3 text-sm leading-relaxed text-zinc-200">{finding.message}</p>
      {(finding.evidence || sql) && (
        <pre className="mt-3 overflow-x-auto rounded-lg border border-zinc-800 bg-black/40 p-3 font-mono text-[0.8rem] leading-relaxed text-zinc-300">
          <code>{finding.evidence || sql}</code>
        </pre>
      )}
      {sql && finding.evidence && sql !== finding.evidence && (
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-zinc-500 hover:text-zinc-300">
            Full SQL
          </summary>
          <pre className="mt-2 overflow-x-auto rounded-lg border border-zinc-800 bg-black/30 p-3 font-mono text-[0.75rem] text-zinc-400">
            <code>{sql}</code>
          </pre>
        </details>
      )}
    </article>
  )
}

function QueryPassCard({ query }: { query: QueryResult }) {
  return (
    <article className="rounded-xl border border-emerald-900/40 bg-emerald-950/20 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-md bg-emerald-500/15 px-2 py-0.5 text-[0.65rem] font-semibold tracking-wide text-emerald-300 uppercase ring-1 ring-emerald-500/30">
          pass
        </span>
        <span className="font-mono text-xs text-zinc-400">{query.query_id}</span>
      </div>
      {query.sql && (
        <pre className="mt-3 overflow-x-auto rounded-lg border border-zinc-800/80 bg-black/30 p-3 font-mono text-[0.8rem] text-zinc-400">
          <code>{query.sql}</code>
        </pre>
      )}
    </article>
  )
}

export default function App() {
  const [corpus, setCorpus] = useState(SAMPLE_JSONL.trim())
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [report, setReport] = useState<ScanReport | null>(null)
  const [apiOk, setApiOk] = useState<boolean | null>(null)
  const [filter, setFilter] = useState<'all' | 'block' | 'review' | 'pass'>('all')

  useEffect(() => {
    void checkHealth().then(setApiOk)
  }, [])

  const sqlById = useMemo(() => {
    const map = new Map<string, string>()
    report?.queries?.forEach((q) => {
      if (q.sql) map.set(q.query_id, q.sql)
    })
    return map
  }, [report])

  const onScan = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await runScan(corpus)
      setReport(result)
    } catch (e) {
      setReport(null)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [corpus])

  const passQueries = useMemo(
    () => report?.queries?.filter((q) => !q.findings?.length) ?? [],
    [report],
  )

  const flatFindings = report?.findings ?? []

  const visibleFindings = useMemo(() => {
    if (filter === 'all') return flatFindings
    if (filter === 'block') return flatFindings.filter((f) => f.severity === 'block')
    if (filter === 'review')
      return flatFindings.filter((f) => f.severity === 'review' || f.severity === 'advisory')
    return []
  }, [flatFindings, filter])

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 py-6 sm:px-6 lg:px-8">
      {/* Header */}
      <header className="mb-8 flex flex-col gap-4 border-b border-zinc-800/80 pb-6 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900/70 px-3 py-1 font-mono text-[0.7rem] text-zinc-400">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
            nishanttyagi-agenteval · sql scan
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
            SQL Agent Safety Scanner
          </h1>
          <p className="mt-2 max-w-xl text-sm leading-relaxed text-zinc-400">
            Paste a JSONL corpus of agent-generated SQL. AgentEval runs structural safety rules
            (Tiers 1–5) and returns a provenance-linked report — BLOCK / REVIEW / PASS.
          </p>
        </div>
        <div className="flex flex-col items-start gap-1 sm:items-end">
          <span className="font-mono text-[0.65rem] tracking-wide text-zinc-500 uppercase">
            API
          </span>
          <span className="max-w-[16rem] truncate font-mono text-xs text-zinc-400" title={getApiUrl()}>
            {getApiUrl()}
          </span>
          <span
            className={`inline-flex items-center gap-1.5 font-mono text-[0.7rem] ${
              apiOk === null
                ? 'text-zinc-500'
                : apiOk
                  ? 'text-emerald-400'
                  : 'text-red-400'
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                apiOk === null ? 'bg-zinc-500' : apiOk ? 'bg-emerald-400' : 'bg-red-400'
              }`}
            />
            {apiOk === null ? 'checking…' : apiOk ? 'reachable' : 'unreachable'}
          </span>
        </div>
      </header>

      <main className="grid flex-1 gap-6 lg:grid-cols-5">
        {/* Editor column */}
        <section className="flex flex-col lg:col-span-2">
          <div className="mb-2 flex items-center justify-between">
            <label htmlFor="corpus" className="text-sm font-medium text-zinc-300">
              Corpus (JSONL)
            </label>
            <button
              type="button"
              onClick={() => setCorpus(SAMPLE_JSONL.trim())}
              className="text-xs text-zinc-500 transition hover:text-emerald-400"
            >
              Reset sample
            </button>
          </div>
          <div className="flex min-h-[22rem] flex-1 flex-col overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 shadow-inner shadow-black/40 ring-1 ring-white/5">
            <div className="flex items-center gap-1.5 border-b border-zinc-800 bg-zinc-900/80 px-3 py-2">
              <span className="h-2.5 w-2.5 rounded-full bg-red-500/70" />
              <span className="h-2.5 w-2.5 rounded-full bg-amber-500/70" />
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-500/70" />
              <span className="ml-2 font-mono text-[0.65rem] text-zinc-500">queries.jsonl</span>
            </div>
            <textarea
              id="corpus"
              value={corpus}
              onChange={(e) => setCorpus(e.target.value)}
              spellCheck={false}
              className="scroll-pane min-h-[18rem] flex-1 resize-y bg-transparent p-4 font-mono text-[0.8rem] leading-relaxed text-zinc-200 outline-none placeholder:text-zinc-600"
              placeholder='{"id": "q1", "sql": "SELECT ..."}'
            />
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => void onScan()}
              disabled={loading || !corpus.trim()}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-zinc-950 shadow-lg shadow-emerald-500/20 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? (
                <>
                  <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-zinc-900/30 border-t-zinc-900" />
                  Running…
                </>
              ) : (
                <>Run Scan</>
              )}
            </button>
            <span className="font-mono text-xs text-zinc-500">
              {corpus.split(/\r?\n/).filter((l) => l.trim()).length} lines
            </span>
          </div>
          {error && (
            <div
              role="alert"
              className="mt-4 rounded-lg border border-red-500/30 bg-red-950/40 px-4 py-3 text-sm text-red-200"
            >
              {error}
            </div>
          )}
        </section>

        {/* Results column */}
        <section className="flex flex-col lg:col-span-3">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-sm font-medium text-zinc-300">Results</h2>
            {loading && <Spinner />}
            {report && !loading && (
              <span className="font-mono text-[0.65rem] text-zinc-500">
                run {report.run_id} · v{report.agenteval_version} · {report.dialect}
              </span>
            )}
          </div>

          {loading && <SkeletonResults />}

          {!loading && !report && !error && (
            <div className="flex min-h-[22rem] flex-col items-center justify-center rounded-xl border border-dashed border-zinc-800 bg-zinc-900/30 px-6 text-center">
              <p className="text-sm text-zinc-400">
                Run a scan to see BLOCK / REVIEW / PASS counts and per-finding evidence.
              </p>
              <p className="mt-2 font-mono text-xs text-zinc-600">
                POST {getApiUrl()}/scan
              </p>
            </div>
          )}

          {!loading && report && (
            <div className="space-y-5">
              {/* Summary bar */}
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <Badge
                  label="Blocked"
                  value={report.counts.blocked_queries}
                  tone="block"
                />
                <Badge
                  label="Review"
                  value={report.counts.review_queries}
                  tone="review"
                />
                <Badge label="Pass" value={report.counts.pass_queries} tone="pass" />
              </div>

              <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-500">
                <span className="font-mono">
                  {report.counts.findings} findings · {report.counts.block_violations} block
                  violations · {report.counts.queries} queries
                </span>
                <span className="hidden sm:inline">·</span>
                <span className="font-mono">
                  tiers{' '}
                  {Object.entries(report.tier_activation || {})
                    .filter(([, on]) => on)
                    .map(([t]) => t)
                    .join(', ') || '—'}
                </span>
              </div>

              {/* Filter chips */}
              <div className="flex flex-wrap gap-2">
                {(
                  [
                    ['all', 'All findings'],
                    ['block', 'BLOCK'],
                    ['review', 'REVIEW'],
                    ['pass', 'PASS'],
                  ] as const
                ).map(([key, label]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setFilter(key)}
                    className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                      filter === key
                        ? 'bg-zinc-100 text-zinc-900'
                        : 'bg-zinc-900 text-zinc-400 ring-1 ring-zinc-800 hover:text-zinc-200'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              <div className="scroll-pane max-h-[min(60vh,36rem)] space-y-3 overflow-y-auto pr-1">
                {filter !== 'pass' &&
                  visibleFindings.map((f, i) => (
                    <FindingCard
                      key={`${f.query_id}-${f.rule_id}-${i}`}
                      finding={f}
                      sql={f.query_id ? sqlById.get(f.query_id) : undefined}
                    />
                  ))}
                {filter === 'pass' &&
                  passQueries.map((q) => <QueryPassCard key={q.query_id} query={q} />)}
                {filter === 'all' && visibleFindings.length === 0 && (
                  <p className="rounded-lg border border-emerald-900/40 bg-emerald-950/20 px-4 py-6 text-center text-sm text-emerald-300">
                    No findings — all queries passed.
                  </p>
                )}
                {filter !== 'all' && filter !== 'pass' && visibleFindings.length === 0 && (
                  <p className="px-2 py-6 text-center text-sm text-zinc-500">
                    No {filter} findings in this run.
                  </p>
                )}
                {filter === 'pass' && passQueries.length === 0 && (
                  <p className="px-2 py-6 text-center text-sm text-zinc-500">
                    No clean PASS queries in this run.
                  </p>
                )}
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className="mt-10 border-t border-zinc-900 pt-4 text-center font-mono text-[0.65rem] text-zinc-600">
        AgentEval portfolio demo · free-tier deploy · no auth · stateless request/response
      </footer>
    </div>
  )
}
