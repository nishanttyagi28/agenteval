import type { ScanReport } from './types'

const API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '')
  || 'http://localhost:8000'

export function getApiUrl(): string {
  return API_URL
}

export async function runScan(jsonlText: string, dialect = 'postgres'): Promise<ScanReport> {
  const lines = jsonlText
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)

  if (lines.length === 0) {
    throw new Error('Corpus is empty — paste JSONL lines with {"id","sql"} objects.')
  }

  const queries = lines.map((line, i) => {
    let rec: Record<string, unknown>
    try {
      rec = JSON.parse(line) as Record<string, unknown>
    } catch {
      throw new Error(`Invalid JSON on line ${i + 1}`)
    }
    return {
      id: (rec.id as string | undefined) || `line_${i + 1}`,
      sql: (rec.sql as string | undefined) || '',
      question: (rec.question as string | undefined) || undefined,
      session_id: (rec.session_id as string | undefined) || undefined,
      metadata: (rec.metadata as Record<string, unknown> | undefined) || undefined,
    }
  })

  const res = await fetch(`${API_URL}/scan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({ queries, dialect }),
  })

  if (!res.ok) {
    let detail = res.statusText
    try {
      const err = (await res.json()) as { detail?: string }
      if (err.detail) detail = err.detail
    } catch {
      /* ignore */
    }
    throw new Error(`Scan failed (${res.status}): ${detail}`)
  }

  return (await res.json()) as ScanReport
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`, { method: 'GET' })
    return res.ok
  } catch {
    return false
  }
}
