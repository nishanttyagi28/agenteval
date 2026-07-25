export type Severity = 'block' | 'review' | 'advisory' | string

export interface Finding {
  query_id?: string
  rule_id: string
  severity: Severity
  message: string
  evidence: string
}

export interface QueryResult {
  query_id: string
  parsed: boolean
  findings: Finding[]
  sql?: string
}

export interface ScanCounts {
  queries: number
  blocked_queries: number
  review_queries: number
  pass_queries: number
  block_violations: number
  findings: number
}

export interface ScanReport {
  run_id: string
  agenteval_version: string
  corpus_hash: string
  dialect: string
  tier_activation: Record<string, boolean>
  counts: ScanCounts
  findings: Finding[]
  queries: QueryResult[]
  created_at: string
  policy_path?: string | null
  notices?: string[]
}
