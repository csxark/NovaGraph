const BASE_URL = '/api'

/**
 * Centralised fetch wrapper – throws Error with backend detail on non-2xx.
 */
async function apiFetch(path, options = {}) {
  const response = await fetch(`${BASE_URL}${path}`, options)
  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const body = await response.json()
      detail = body.detail ?? body.message ?? detail
    } catch (_) { /* ignore parse errors */ }
    throw new Error(detail)
  }
  return response.json()
}

/**
 * Upload a PDF file.
 * @param {File} file
 * @returns {Promise<{ job_id: string, paper_id: string, title: string }>}
 */
export async function uploadPDF(file) {
  const form = new FormData()
  form.append('file', file)
  return apiFetch('/upload', { method: 'POST', body: form })
}

/**
 * Get the processing job status.
 * @param {string} jobId
 * @returns {Promise<{
 *   job_id: string,
 *   paper_id: string,
 *   status: 'queued'|'processing'|'completed'|'failed',
 *   stages: Record<string, { status: string, started_at: number|null, completed_at: number|null }>,
 *   error: string|null,
 *   title: string
 * }>}
 */
export async function getJobStatus(jobId) {
  return apiFetch(`/status/${jobId}`)
}

/**
 * Query the knowledge graph.
 * @param {string} paperId
 * @param {string} query
 * @param {number} [topK=5]
 * @param {boolean} [includeTrace=false]
 * @returns {Promise<{ answer: string, sources: any[], query_id: string, trace: any|null }>}
 */
export async function queryGraph(paperId, query, topK = 5, includeTrace = false) {
  return apiFetch('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paper_id: paperId, query, top_k: topK, include_trace: includeTrace })
  })
}

/**
 * Fetch graph data (nodes + edges) for a paper.
 * @param {string} paperId
 * @returns {Promise<{ nodes: any[], edges: any[], paper_id: string }>}
 */
export async function getGraph(paperId) {
  return apiFetch(`/graph/${paperId}`)
}

/**
 * Fetch detailed query trace by query_id.
 * @param {string} queryId
 * @returns {Promise<any>}
 */
export async function getTrace(queryId) {
  return apiFetch(`/trace/${queryId}`)
}

/**
 * Health check.
 * @returns {Promise<{ status: string, version: string }>}
 */
export async function getHealth() {
  return apiFetch('/health')
}
