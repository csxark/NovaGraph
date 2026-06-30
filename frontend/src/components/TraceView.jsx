import React, { useState } from 'react'
import {
  Activity, ChevronDown, ChevronUp,
  Search, Network, Layers, CheckCircle2,
  ArrowRight, Tag, AlertCircle
} from 'lucide-react'

/* ── Collapsible section ─────────────────────────────────── */
function Section({ title, icon: Icon, color, badge, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen)

  const colorMap = {
    blue:   { header: 'text-sky-400',     badge: 'bg-sky-500/15 text-sky-400 border-sky-500/25',    icon: 'text-sky-400'     },
    violet: { header: 'text-violet-400',  badge: 'bg-violet-500/15 text-violet-400 border-violet-500/25', icon: 'text-violet-400'  },
    emerald:{ header: 'text-emerald-400', badge: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/25', icon: 'text-emerald-400' },
  }
  const c = colorMap[color] ?? colorMap.blue

  return (
    <div className="rounded-xl border border-slate-700/40 overflow-hidden">
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center gap-3 px-4 py-3 bg-surface-700/40 hover:bg-surface-700/60 transition-colors text-left"
      >
        <Icon size={15} className={c.icon} />
        <span className={`text-sm font-semibold ${c.header} flex-1`}>{title}</span>
        {badge != null && (
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${c.badge}`}>
            {badge}
          </span>
        )}
        {open
          ? <ChevronUp   size={14} className="text-slate-500" />
          : <ChevronDown size={14} className="text-slate-500" />
        }
      </button>
      <div className={`collapse-content ${open ? 'open' : 'closed'}`}>
        <div className="px-4 py-3 bg-surface-800/40">
          {children}
        </div>
      </div>
    </div>
  )
}

/* ── Vector Agent section ────────────────────────────────── */
function VectorAgentSection({ data }) {
  const chunks = data?.chunks ?? data?.results ?? []

  if (chunks.length === 0) {
    return <EmptySection message="No vector results for this query" />
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest border-b border-slate-700/40">
            <th className="text-left py-2 pr-3 w-1/2">Chunk</th>
            <th className="text-left py-2 pr-3">Type</th>
            <th className="text-right py-2">Score</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-700/20">
          {chunks.map((chunk, i) => {
            const score   = chunk.score ?? chunk.similarity ?? 0
            const isTop   = i === 0
            return (
              <tr
                key={i}
                className={`transition-colors ${isTop ? 'bg-sky-500/5' : 'hover:bg-slate-700/20'}`}
              >
                <td className="py-2 pr-3">
                  <div className={`font-medium leading-snug ${isTop ? 'text-sky-300' : 'text-slate-300'} line-clamp-2`}>
                    {chunk.name ?? chunk.title ?? chunk.text?.slice(0, 80) ?? `Chunk ${i + 1}`}
                    {isTop && <span className="ml-1 text-[9px] text-sky-500">TOP</span>}
                  </div>
                </td>
                <td className="py-2 pr-3">
                  <span className="px-1.5 py-0.5 rounded-md bg-slate-700/60 text-slate-400 text-[10px]">
                    {chunk.type ?? chunk.entity_type ?? '—'}
                  </span>
                </td>
                <td className="py-2 text-right">
                  <div className="inline-flex items-center gap-1.5">
                    <div className="w-12 h-1.5 rounded-full bg-slate-700 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-sky-500 to-primary-500"
                        style={{ width: `${Math.round(score * 100)}%` }}
                      />
                    </div>
                    <span className={`font-mono ${isTop ? 'text-sky-400' : 'text-slate-500'}`}>
                      {score.toFixed(3)}
                    </span>
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/* ── Graph Agent section ─────────────────────────────────── */
function GraphAgentSection({ data }) {
  const nodes = data?.nodes ?? data?.entities ?? []
  const edgeCount  = data?.edge_count ?? data?.edges?.length ?? 0
  const depth      = data?.depth ?? data?.traversal_depth ?? 0

  if (nodes.length === 0 && edgeCount === 0) {
    return <EmptySection message="No graph traversal data for this query" />
  }

  return (
    <div className="space-y-3">
      {/* Stats row */}
      <div className="grid grid-cols-3 gap-2">
        <Stat label="Nodes"  value={nodes.length} color="text-violet-400" />
        <Stat label="Edges"  value={edgeCount}    color="text-indigo-400" />
        <Stat label="Depth"  value={depth}        color="text-purple-400" />
      </div>

      {/* Node list */}
      {nodes.length > 0 && (
        <div className="space-y-1.5 max-h-48 overflow-y-auto">
          {nodes.slice(0, 12).map((node, i) => (
            <div
              key={i}
              className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-surface-700/50 text-xs"
            >
              <div className="w-2 h-2 rounded-full bg-violet-400 shrink-0" />
              <span className="text-slate-300 flex-1 truncate">{node.name ?? node.id}</span>
              {node.type && (
                <span className="text-[10px] text-slate-500 shrink-0">{node.type}</span>
              )}
            </div>
          ))}
          {nodes.length > 12 && (
            <p className="text-[10px] text-slate-600 text-center py-1">+{nodes.length - 12} more nodes</p>
          )}
        </div>
      )}
    </div>
  )
}

/* ── Entity Resolver section ─────────────────────────────── */
function EntityResolverSection({ data }) {
  const expanded = data?.expanded_terms ?? data?.expansions ?? []
  const resolved = data?.resolved_entities ?? data?.entities ?? []

  if (expanded.length === 0 && resolved.length === 0) {
    return <EmptySection message="No entity resolution data for this query" />
  }

  return (
    <div className="space-y-3">
      {/* Expanded terms */}
      {expanded.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-2">
            Expanded Terms ({expanded.length})
          </p>
          <div className="flex flex-wrap gap-1.5">
            {expanded.map((term, i) => (
              <span key={i}
                className="px-2 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs">
                {typeof term === 'string' ? term : term.term ?? term.label}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Resolved entities */}
      {resolved.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-2">
            Resolved Entities ({resolved.length})
          </p>
          <div className="flex flex-wrap gap-1.5">
            {resolved.map((entity, i) => (
              <div key={i} className="flex items-center gap-1 px-2 py-1 rounded-lg bg-surface-700/60 border border-slate-700/40 text-xs">
                <CheckCircle2 size={10} className="text-emerald-400 shrink-0" />
                <span className="text-slate-300">{typeof entity === 'string' ? entity : entity.name ?? entity.id}</span>
                {entity.type && (
                  <span className="text-[10px] text-slate-500 ml-1">{entity.type}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

/* ── Contribution indicator ──────────────────────────────── */
function ContributionBanner({ trace }) {
  const contributors = []
  const va = trace?.vector_agent ?? trace?.vectorAgent
  const ga = trace?.graph_agent  ?? trace?.graphAgent
  const er = trace?.entity_resolver ?? trace?.entityResolver

  const chunks = va?.chunks ?? va?.results ?? []
  const nodes  = ga?.nodes  ?? ga?.entities ?? []
  const resolved = er?.resolved_entities ?? er?.entities ?? []

  if (chunks.length  > 0) contributors.push({ label: 'Vector Agent',    color: 'bg-sky-500/20 text-sky-400 border-sky-500/25' })
  if (nodes.length   > 0) contributors.push({ label: 'Graph Agent',     color: 'bg-violet-500/20 text-violet-400 border-violet-500/25' })
  if (resolved.length > 0) contributors.push({ label: 'Entity Resolver', color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/25' })

  if (contributors.length === 0) return null

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-xs text-slate-500">Answer from:</span>
      {contributors.map((c, i) => (
        <React.Fragment key={c.label}>
          {i > 0 && <ArrowRight size={10} className="text-slate-600" />}
          <span className={`px-2 py-0.5 rounded-full text-[11px] font-semibold border ${c.color}`}>
            {c.label}
          </span>
        </React.Fragment>
      ))}
    </div>
  )
}

function Stat({ label, value, color }) {
  return (
    <div className="rounded-lg bg-surface-700/50 py-2 px-2 text-center">
      <p className={`text-lg font-bold ${color}`}>{value}</p>
      <p className="text-[10px] text-slate-500">{label}</p>
    </div>
  )
}

function EmptySection({ message }) {
  return (
    <div className="flex items-center gap-2 py-1 text-xs text-slate-600">
      <AlertCircle size={12} />
      {message}
    </div>
  )
}

/* ── Trace card ──────────────────────────────────────────── */
function TraceCard({ trace }) {
  const va = trace?.vector_agent   ?? trace?.vectorAgent   ?? {}
  const ga = trace?.graph_agent    ?? trace?.graphAgent    ?? {}
  const er = trace?.entity_resolver ?? trace?.entityResolver ?? {}

  const vaChunks = (va?.chunks ?? va?.results ?? []).length
  const gaNodes  = (ga?.nodes  ?? ga?.entities ?? []).length
  const erTerms  = (er?.resolved_entities ?? er?.entities ?? []).length

  return (
    <div className="animate-slide-up space-y-3">
      {/* Query header */}
      <div className="glass rounded-xl px-4 py-3">
        <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-1">Query</p>
        <p className="text-sm text-slate-200 leading-snug">{trace.query ?? '—'}</p>
        {trace.timestamp && (
          <p className="text-[10px] text-slate-600 mt-1.5">
            {new Date(trace.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </p>
        )}
      </div>

      {/* Contribution */}
      <ContributionBanner trace={trace} />

      {/* Sections */}
      <Section
        title="Vector Agent"
        icon={Search}
        color="blue"
        badge={vaChunks > 0 ? vaChunks : null}
        defaultOpen
      >
        <VectorAgentSection data={va} />
      </Section>

      <Section
        title="Graph Agent"
        icon={Network}
        color="violet"
        badge={gaNodes > 0 ? gaNodes : null}
      >
        <GraphAgentSection data={ga} />
      </Section>

      <Section
        title="Entity Resolver"
        icon={Layers}
        color="emerald"
        badge={erTerms > 0 ? erTerms : null}
      >
        <EntityResolverSection data={er} />
      </Section>
    </div>
  )
}

/* ── Main TraceView ──────────────────────────────────────── */
export default function TraceView({ traceHistory }) {
  const [selectedIdx, setSelectedIdx] = useState(0)

  if (!traceHistory || traceHistory.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4 text-center px-8">
        <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-emerald-600/20 to-violet-600/20 border border-emerald-500/20 flex items-center justify-center">
          <Activity size={36} className="text-emerald-400/60" />
        </div>
        <div>
          <p className="text-base font-semibold text-slate-300 mb-1">No traces yet</p>
          <p className="text-sm text-slate-500 max-w-sm">
            Ask questions about your paper in the Chat tab to see detailed agent reasoning traces here.
          </p>
        </div>
      </div>
    )
  }

  const selectedTrace = traceHistory[selectedIdx]

  return (
    <div className="flex flex-col h-full">
      {/* Query selector */}
      <div className="shrink-0 px-4 pt-4 pb-3 border-b border-slate-700/40">
        <label className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest block mb-2">
          Select Query Trace
        </label>
        <select
          value={selectedIdx}
          onChange={e => setSelectedIdx(Number(e.target.value))}
          className="w-full bg-surface-700/60 border border-slate-600/50 rounded-xl px-3 py-2.5 text-sm text-slate-200 outline-none focus:border-primary-500/60 transition-colors"
        >
          {traceHistory.map((t, i) => (
            <option key={i} value={i}>
              {`[${i + 1}] ${(t.query ?? 'Query').slice(0, 60)}${(t.query?.length ?? 0) > 60 ? '…' : ''}`}
            </option>
          ))}
        </select>
      </div>

      {/* Trace content */}
      <div className="flex-1 overflow-y-auto p-4">
        {selectedTrace ? (
          <TraceCard trace={selectedTrace} />
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className="text-sm text-slate-600">Select a query to inspect its trace.</p>
          </div>
        )}
      </div>
    </div>
  )
}
