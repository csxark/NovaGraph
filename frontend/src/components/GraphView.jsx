import React, { useRef, useEffect, useState, useCallback } from 'react'
import * as d3 from 'd3'
import { GitBranch, Info, X, Loader2 } from 'lucide-react'

/* ── Node colour map ──────────────────────────────────────── */
const NODE_COLORS = {
  Concept:     '#8b5cf6',
  Method:      '#0ea5e9',
  Evidence:    '#10b981',
  Finding:     '#f59e0b',
  Entity:      '#ef4444',
  Reference:   '#6366f1',
  Proposition: '#ec4899',
  Assumption:  '#14b8a6',
}
const DEFAULT_COLOR = '#94a3b8'

function nodeColor(type) {
  return NODE_COLORS[type] ?? DEFAULT_COLOR
}

/* ── Legend ───────────────────────────────────────────────── */
function Legend() {
  return (
    <div className="absolute bottom-4 left-4 glass rounded-xl p-3 max-w-[180px]">
      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Node Types</p>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
        {Object.entries(NODE_COLORS).map(([type, color]) => (
          <div key={type} className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
            <span className="text-[10px] text-slate-400 leading-none">{type}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── Node detail sidebar ─────────────────────────────────── */
function NodeDetail({ node, onClose }) {
  if (!node) return null
  const color = nodeColor(node.type)

  return (
    <div className="absolute top-4 right-4 w-64 glass rounded-xl p-4 animate-slide-up">
      <div className="flex items-start justify-between gap-2 mb-3">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: color }} />
          <p className="text-sm font-bold text-slate-100 leading-snug truncate">{node.name ?? node.id}</p>
        </div>
        <button
          onClick={onClose}
          className="shrink-0 p-1 rounded-lg hover:bg-slate-700/60 text-slate-500 hover:text-slate-300 transition-colors"
        >
          <X size={13} />
        </button>
      </div>

      <div className="space-y-2 text-xs">
        <DetailRow label="Type" value={
          <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold"
                style={{ background: `${color}22`, color, border: `1px solid ${color}44` }}>
            {node.type ?? 'Unknown'}
          </span>
        } />
        {node.confidence != null && (
          <DetailRow label="Confidence" value={
            <div className="flex items-center gap-2 flex-1">
              <div className="flex-1 h-1.5 rounded-full bg-slate-700">
                <div
                  className="h-full rounded-full"
                  style={{ width: `${Math.round((node.confidence ?? 0) * 100)}%`, background: color }}
                />
              </div>
              <span className="text-slate-400">{((node.confidence ?? 0) * 100).toFixed(0)}%</span>
            </div>
          } />
        )}
        {node.description && (
          <div>
            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-1">Description</p>
            <p className="text-slate-300 leading-relaxed">{node.description}</p>
          </div>
        )}
        {node.properties && Object.entries(node.properties).map(([k, v]) => (
          k !== 'name' && k !== 'type' && k !== 'description' && k !== 'confidence' && (
            <DetailRow key={k} label={k} value={String(v)} />
          )
        ))}
      </div>
    </div>
  )
}

function DetailRow({ label, value }) {
  return (
    <div className="flex items-start gap-2">
      <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest w-20 shrink-0 mt-0.5">{label}</span>
      <div className="flex-1 text-slate-300">{value}</div>
    </div>
  )
}

/* ── Stats bar ────────────────────────────────────────────── */
function StatsBar({ graphData, paperId }) {
  const nodeCount = graphData?.nodes?.length ?? 0
  const edgeCount = graphData?.edges?.length ?? 0
  return (
    <div className="absolute top-4 left-4 flex items-center gap-2">
      <div className="glass rounded-lg px-3 py-1.5 flex items-center gap-3 text-xs">
        <span className="text-slate-400">{nodeCount} <span className="text-slate-600">nodes</span></span>
        <span className="w-px h-3 bg-slate-700" />
        <span className="text-slate-400">{edgeCount} <span className="text-slate-600">edges</span></span>
        {paperId && (
          <>
            <span className="w-px h-3 bg-slate-700" />
            <span className="text-slate-600 font-mono text-[10px]">{paperId.slice(0, 12)}…</span>
          </>
        )}
      </div>
    </div>
  )
}

/* ── D3 Graph ─────────────────────────────────────────────── */
export default function GraphView({ graphData, paperId, isLoading }) {
  const svgRef       = useRef(null)
  const tooltipRef   = useRef(null)
  const simulationRef = useRef(null)
  const [selectedNode, setSelectedNode] = useState(null)

  /* ── D3 rendering ─────────────────────────────────────── */
  useEffect(() => {
    if (!graphData || !svgRef.current) return

    const nodes = (graphData.nodes ?? []).map(n => ({ ...n }))
    const edges = (graphData.edges ?? []).map(e => ({ ...e }))

    const container = svgRef.current.parentElement
    const W = container.clientWidth  || 800
    const H = container.clientHeight || 600

    /* Clear previous */
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', W)
      .attr('height', H)
      .attr('class', 'graph-svg')

    /* Zoom */
    const gRoot = svg.append('g')
    const zoom = d3.zoom()
      .scaleExtent([0.2, 4])
      .on('zoom', (event) => gRoot.attr('transform', event.transform))
    svg.call(zoom)

    /* Arrow markers */
    const defs = svg.append('defs')
    Object.entries(NODE_COLORS).forEach(([type, color]) => {
      defs.append('marker')
        .attr('id',           `arrow-${type}`)
        .attr('viewBox',      '0 -5 10 10')
        .attr('refX',         22)
        .attr('refY',         0)
        .attr('markerWidth',  6)
        .attr('markerHeight', 6)
        .attr('orient',       'auto')
        .append('path')
        .attr('d',    'M0,-5L10,0L0,5')
        .attr('fill', color)
        .attr('opacity', 0.6)
    })
    defs.append('marker')
      .attr('id',           'arrow-default')
      .attr('viewBox',      '0 -5 10 10')
      .attr('refX',         22)
      .attr('refY',         0)
      .attr('markerWidth',  6)
      .attr('markerHeight', 6)
      .attr('orient',       'auto')
      .append('path')
      .attr('d',    'M0,-5L10,0L0,5')
      .attr('fill', DEFAULT_COLOR)
      .attr('opacity', 0.5)

    /* Build id→index map */
    const nodeById = new Map(nodes.map((n, i) => [n.id, i]))

    /* Resolve source/target to objects */
    const linkedEdges = edges
      .map(e => ({
        ...e,
        source: nodeById.has(e.source ?? e.from) ? nodes[nodeById.get(e.source ?? e.from)] : e.source ?? e.from,
        target: nodeById.has(e.target ?? e.to)   ? nodes[nodeById.get(e.target ?? e.to)]   : e.target ?? e.to,
      }))
      .filter(e => typeof e.source === 'object' && typeof e.target === 'object')

    /* Simulation */
    const simulation = d3.forceSimulation(nodes)
      .force('link',    d3.forceLink(linkedEdges).id(d => d.id).distance(100))
      .force('charge',  d3.forceManyBody().strength(-200))
      .force('center',  d3.forceCenter(W / 2, H / 2))
      .force('collide', d3.forceCollide(30))
    simulationRef.current = simulation

    /* Edge lines */
    const linkGroup = gRoot.append('g').attr('class', 'links')
    const linkEl = linkGroup.selectAll('line')
      .data(linkedEdges)
      .enter().append('line')
      .attr('class',        'graph-link')
      .attr('stroke',       d => {
        const tgt = typeof d.target === 'object' ? d.target : nodes.find(n => n.id === d.target)
        return nodeColor(tgt?.type)
      })
      .attr('stroke-width', 1.5)
      .attr('marker-end',   d => {
        const tgt = typeof d.target === 'object' ? d.target : nodes.find(n => n.id === d.target)
        const type = tgt?.type
        return `url(#${NODE_COLORS[type] ? `arrow-${type}` : 'arrow-default'})`
      })

    /* Edge labels */
    const edgeLabelEl = gRoot.append('g').attr('class', 'edge-labels')
      .selectAll('text')
      .data(linkedEdges)
      .enter().append('text')
      .attr('class', 'edge-label')
      .attr('font-size', 9)
      .attr('fill', '#475569')
      .attr('text-anchor', 'middle')
      .text(d => d.type ?? d.relationship ?? d.label ?? '')

    /* Node groups */
    const nodeGroup = gRoot.append('g').attr('class', 'nodes')
    const nodeEl = nodeGroup.selectAll('g')
      .data(nodes)
      .enter().append('g')
      .attr('class', 'graph-node')
      .call(
        d3.drag()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart()
            d.fx = d.x; d.fy = d.y
          })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0)
            d.fx = null; d.fy = null
          })
      )

    nodeEl.append('circle')
      .attr('r',    12)
      .attr('fill', d => nodeColor(d.type))
      .attr('fill-opacity', 0.85)
      .attr('stroke', d => nodeColor(d.type))
      .attr('stroke-width', 0)

    /* Node label */
    nodeEl.append('text')
      .attr('dy',          20)
      .attr('text-anchor', 'middle')
      .attr('font-size',   9)
      .attr('fill',        '#94a3b8')
      .text(d => {
        const label = d.name ?? d.id ?? ''
        return label.length > 18 ? label.slice(0, 16) + '…' : label
      })

    /* Tooltip + click */
    const tooltip = d3.select(tooltipRef.current)

    nodeEl
      .on('mouseover', (event, d) => {
        tooltip
          .style('opacity', 1)
          .html(`
            <div class="tooltip-title">${d.name ?? d.id}</div>
            <span class="tooltip-type" style="background:${nodeColor(d.type)}22;color:${nodeColor(d.type)};border:1px solid ${nodeColor(d.type)}44">${d.type ?? 'Node'}</span>
            ${d.description ? `<div class="tooltip-desc">${d.description.slice(0, 120)}${d.description.length > 120 ? '…' : ''}</div>` : ''}
          `)
      })
      .on('mousemove', (event) => {
        const [mx, my] = d3.pointer(event, document.body)
        tooltip
          .style('left', `${mx + 14}px`)
          .style('top',  `${my - 10}px`)
      })
      .on('mouseleave', () => tooltip.style('opacity', 0))
      .on('click', (event, d) => {
        event.stopPropagation()
        setSelectedNode(d)
        nodeGroup.selectAll('g').classed('selected', false)
        d3.select(event.currentTarget).classed('selected', true)
      })

    svg.on('click', () => {
      setSelectedNode(null)
      nodeGroup.selectAll('g').classed('selected', false)
    })

    /* Tick */
    simulation.on('tick', () => {
      linkEl
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y)

      edgeLabelEl
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2 - 4)

      nodeEl.attr('transform', d => `translate(${d.x},${d.y})`)
    })

    /* Initial zoom-to-fit after a short delay */
    setTimeout(() => {
      const bbox = gRoot.node().getBBox()
      if (bbox.width === 0) return
      const scale = Math.min(0.9, 0.9 * Math.min(W / bbox.width, H / bbox.height))
      const tx = W / 2 - scale * (bbox.x + bbox.width  / 2)
      const ty = H / 2 - scale * (bbox.y + bbox.height / 2)
      svg.transition().duration(600).call(
        zoom.transform,
        d3.zoomIdentity.translate(tx, ty).scale(scale)
      )
    }, 600)

    return () => {
      simulation.stop()
      tooltip.style('opacity', 0)
    }
  }, [graphData])

  /* ── Empty / loading states ──────────────────────────── */
  if (!paperId) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4 text-center px-8">
        <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-primary-600/20 to-accent-600/20 border border-primary-500/20 flex items-center justify-center">
          <GitBranch size={36} className="text-primary-400/60" />
        </div>
        <div>
          <p className="text-base font-semibold text-slate-300 mb-1">No graph loaded</p>
          <p className="text-sm text-slate-500">Upload and process a PDF to visualize its knowledge graph.</p>
        </div>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-600/20 to-accent-600/20 flex items-center justify-center">
          <Loader2 size={28} className="text-primary-400 animate-spin" />
        </div>
        <p className="text-sm text-slate-400">Loading knowledge graph…</p>
      </div>
    )
  }

  if (!graphData || (graphData.nodes?.length === 0)) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4 text-center px-8">
        <div className="w-20 h-20 rounded-3xl bg-surface-700/60 border border-slate-700/40 flex items-center justify-center">
          <Info size={32} className="text-slate-500" />
        </div>
        <div>
          <p className="text-base font-semibold text-slate-400 mb-1">Empty graph</p>
          <p className="text-sm text-slate-600">No nodes were extracted from this paper.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative w-full h-full overflow-hidden bg-surface-900/40">
      {/* D3 SVG */}
      <svg ref={svgRef} className="w-full h-full" />

      {/* D3 tooltip (positioned by JS) */}
      <div ref={tooltipRef} className="node-tooltip" style={{ opacity: 0 }} />

      {/* Overlays */}
      <StatsBar graphData={graphData} paperId={paperId} />
      <NodeDetail node={selectedNode} onClose={() => setSelectedNode(null)} />
      <Legend />
    </div>
  )
}
