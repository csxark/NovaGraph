import React, { useState, useCallback, useEffect } from 'react'
import { Brain, GitBranch, Github, Zap, Activity } from 'lucide-react'
import UploadPanel from './components/UploadPanel.jsx'
import ChatInterface from './components/ChatInterface.jsx'
import GraphView from './components/GraphView.jsx'
import TraceView from './components/TraceView.jsx'
import { getGraph } from './api/client.js'

const TABS = [
  { id: 'chat',  label: 'Chat',        icon: Brain   },
  { id: 'graph', label: 'Graph',       icon: GitBranch },
  { id: 'trace', label: 'Trace',       icon: Activity  },
]

export default function App() {
  const [uploadedPaperId, setUploadedPaperId] = useState(null)
  const [paperTitle,      setPaperTitle]      = useState('')
  const [messages,        setMessages]        = useState([])
  const [traceHistory,    setTraceHistory]    = useState([])
  const [graphData,       setGraphData]       = useState(null)
  const [activeTab,       setActiveTab]       = useState('chat')
  const [graphLoading,    setGraphLoading]    = useState(false)

  /* When a paper is successfully processed, fetch its graph */
  useEffect(() => {
    if (!uploadedPaperId) return
    setGraphLoading(true)
    setGraphData(null)
    getGraph(uploadedPaperId)
      .then(data  => setGraphData(data))
      .catch(err  => console.error('Graph fetch failed:', err))
      .finally(() => setGraphLoading(false))
  }, [uploadedPaperId])

  const handleUploadComplete = useCallback((paperId, title) => {
    setUploadedPaperId(paperId)
    setPaperTitle(title)
    setMessages([])
    setTraceHistory([])
    setGraphData(null)
    setActiveTab('chat')
  }, [])

  const handleNewMessage = useCallback((msg) => {
    setMessages(prev => [...prev, msg])
    /* Capture trace if present */
    if (msg.trace) {
      setTraceHistory(prev => [
        { ...msg.trace, query: msg.query, timestamp: msg.timestamp },
        ...prev
      ])
    }
  }, [])

  const nodeCount = graphData?.nodes?.length ?? 0
  const edgeCount = graphData?.edges?.length ?? 0

  return (
    <div className="relative min-h-screen bg-surface-900 text-slate-100 overflow-hidden flex flex-col">
      {/* Ambient background orbs */}
      <div className="bg-orb w-96 h-96 opacity-10"
           style={{ background: 'radial-gradient(circle, #0ea5e9, transparent)', top: '-80px', left: '-80px' }} />
      <div className="bg-orb w-80 h-80 opacity-10"
           style={{ background: 'radial-gradient(circle, #8b5cf6, transparent)', bottom: '-60px', right: '-60px', animationDelay: '4s' }} />
      <div className="bg-orb w-64 h-64 opacity-5"
           style={{ background: 'radial-gradient(circle, #10b981, transparent)', top: '40%', left: '50%', animationDelay: '8s' }} />

      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="relative z-10 glass border-b border-slate-700/50 px-6 py-3 flex items-center gap-4 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shadow-lg glow-primary">
            <Brain size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold gradient-text leading-tight">
              GraphRAG Research Assistant
            </h1>
            <p className="text-xs text-slate-500 leading-tight">
              Knowledge-graph powered AI research
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 ml-auto">
          {uploadedPaperId && (
            <div className="flex items-center gap-2 text-xs">
              <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-medium">
                <Zap size={11} />
                {nodeCount} nodes
              </span>
              <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-primary-500/10 border border-primary-500/20 text-primary-400 font-medium">
                <GitBranch size={11} />
                {edgeCount} edges
              </span>
            </div>
          )}
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="ml-2 p-2 rounded-lg hover:bg-slate-700/60 transition-colors text-slate-400 hover:text-slate-200"
          >
            <Github size={18} />
          </a>
        </div>
      </header>

      {/* ── Main body ────────────────────────────────────────────── */}
      <div className="relative z-10 flex flex-1 min-h-0">

        {/* Left sidebar */}
        <aside className="w-80 shrink-0 flex flex-col gap-3 p-4 border-r border-slate-700/40 overflow-y-auto">
          <UploadPanel onUploadComplete={handleUploadComplete} />

          {/* Paper info card */}
          {uploadedPaperId && (
            <div className="glass rounded-xl p-4 animate-fade-in">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-2">Loaded Paper</p>
              <p className="text-sm font-semibold text-slate-100 line-clamp-3 leading-snug mb-3">
                {paperTitle || 'Untitled Document'}
              </p>
              <div className="grid grid-cols-3 gap-2 text-center">
                <Stat label="Nodes"  value={graphLoading ? '…' : nodeCount} color="text-accent-400" />
                <Stat label="Edges"  value={graphLoading ? '…' : edgeCount} color="text-primary-400" />
                <Stat label="Chats"  value={messages.filter(m => m.role === 'user').length} color="text-emerald-400" />
              </div>
              <div className="mt-3 pt-3 border-t border-slate-700/50">
                <p className="text-[10px] text-slate-500 font-mono truncate">
                  ID: {uploadedPaperId}
                </p>
              </div>
            </div>
          )}

          {/* Quick-start tips */}
          {!uploadedPaperId && (
            <div className="glass rounded-xl p-4 animate-fade-in">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">Quick Start</p>
              <ol className="space-y-2.5 text-xs text-slate-400">
                {[
                  'Upload a research PDF (≤ 50 MB)',
                  'Wait for knowledge graph extraction',
                  'Chat with the paper using AI',
                  'Explore the interactive graph',
                  'Inspect agent traces for transparency'
                ].map((tip, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <span className="shrink-0 w-5 h-5 rounded-full bg-gradient-to-br from-primary-600 to-accent-600 text-white text-[10px] flex items-center justify-center font-bold mt-0.5">
                      {i + 1}
                    </span>
                    {tip}
                  </li>
                ))}
              </ol>
            </div>
          )}
        </aside>

        {/* Main panel */}
        <main className="flex-1 flex flex-col min-w-0 min-h-0">
          {/* Tab bar */}
          <div className="shrink-0 flex items-center gap-1 px-4 pt-3 pb-0 border-b border-slate-700/40">
            {TABS.map(tab => {
              const Icon = tab.icon
              const isActive = activeTab === tab.id
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg transition-all duration-200
                    ${isActive
                      ? 'text-primary-400 bg-surface-800/80 border border-b-transparent border-slate-700/40'
                      : 'text-slate-500 hover:text-slate-300 hover:bg-slate-700/30'
                    }
                  `}
                >
                  <Icon size={15} />
                  {tab.label}
                  {tab.id === 'trace' && traceHistory.length > 0 && (
                    <span className="ml-0.5 px-1.5 py-0.5 text-[10px] font-bold bg-accent-600 text-white rounded-full leading-none">
                      {traceHistory.length}
                    </span>
                  )}
                </button>
              )
            })}
          </div>

          {/* Panel content */}
          <div className="flex-1 min-h-0 overflow-hidden bg-surface-800/30">
            {activeTab === 'chat' && (
              <ChatInterface
                paperId={uploadedPaperId}
                messages={messages}
                onNewMessage={handleNewMessage}
                isDisabled={!uploadedPaperId}
              />
            )}
            {activeTab === 'graph' && (
              <GraphView
                graphData={graphData}
                paperId={uploadedPaperId}
                isLoading={graphLoading}
              />
            )}
            {activeTab === 'trace' && (
              <TraceView traceHistory={traceHistory} />
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

function Stat({ label, value, color }) {
  return (
    <div className="rounded-lg bg-surface-700/50 py-2 px-1">
      <p className={`text-lg font-bold leading-none ${color}`}>{value}</p>
      <p className="text-[10px] text-slate-500 mt-0.5">{label}</p>
    </div>
  )
}
