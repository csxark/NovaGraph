import React, { useState, useRef, useEffect, useCallback } from 'react'
import {
  Send, Brain, ChevronDown, ChevronUp,
  User, BookOpen, Tag, Clock
} from 'lucide-react'
import { queryGraph } from '../api/client.js'

const ENTITY_TYPE_COLORS = {
  Concept:     'bg-violet-500/20 text-violet-300 border-violet-500/30',
  Method:      'bg-sky-500/20    text-sky-300    border-sky-500/30',
  Evidence:    'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  Finding:     'bg-amber-500/20  text-amber-300   border-amber-500/30',
  Entity:      'bg-rose-500/20   text-rose-300    border-rose-500/30',
  Reference:   'bg-indigo-500/20 text-indigo-300  border-indigo-500/30',
  Proposition: 'bg-pink-500/20   text-pink-300    border-pink-500/30',
  Assumption:  'bg-teal-500/20   text-teal-300    border-teal-500/30',
}

function entityBadgeClass(type) {
  return ENTITY_TYPE_COLORS[type] ?? 'bg-slate-600/40 text-slate-300 border-slate-500/30'
}

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/* ── Bouncing dots ──────────────────────────────────────────── */
function LoadingDots() {
  return (
    <div className="flex items-center gap-2 px-4 py-3 glass rounded-2xl rounded-bl-sm w-fit max-w-xs">
      <Brain size={14} className="text-accent-400 shrink-0" />
      <div className="dot-flashing flex gap-1">
        <span /><span /><span />
      </div>
    </div>
  )
}

/* ── Source list ────────────────────────────────────────────── */
function SourceList({ sources }) {
  const [open, setOpen] = useState(false)
  if (!sources || sources.length === 0) return null

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors"
      >
        <BookOpen size={11} />
        {sources.length} source{sources.length !== 1 ? 's' : ''}
        {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
      </button>

      {open && (
        <div className="mt-2 space-y-2 animate-fade-in">
          {sources.map((src, i) => (
            <div
              key={i}
              className="p-2.5 rounded-xl bg-surface-700/60 border border-slate-700/40 text-xs"
            >
              <div className="flex items-start justify-between gap-2 mb-1">
                <p className="font-semibold text-slate-200 leading-snug flex-1">
                  {src.name ?? src.title ?? `Source ${i + 1}`}
                </p>
                {src.type && (
                  <span className={`shrink-0 px-2 py-0.5 rounded-full text-[10px] font-semibold border source-badge ${entityBadgeClass(src.type)}`}>
                    <Tag size={8} className="inline mr-0.5" />
                    {src.type}
                  </span>
                )}
              </div>
              {src.description && (
                <p className="text-slate-400 leading-relaxed line-clamp-3">{src.description}</p>
              )}
              {src.score != null && (
                <div className="mt-1.5 flex items-center gap-1.5">
                  <div className="flex-1 h-1 rounded-full bg-slate-700">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-primary-500 to-accent-500"
                      style={{ width: `${Math.round(src.score * 100)}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-slate-500">{(src.score * 100).toFixed(0)}%</span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ── Single message bubble ──────────────────────────────────── */
function MessageBubble({ msg }) {
  const isUser = msg.role === 'user'

  return (
    <div className={`flex gap-3 msg-enter ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar */}
      <div className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 mt-0.5
        ${isUser
          ? 'bg-gradient-to-br from-primary-600 to-primary-500'
          : 'bg-gradient-to-br from-accent-600 to-accent-500'}`}
      >
        {isUser ? <User size={14} className="text-white" /> : <Brain size={14} className="text-white" />}
      </div>

      {/* Content */}
      <div className={`flex flex-col gap-1 max-w-[78%] ${isUser ? 'items-end' : 'items-start'}`}>
        <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap
          ${isUser
            ? 'bg-gradient-to-br from-primary-600/80 to-primary-700/80 text-white rounded-br-sm border border-primary-500/30'
            : 'glass text-slate-100 rounded-bl-sm'
          }`}
        >
          {msg.content}
        </div>

        {!isUser && <SourceList sources={msg.sources} />}

        <div className="flex items-center gap-1 text-[10px] text-slate-600 px-1">
          <Clock size={9} />
          {formatTime(msg.timestamp)}
        </div>
      </div>
    </div>
  )
}

/* ── Empty / disabled state ─────────────────────────────────── */
function EmptyState({ isDisabled }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center px-8">
      <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-accent-600/20 to-primary-600/20 border border-accent-500/20 flex items-center justify-center">
        <Brain size={36} className="text-accent-400/70" />
      </div>
      <div>
        <p className="text-base font-semibold text-slate-300 mb-1">
          {isDisabled ? 'No paper loaded' : 'Start a conversation'}
        </p>
        <p className="text-sm text-slate-500 max-w-sm">
          {isDisabled
            ? 'Upload a PDF in the sidebar to extract its knowledge graph and start asking questions.'
            : 'Ask anything about the paper — methods, findings, comparisons, or explanations.'
          }
        </p>
      </div>
      {isDisabled && (
        <div className="flex flex-wrap justify-center gap-2 mt-2">
          {['What is the main contribution?', 'Explain the methodology', 'What are the key findings?'].map(q => (
            <span key={q} className="px-3 py-1.5 rounded-lg bg-surface-700/50 border border-slate-700/50 text-xs text-slate-500">
              {q}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

/* ── Main ChatInterface ─────────────────────────────────────── */
export default function ChatInterface({ paperId, messages, onNewMessage, isDisabled }) {
  const [input,     setInput]     = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const bottomRef   = useRef(null)
  const textareaRef = useRef(null)

  /* Auto-scroll */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const handleSubmit = useCallback(async () => {
    const query = input.trim()
    if (!query || isLoading || isDisabled) return

    const userMsg = {
      id:        `user-${Date.now()}`,
      role:      'user',
      content:   query,
      timestamp: Date.now(),
    }
    onNewMessage(userMsg)
    setInput('')
    setIsLoading(true)

    try {
      const res = await queryGraph(paperId, query, 5, true)
      const assistantMsg = {
        id:        `ai-${Date.now()}`,
        role:      'assistant',
        content:   res.answer ?? 'No answer returned.',
        sources:   res.sources ?? [],
        query,
        trace:     res.trace ?? null,
        timestamp: Date.now(),
      }
      onNewMessage(assistantMsg)
    } catch (err) {
      const errMsg = {
        id:        `err-${Date.now()}`,
        role:      'assistant',
        content:   `⚠️ Error: ${err.message}`,
        sources:   [],
        timestamp: Date.now(),
      }
      onNewMessage(errMsg)
    } finally {
      setIsLoading(false)
    }
  }, [input, isLoading, isDisabled, paperId, onNewMessage])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const hasMessages = messages.length > 0

  return (
    <div className="flex flex-col h-full">
      {/* Message list */}
      <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5">
        {!hasMessages && <EmptyState isDisabled={isDisabled} />}

        {hasMessages && messages.map(msg => (
          <MessageBubble key={msg.id} msg={msg} />
        ))}

        {isLoading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent-600 to-accent-500 flex items-center justify-center shrink-0">
              <Brain size={14} className="text-white" />
            </div>
            <LoadingDots />
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Suggested questions (shown when paper loaded but no messages yet) */}
      {!isDisabled && !hasMessages && (
        <div className="shrink-0 px-6 pb-3 flex flex-wrap gap-2">
          {[
            'What is the main contribution of this paper?',
            'Describe the experimental setup',
            'What datasets were used?',
            'What are the limitations?',
          ].map(q => (
            <button
              key={q}
              onClick={() => { setInput(q); textareaRef.current?.focus() }}
              className="px-3 py-1.5 rounded-xl bg-surface-700/60 border border-slate-700/50 text-xs text-slate-400 hover:text-slate-200 hover:border-primary-500/40 hover:bg-primary-500/10 transition-all"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {/* Input bar */}
      <div className="shrink-0 px-4 pb-4 pt-2 border-t border-slate-700/40">
        <div className={`flex items-end gap-3 glass rounded-2xl px-4 py-3 transition-all
          ${isDisabled ? 'opacity-50 cursor-not-allowed' : 'focus-within:border-primary-500/50 focus-within:glow-primary'}`}
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isDisabled || isLoading}
            placeholder={isDisabled ? 'Upload a paper first…' : 'Ask about this paper… (Enter to send)'}
            rows={1}
            className="flex-1 bg-transparent resize-none text-sm text-slate-100 placeholder-slate-600 outline-none leading-relaxed max-h-32"
            style={{ overflowY: 'auto' }}
            onInput={e => {
              e.target.style.height = 'auto'
              e.target.style.height = `${Math.min(e.target.scrollHeight, 128)}px`
            }}
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading || isDisabled}
            className={`shrink-0 w-9 h-9 rounded-xl flex items-center justify-center transition-all
              ${(!input.trim() || isLoading || isDisabled)
                ? 'bg-slate-700/60 text-slate-600 cursor-not-allowed'
                : 'bg-gradient-to-br from-primary-600 to-accent-600 text-white hover:from-primary-500 hover:to-accent-500 shadow-lg glow-primary'
              }`}
          >
            {isLoading
              ? <span className="spinner spinner-sm" />
              : <Send size={15} />
            }
          </button>
        </div>
        <p className="text-[10px] text-slate-700 text-center mt-2">
          Shift + Enter for new line · Enter to send
        </p>
      </div>
    </div>
  )
}
