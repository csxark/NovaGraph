import React, { useState, useRef, useCallback, useEffect } from 'react'
import {
  Upload, FileText, CheckCircle, XCircle, Loader2,
  AlertCircle, RefreshCw, File
} from 'lucide-react'
import { uploadPDF, getJobStatus } from '../api/client.js'

const STAGE_LABELS = {
  pdf_parse:         'Parsing PDF',
  domain_detection:  'Detecting Domain',
  entity_extraction: 'Extracting Entities',
  embedding:         'Generating Embeddings',
  neo4j_write:       'Writing Graph',
  pinecone_upsert:   'Indexing Vectors',
}

const STAGE_ORDER = Object.keys(STAGE_LABELS)

const MAX_SIZE_MB = 50

function formatBytes(bytes) {
  if (bytes < 1024)          return `${bytes} B`
  if (bytes < 1024 * 1024)   return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatDuration(startMs, endMs) {
  if (!startMs || !endMs) return null
  const secs = ((endMs - startMs) / 1000).toFixed(1)
  return `${secs}s`
}

export default function UploadPanel({ onUploadComplete }) {
  const [isDragging,  setIsDragging]  = useState(false)
  const [file,        setFile]        = useState(null)
  const [jobId,       setJobId]       = useState(null)
  const [status,      setStatus]      = useState('idle')  // idle | uploading | polling | completed | failed
  const [stages,      setStages]      = useState({})
  const [error,       setError]       = useState(null)
  const [isPolling,   setIsPolling]   = useState(false)
  const [uploadedPaperId, setUploadedPaperId] = useState(null)

  const inputRef     = useRef(null)
  const pollInterval = useRef(null)

  /* ── Polling ──────────────────────────────────────────────── */
  useEffect(() => {
    if (!isPolling || !jobId) return
    pollInterval.current = setInterval(async () => {
      try {
        const data = await getJobStatus(jobId)
        setStages(data.stages ?? {})
        if (data.status === 'completed') {
          clearInterval(pollInterval.current)
          setIsPolling(false)
          setStatus('completed')
          setUploadedPaperId(data.paper_id)
          onUploadComplete(data.paper_id, data.title ?? file?.name ?? 'Untitled')
        } else if (data.status === 'failed') {
          clearInterval(pollInterval.current)
          setIsPolling(false)
          setStatus('failed')
          setError(data.error ?? 'Processing failed. Please try again.')
        }
      } catch (err) {
        clearInterval(pollInterval.current)
        setIsPolling(false)
        setStatus('failed')
        setError(err.message)
      }
    }, 2000)

    return () => clearInterval(pollInterval.current)
  }, [isPolling, jobId, onUploadComplete, file])

  /* ── File validation ──────────────────────────────────────── */
  const validateFile = useCallback((f) => {
    if (!f) return 'No file selected.'
    if (f.type !== 'application/pdf' && !f.name.endsWith('.pdf'))
      return 'Only PDF files are accepted.'
    if (f.size > MAX_SIZE_MB * 1024 * 1024)
      return `File exceeds ${MAX_SIZE_MB} MB limit.`
    return null
  }, [])

  const handleFileSelect = useCallback((f) => {
    const err = validateFile(f)
    if (err) { setError(err); return }
    setError(null)
    setFile(f)
    setStatus('idle')
    setStages({})
    setJobId(null)
    setIsPolling(false)
    clearInterval(pollInterval.current)
  }, [validateFile])

  /* ── Drag handlers ────────────────────────────────────────── */
  const onDragOver  = (e) => { e.preventDefault(); setIsDragging(true)  }
  const onDragLeave = (e) => { e.preventDefault(); setIsDragging(false) }
  const onDrop      = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) handleFileSelect(dropped)
  }

  /* ── Upload ───────────────────────────────────────────────── */
  const handleUpload = async () => {
    if (!file) return
    setError(null)
    setStatus('uploading')
    try {
      const res = await uploadPDF(file)
      setJobId(res.job_id)
      setStatus('polling')
      setIsPolling(true)
    } catch (err) {
      setStatus('failed')
      setError(err.message)
    }
  }

  /* ── Reset ────────────────────────────────────────────────── */
  const handleReset = () => {
    clearInterval(pollInterval.current)
    setFile(null)
    setJobId(null)
    setStatus('idle')
    setStages({})
    setError(null)
    setIsPolling(false)
  }

  /* ── Derived ──────────────────────────────────────────────── */
  const isProcessing = status === 'uploading' || status === 'polling'

  const completedStages = Object.values(stages).filter(s => s?.status === 'completed').length
  const totalStages     = STAGE_ORDER.length
  const progressPct     = totalStages > 0 ? Math.round((completedStages / totalStages) * 100) : 0

  return (
    <div className="glass rounded-2xl p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold text-slate-200 flex items-center gap-2">
          <Upload size={15} className="text-primary-400" />
          Upload Paper
        </h2>
        {(file || status !== 'idle') && (
          <button
            onClick={handleReset}
            className="text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1 transition-colors"
          >
            <RefreshCw size={11} />
            Reset
          </button>
        )}
      </div>

      {/* Drop zone */}
      {!file && (
        <div
          className={`drag-zone border-2 border-dashed rounded-xl p-6 flex flex-col items-center gap-3 cursor-pointer transition-all
            ${isDragging ? 'dragging border-primary-500' : 'border-slate-600 hover:border-slate-500'}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
        >
          <div className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-all
            ${isDragging ? 'bg-primary-500/20' : 'bg-slate-700/60'}`}>
            <Upload size={22} className={isDragging ? 'text-primary-400' : 'text-slate-400'} />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-slate-300">
              {isDragging ? 'Drop to upload' : 'Drop PDF here'}
            </p>
            <p className="text-xs text-slate-500 mt-0.5">or click to browse · max {MAX_SIZE_MB} MB</p>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,application/pdf"
            className="hidden"
            onChange={e => { if (e.target.files[0]) handleFileSelect(e.target.files[0]) }}
          />
        </div>
      )}

      {/* File info */}
      {file && status === 'idle' && (
        <div className="animate-fade-in flex items-start gap-3 p-3 rounded-xl bg-surface-700/50 border border-slate-600/50">
          <div className="w-10 h-10 rounded-lg bg-primary-500/15 flex items-center justify-center shrink-0">
            <FileText size={18} className="text-primary-400" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-slate-200 truncate">{file.name}</p>
            <p className="text-xs text-slate-500 mt-0.5">{formatBytes(file.size)}</p>
          </div>
        </div>
      )}

      {/* Upload button */}
      {file && status === 'idle' && (
        <button
          onClick={handleUpload}
          className="w-full py-2.5 rounded-xl bg-gradient-to-r from-primary-600 to-accent-600 hover:from-primary-500 hover:to-accent-500 text-white text-sm font-semibold transition-all shadow-lg glow-primary"
        >
          Process Paper
        </button>
      )}

      {/* Processing state */}
      {isProcessing && (
        <div className="animate-fade-in space-y-3">
          {/* File row */}
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <File size={12} />
            <span className="truncate">{file?.name}</span>
          </div>

          {/* Progress bar */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-slate-500">
              <span>Processing…</span>
              <span>{progressPct}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-slate-700 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-primary-500 to-accent-500 rounded-full transition-all duration-500"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>

          {/* Stage list */}
          <div className="space-y-0">
            {STAGE_ORDER.map((key, i) => {
              const stage = stages[key]
              const stageStatus = stage?.status ?? 'pending'
              const duration = formatDuration(stage?.started_at, stage?.completed_at)
              const isActive   = stageStatus === 'processing'
              const isDone     = stageStatus === 'completed'
              const isFailed   = stageStatus === 'failed'
              const isLast     = i === STAGE_ORDER.length - 1

              return (
                <div key={key} className="flex gap-3">
                  {/* connector + icon column */}
                  <div className="flex flex-col items-center">
                    <StageIcon status={stageStatus} />
                    {!isLast && (
                      <div className={`step-connector ${isDone ? 'done' : isActive ? 'active' : ''}`} />
                    )}
                  </div>
                  {/* label */}
                  <div className={`pb-3 pt-0.5 flex-1 flex justify-between items-start
                    ${isActive ? 'text-primary-300' : isDone ? 'text-emerald-400' : isFailed ? 'text-red-400' : 'text-slate-500'}
                  `}>
                    <span className="text-xs font-medium">{STAGE_LABELS[key]}</span>
                    {duration && <span className="text-[10px] text-slate-600 ml-2 shrink-0">{duration}</span>}
                    {isActive && <span className="spinner spinner-sm ml-2 shrink-0" />}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Completed state */}
      {status === 'completed' && (
        <div className="animate-slide-up flex flex-col items-center gap-3 py-2">
          <div className="w-12 h-12 rounded-2xl bg-emerald-500/15 flex items-center justify-center glow-success">
            <CheckCircle size={24} className="text-emerald-400" />
          </div>
          <div className="text-center">
            <p className="text-sm font-semibold text-emerald-400">Graph Ready!</p>
            <p className="text-xs text-slate-500 mt-0.5">You can now chat with this paper</p>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <File size={11} className="text-slate-500" />
            <span className="text-slate-500 truncate max-w-[180px]">{file?.name}</span>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="animate-slide-up p-3 rounded-xl bg-red-500/10 border border-red-500/20 flex items-start gap-2.5">
          <AlertCircle size={14} className="text-red-400 shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-red-400">Error</p>
            <p className="text-xs text-red-300/80 mt-0.5 break-words">{error}</p>
          </div>
        </div>
      )}

      {status === 'failed' && (
        <button
          onClick={handleReset}
          className="w-full py-2 rounded-xl border border-slate-600 text-slate-400 hover:text-slate-200 hover:border-slate-500 text-xs font-medium transition-all flex items-center justify-center gap-2"
        >
          <RefreshCw size={12} />
          Try Again
        </button>
      )}
    </div>
  )
}

function StageIcon({ status }) {
  const base = 'w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5'
  if (status === 'completed')  return (
    <div className={`${base} bg-emerald-500/20`}>
      <CheckCircle size={13} className="text-emerald-400" />
    </div>
  )
  if (status === 'processing') return (
    <div className={`${base} bg-primary-500/20`}>
      <Loader2 size={13} className="text-primary-400 animate-spin" />
    </div>
  )
  if (status === 'failed')     return (
    <div className={`${base} bg-red-500/20`}>
      <XCircle size={13} className="text-red-400" />
    </div>
  )
  return (
    <div className={`${base} bg-slate-700/60`}>
      <div className="w-2 h-2 rounded-full bg-slate-600" />
    </div>
  )
}
