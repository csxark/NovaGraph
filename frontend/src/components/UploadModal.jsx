import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, 
  UploadCloud, 
  FileText, 
  CheckCircle2, 
  Loader2, 
  AlertCircle,
  RefreshCw 
} from 'lucide-react';
import useAppStore from '../store/appStore';
import { api } from '../lib/api';

export default function UploadModal({ isOpen, onClose }) {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  // Zustand Store
  const {
    setCurrentDocument,
    setIngestionStatus,
    setGraphData,
    addMessage,
    resetAll
  } = useAppStore();

  // Local State
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [timelineStep, setTimelineStep] = useState(1); // 1 to 5
  const [progressPercent, setProgressPercent] = useState(0);
  const [statusText, setStatusText] = useState('');
  const [errorMessage, setErrorMessage] = useState(null);
  const [localDocId, setLocalDocId] = useState(null);
  const [status, setStatus] = useState('idle'); // 'idle' | 'uploading' | 'processing' | 'completed' | 'failed'

  // Polling Effect
  useEffect(() => {
    if (!localDocId || status !== 'processing') return;

    const pollInterval = setInterval(async () => {
      try {
        const res = await api.getStatus(localDocId);
        const jobStatus = res.status;
        const stages = res.stages || [];

        if (jobStatus === 'completed') {
          setStatus('completed');
          setIngestionStatus('completed');
          setProgressPercent(100);
          setTimelineStep(5);
          setStatusText('Ready');
          clearInterval(pollInterval);

          // Fetch graph data
          try {
            const graph = await api.getGraph(localDocId);
            if (graph && graph.nodes) {
              setGraphData(graph.nodes, graph.edges || []);
              addMessage({
                role: 'assistant',
                content: `👋 Your paper is ready! I extracted ${graph.nodes.length} entities and ${graph.edges?.length || 0} relationships from the knowledge graph. What would you like to know?`,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              });
            } else {
              addMessage({
                role: 'assistant',
                content: '⚠️ Paper ingested but graph returned no nodes. The extraction may have failed silently — check backend logs.',
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              });
            }
          } catch (graphErr) {
            console.error('Graph fetch failed after ingestion:', graphErr);
            addMessage({
              role: 'assistant',
              content: `⚠️ Paper ingested but graph fetch failed: ${graphErr.message}`,
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            });
          }

          // Close modal and navigate
          setTimeout(() => {
            onClose();
            navigate('/chat');
          }, 800);
          return;
        }

        if (jobStatus === 'failed') {
          setStatus('failed');
          setIngestionStatus('failed');
          setErrorMessage(res.error || 'Ingestion failed');
          clearInterval(pollInterval);
          return;
        }

        // Map stages to timeline steps
        const pdfStage = stages.find(s => s.stage === 'pdf_parse');
        const extractStage = stages.find(s => s.stage === 'entity_extraction');
        const embedStage = stages.find(s => s.stage === 'embedding');
        const neo4jStage = stages.find(s => s.stage === 'neo4j_write');
        const pineconeStage = stages.find(s => s.stage === 'pinecone_upsert');

        setIngestionStatus('processing');

        if (pdfStage?.status !== 'completed') {
          setTimelineStep(1);
          setProgressPercent(15);
          setStatusText('Parsing PDF...');
        } else if (extractStage?.status !== 'completed') {
          setTimelineStep(2);
          setProgressPercent(40);
          setStatusText('Extracting entities...');
        } else if (embedStage?.status !== 'completed') {
          setTimelineStep(3);
          setProgressPercent(65);
          setStatusText('Generating embeddings...');
        } else if (neo4jStage?.status !== 'completed' || pineconeStage?.status !== 'completed') {
          setTimelineStep(4);
          setProgressPercent(85);
          setStatusText('Building knowledge graph...');
        } else {
          setTimelineStep(4);
          setProgressPercent(95);
          setStatusText('Finalizing ingestion...');
        }
      } catch (err) {
        console.error('Polling status error:', err);
      }
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [localDocId, status]);

  const handleFileUpload = async (selectedFile) => {
    if (!selectedFile) return;

    // File Validation
    if (selectedFile.type !== 'application/pdf') {
      setErrorMessage('Invalid file type. Please upload a PDF file.');
      return;
    }

    const maxSizeInBytes = 50 * 1024 * 1024; // 50MB
    if (selectedFile.size > maxSizeInBytes) {
      setErrorMessage('File size exceeds the 50MB limit.');
      return;
    }

    setFile(selectedFile);
    setErrorMessage(null);
    setStatus('uploading');
    setProgressPercent(5);
    setStatusText('Initiating upload...');
    setTimelineStep(1);

    try {
      // Clean previous session state locally, but don't call api.reset() backend-side
      resetAll();

      const res = await api.uploadPaper(selectedFile);
      
      setLocalDocId(res.doc_id);
      setCurrentDocument(res.doc_id, selectedFile.name, res.doc_id);
      
      setStatus('processing');
      setStatusText('Processing document...');
    } catch (err) {
      setStatus('failed');
      setIngestionStatus('failed');
      setErrorMessage(err.message || 'Failed to upload paper');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      handleFileUpload(droppedFile);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      handleFileUpload(selectedFile);
    }
  };

  const handleRetry = () => {
    setFile(null);
    setLocalDocId(null);
    setStatus('idle');
    setTimelineStep(1);
    setProgressPercent(0);
    setStatusText('');
    setErrorMessage(null);
  };

  if (!isOpen) return null;

  const steps = [
    { label: 'Parsing', desc: 'PDF Parsing' },
    { label: 'Extraction', desc: 'Entity Extraction' },
    { label: 'Embedding', desc: 'Vector Embeddings' },
    { label: 'Graph Build', desc: 'Neo4j/Pinecone Write' },
    { label: 'Ready', desc: 'Ingestion Complete' }
  ];

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        {/* Backdrop overlay */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={() => {
            if (status !== 'uploading' && status !== 'processing') {
              onClose();
            }
          }}
          className="absolute inset-0 bg-[#040812]/80 backdrop-blur-md"
        />

        {/* Modal content box */}
        <motion.div 
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="relative w-full max-w-3xl bg-[#0B1220] border border-white/10 rounded-2xl p-8 shadow-2xl z-10 overflow-hidden text-white"
        >
          {/* Close button */}
          {(status !== 'uploading' && status !== 'processing') && (
            <button 
              onClick={onClose}
              className="absolute top-4 right-4 p-2 text-slate-400 hover:text-white rounded-lg hover:bg-white/5 transition-colors"
            >
              <X size={18} />
            </button>
          )}

          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
              Ingest Research Paper
            </h2>
            <p className="text-sm text-slate-400 mt-2">
              Upload your paper to build a dedicated semantic knowledge graph.
            </p>
          </div>

          {/* Idle Drag & Drop Area */}
          {status === 'idle' && (
            <div 
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={triggerFileInput}
              className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${
                isDragging 
                  ? 'border-[#22D3EE] bg-[#22D3EE]/5 shadow-[0_0_15px_rgba(34,211,238,0.15)]' 
                  : 'border-white/10 hover:border-[#3B82F6]/50 hover:bg-white/[0.02]'
              }`}
            >
              <input 
                ref={fileInputRef}
                type="file" 
                accept=".pdf"
                className="hidden"
                onChange={handleFileChange}
              />
              <UploadCloud size={48} className="mx-auto text-slate-400 mb-4" />
              <p className="text-base font-semibold text-slate-200">
                Drag and drop your PDF here, or <span className="text-[#22D3EE] hover:underline">browse</span>
              </p>
              <p className="text-xs text-slate-500 mt-2">
                Supports PDF up to 50MB
              </p>
              {errorMessage && (
                <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 text-red-400 text-xs rounded-lg inline-flex items-center gap-2">
                  <AlertCircle size={14} />
                  {errorMessage}
                </div>
              )}
            </div>
          )}

          {/* Uploading or Processing State */}
          {(status === 'uploading' || status === 'processing' || status === 'completed') && (
            <div className="space-y-8">
              {/* File Info Card */}
              {file && (
                <div className="flex items-center gap-4 p-4 bg-white/5 rounded-xl border border-white/5">
                  <div className="w-12 h-12 rounded-lg bg-red-500/10 flex items-center justify-center text-red-400 shrink-0">
                    <FileText size={24} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-slate-200 truncate">{file.name}</p>
                    <p className="text-xs text-slate-400 mt-0.5">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                  </div>
                  {status === 'completed' && (
                    <div className="text-emerald-400 flex items-center gap-1.5 text-xs font-semibold">
                      <CheckCircle2 size={16} />
                      Completed
                    </div>
                  )}
                </div>
              )}

              {/* Progress Timeline */}
              <div className="relative pt-4">
                <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-white/5 -translate-y-4 z-0" />
                
                {/* Horizontal Progress Bar background filler */}
                <div 
                  className="absolute top-1/2 left-0 h-0.5 bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] -translate-y-4 z-0 transition-all duration-500" 
                  style={{ width: `${Math.max(0, (timelineStep - 1) / 4) * 100}%` }}
                />

                <div className="relative z-10 flex justify-between">
                  {steps.map((step, idx) => {
                    const stepNum = idx + 1;
                    const isCompleted = timelineStep > stepNum || status === 'completed';
                    const isActive = timelineStep === stepNum && status !== 'completed';
                    
                    return (
                      <div key={idx} className="flex flex-col items-center">
                        <div 
                          className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${
                            isCompleted 
                              ? 'bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] text-white shadow-md' 
                              : isActive 
                                ? 'bg-[#0B1220] border-2 border-[#22D3EE] shadow-[0_0_15px_rgba(34,211,238,0.4)] text-[#22D3EE]' 
                                : 'bg-[#0B1220] border-2 border-white/5 text-slate-500'
                          }`}
                        >
                          {isCompleted ? (
                            <CheckCircle2 size={16} className="text-white" />
                          ) : isActive ? (
                            <motion.div
                              animate={{ scale: [0.9, 1.1, 0.9] }}
                              transition={{ repeat: Infinity, duration: 1.5 }}
                              className="w-2.5 h-2.5 rounded-full bg-[#22D3EE]"
                            />
                          ) : (
                            <span className="text-xs font-bold">{stepNum}</span>
                          )}
                        </div>
                        <span className={`text-[11px] mt-2 font-medium ${isCompleted || isActive ? 'text-slate-200' : 'text-slate-500'}`}>
                          {step.label}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Progress Bar & Status Text */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-400 flex items-center gap-2">
                    {status !== 'completed' && <Loader2 size={12} className="animate-spin text-[#22D3EE]" />}
                    {statusText}
                  </span>
                  <span className="font-semibold text-slate-200">{progressPercent}%</span>
                </div>
                <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${progressPercent}%` }}
                    transition={{ ease: 'easeOut', duration: 0.3 }}
                    className="h-full bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] rounded-full"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Failed State */}
          {status === 'failed' && (
            <div className="space-y-6">
              <div className="p-6 bg-red-500/10 border border-red-500/20 rounded-xl flex items-start gap-4">
                <AlertCircle size={24} className="text-red-400 shrink-0" />
                <div className="flex-1">
                  <h4 className="text-sm font-bold text-red-400">Ingestion Failed</h4>
                  <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                    {errorMessage || 'An error occurred while uploading and parsing your research paper. Please check the file and try again.'}
                  </p>
                </div>
              </div>

              <div className="flex justify-end gap-3">
                <button 
                  onClick={onClose}
                  className="px-5 py-2.5 rounded-full border border-white/10 text-sm font-semibold text-slate-300 hover:bg-white/5 transition-colors"
                >
                  Cancel
                </button>
                <button 
                  onClick={handleRetry}
                  className="flex items-center gap-2 px-6 py-2.5 rounded-full bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] text-white text-sm font-semibold hover:shadow-[0_0_15px_rgba(59,130,246,0.3)] transition-all duration-200"
                >
                  <RefreshCw size={14} />
                  Retry Upload
                </button>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
