import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate, Link } from 'react-router-dom';
import * as d3 from 'd3';
import { 
  Home, 
  FileText, 
  MessageSquare, 
  GitBranch, 
  Bookmark, 
  Settings, 
  ChevronRight, 
  ChevronDown,
  Sparkles, 
  Plus, 
  Trash2, 
  Send, 
  Search, 
  RotateCcw, 
  Download, 
  X,
  Maximize2,
  ZoomIn,
  ZoomOut,
  UploadCloud,
  Loader2,
  AlertTriangle
} from 'lucide-react';
import useAppStore from '../store/appStore';
import { api } from '../lib/api';

const NODE_COLORS = {
  Concept: '#3B82F6',
  Method: '#22D3EE', 
  Evidence: '#10B981',
  Finding: '#10B981',
  Entity: '#6366F1',
  Reference: '#6366F1',
  Proposition: '#F59E0B',
  Assumption: '#F59E0B',
};
const DEFAULT_NODE_COLOR = '#94a3b8';

export default function DashboardPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  // Zustand Store
  const {
    currentPaperId,
    currentPaperName,
    ingestionStatus,
    jobId,
    graphNodes,
    graphEdges,
    messages,
    setCurrentPaper,
    setIngestionStatus,
    setGraphData,
    addMessage,
    resetAll
  } = useAppStore();

  // Local UI States
  const [activeNavItem, setActiveNavItem] = useState('Home');
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [activeChatTab, setActiveChatTab] = useState('Chat');
  const [chatInput, setChatInput] = useState('');
  const [isChatTyping, setIsChatTyping] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Timeline & Progress State derived from API
  const [timelineStep, setTimelineStep] = useState(1); // 1-5
  const [progressPercent, setProgressPercent] = useState(0);
  const [statusText, setStatusText] = useState('');

  // Graph states
  const [selectedNode, setSelectedNode] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTypeFilter, setSelectedTypeFilter] = useState('All Types');
  const [selectedRelationFilter, setSelectedRelationFilter] = useState('All Relations');

  // DOM Refs
  const svgRef = useRef(null);
  const zoomBehaviorRef = useRef(null);
  const chatEndRef = useRef(null);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isChatTyping]);

  // Status Polling Effect
  useEffect(() => {
    if (!jobId || ingestionStatus === 'completed' || ingestionStatus === 'failed') return;

    let pollInterval = setInterval(async () => {
      try {
        const res = await api.getStatus(jobId);
        const status = res.status;
        const stages = res.stages || [];

        // Check if finished
        if (status === 'completed') {
          setIngestionStatus('completed');
          setProgressPercent(100);
          setTimelineStep(5);
          setStatusText('Ready');
          clearInterval(pollInterval);
          // Fetch graph data
          try {
            const graph = await api.getGraph(res.paper_id);
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
          return;
        }

        if (status === 'failed') {
          setIngestionStatus('failed');
          setStatusText(res.error || 'Ingestion failed');
          clearInterval(pollInterval);
          return;
        }

        // Map status stage statuses to timeline steps
        // 1. pdf_parse -> Parsing
        // 2. domain_detection + entity_extraction -> Extraction
        // 3. embedding -> Embedding
        // 4. neo4j_write + pinecone_upsert -> Graph Build
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
        console.error('Polling error:', err);
      }
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [jobId, ingestionStatus]);

  // Upload handler
  const handleUpload = async (file) => {
    if (!file) return;
    setErrorMessage(null);
    setIngestionStatus('pending');
    setTimelineStep(1);
    setProgressPercent(5);
    setStatusText('Initiating upload...');

    try {
      // Always reset current session first to ensure starting clean
      await api.reset();
      resetAll();

      const res = await api.uploadPaper(file);
      setCurrentPaper(res.paper_id, file.name, res.job_id);
    } catch (err) {
      setIngestionStatus('failed');
      setErrorMessage(err.message || 'Failed to upload paper');
    }
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type === 'application/pdf') {
      handleUpload(file);
    }
  };

  // Chat submission
  const handleChatSend = async () => {
    if (!chatInput.trim() || ingestionStatus !== 'completed') return;

    const userMsg = {
      role: 'user',
      content: chatInput,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    addMessage(userMsg);
    
    const queryText = chatInput;
    setChatInput('');
    setIsChatTyping(true);

    try {
      const res = await api.query({
        query: queryText,
        paper_id: currentPaperId,
        top_k: 5,
        include_trace: true
      });

      addMessage({
        role: 'assistant',
        content: res.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        query_id: res.query_id
      });
    } catch (err) {
      addMessage({
        role: 'assistant',
        content: `Error: ${err.message || 'Failed to query the assistant.'}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      });
    } finally {
      setIsChatTyping(false);
    }
  };

  // Reset Session
  const handleReset = async () => {
    try {
      await api.reset();
      resetAll();
      setTimelineStep(1);
      setProgressPercent(0);
      setStatusText('');
      setErrorMessage(null);
    } catch (err) {
      console.error('Reset failed:', err);
    }
  };

  // Zoom Helpers
  const handleZoomIn = () => {
    if (svgRef.current && zoomBehaviorRef.current) {
      d3.select(svgRef.current).transition().duration(300).call(zoomBehaviorRef.current.scaleBy, 1.3);
    }
  };

  const handleZoomOut = () => {
    if (svgRef.current && zoomBehaviorRef.current) {
      d3.select(svgRef.current).transition().duration(300).call(zoomBehaviorRef.current.scaleBy, 0.7);
    }
  };

  const handleResetZoom = () => {
    if (svgRef.current && zoomBehaviorRef.current) {
      const container = svgRef.current.parentElement;
      const width = container.clientWidth || 800;
      const height = container.clientHeight || 400;
      d3.select(svgRef.current)
        .transition()
        .duration(400)
        .call(zoomBehaviorRef.current.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(1));
    }
  };

  // Render D3 Graph
  useEffect(() => {
    if (!svgRef.current || graphNodes.length === 0) return;

    // Filter nodes/edges
    let filteredNodes = graphNodes.filter(n => {
      const nameMatch = (n.name || '').toLowerCase().includes(searchQuery.toLowerCase());
      const typeMatch = selectedTypeFilter === 'All Types' || n.type === selectedTypeFilter;
      return nameMatch && typeMatch;
    });

    const activeIds = new Set(filteredNodes.map(n => n.id));

    let filteredEdges = graphEdges.filter(e => {
      const sourceId = typeof e.source === 'object' ? e.source.id : e.source;
      const targetId = typeof e.target === 'object' ? e.target.id : e.target;
      const matchesNodes = activeIds.has(sourceId) && activeIds.has(targetId);
      const matchesRelation = selectedRelationFilter === 'All Relations' || e.type === selectedRelationFilter;
      return matchesNodes && matchesRelation;
    });

    const container = svgRef.current.parentElement;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 400;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', `0 0 ${width} ${height}`);

    // Dot grid background pattern
    const defs = svg.append('defs');
    const pattern = defs.append('pattern')
      .attr('id', 'dot-grid')
      .attr('width', 20)
      .attr('height', 20)
      .attr('patternUnits', 'userSpaceOnUse');

    pattern.append('circle')
      .attr('cx', 2)
      .attr('cy', 2)
      .attr('r', 1)
      .attr('fill', 'rgba(255, 255, 255, 0.08)');

    svg.append('rect')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('fill', 'url(#dot-grid)');

    const gContainer = svg.append('g');

    // Zoom behaviour
    const zoom = d3.zoom()
      .scaleExtent([0.1, 5])
      .on('zoom', (event) => {
        gContainer.attr('transform', event.transform);
      });

    zoomBehaviorRef.current = zoom;
    svg.call(zoom);
    svg.call(zoom.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(1));

    // Force simulation
    const simulation = d3.forceSimulation(filteredNodes)
      .force('link', d3.forceLink(filteredEdges).id(d => d.id).distance(120))
      .force('charge', d3.forceManyBody().strength(-400))
      .force('collision', d3.forceCollide().radius(50))
      .force('x', d3.forceX(0))
      .force('y', d3.forceY(0));

    // Render Edges
    const link = gContainer.append('g')
      .selectAll('line')
      .data(filteredEdges)
      .enter().append('line')
      .attr('stroke', 'rgba(255, 255, 255, 0.15)')
      .attr('stroke-width', 1.5);

    // Edge Labels
    const edgeLabels = gContainer.append('g')
      .selectAll('text')
      .data(filteredEdges)
      .enter().append('text')
      .attr('class', 'text-[9px] fill-graphora-textMuted select-none')
      .attr('text-anchor', 'middle')
      .text(d => d.type || 'uses');

    // Render Nodes
    const node = gContainer.append('g')
      .selectAll('g')
      .data(filteredNodes)
      .enter().append('g')
      .attr('class', 'cursor-pointer select-none')
      .call(d3.drag()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
      )
      .on('click', (event, d) => {
        setSelectedNode(d);
      });

    // Outer Hover Ring
    node.append('circle')
      .attr('r', 20)
      .attr('fill', 'transparent')
      .attr('stroke', d => NODE_COLORS[d.type] || DEFAULT_NODE_COLOR)
      .attr('stroke-width', 2)
      .attr('class', 'transition-all duration-300 opacity-0 hover:opacity-40')
      .style('filter', 'blur(4px)');

    // Inner Circle Fills
    node.append('circle')
      .attr('r', 16)
      .attr('fill', d => NODE_COLORS[d.type] || DEFAULT_NODE_COLOR)
      .attr('stroke', 'rgba(0,0,0,0.4)')
      .attr('stroke-width', 1.5);

    // Centered label inside node
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '.3em')
      .attr('fill', '#FFFFFF')
      .attr('class', 'text-[8px] font-bold pointer-events-none')
      .text(d => {
        const label = d.name || '';
        return label.length > 8 ? label.substring(0, 6) + '..' : label;
      });

    node.append('title')
      .text(d => `${d.name} (${d.type})\n${d.description || ''}`);

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      edgeLabels
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2 - 5);

      node
        .attr('transform', d => `translate(${d.x}, ${d.y})`);
    });

    return () => {
      simulation.stop();
    };

  }, [graphNodes, graphEdges, searchQuery, selectedTypeFilter, selectedRelationFilter]);

  // Unique relation filter values derived from data
  const relationTypes = Array.from(new Set(graphEdges.map(e => e.type).filter(Boolean)));
  const nodeTypes = Array.from(new Set(graphNodes.map(n => n.type).filter(Boolean)));

  return (
    <div className="flex h-screen bg-[#050A13] text-white overflow-hidden font-sans relative">
      
      {/* Hidden File Input */}
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={(e) => handleUpload(e.target.files?.[0])} 
        accept=".pdf" 
        className="hidden" 
      />

      {/* FIXED SIDEBAR */}
      <aside className={`shrink-0 h-full bg-[#0B1220] border-r border-white/5 flex flex-col justify-between transition-all duration-300 ${isSidebarCollapsed ? 'w-20' : 'w-[260px]'} relative z-30`}>
        <div>
          {/* Header */}
          <div className="h-20 px-6 flex items-center justify-between border-b border-white/5">
            {!isSidebarCollapsed && (
              <Link to="/" className="flex items-center gap-3">
                <div className="w-8 h-8 flex items-center justify-center bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] rounded-lg">
                  <GitBranch size={18} className="text-white" />
                </div>
                <span className="text-lg font-bold tracking-tight text-white">Graphora</span>
              </Link>
            )}
            
            {isSidebarCollapsed && (
              <div className="w-8 h-8 mx-auto flex items-center justify-center bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] rounded-lg">
                <GitBranch size={18} className="text-white" />
              </div>
            )}

            <button 
              onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
              className="p-1 rounded bg-white/5 text-graphora-textSec hover:text-white"
            >
              {isSidebarCollapsed ? <ChevronRight size={16} /> : '<<'}
            </button>
          </div>

          {/* New Upload Button */}
          <div className="p-4">
            <button 
              onClick={handleNewUploadClick}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-full bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] text-white text-sm font-semibold hover:shadow-[0_0_15px_rgba(59,130,246,0.3)] transition-all duration-200"
            >
              <Plus size={16} />
              {!isSidebarCollapsed && 'New Upload'}
            </button>
          </div>

          {/* Nav Items */}
          <nav className="px-2 space-y-1">
            {[
              { name: 'Home', icon: Home, route: '/dashboard' },
              { name: 'Graph Viewer', icon: GitBranch, route: '/graph' },
            ].map((item) => {
              const Icon = item.icon;
              const isActive = activeNavItem === item.name;
              return (
                <button
                  key={item.name}
                  onClick={() => {
                    setActiveNavItem(item.name);
                    navigate(item.route);
                  }}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm transition-all ${
                    isActive 
                      ? 'text-graphora-blue border-l-[3px] border-graphora-blue bg-graphora-blue/5' 
                      : 'text-graphora-textSec hover:text-white hover:bg-white/5'
                  }`}
                >
                  <Icon size={18} />
                  {!isSidebarCollapsed && <span>{item.name}</span>}
                </button>
              );
            })}
          </nav>
        </div>

        {/* User + Subscription cards */}
        <div className="p-4 space-y-4">
          <div className="flex items-center gap-3 p-2 bg-white/5 rounded-xl border border-white/5">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-graphora-blue to-[#6366F1] flex items-center justify-center text-xs font-bold text-white shadow-md">
              US
            </div>
            {!isSidebarCollapsed && (
              <div className="flex-1 min-w-0">
                <p className="text-sm font-bold text-slate-100 truncate">Workspace User</p>
                <p className="text-xs text-graphora-textMuted">Researcher</p>
              </div>
            )}
          </div>

          {!isSidebarCollapsed && (
            <div className="p-3 bg-white/5 rounded-xl border border-white/5 text-xs">
              <div className="flex justify-between font-bold mb-1.5">
                <span className="text-white">Pro Plan</span>
                <span className="text-graphora-cyan">Unlimited</span>
              </div>
              <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden mb-1">
                <div className="h-full bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] w-[80%]" />
              </div>
              <p className="text-[10px] text-graphora-textMuted leading-tight">
                Unlimited research paper graphs
              </p>
            </div>
          )}
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="flex-1 flex flex-col h-full overflow-y-auto p-6 space-y-6 relative z-10">
        
        {/* ROW 1: Upload & Process + Chat Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          
          {/* UPLOAD & PROCESS CARD */}
          <div className="lg:col-span-5 bg-[#0B1220] border border-white/5 rounded-[22px] p-6 flex flex-col justify-between hover:border-graphora-blue/20 transition-all duration-300">
            <div>
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold text-white">Upload & Process</h2>
                {currentPaperId && ingestionStatus !== 'failed' && (
                  <span className="text-xs text-graphora-textMuted font-mono uppercase">
                    Stage {timelineStep} of 5
                  </span>
                )}
              </div>

              {!currentPaperId ? (
                /* Drag & Drop File Zone */
                <div 
                  onDragOver={onDragOver}
                  onDragLeave={onDragLeave}
                  onDrop={onDrop}
                  onClick={handleNewUploadClick}
                  className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center gap-4 cursor-pointer transition-all duration-200 ${
                    isDragging 
                      ? 'border-graphora-blue bg-graphora-blue/5' 
                      : 'border-white/10 hover:border-graphora-blue/30 bg-white/5'
                  }`}
                >
                  <UploadCloud size={40} className="text-graphora-textSec" />
                  <div className="text-center">
                    <p className="text-sm font-semibold text-white">Drag & Drop PDF file here</p>
                    <p className="text-xs text-graphora-textMuted mt-1">or click to browse from device</p>
                  </div>
                </div>
              ) : (
                /* Active Pipeline View */
                <div className="space-y-6">
                  {/* Timeline progress steps */}
                  <div className="flex items-center justify-between relative">
                    <div className="absolute left-[18px] right-[18px] top-[18px] h-[2px] bg-white/10 -z-10" />
                    <div 
                      className="absolute left-[18px] top-[18px] h-[2px] bg-graphora-blue transition-all duration-500 -z-10" 
                      style={{ width: `${(timelineStep - 1) * 25}%` }}
                    />

                    {[
                      { num: 1, label: 'Parsing' },
                      { num: 2, label: 'Extraction' },
                      { num: 3, label: 'Embedding' },
                      { num: 4, label: 'Graph Build' },
                      { num: 5, label: 'Ready' }
                    ].map((step) => {
                      const isCompleted = step.num < timelineStep;
                      const isActive = step.num === timelineStep;
                      return (
                        <div key={step.num} className="flex flex-col items-center relative">
                          <div 
                            className={`w-9 h-9 rounded-full flex items-center justify-center text-xs font-semibold transition-all ${
                              isCompleted 
                                ? 'bg-graphora-blue text-white shadow-lg shadow-graphora-blue/20' 
                                : isActive 
                                  ? 'border-2 border-graphora-blue bg-[#0B1220] text-graphora-blue shadow-[0_0_15px_rgba(59,130,246,0.3)] animate-pulse'
                                  : 'border border-white/20 bg-white/5 text-graphora-textMuted'
                            }`}
                          >
                            {isCompleted ? '✓' : step.num}
                          </div>
                          <span className="text-[10px] mt-2 text-graphora-textSec font-medium">{step.label}</span>
                        </div>
                      );
                    })}
                  </div>

                  {/* Uploaded File Details */}
                  <div className="bg-white/5 border border-white/5 rounded-xl p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-500 font-bold text-xs">
                        PDF
                      </div>
                      <div className="min-w-0">
                        <h4 className="text-sm font-bold text-white truncate max-w-[180px]">
                          {currentPaperName}
                        </h4>
                        <p className="text-xs text-graphora-textSec font-mono">
                          ID: {currentPaperId.substring(0, 12)}...
                        </p>
                      </div>
                    </div>
                    
                    <button 
                      onClick={handleReset}
                      className="p-1.5 rounded-lg hover:bg-white/10 text-graphora-textMuted hover:text-white"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>

                  {/* Status Progress */}
                  {ingestionStatus !== 'failed' ? (
                    <div className="space-y-2">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-graphora-cyan font-mono">{statusText}</span>
                        <span className="text-white font-mono">{progressPercent}%</span>
                      </div>
                      <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-graphora-blue to-graphora-cyan transition-all duration-300"
                          style={{ width: `${progressPercent}%` }}
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-red-400 text-xs bg-red-500/10 p-3 rounded-lg border border-red-500/25">
                      <AlertTriangle size={16} />
                      <span>{statusText || errorMessage || 'Ingestion failed.'}</span>
                      <button 
                        onClick={handleReset} 
                        className="ml-auto underline font-bold"
                      >
                        Retry
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            <p className="text-xs text-graphora-textMuted mt-6">
              This may take a few minutes depending on the size of the paper.
            </p>
          </div>

          {/* CHAT PANEL CARD */}
          <div className="lg:col-span-7 bg-[#0B1220] border border-white/5 rounded-[22px] p-6 flex flex-col justify-between hover:border-graphora-blue/20 transition-all duration-300 max-h-[460px]">
            <div>
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold text-white">Chat</h2>
                
                <div className="flex bg-white/5 border border-white/5 rounded-lg p-0.5">
                  {['Chat', 'Graph'].map((tab) => (
                    <button
                      key={tab}
                      onClick={() => {
                        setActiveChatTab(tab);
                        if (tab === 'Graph') navigate('/graph');
                      }}
                      className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                        activeChatTab === tab 
                          ? 'bg-white/10 text-white shadow-sm' 
                          : 'text-graphora-textSec hover:text-white'
                      }`}
                    >
                      {tab}
                    </button>
                  ))}
                </div>
              </div>

              {currentPaperId && (
                <div className="flex items-center justify-between px-4 py-2 rounded-lg border border-white/5 bg-white/5 mb-4">
                  <span className="text-xs font-bold text-white font-mono truncate">
                    Paper: {currentPaperName}
                  </span>
                </div>
              )}

              {/* Chat Scroll Area */}
              <div className="space-y-4 max-h-[180px] overflow-y-auto pr-1">
                {messages.length === 0 ? (
                  <div className="h-28 flex flex-col items-center justify-center text-graphora-textSec text-xs text-center">
                    <MessageSquare size={24} className="text-white/15 mb-2" />
                    <span>Upload a research paper to query insights and relationships.</span>
                  </div>
                ) : (
                  <AnimatePresence>
                    {messages.map((msg, i) => (
                      <motion.div 
                        key={i}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`flex gap-3 items-start ${msg.role === 'user' ? 'justify-end' : ''}`}
                      >
                        {msg.role !== 'user' && (
                          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-graphora-blue to-graphora-purple flex items-center justify-center shrink-0">
                            <Sparkles size={14} className="text-white" />
                          </div>
                        )}
                        <div className={`p-3 rounded-2xl text-xs max-w-[85%] leading-relaxed ${
                          msg.role === 'user'
                            ? 'bg-graphora-blue text-white rounded-br-none font-medium'
                            : 'bg-white/5 border border-white/5 text-graphora-textSec rounded-bl-none'
                        }`}>
                          {msg.content}
                          <span className="block mt-1.5 text-[9px] text-graphora-textMuted">{msg.timestamp}</span>
                        </div>
                      </motion.div>
                    ))}
                    {isChatTyping && (
                      <div className="flex gap-3 items-start animate-pulse">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-graphora-blue to-graphora-purple flex items-center justify-center shrink-0">
                          <Sparkles size={14} className="text-white" />
                        </div>
                        <div className="p-3 bg-white/5 border border-white/5 text-graphora-textSec rounded-2xl rounded-bl-none text-xs">
                          Thinking...
                        </div>
                      </div>
                    )}
                  </AnimatePresence>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Suggestions chips */}
              {ingestionStatus === 'completed' && (
                <div className="flex flex-wrap gap-2 mt-4">
                  {[
                    'What is the main contribution of this paper?',
                    'How does the model architecture work?',
                    'What datasets were used?',
                  ].map((sug) => (
                    <button
                      key={sug}
                      onClick={() => setChatInput(sug)}
                      className="px-3 py-1.5 rounded-full border border-white/10 bg-white/5 hover:border-graphora-blue/50 hover:bg-graphora-blue/5 text-[10px] text-graphora-textSec hover:text-white transition-all text-left max-w-full truncate"
                    >
                      {sug}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Input sticky */}
            <div className="flex items-center gap-2 mt-4 pt-3 border-t border-white/5">
              <input 
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleChatSend()}
                disabled={ingestionStatus !== 'completed'}
                placeholder={ingestionStatus === 'completed' ? 'Ask anything about your paper...' : 'Upload and ingest paper to enable chat...'}
                className="flex-1 bg-white/5 border border-white/5 focus:border-graphora-blue/50 rounded-xl px-4 py-3 text-xs text-white placeholder-graphora-textMuted transition-all focus:outline-none focus:ring-1 focus:ring-graphora-blue/50 disabled:opacity-50"
              />
              <button 
                onClick={handleChatSend}
                disabled={ingestionStatus !== 'completed'}
                className="w-10 h-10 rounded-full bg-gradient-to-r from-graphora-blue to-graphora-cyan flex items-center justify-center text-white shadow-md shadow-graphora-blue/20 hover:scale-105 transition-transform disabled:opacity-50 disabled:hover:scale-100"
              >
                <Send size={16} />
              </button>
            </div>
          </div>

        </div>

        {/* ROW 2: FULL WIDTH GRAPH VIEWER */}
        <div className="bg-[#0B1220] border border-white/5 rounded-[22px] p-6 flex flex-col justify-between hover:border-graphora-blue/20 transition-all duration-300 relative min-h-[500px]">
          
          {/* Header Controls */}
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-white/5 pb-4 mb-4">
            <h2 className="text-xl font-semibold text-white">Graph Viewer</h2>
            
            {graphNodes.length > 0 && (
              <div className="flex flex-wrap items-center gap-3">
                {/* Search */}
                <div className="relative">
                  <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-graphora-textMuted" />
                  <input 
                    type="text" 
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search nodes..." 
                    className="bg-white/5 border border-white/5 focus:border-graphora-blue/50 rounded-full pl-9 pr-4 py-2 text-xs text-white placeholder-graphora-textMuted w-44 focus:outline-none"
                  />
                </div>

                {/* Type Filter */}
                <select 
                  value={selectedTypeFilter}
                  onChange={(e) => setSelectedTypeFilter(e.target.value)}
                  className="bg-white/5 border border-white/5 rounded-full px-4 py-2 text-xs text-white focus:outline-none cursor-pointer"
                >
                  <option value="All Types">All Types</option>
                  {nodeTypes.map(t => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>

                {/* Relation Filter */}
                <select 
                  value={selectedRelationFilter}
                  onChange={(e) => setSelectedRelationFilter(e.target.value)}
                  className="bg-white/5 border border-white/5 rounded-full px-4 py-2 text-xs text-white focus:outline-none cursor-pointer"
                >
                  <option value="All Relations">All Relations</option>
                  {relationTypes.map(r => (
                    <option key={r} value={r}>{r}</option>
                  ))}
                </select>

                {/* Reset button */}
                <button 
                  onClick={() => {
                    setSearchQuery('');
                    setSelectedTypeFilter('All Types');
                    setSelectedRelationFilter('All Relations');
                    handleResetZoom();
                  }}
                  className="flex items-center gap-1 px-4 py-2 rounded-full border border-white/10 hover:bg-white/5 text-xs text-graphora-textSec"
                >
                  <RotateCcw size={12} />
                  Reset
                </button>

                {/* Export button */}
                <button className="flex items-center gap-1.5 px-4 py-2 rounded-full border border-white/10 hover:bg-white/5 text-xs text-graphora-textSec">
                  <Download size={12} />
                  Export
                </button>
              </div>
            )}
          </div>

          {/* Graph visualizer canvas container */}
          <div className="flex-1 w-full h-[380px] relative rounded-xl bg-black/20 border border-white/5 overflow-hidden flex items-center justify-center">
            
            {graphNodes.length === 0 ? (
              /* Empty state */
              <div className="text-center p-8 max-w-md">
                <GitBranch size={48} className="text-white/15 mx-auto mb-3" />
                <p className="text-sm font-semibold text-slate-300">No graph data yet</p>
                <p className="text-xs text-graphora-textMuted mt-1">Upload a paper to generate the knowledge graph.</p>
              </div>
            ) : (
              <>
                <svg ref={svgRef} className="w-full h-full" />

                {/* FLOATING NODE INSPECTOR */}
                {selectedNode && (
                  <div className="absolute left-4 top-4 bottom-4 w-[240px] bg-[#0B1220]/95 border border-white/10 rounded-2xl p-4 flex flex-col justify-between backdrop-blur-md shadow-2xl z-20 overflow-y-auto">
                    <div>
                      <div className="flex justify-between items-start gap-2 mb-3">
                        <h3 className="text-sm font-bold text-white truncate">{selectedNode.name}</h3>
                        <button 
                          onClick={() => setSelectedNode(null)}
                          className="p-1 rounded-lg hover:bg-white/10 text-graphora-textMuted"
                        >
                          <X size={14} />
                        </button>
                      </div>

                      <span 
                        className="inline-block px-2.5 py-1 rounded-full text-[10px] font-semibold mb-4 text-white"
                        style={{ 
                          backgroundColor: `${NODE_COLORS[selectedNode.type] || DEFAULT_NODE_COLOR}33`, 
                          border: `1px solid ${NODE_COLORS[selectedNode.type] || DEFAULT_NODE_COLOR}` 
                        }}
                      >
                        {selectedNode.type}
                      </span>

                      <div className="space-y-3">
                        <p className="text-xs text-graphora-textSec leading-relaxed">
                          {selectedNode.description || 'No description available for this extracted entity.'}
                        </p>

                        <div className="border-t border-white/5 pt-3">
                          <h4 className="text-[10px] font-bold text-graphora-textMuted uppercase tracking-wider mb-2">Properties</h4>
                          <div className="grid grid-cols-2 gap-2 text-[10px]">
                            {selectedNode.domains && selectedNode.domains.length > 0 && (
                              <>
                                <div className="text-graphora-textMuted">Domains:</div>
                                <div className="text-white truncate">{selectedNode.domains.join(', ')}</div>
                              </>
                            )}
                            {selectedNode.properties && Object.entries(selectedNode.properties).map(([k, v]) => (
                              k !== 'name' && k !== 'type' && k !== 'description' && k !== 'domains' && k !== 'id' && (
                                <React.Fragment key={k}>
                                  <div className="text-graphora-textMuted truncate">{k}:</div>
                                  <div className="text-white truncate">{String(v)}</div>
                                </React.Fragment>
                              )
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>

                    <button 
                      onClick={() => {
                        if (ingestionStatus === 'completed') {
                          setChatInput(`Tell me about the ${selectedNode.name || 'entity'}`);
                        }
                      }}
                      className="w-full mt-4 flex items-center justify-center gap-1.5 py-2.5 rounded-lg border border-white/10 hover:bg-white/5 text-[11px] font-semibold text-white transition-colors"
                    >
                      View in Chat
                    </button>
                  </div>
                )}

                {/* FLOATING ZOOM CONTROLS */}
                <div className="absolute right-4 bottom-4 flex flex-col gap-2 p-1.5 rounded-xl border border-white/5 bg-[#0B1220]/80 backdrop-blur-md z-20">
                  <button 
                    onClick={handleZoomIn}
                    className="w-8 h-8 rounded-lg hover:bg-white/5 flex items-center justify-center text-graphora-textSec hover:text-white"
                  >
                    <ZoomIn size={16} />
                  </button>
                  <button 
                    onClick={handleZoomOut}
                    className="w-8 h-8 rounded-lg hover:bg-white/5 flex items-center justify-center text-graphora-textSec hover:text-white"
                  >
                    <ZoomOut size={16} />
                  </button>
                  <button 
                    onClick={handleResetZoom}
                    className="w-8 h-8 rounded-lg hover:bg-white/5 flex items-center justify-center text-graphora-textSec hover:text-white text-xs font-semibold"
                  >
                    Fit
                  </button>
                  <button 
                    onClick={() => navigate('/graph')}
                    className="w-8 h-8 rounded-lg hover:bg-white/5 flex items-center justify-center text-graphora-textSec hover:text-white"
                  >
                    <Maximize2 size={15} />
                  </button>
                </div>
              </>
            )}

          </div>

        </div>

      </main>

    </div>
  );
}

function handleNewUploadClick() {
  const fileInput = document.querySelector('input[type="file"]');
  if (fileInput) {
    fileInput.click();
  }
}
