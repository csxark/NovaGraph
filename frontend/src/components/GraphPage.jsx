import React, { useState, useEffect, useRef } from 'react';
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
  Plus, 
  Search, 
  RotateCcw, 
  Download, 
  X,
  Minimize2,
  ZoomIn,
  ZoomOut
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

export default function GraphPage() {
  const navigate = useNavigate();
  
  // Zustand Store
  const {
    currentPaperId,
    currentPaperName,
    graphNodes,
    graphEdges,
    setGraphData,
    resetAll
  } = useAppStore();

  // Navigation / Collapse states
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [activeNavItem, setActiveNavItem] = useState('Graph Viewer');

  // Search & Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTypeFilter, setSelectedTypeFilter] = useState('All Types');
  const [selectedRelationFilter, setSelectedRelationFilter] = useState('All Relations');

  // Node details inspector
  const [selectedNode, setSelectedNode] = useState(null);

  // D3 elements
  const svgRef = useRef(null);
  const zoomBehaviorRef = useRef(null);

  // Pull graph data if not already populated but paper is active
  useEffect(() => {
    if (!currentPaperId) return;
    if (graphNodes.length > 0) {
      if (!selectedNode && graphNodes.length > 0) {
        setSelectedNode(graphNodes[0]);
      }
      return;
    }

    api.getGraph(currentPaperId)
      .then(data => {
        setGraphData(data.nodes || [], data.edges || []);
        if (data.nodes && data.nodes.length > 0) {
          setSelectedNode(data.nodes[0]);
        }
      })
      .catch(err => {
        console.error('Failed to fetch graph:', err);
      });
  }, [currentPaperId, graphNodes]);

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
      const width = container.clientWidth || 1000;
      const height = container.clientHeight || 700;
      d3.select(svgRef.current)
        .transition()
        .duration(400)
        .call(zoomBehaviorRef.current.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(1));
    }
  };

  // Reset Session
  const handleReset = async () => {
    try {
      await api.reset();
      resetAll();
      navigate('/dashboard');
    } catch (err) {
      console.error('Reset failed:', err);
    }
  };

  // Render full screen D3 Graph
  useEffect(() => {
    if (!svgRef.current || graphNodes.length === 0) return;

    // Filter nodes and edges based on filters
    let filteredNodes = graphNodes.filter(n => {
      const name = n.name || '';
      const type = n.type || '';
      const nameMatch = name.toLowerCase().includes(searchQuery.toLowerCase());
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
    const width = container.clientWidth || 1000;
    const height = container.clientHeight || 700;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', `0 0 ${width} ${height}`);

    // Dot grid background
    const defs = svg.append('defs');
    const pattern = defs.append('pattern')
      .attr('id', 'full-dot-grid')
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
      .attr('fill', 'url(#full-dot-grid)');

    const gContainer = svg.append('g');

    // Zoom setup
    const zoom = d3.zoom()
      .scaleExtent([0.1, 5])
      .on('zoom', (event) => {
        gContainer.attr('transform', event.transform);
      });

    zoomBehaviorRef.current = zoom;
    svg.call(zoom);
    svg.call(zoom.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(1));

    // Simulation
    const simulation = d3.forceSimulation(filteredNodes)
      .force('link', d3.forceLink(filteredEdges).id(d => d.id).distance(140))
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

    // Node outer glow ring
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

    // Centered labels inside
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
      
      {/* 260px FIXED SIDEBAR */}
      <aside className={`shrink-0 h-full bg-[#0B1220] border-r border-white/5 flex flex-col justify-between transition-all duration-300 ${isSidebarCollapsed ? 'w-20' : 'w-[260px]'} relative z-30`}>
        <div>
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

          <div className="p-4">
            <button 
              onClick={() => navigate('/dashboard')}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-full bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] text-white text-sm font-semibold hover:shadow-[0_0_15px_rgba(59,130,246,0.3)] transition-all duration-200"
            >
              <Plus size={16} />
              {!isSidebarCollapsed && 'New Upload'}
            </button>
          </div>

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
        </div>
      </aside>

      {/* FULL GRAPH CANVAS CONTAINER */}
      <main className="flex-1 flex flex-col h-full overflow-hidden relative z-10">
        
        {/* Header Bar */}
        <div className="h-20 border-b border-white/5 px-6 flex items-center justify-between shrink-0 bg-[#0B1220]">
          <h1 className="text-lg font-bold text-white flex items-center gap-2">
            <GitBranch size={20} className="text-graphora-blue" />
            Full Screen Graph
          </h1>

          {/* Search and Filters Controls */}
          {graphNodes.length > 0 && (
            <div className="flex items-center gap-3">
              <div className="relative">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-graphora-textMuted" />
                <input 
                  type="text" 
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search nodes..." 
                  className="bg-white/5 border border-white/5 focus:border-graphora-blue/50 rounded-full pl-9 pr-4 py-2 text-xs text-white placeholder-graphora-textMuted w-52 focus:outline-none"
                />
              </div>

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

              <button className="flex items-center gap-1.5 px-4 py-2 rounded-full border border-white/10 hover:bg-white/5 text-xs text-graphora-textSec">
                <Download size={12} />
                Export
              </button>
            </div>
          )}
        </div>

        {/* Graph Viewer Canvas */}
        <div className="flex-1 w-full relative bg-black/10 overflow-hidden flex items-center justify-center">
          
          {graphNodes.length === 0 ? (
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
                <div className="absolute left-6 top-6 bottom-6 w-[240px] bg-[#0B1220]/95 border border-white/10 rounded-2xl p-4 flex flex-col justify-between backdrop-blur-md shadow-2xl z-20 overflow-y-auto">
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
                    onClick={() => navigate('/dashboard')}
                    className="w-full mt-4 flex items-center justify-center gap-1.5 py-2.5 rounded-lg border border-white/10 hover:bg-white/5 text-[11px] font-semibold text-white transition-colors"
                  >
                    View in Chat
                  </button>
                </div>
              )}

              {/* FLOATING ZOOM CONTROLS */}
              <div className="absolute right-6 bottom-6 flex flex-col gap-2 p-1.5 rounded-xl border border-white/5 bg-[#0B1220]/80 backdrop-blur-md z-20">
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
                  onClick={() => navigate('/dashboard')}
                  className="w-8 h-8 rounded-lg hover:bg-white/5 flex items-center justify-center text-graphora-textSec hover:text-white"
                >
                  <Minimize2 size={15} />
                </button>
              </div>
            </>
          )}

        </div>

      </main>

    </div>
  );
}
