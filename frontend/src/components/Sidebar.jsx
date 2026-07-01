import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Home, 
  FileText, 
  MessageSquare, 
  GitBranch, 
  Plus, 
  ChevronRight,
  ChevronLeft 
} from 'lucide-react';
import useAppStore from '../store/appStore';

export default function Sidebar({ activePage, onNewPaper }) {
  const navigate = useNavigate();
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const { currentPaperName, currentDocId } = useAppStore();

  const navItems = [
    { name: 'Chat', icon: MessageSquare, path: '/chat', id: 'chat' },
    { name: 'Graph Viewer', icon: GitBranch, path: '/graph', id: 'graph' }
  ];

  return (
    <motion.aside 
      animate={{ width: isSidebarCollapsed ? 72 : 260 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="shrink-0 h-full bg-[#0B1220] border-r border-white/5 flex flex-col justify-between relative z-30 overflow-hidden"
    >
      <div>
        {/* Header (Logo & Toggle) */}
        <div className="h-20 px-6 flex items-center justify-between border-b border-white/5">
          {!isSidebarCollapsed ? (
            <Link to="/" className="flex items-center gap-3 shrink-0">
              <div className="w-8 h-8 flex items-center justify-center bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] rounded-lg">
                <GitBranch size={18} className="text-white" />
              </div>
              <span className="text-lg font-bold tracking-tight text-white">Graphora</span>
            </Link>
          ) : (
            <Link to="/" className="w-8 h-8 mx-auto flex items-center justify-center bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] rounded-lg shrink-0">
              <GitBranch size={18} className="text-white" />
            </Link>
          )}

          <button 
            onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            className={`p-1.5 rounded bg-white/5 text-slate-400 hover:text-white transition-colors ${isSidebarCollapsed ? 'mx-auto mt-1' : ''}`}
          >
            {isSidebarCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
          </button>
        </div>

        {/* New Upload Button */}
        <div className="p-4">
          <button 
            onClick={onNewPaper}
            className="w-full flex items-center justify-center gap-2 py-3 rounded-full bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] text-white text-sm font-semibold hover:shadow-[0_0_15px_rgba(59,130,246,0.3)] transition-all duration-200"
          >
            <Plus size={16} />
            {!isSidebarCollapsed && <span className="truncate">New Paper</span>}
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="px-2 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => navigate(item.path)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm transition-all ${
                  isActive 
                    ? 'text-white border-l-[3px] border-[#3B82F6] bg-[#3B82F6]/10' 
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <Icon size={18} className={isActive ? 'text-[#3B82F6]' : ''} />
                {!isSidebarCollapsed && <span>{item.name}</span>}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer Info Cards */}
      <div className="p-4 space-y-4">
        {/* Paper Info Card */}
        {currentPaperName && (
          <div className={`p-3 bg-white/5 rounded-xl border border-white/5 flex ${isSidebarCollapsed ? 'justify-center' : 'flex-col'} gap-1.5`}>
            {isSidebarCollapsed ? (
              <div className="text-slate-400 hover:text-white" title={currentPaperName}>
                <FileText size={16} className="text-[#22D3EE]" />
              </div>
            ) : (
              <>
                <div className="flex items-center gap-2 text-xs text-slate-300 font-medium">
                  <FileText size={14} className="text-[#22D3EE] shrink-0" />
                  <span className="truncate" title={currentPaperName}>
                    {currentPaperName.length > 20 ? `${currentPaperName.substring(0, 17)}...` : currentPaperName}
                  </span>
                </div>
                {currentDocId && (
                  <div className="text-[10px] text-slate-500 font-mono pl-5">
                    ID: {currentDocId.substring(0, 8)}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* User Card */}
        <div className="flex items-center gap-3 p-2 bg-white/5 rounded-xl border border-white/5">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#3B82F6] to-[#6366F1] flex items-center justify-center text-xs font-bold text-white shadow-md shrink-0">
            US
          </div>
          {!isSidebarCollapsed && (
            <div className="flex-1 min-w-0">
              <p className="text-sm font-bold text-slate-100 truncate">Workspace User</p>
              <p className="text-xs text-slate-400 truncate">Researcher</p>
            </div>
          )}
        </div>
      </div>
    </motion.aside>
  );
}
