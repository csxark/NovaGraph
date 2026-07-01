import React, { useState, useEffect, useRef } from 'react';
import { Send, FileText, Loader2, Sparkles } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import Sidebar from '../components/Sidebar';
import UploadModal from '../components/UploadModal';
import PromptArchitectCards from '../components/PromptArchitectCards';
import useAppStore from '../store/appStore';
import { api } from '../lib/api';

export default function ChatPage() {
  const [chatInput, setChatInput] = useState('');
  const [isChatTyping, setIsChatTyping] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Zustand Store
  const {
    currentDocId,
    currentPaperName,
    ingestionStatus,
    messages,
    addMessage
  } = useAppStore();

  const chatEndRef = useRef(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isChatTyping]);

  const handleSend = async (textToSend) => {
    const text = textToSend || chatInput;
    if (!text.trim() || ingestionStatus !== 'completed') return;

    // Add user message
    const userMsg = {
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    addMessage(userMsg);
    
    if (!textToSend) {
      setChatInput('');
    }
    
    setIsChatTyping(true);

    try {
      const res = await api.query({
        query: text,
        doc_id: currentDocId,
        top_k: 5,
        include_trace: true
      });

      addMessage({
        role: 'assistant',
        content: res.answer,
        response_type: res.response_type || 'paper_answer',
        prompts: res.prompts || [],
        domain: res.domain || '',
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

  const handleChipClick = (suggestion) => {
    handleSend(suggestion);
  };

  const openModal = () => setIsModalOpen(true);
  const closeModal = () => setIsModalOpen(false);

  const suggestionChips = [
    "What is the main contribution?",
    "Summarize the methodology",
    "What are the key results?",
    "What are the limitations?"
  ];

  return (
    <div className="flex h-screen overflow-hidden text-white bg-[#050A13] font-sans">
      {/* Sidebar */}
      <Sidebar activePage="chat" onNewPaper={openModal} />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Header Bar */}
        <header className="h-16 border-b border-white/5 flex items-center justify-between px-6 shrink-0 bg-[#0B1220]/50 backdrop-blur-md">
          <div className="flex items-center gap-2">
            <h1 className="text-base font-bold text-slate-100">Chat with Paper</h1>
          </div>
          {currentPaperName && (
            <div className="flex items-center gap-2 px-3 py-1 bg-white/5 rounded-full border border-white/5 max-w-xs md:max-w-md">
              <FileText size={14} className="text-[#22D3EE] shrink-0" />
              <span className="text-xs text-slate-300 truncate" title={currentPaperName}>
                {currentPaperName}
              </span>
            </div>
          )}
        </header>

        {/* Scrollable Message List */}
        <main className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin">
          {messages.length === 0 ? (
            /* Empty State */
            <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto text-center px-4">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] flex items-center justify-center mb-6 shadow-[0_0_20px_rgba(34,211,238,0.2)]">
                <Sparkles size={24} className="text-white" />
              </div>
              <h2 className="text-xl font-bold text-slate-100">Ask anything about this paper</h2>
              <p className="text-sm text-slate-400 mt-2 max-w-md leading-relaxed">
                Analyze and query this document. Graphora has constructed a semantic entity map to answer with precision and citations.
              </p>

              {/* Suggestion Chips */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-8 w-full max-w-lg">
                {suggestionChips.map((chip, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleChipClick(chip)}
                    disabled={ingestionStatus !== 'completed'}
                    className="p-3 text-left text-xs bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/10 rounded-xl transition-all duration-200 text-slate-300 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {chip}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* Active Message bubbles */
            <div className="max-w-3xl mx-auto space-y-6 pb-4">
              {messages.map((msg, idx) => {
                const isUser = msg.role === 'user';
                return (
                  <div 
                    key={idx} 
                    className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}
                  >
                    {/* Assistant Avatar */}
                    {!isUser && (
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#3B82F6] to-[#6366F1] flex items-center justify-center text-xs font-bold text-white shadow-md shrink-0 self-start mt-0.5">
                        AI
                      </div>
                    )}
                    
                    {/* Message Bubble */}
                    <div className="flex flex-col gap-1 max-w-[80%]">
                      <div 
                        className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                          isUser 
                            ? 'bg-[#3B82F6] text-white rounded-tr-none' 
                            : 'bg-white/5 border border-white/10 text-slate-100 rounded-tl-none'
                        }`}
                      >
                        {isUser ? (
                          <span className="whitespace-pre-wrap">{msg.content}</span>
                        ) : (
                          <div className="prose prose-invert prose-xs max-w-none text-slate-200">
                            {msg.response_type === 'prompt_architect' ? (
                              <PromptArchitectCards prompts={msg.prompts} domain={msg.domain} />
                            ) : (
                              <ReactMarkdown>{msg.content}</ReactMarkdown>
                            )}
                          </div>
                        )}
                      </div>
                      <span className={`text-[10px] text-slate-500 ${isUser ? 'text-right pr-1' : 'pl-1'}`}>
                        {msg.timestamp}
                      </span>
                    </div>

                    {/* User Avatar */}
                    {isUser && (
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#10B981] to-[#3B82F6] flex items-center justify-center text-xs font-bold text-white shadow-md shrink-0 self-start mt-0.5">
                        ME
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Typing Indicator */}
              {isChatTyping && (
                <div className="flex gap-3 justify-start">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#3B82F6] to-[#6366F1] flex items-center justify-center text-xs font-bold text-white shadow-md shrink-0 self-start mt-0.5 animate-pulse">
                    AI
                  </div>
                  <div className="bg-white/5 border border-white/5 rounded-2xl rounded-tl-none px-4 py-3 flex gap-1 items-center">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#22D3EE] animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-1.5 h-1.5 rounded-full bg-[#22D3EE] animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-1.5 h-1.5 rounded-full bg-[#22D3EE] animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              )}

              <div ref={chatEndRef} />
            </div>
          )}
        </main>

        {/* Input Bar */}
        <footer className="h-20 border-t border-white/5 px-6 flex items-center bg-[#0B1220]/50 backdrop-blur-md shrink-0">
          <div className="max-w-3xl mx-auto w-full flex items-center gap-3 bg-white/5 border border-white/5 rounded-full px-4 py-1.5 focus-within:border-white/10 focus-within:bg-white/[0.08] transition-all">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              disabled={ingestionStatus !== 'completed'}
              placeholder={
                ingestionStatus === 'completed' 
                  ? "Ask anything about this paper..." 
                  : "Please upload and process a paper to chat..."
              }
              className="flex-1 bg-transparent text-sm text-white placeholder-slate-500 focus:outline-none py-1.5 disabled:cursor-not-allowed"
            />
            <button
              onClick={() => handleSend()}
              disabled={ingestionStatus !== 'completed' || !chatInput.trim()}
              className="w-8 h-8 rounded-full bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] flex items-center justify-center text-white disabled:opacity-30 disabled:cursor-not-allowed hover:shadow-[0_0_10px_rgba(34,211,238,0.2)] transition-all shrink-0"
            >
              <Send size={14} />
            </button>
          </div>
        </footer>
      </div>

      {/* Upload Modal */}
      <UploadModal isOpen={isModalOpen} onClose={closeModal} />
    </div>
  );
}
