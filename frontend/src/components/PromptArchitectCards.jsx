import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Copy, Check, ChevronDown, AlertCircle } from 'lucide-react';

const CARD_STYLES = {
  PLAN:     { badge: 'bg-blue-500/20 text-blue-400 border-blue-500/30',     number: 'from-blue-500 to-cyan-400' },
  BUILD:    { badge: 'bg-green-500/20 text-green-400 border-green-500/30',  number: 'from-green-500 to-emerald-400' },
  OPTIMIZE: { badge: 'bg-orange-500/20 text-orange-400 border-orange-500/30', number: 'from-orange-500 to-amber-400' },
};

function PromptCard({ prompt, index }) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [copiedState, setCopiedState] = useState(null); // null | 'copied' | 'error'

  const style = CARD_STYLES[prompt.title] || CARD_STYLES.PLAN;
  const safeDomain = (prompt.domain || '').replace(/[<>]/g, '').substring(0, 30);

  const handleCopy = async () => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(prompt.content);
      } else {
        const textarea = document.createElement('textarea');
        textarea.value = prompt.content;
        textarea.style.cssText = 'position:fixed;left:-9999px;top:-9999px;';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        const success = document.execCommand('copy');
        document.body.removeChild(textarea);
        if (!success) throw new Error('execCommand failed');
      }
      setCopiedState('copied');
    } catch {
      setCopiedState('error');
    } finally {
      setTimeout(() => setCopiedState(null), 2000);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.15, duration: 0.4 }}
      className="bg-white/5 border border-white/8 rounded-[18px] overflow-hidden"
    >
      {/* Header */}
      <button
        onClick={() => setIsExpanded(v => !v)}
        className="w-full flex items-center gap-3 p-4 hover:bg-white/5 transition-colors text-left"
      >
        {/* Number badge */}
        <div className={`w-7 h-7 rounded-full bg-gradient-to-br ${style.number} flex items-center justify-center shrink-0`}>
          <span className="text-white text-xs font-bold">{index + 1}</span>
        </div>

        {/* Title */}
        <span className="text-white font-bold text-sm tracking-wide flex-1">
          {prompt.title}
        </span>

        {/* Domain badge */}
        {safeDomain && (
          <span className={`text-[10px] border rounded-full px-2 py-0.5 ${style.badge}`}>
            {safeDomain || 'General'}
          </span>
        )}

        {/* Chevron */}
        <motion.div
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown size={14} className="text-white/40" />
        </motion.div>
      </button>

      {/* Purpose line */}
      {prompt.purpose && (
        <p className="px-4 pb-2 text-[11px] text-white/40 italic leading-relaxed">
          {prompt.purpose}
        </p>
      )}

      {/* Content */}
      {isExpanded && (
        <div className="px-4 pb-4">
          <div className="relative">
            <pre className="bg-black/40 rounded-xl p-4 font-mono text-[11px] text-slate-300 leading-relaxed whitespace-pre-wrap max-h-72 overflow-y-auto border border-white/5 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-white/10">
              {prompt.content}
            </pre>

            {/* Copy button */}
            <button
              onClick={handleCopy}
              className="absolute top-2 right-2 p-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors flex items-center gap-1"
              title={copiedState === 'error' ? 'Copy failed' : 'Copy prompt'}
            >
              {copiedState === 'copied' ? (
                <Check size={11} className="text-green-400" />
              ) : copiedState === 'error' ? (
                <AlertCircle size={11} className="text-red-400" />
              ) : (
                <Copy size={11} className="text-white/50" />
              )}
              <span className="text-[9px] text-white/40">
                {copiedState === 'copied' ? 'Copied!' : copiedState === 'error' ? 'Failed' : 'Copy'}
              </span>
            </button>
          </div>
        </div>
      )}
    </motion.div>
  );
}

export default function PromptArchitectCards({ prompts = [], domain = '' }) {
  if (!prompts || prompts.length === 0) return null;

  return (
    <div className="flex flex-col gap-3 w-full max-w-2xl">
      {/* Header label */}
      <div className="flex items-center gap-2 mb-1">
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-blue-500/30 to-transparent" />
        <span className="text-[10px] text-white/30 uppercase tracking-widest px-2">Prompt Architect</span>
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-blue-500/30 to-transparent" />
      </div>

      {prompts.map((prompt, index) => (
        <PromptCard
          key={`${prompt.title}-${index}`}
          prompt={prompt}
          index={index}
        />
      ))}
    </div>
  );
}
