import React, { useState } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { 
  ArrowRight, 
  Upload, 
  Calendar,
  FileText, 
  Database, 
  GitBranch, 
  Search, 
  CheckCircle 
} from 'lucide-react';
import UploadModal from '../components/UploadModal';

export default function LandingPage() {
  const navigate = useNavigate();
  const { scrollY } = useScroll();
  
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  // Navbar glass effect on scroll
  const navBg = useTransform(
    scrollY, 
    [0, 50], 
    ['rgba(4, 8, 18, 0)', 'rgba(11, 18, 32, 0.85)']
  );
  const navBorder = useTransform(
    scrollY,
    [0, 50],
    ['rgba(255, 255, 255, 0)', '1px solid rgba(255, 255, 255, 0.08)']
  );

  return (
    <div className="relative min-h-screen text-white overflow-x-hidden font-sans" style={{ backgroundColor: '#040812' }}>
      
      {/* ── Background & Ambient Glows ── */}
      {/* Radial Blue Glow in Center */}
      <div 
        className="absolute top-[20%] left-1/2 -translate-x-1/2 w-[800px] h-[500px] rounded-full pointer-events-none opacity-20"
        style={{
          background: 'radial-gradient(circle, #22D3EE 0%, #3B82F6 40%, transparent 70%)',
          filter: 'blur(80px)'
        }}
      />
      
      {/* Subtle Grid Pattern */}
      <div 
        className="absolute inset-0 pointer-events-none opacity-10"
        style={{
          backgroundImage: `
            linear-gradient(to right, rgba(255, 255, 255, 0.05) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(255, 255, 255, 0.05) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px'
        }}
      />
      
      {/* Noise Overlay */}
      <div 
        className="absolute inset-0 pointer-events-none opacity-[0.02]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`
        }}
      />

      {/* FIXED NAVBAR */}
      <motion.nav 
        style={{ background: navBg, borderBottom: navBorder, backdropFilter: 'blur(16px)' }}
        className="fixed top-0 left-0 right-0 h-20 z-50 transition-all duration-300 flex items-center justify-between px-10 max-w-[1600px] mx-auto"
      >
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2 group">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] flex items-center justify-center shadow-lg">
            <GitBranch size={16} className="text-white" />
          </div>
          <span className="text-lg font-bold tracking-tight text-white">
            Graphora
          </span>
        </Link>

        {/* Nav Links */}
        <div className="hidden md:flex items-center gap-8">
          {['Product', 'Features', 'Use Cases', 'Pricing', 'Docs'].map((link) => (
            <a 
              key={link} 
              href={`#${link.toLowerCase()}`}
              className="text-sm font-medium text-slate-400 hover:text-white transition-colors duration-200"
            >
              {link}
            </a>
          ))}
        </div>

        {/* CTA Button */}
        <div>
          <button 
            onClick={() => setIsModalOpen(true)}
            className="flex items-center gap-2 px-5 py-2 rounded-full border border-white/10 bg-white/5 hover:bg-gradient-to-r hover:from-[#3B82F6] hover:to-[#22D3EE] hover:border-transparent hover:shadow-[0_0_15px_rgba(34,211,238,0.3)] transition-all duration-300 text-xs font-semibold text-white group"
          >
            Get Started
            <div className="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center group-hover:bg-white/20 transition-colors">
              <ArrowRight size={12} className="text-white" />
            </div>
          </button>
        </div>
      </motion.nav>

      {/* HERO SECTION */}
      <div className="pt-32 pb-16 px-10 max-w-[1600px] mx-auto min-h-screen flex flex-col justify-between relative z-10">
        
        {/* Main 3-Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center flex-1 my-auto">
          
          {/* Left Column (25% -> 3 cols) — Upload Flow Illustration */}
          <div className="lg:col-span-3 flex flex-col items-center justify-center relative min-h-[350px]">
            <div className="relative w-64 h-64 flex items-center justify-center">
              
              {/* Fiber Optic Flow Lines (streaming to the right) */}
              <svg className="absolute w-[400px] h-[300px] left-10 pointer-events-none text-[#22D3EE]" viewBox="0 0 200 150">
                <defs>
                  <linearGradient id="fiberGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.8" />
                    <stop offset="50%" stopColor="#22D3EE" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="#22D3EE" stopOpacity="0" />
                  </linearGradient>
                </defs>
                {/* Custom glowing paths fanning out to the center-right */}
                <path d="M 30,75 C 60,60 110,10 180,20" fill="none" stroke="url(#fiberGrad)" strokeWidth="1" />
                <path d="M 30,75 C 70,70 120,30 180,45" fill="none" stroke="url(#fiberGrad)" strokeWidth="1" />
                <path d="M 30,75 C 80,80 130,55 180,70" fill="none" stroke="url(#fiberGrad)" strokeWidth="1.5" />
                <path d="M 30,75 C 80,90 130,95 180,95" fill="none" stroke="url(#fiberGrad)" strokeWidth="1" />
                <path d="M 30,75 C 70,110 120,130 180,120" fill="none" stroke="url(#fiberGrad)" strokeWidth="1" />
                <path d="M 30,75 C 60,120 110,150 180,140" fill="none" stroke="url(#fiberGrad)" strokeWidth="1" />

                {/* Animated Particles flowing on lines */}
                <motion.circle 
                  r="2" fill="#22D3EE"
                  animate={{
                    offsetDistance: ["0%", "100%"]
                  }}
                  transition={{
                    repeat: Infinity,
                    duration: 3,
                    ease: "linear"
                  }}
                  style={{
                    motionPath: "path('M 30,75 C 80,80 130,55 180,70')"
                  }}
                />
              </svg>

              {/* Glowing anchor node on left */}
              <div className="relative z-10 w-24 h-24 rounded-full flex flex-col items-center justify-center bg-[#0B1220] border-2 border-[#22D3EE] shadow-[0_0_30px_rgba(34,211,238,0.4)]">
                <FileText size={32} className="text-[#22D3EE]" />
                <span className="absolute -bottom-8 whitespace-nowrap text-xs text-slate-400 font-mono">
                  Research Paper
                </span>
              </div>

            </div>
          </div>

          {/* Center Column (50% -> 6 cols) — Copy */}
          <div className="lg:col-span-6 flex flex-col items-center text-center px-4">
            <span className="text-xs font-bold text-[#22D3EE] tracking-[3px] uppercase mb-4">
              AI-POWERED RESEARCH INTELLIGENCE
            </span>

            <h1 className="text-4xl md:text-5xl lg:text-[72px] font-bold tracking-tight leading-[1.08] mb-6 text-white">
              See Every Insight.<br />
              Connect Every Idea.<br />
              Query with <span className="bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] bg-clip-text text-transparent">Intelligence.</span>
            </h1>

            <p className="text-base text-slate-400 max-w-[480px] mb-8 leading-relaxed">
              Graphora converts any research paper into a structured knowledge graph and empowers multi-agent retrieval to deliver accurate, evidence-backed answers.
            </p>

            {/* Buttons Row */}
            <div className="flex flex-col sm:flex-row gap-4 mb-4 justify-center">
              <button 
                onClick={() => setIsModalOpen(true)}
                className="flex items-center justify-center gap-2 px-8 py-3.5 rounded-full bg-gradient-to-r from-[#3B82F6] to-[#22D3EE] text-white font-semibold text-sm hover:shadow-[0_0_25px_rgba(59,130,246,0.4)] transition-all duration-300"
              >
                <Upload size={16} />
                Upload a Paper
              </button>
              
              <button 
                onClick={() => setIsModalOpen(true)}
                className="flex items-center justify-center gap-2 px-8 py-3.5 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 transition-colors duration-200 text-sm font-semibold text-white"
              >
                <Calendar size={16} className="text-[#22D3EE]" />
                Book a Demo
              </button>
            </div>

            <p className="text-xs text-slate-500 mt-4">
              Works with any field of research
            </p>
          </div>

          {/* Right Column (25% -> 3 cols) — 3D Graph Sphere */}
          <div className="lg:col-span-3 flex flex-col items-center justify-center relative min-h-[350px]">
            <div className="relative w-64 h-64 flex items-center justify-center">
              
              {/* Globe grid structure using CSS animations */}
              <div className="absolute w-52 h-52 rounded-full border border-[#22D3EE]/25 animate-[spin_20s_linear_infinite]" style={{ transformStyle: 'preserve-3d' }}>
                {/* Horizontal latitude lines */}
                <div className="absolute inset-0 rounded-full border border-[#22D3EE]/20 rotate-x-45" />
                <div className="absolute inset-0 rounded-full border border-[#22D3EE]/20 rotate-x-90" />
                {/* Vertical longitude lines */}
                <div className="absolute inset-0 rounded-full border border-[#22D3EE]/20 rotate-y-45" />
                <div className="absolute inset-0 rounded-full border border-[#22D3EE]/20 rotate-y-90" />
              </div>

              {/* Pulsing Nodes on Globe intersections */}
              <div className="absolute inset-0 flex items-center justify-center">
                <svg viewBox="0 0 100 100" className="w-48 h-48">
                  {/* Connective lines */}
                  <line x1="50" y1="50" x2="25" y2="25" stroke="rgba(34,211,238,0.2)" strokeWidth="1" />
                  <line x1="50" y1="50" x2="75" y2="25" stroke="rgba(34,211,238,0.2)" strokeWidth="1" />
                  <line x1="50" y1="50" x2="25" y2="75" stroke="rgba(34,211,238,0.2)" strokeWidth="1" />
                  <line x1="50" y1="50" x2="75" y2="75" stroke="rgba(34,211,238,0.2)" strokeWidth="1" />
                  
                  {/* Nodes */}
                  <motion.circle 
                    cx="50" cy="50" r="8" fill="#3B82F6" 
                    animate={{ r: [8, 10, 8] }}
                    transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}
                  />
                  <motion.circle cx="25" cy="25" r="5" fill="#22D3EE" />
                  <motion.circle cx="75" cy="25" r="5" fill="#6366F1" />
                  <motion.circle cx="25" cy="75" r="4" fill="#10B981" />
                  <motion.circle cx="75" cy="75" r="6" fill="#F59E0B" />
                </svg>
              </div>

              {/* Floating Label Chips */}
              {/* Concepts */}
              <div className="absolute top-0 -translate-y-4 px-3.5 py-1.5 rounded-full border border-white/5 bg-[#0B1220]/90 backdrop-blur-md text-[10px] font-bold text-white shadow-xl">
                Concepts
              </div>

              {/* Methods */}
              <div className="absolute right-0 top-1/4 translate-x-8 px-3.5 py-1.5 rounded-full border border-white/5 bg-[#0B1220]/90 backdrop-blur-md text-[10px] font-bold text-white shadow-xl">
                Methods
              </div>

              {/* Results */}
              <div className="absolute right-0 bottom-1/4 translate-x-10 px-3.5 py-1.5 rounded-full border border-[#F59E0B]/30 bg-[#0B1220]/90 backdrop-blur-md text-[10px] font-bold text-[#F59E0B] shadow-xl">
                Results
              </div>

              {/* Datasets */}
              <div className="absolute bottom-0 translate-y-4 px-3.5 py-1.5 rounded-full border border-white/5 bg-[#0B1220]/90 backdrop-blur-md text-[10px] font-bold text-white shadow-xl">
                Datasets
              </div>

              {/* References */}
              <div className="absolute left-0 top-1/2 -translate-x-8 px-3.5 py-1.5 rounded-full border border-white/5 bg-[#0B1220]/90 backdrop-blur-md text-[10px] font-bold text-white shadow-xl">
                References
              </div>

            </div>
          </div>

        </div>

        {/* TRUST CHIPS ROW */}
        <div className="w-full mt-12 mb-4">
          <div className="flex flex-wrap items-center justify-center gap-4">
            
            {/* Any Paper */}
            <div className="flex items-center gap-2.5 px-5 py-2.5 rounded-full border border-white/10 bg-white/5 hover:border-[#3B82F6] hover:shadow-[0_0_15px_rgba(59,130,246,0.2)] transition-all cursor-pointer group">
              <FileText size={15} className="text-[#3B82F6] group-hover:scale-110 transition-transform" />
              <span className="text-[13px] text-slate-300 group-hover:text-white">Any Paper</span>
            </div>

            {/* Structured Knowledge */}
            <div className="flex items-center gap-2.5 px-5 py-2.5 rounded-full border border-white/10 bg-white/5 hover:border-[#22D3EE] hover:shadow-[0_0_15px_rgba(34,211,238,0.2)] transition-all cursor-pointer group">
              <Database size={15} className="text-[#22D3EE] group-hover:scale-110 transition-transform" />
              <span className="text-[13px] text-slate-300 group-hover:text-white">Structured Knowledge</span>
            </div>

            {/* Knowledge Graph */}
            <div className="flex items-center gap-2.5 px-5 py-2.5 rounded-full border border-white/10 bg-white/5 hover:border-[#6366F1] hover:shadow-[0_0_15px_rgba(99,102,241,0.2)] transition-all cursor-pointer group">
              <GitBranch size={15} className="text-[#6366F1] group-hover:scale-110 transition-transform" />
              <span className="text-[13px] text-slate-300 group-hover:text-white">Knowledge Graph</span>
            </div>

            {/* Intelligent Retrieval */}
            <div className="flex items-center gap-2.5 px-5 py-2.5 rounded-full border border-white/10 bg-white/5 hover:border-[#3B82F6] hover:shadow-[0_0_15px_rgba(59,130,246,0.2)] transition-all cursor-pointer group">
              <Search size={15} className="text-[#3B82F6] group-hover:scale-110 transition-transform" />
              <span className="text-[13px] text-slate-300 group-hover:text-white">Intelligent Retrieval</span>
            </div>

            {/* Evidence-Backed */}
            <div className="flex items-center gap-2.5 px-5 py-2.5 rounded-full border border-white/10 bg-white/5 hover:border-[#10B981] hover:shadow-[0_0_15px_rgba(16,185,129,0.2)] transition-all cursor-pointer group">
              <CheckCircle size={15} className="text-[#10B981] group-hover:scale-110 transition-transform" />
              <span className="text-[13px] text-slate-300 group-hover:text-white">Evidence-Backed</span>
            </div>

          </div>
        </div>

      </div>

      {/* SECTION DIVIDER */}
      <div className="relative w-full flex items-center justify-center py-8">
        <div 
          className="w-1/3 h-[1px]"
          style={{
            background: 'linear-gradient(to right, transparent, rgba(34,211,238,0.2), transparent)'
          }}
        />
        <div className="flex gap-1.5 px-4">
          <div className="w-1.5 h-1.5 rounded-full bg-[#22D3EE]" />
          <div className="w-1.5 h-1.5 rounded-full bg-[#3B82F6]" />
        </div>
        <div 
          className="w-1/3 h-[1px]"
          style={{
            background: 'linear-gradient(to right, transparent, rgba(34,211,238,0.2), transparent)'
          }}
        />
      </div>

      {/* Render Upload Modal */}
      <UploadModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />

    </div>
  );
}
