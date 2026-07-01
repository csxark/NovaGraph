import { create } from 'zustand';

const useAppStore = create((set) => ({
  // Current paper
  currentPaperId: null,
  currentPaperName: null,
  ingestionStatus: null, // 'pending' | 'processing' | 'completed' | 'failed'
  jobId: null,

  // Graph data
  graphNodes: [],
  graphEdges: [],

  // Chat
  messages: [],

  // Actions
  setCurrentPaper: (paperId, paperName, jobId) => set({
    currentPaperId: paperId,
    currentPaperName: paperName,
    jobId,
    ingestionStatus: 'pending',
    graphNodes: [],
    graphEdges: [],
    messages: [],
  }),

  setIngestionStatus: (status) => set({ ingestionStatus: status }),

  setGraphData: (nodes, edges) => set({ graphNodes: nodes, graphEdges: edges }),

  addMessage: (message) => set((state) => ({
    messages: [...state.messages, message],
  })),

  resetAll: () => set({
    currentPaperId: null,
    currentPaperName: null,
    ingestionStatus: null,
    jobId: null,
    graphNodes: [],
    graphEdges: [],
    messages: [],
  }),
}));

export default useAppStore;
