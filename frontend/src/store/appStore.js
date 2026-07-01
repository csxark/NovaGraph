import { create } from 'zustand';

// Generate or retrieve user isolation ID from localStorage
let initialUserId = localStorage.getItem('graphora_user_id');
if (!initialUserId) {
  initialUserId = crypto.randomUUID();
  localStorage.setItem('graphora_user_id', initialUserId);
}

const useAppStore = create((set) => ({
  // User Isolation ID
  userId: initialUserId,

  // Current document state
  currentDocId: null,
  currentPaperName: null,
  ingestionStatus: null, // 'pending' | 'processing' | 'completed' | 'failed'
  jobId: null,

  // Graph data
  graphNodes: [],
  graphEdges: [],

  // Chat
  messages: [],

  // Actions
  setCurrentDocument: (docId, paperName, jobId) => set({
    currentDocId: docId,
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
    currentDocId: null,
    currentPaperName: null,
    ingestionStatus: null,
    jobId: null,
    graphNodes: [],
    graphEdges: [],
    messages: [],
  }),
}));

export default useAppStore;
