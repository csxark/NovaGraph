import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import ChatPage from './pages/ChatPage';
import GraphPage from './pages/GraphPage';
import useAppStore from './store/appStore';

function RouteGuard({ children }) {
  const currentDocId = useAppStore((state) => state.currentDocId);
  if (!currentDocId) {
    return <Navigate to="/" replace />;
  }
  return children;
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/chat" element={
          <RouteGuard>
            <ChatPage />
          </RouteGuard>
        } />
        <Route path="/graph" element={
          <RouteGuard>
            <GraphPage />
          </RouteGuard>
        } />
        <Route path="/dashboard" element={<Navigate to="/chat" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}
