import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import DashboardPage from './components/DashboardPage';
import GraphPage from './components/GraphPage';

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/graph" element={<GraphPage />} />
      </Routes>
    </Router>
  );
}
