import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import App from './App';
import Onboarding from './pages/Onboarding';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import IJRStrategy from './pages/strategies/IJRStrategy';
import LeveragedMidcapStrategy from './pages/strategies/LeveragedMidcapStrategy';
import TrendingValueStrategy from './pages/strategies/TrendingValueStrategy';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/create-account" element={<Onboarding />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/strategy/ijr" element={<IJRStrategy />} />
        <Route path="/strategy/leveraged-midcap" element={<LeveragedMidcapStrategy />} />
        <Route path="/strategy/trending-value" element={<TrendingValueStrategy />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);