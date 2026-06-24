import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import AppShell from './components/layout/AppShell';
import DashboardPage from './pages/DashboardPage';

import HistoryPage from './pages/HistoryPage';
import ScenarioPlannerPage from './pages/ScenarioPlannerPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<AppShell><DashboardPage /></AppShell>} />
        <Route path="/history" element={<AppShell><HistoryPage /></AppShell>} />
        <Route path="/scenarios" element={<AppShell><ScenarioPlannerPage /></AppShell>} />
      </Routes>
    </BrowserRouter>
  );
}
