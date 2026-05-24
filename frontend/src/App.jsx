import { useState } from 'react';
import WorkoutLogger from './components/WorkoutLogger';
import Dashboard from './components/Dashboard';
import Recommendation from './components/Recommendation';
import ModelExplainer from './components/ModelExplainer';
import './App.css';

const TABS = [
  { id: 'log', label: 'Log', icon: '🏋️' },
  { id: 'dashboard', label: 'Dashboard', icon: '📊' },
  { id: 'ai', label: 'AI Coach', icon: '🤖' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [showExplainer, setShowExplainer] = useState(false);
  const [toast, setToast] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const handleWorkoutLogged = (result) => {
    setToast(`Workout logged! Recovery: ${result.recovery.recovery_estimate}`);
    setTimeout(() => setToast(null), 3000);
    setRefreshKey((k) => k + 1);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="app-logo">
          <img src="/smartlift-icon.svg" alt="SmartLift" />
          <h1>SmartLift</h1>
        </div>
        <div className="header-actions">
          <button
            className="btn btn--icon btn--secondary"
            onClick={() => setShowExplainer(true)}
            title="How it works"
          >
            🧠
          </button>
        </div>
      </header>

      {/* Navigation */}
      <nav className="nav-tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="nav-tab-icon">{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </nav>

      {/* Pages */}
      {activeTab === 'log' && (
        <WorkoutLogger onWorkoutLogged={handleWorkoutLogged} />
      )}
      {activeTab === 'dashboard' && (
        <Dashboard key={refreshKey} />
      )}
      {activeTab === 'ai' && (
        <Recommendation key={refreshKey} />
      )}

      {/* Model Explainer Modal */}
      <ModelExplainer
        isOpen={showExplainer}
        onClose={() => setShowExplainer(false)}
      />

      {/* Toast */}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
