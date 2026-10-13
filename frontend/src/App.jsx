import { useEffect, useRef, useState } from 'react';
import WorkoutLogger from './components/WorkoutLogger';
import Dashboard from './components/Dashboard';
import Recommendation from './components/Recommendation';
import ModelExplainer from './components/ModelExplainer';

const TABS = [
  { id: 'log', label: 'Log' },
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'ai', label: 'AI Coach' },
];

const TOAST_MS = 3000;

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [showExplainer, setShowExplainer] = useState(false);
  const [toast, setToast] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const toastTimer = useRef(null);

  useEffect(() => () => clearTimeout(toastTimer.current), []);

  const handleWorkoutLogged = (result) => {
    clearTimeout(toastTimer.current);
    setToast(`Workout logged. Recovery: ${result.recovery.recovery_estimate}`);
    toastTimer.current = setTimeout(() => setToast(null), TOAST_MS);
    setRefreshKey((k) => k + 1);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-logo">
          <img src="/smartlift-icon.svg" alt="" width="28" height="28" />
          <span className="app-name">SmartLift</span>
        </div>

        <nav className="tab-bar" aria-label="Sections">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className="tab"
              aria-current={activeTab === tab.id ? 'page' : undefined}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <button
          type="button"
          className="btn btn--quiet btn--sm"
          onClick={() => setShowExplainer(true)}
        >
          How it works
        </button>
      </header>

      <main className="app-main">
        {activeTab === 'log' && (
          <WorkoutLogger onWorkoutLogged={handleWorkoutLogged} />
        )}
        {activeTab === 'dashboard' && <Dashboard key={refreshKey} />}
        {activeTab === 'ai' && <Recommendation key={refreshKey} />}
      </main>

      <ModelExplainer
        isOpen={showExplainer}
        onClose={() => setShowExplainer(false)}
      />

      <div className="toast-region" role="status" aria-live="polite">
        {toast && <div className="toast">{toast}</div>}
      </div>
    </div>
  );
}
