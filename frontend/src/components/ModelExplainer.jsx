import { useState } from 'react';

const FEATURES = [
    { name: 'Weight Lifted', pct: 35.5, desc: 'The primary load used in the exercise' },
    { name: 'Bodyweight', pct: 31.3, desc: 'User bodyweight influences relative strength' },
    { name: 'Progression', pct: 13.0, desc: 'Recent trend direction (improving / stalling)' },
    { name: 'Exercise Type', pct: 9.9, desc: 'Bench press, squat, or deadlift' },
    { name: 'Volume Score', pct: 6.2, desc: 'Weekly training stress accumulation' },
    { name: 'RIR', pct: 1.8, desc: 'Reps in reserve — proximity to failure' },
    { name: 'Sets', pct: 1.3, desc: 'Number of working sets per session' },
    { name: 'Frequency', pct: 1.0, desc: 'Training sessions per week' },
];

export default function ModelExplainer({ isOpen, onClose }) {
    const [expandedFeature, setExpandedFeature] = useState(null);

    if (!isOpen) return null;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="modal-handle" />

                <h2 className="modal-title">🧠 How SmartLift Works</h2>

                {/* Model Overview */}
                <div className="explainer-section">
                    <h3>The Model</h3>
                    <p>
                        SmartLift uses a <strong>Gradient Boosting Regression</strong> model trained on
                        strength progression data. It learns patterns from thousands of training sessions
                        to predict how many reps you can perform at a given weight.
                    </p>
                </div>

                <div className="explainer-section">
                    <h3>Why Gradient Boosting?</h3>
                    <p>
                        Gradient Boosting excels at structured, tabular data — exactly what workout logs are.
                        It builds an ensemble of decision trees sequentially, with each tree correcting errors
                        from the previous ones. This gives us strong predictive accuracy while maintaining
                        interpretable feature importance.
                    </p>
                </div>

                {/* Feature Importance */}
                <div className="explainer-section">
                    <h3>Feature Importance</h3>
                    <p style={{ marginBottom: 16 }}>
                        These are the factors the model weighs when making predictions, ranked by importance:
                    </p>
                    <div className="feature-importance">
                        {FEATURES.map((f, i) => (
                            <div key={i}>
                                <div
                                    className="feature-row"
                                    onClick={() => setExpandedFeature(expandedFeature === i ? null : i)}
                                    style={{ cursor: 'pointer' }}
                                >
                                    <span className="feature-name">{f.name}</span>
                                    <div className="feature-bar-bg">
                                        <div
                                            className="feature-bar"
                                            style={{ width: `${(f.pct / 35.5) * 100}%` }}
                                        />
                                    </div>
                                    <span className="feature-pct">{f.pct}%</span>
                                </div>
                                {expandedFeature === i && (
                                    <div style={{
                                        fontSize: '0.78rem',
                                        color: '#9898a8',
                                        padding: '4px 0 8px 0',
                                        marginLeft: 120,
                                        animation: 'fadeSlideIn 0.2s ease',
                                    }}>
                                        {f.desc}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                {/* How Predictions Work */}
                <div className="explainer-section">
                    <h3>Prediction Pipeline</h3>
                    <p>
                        When you log a workout, the model takes your most recent performance data and predicts
                        how many reps you'll achieve at various weights. The recommendation engine then:
                    </p>
                    <ul style={{ color: '#9898a8', fontSize: '0.85rem', lineHeight: 1.8, paddingLeft: 20 }}>
                        <li>Predicts reps at your current working weight</li>
                        <li>If predicted reps {'>'} 9 → increases weight by 5 lbs</li>
                        <li>If predicted reps {'<'} 5 → decreases weight by 5 lbs</li>
                        <li>Targets the 5–9 rep hypertrophy range</li>
                        <li>Sets target RIR at 2 for productive training stimulus</li>
                    </ul>
                </div>

                {/* 1RM Estimation */}
                <div className="explainer-section">
                    <h3>1RM Estimation</h3>
                    <p>
                        Estimated one-rep max is calculated using the <strong>Epley formula</strong> with
                        an RIR adjustment:
                    </p>
                    <div style={{
                        background: '#1a1a26',
                        border: '1px solid rgba(168,85,247,0.2)',
                        borderRadius: 10,
                        padding: '12px 16px',
                        marginTop: 8,
                        fontFamily: 'monospace',
                        fontSize: '0.85rem',
                        color: '#c084fc',
                    }}>
                        e1RM = weight × (1 + (reps + RIR) / 30)
                    </div>
                </div>

                {/* Volume Score */}
                <div className="explainer-section">
                    <h3>Volume Score (0–100)</h3>
                    <p>
                        The volume score measures weekly training stress. It's calculated from:
                    </p>
                    <ul style={{ color: '#9898a8', fontSize: '0.85rem', lineHeight: 1.8, paddingLeft: 20 }}>
                        <li><strong>Set intensity:</strong> Lower RIR = higher stress per set</li>
                        <li><strong>Training frequency:</strong> More sessions = higher volume</li>
                        <li><strong>Muscle group:</strong> Compound lifts (squat, deadlift) generate more fatigue</li>
                    </ul>
                </div>

                {/* Recovery */}
                <div className="explainer-section">
                    <h3>Recovery Estimation</h3>
                    <p>
                        After each workout, SmartLift estimates the recovery time needed before your next
                        session. This is based on the muscle group (legs need longer), session intensity
                        (lower RIR = more recovery), and total set volume. All fatigue is designed to be
                        fully recoverable before the next session — no deload weeks needed.
                    </p>
                </div>

                {/* Model Performance */}
                <div className="explainer-section">
                    <h3>Model Performance</h3>
                    <div className="stat-grid" style={{ marginTop: 8 }}>
                        <div className="stat-card">
                            <div className="stat-value">1.12</div>
                            <div className="stat-label">MAE (reps)</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">0.38</div>
                            <div className="stat-label">R² Score</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">5,559</div>
                            <div className="stat-label">Training Samples</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">200</div>
                            <div className="stat-label">Trees (Estimators)</div>
                        </div>
                    </div>
                </div>

                <button
                    className="btn btn--secondary"
                    onClick={onClose}
                    style={{ marginTop: 8 }}
                >
                    Close
                </button>
            </div>
        </div>
    );
}
