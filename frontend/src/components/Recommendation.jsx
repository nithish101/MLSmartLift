import { useState, useEffect } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Cell, Legend,
} from 'recharts';
import { api } from '../api';

const EXERCISES = [
    { id: 'bench_press', label: 'Bench' },
    { id: 'squat', label: 'Squat' },
    { id: 'deadlift', label: 'Deadlift' },
];

export default function Recommendation() {
    const [exercise, setExercise] = useState('bench_press');
    const [rec, setRec] = useState(null);
    const [pva, setPva] = useState([]);
    const [alerts, setAlerts] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        setLoading(true);
        Promise.all([
            api.getRecommendation(exercise),
            api.getPredictedVsActual(exercise),
            api.getAlerts(exercise),
        ]).then(([r, p, a]) => {
            setRec(r);
            setPva(p);
            setAlerts(a);
            setLoading(false);
        }).catch(() => setLoading(false));
    }, [exercise]);

    if (loading) {
        return (
            <div className="page">
                <div className="loading">
                    <div className="spinner"></div>
                    <span>Generating recommendation...</span>
                </div>
            </div>
        );
    }

    const alertIcons = { warning: '⚠️', info: 'ℹ️', tip: '💡' };

    return (
        <div className="page">
            <h2 className="page-title">AI Recommendation</h2>

            {/* Exercise Filter */}
            <div className="exercise-pills">
                {EXERCISES.map((ex) => (
                    <button
                        key={ex.id}
                        className={`exercise-pill ${exercise === ex.id ? 'active' : ''}`}
                        onClick={() => setExercise(ex.id)}
                    >
                        {ex.label}
                    </button>
                ))}
            </div>

            {/* Recommendation Card */}
            {rec && (
                <div className="rec-card">
                    <div className="card-label">Next Workout</div>
                    <div className="rec-weight">
                        {rec.recommended_weight}
                        <span className="rec-unit"> lbs</span>
                    </div>
                    <div className="rec-details">
                        <div className="rec-detail">
                            <div className="rec-detail-value">{rec.expected_reps}</div>
                            <div className="rec-detail-label">Expected Reps</div>
                        </div>
                        <div className="rec-detail">
                            <div className="rec-detail-value">{rec.target_rir}</div>
                            <div className="rec-detail-label">Target RIR</div>
                        </div>
                    </div>

                    {/* Confidence */}
                    <div style={{ marginTop: 16 }}>
                        <div style={{
                            display: 'flex', justifyContent: 'space-between',
                            fontSize: '0.75rem', color: '#9898a8', marginBottom: 4,
                        }}>
                            <span>Model Confidence</span>
                            <span>{Math.round(rec.confidence * 100)}%</span>
                        </div>
                        <div className="confidence-bar">
                            <div
                                className="confidence-fill"
                                style={{ width: `${rec.confidence * 100}%` }}
                            />
                        </div>
                    </div>
                </div>
            )}

            {/* Progression Alerts */}
            {alerts.length > 0 && (
                <>
                    <div className="section-header">
                        <span className="section-title">Progression Alerts</span>
                    </div>
                    {alerts.map((alert, i) => (
                        <div key={i} className={`alert alert--${alert.type}`}>
                            <span className="alert-icon">{alertIcons[alert.type] || 'ℹ️'}</span>
                            <span>{alert.message}</span>
                        </div>
                    ))}
                </>
            )}

            {/* Predicted vs Actual Chart */}
            {pva.length > 0 && (
                <div className="card" style={{ marginTop: 16 }}>
                    <div className="card-header">
                        <span className="card-title">Predicted vs Actual</span>
                        <span className="card-label">Reps</span>
                    </div>
                    <div className="chart-container" style={{ height: 240 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={pva} barGap={2}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis
                                    dataKey="date"
                                    tick={{ fill: '#5a5a6e', fontSize: 9 }}
                                    tickFormatter={(d) => d.slice(5)}
                                    axisLine={false}
                                    tickLine={false}
                                />
                                <YAxis
                                    tick={{ fill: '#5a5a6e', fontSize: 10 }}
                                    axisLine={false}
                                    tickLine={false}
                                />
                                <Tooltip
                                    contentStyle={{
                                        background: '#1e1e2a',
                                        border: '1px solid rgba(168,85,247,0.3)',
                                        borderRadius: 10,
                                        fontSize: '0.78rem',
                                    }}
                                    labelStyle={{ color: '#5a5a6e' }}
                                />
                                <Legend
                                    wrapperStyle={{ fontSize: '0.72rem', color: '#9898a8' }}
                                />
                                <Bar
                                    dataKey="predicted_reps"
                                    name="Predicted"
                                    fill="#a855f7"
                                    radius={[3, 3, 0, 0]}
                                    maxBarSize={20}
                                />
                                <Bar
                                    dataKey="actual_reps"
                                    name="Actual"
                                    fill="#f97316"
                                    radius={[3, 3, 0, 0]}
                                    maxBarSize={20}
                                />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="card-sub" style={{ textAlign: 'center', marginTop: 8 }}>
                        Compare AI predictions to your real performance
                    </div>
                </div>
            )}
        </div>
    );
}
