import { useState, useEffect } from 'react';
import {
    LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Area, AreaChart,
} from 'recharts';
import { api } from '../api';

const EXERCISES = [
    { id: 'bench_press', label: 'Bench' },
    { id: 'squat', label: 'Squat' },
    { id: 'deadlift', label: 'Deadlift' },
];

const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
        <div style={{
            background: '#1e1e2a',
            border: '1px solid rgba(168,85,247,0.3)',
            borderRadius: '10px',
            padding: '10px 14px',
            fontSize: '0.78rem',
        }}>
            <div style={{ color: '#a855f7', fontWeight: 700 }}>
                {d.estimated_1rm?.toFixed(1) || d.score} {d.estimated_1rm ? 'lbs' : ''}
            </div>
            {d.weight && (
                <div style={{ color: '#9898a8', marginTop: 4 }}>
                    {d.weight} lbs × {d.reps} reps | RIR {d.rir}
                </div>
            )}
            <div style={{ color: '#5a5a6e', marginTop: 2 }}>{d.date || d.week}</div>
        </div>
    );
};

export default function Dashboard() {
    const [exercise, setExercise] = useState('bench_press');
    const [progress, setProgress] = useState([]);
    const [volume, setVolume] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        setLoading(true);
        Promise.all([
            api.getProgress(exercise),
            api.getVolumeScore(exercise),
        ]).then(([prog, vol]) => {
            setProgress(prog);
            setVolume(vol);
            setLoading(false);
        }).catch(() => setLoading(false));
    }, [exercise]);

    if (loading) {
        return (
            <div className="page">
                <div className="loading">
                    <div className="spinner"></div>
                    <span>Loading dashboard...</span>
                </div>
            </div>
        );
    }

    // Calculate stats
    const latestE1RM = progress.length ? progress[progress.length - 1].estimated_1rm : 0;
    const firstE1RM = progress.length ? progress[0].estimated_1rm : 0;
    const e1rmChange = latestE1RM - firstE1RM;

    return (
        <div className="page">
            <h2 className="page-title">Progress Dashboard</h2>

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

            {/* Stats Grid */}
            <div className="stat-grid">
                <div className="stat-card">
                    <div className="stat-value">{latestE1RM.toFixed(0)}</div>
                    <div className="stat-label">Est. 1RM (lbs)</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{
                        background: e1rmChange >= 0
                            ? 'linear-gradient(135deg, #34d399, #60a5fa)'
                            : 'linear-gradient(135deg, #f87171, #fbbf24)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                    }}>
                        {e1rmChange >= 0 ? '+' : ''}{e1rmChange.toFixed(0)}
                    </div>
                    <div className="stat-label">Change (lbs)</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{volume?.current_score || 0}</div>
                    <div className="stat-label">Volume Score</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{progress.length}</div>
                    <div className="stat-label">Sessions</div>
                </div>
            </div>

            {/* 1RM Progress Chart */}
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Estimated 1RM Trend</span>
                    <span className="card-label">Over time</span>
                </div>
                {progress.length > 0 ? (
                    <div className="chart-container">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={progress}>
                                <defs>
                                    <linearGradient id="e1rmGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor="#a855f7" stopOpacity={0.3} />
                                        <stop offset="100%" stopColor="#a855f7" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis
                                    dataKey="date"
                                    tick={{ fill: '#5a5a6e', fontSize: 10 }}
                                    tickFormatter={(d) => d.slice(5)}
                                    axisLine={false}
                                    tickLine={false}
                                />
                                <YAxis
                                    tick={{ fill: '#5a5a6e', fontSize: 10 }}
                                    axisLine={false}
                                    tickLine={false}
                                    domain={['auto', 'auto']}
                                />
                                <Tooltip content={<CustomTooltip />} />
                                <Area
                                    type="monotone"
                                    dataKey="estimated_1rm"
                                    stroke="#a855f7"
                                    strokeWidth={2.5}
                                    fill="url(#e1rmGradient)"
                                    dot={{ r: 4, fill: '#a855f7', stroke: '#0a0a0f', strokeWidth: 2 }}
                                    activeDot={{ r: 6, fill: '#c084fc' }}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                ) : (
                    <div className="empty-state">
                        <div className="empty-state-icon">📊</div>
                        <div className="empty-state-text">Log workouts to see progress</div>
                    </div>
                )}
            </div>

            {/* Volume Score */}
            {volume && (
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Weekly Volume</span>
                        <span className="card-label">Stress Score</span>
                    </div>

                    {/* Gauge */}
                    <div className="volume-gauge">
                        <div className="gauge-ring">
                            <svg width="140" height="140" viewBox="0 0 140 140">
                                <circle
                                    cx="70" cy="70" r="60"
                                    fill="none"
                                    stroke="rgba(255,255,255,0.05)"
                                    strokeWidth="10"
                                />
                                <circle
                                    cx="70" cy="70" r="60"
                                    fill="none"
                                    stroke="url(#gaugeGrad)"
                                    strokeWidth="10"
                                    strokeLinecap="round"
                                    strokeDasharray={`${(volume.current_score / 100) * 377} 377`}
                                />
                                <defs>
                                    <linearGradient id="gaugeGrad" x1="0" y1="0" x2="1" y2="1">
                                        <stop offset="0%" stopColor="#a855f7" />
                                        <stop offset="100%" stopColor="#f97316" />
                                    </linearGradient>
                                </defs>
                            </svg>
                            <div className="gauge-value">{volume.current_score}</div>
                        </div>
                        <div className="gauge-label">{volume.interpretation}</div>
                    </div>

                    {/* Weekly Bar Chart */}
                    {volume.weekly_scores?.length > 0 && (
                        <div className="chart-container">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={volume.weekly_scores}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis
                                        dataKey="week"
                                        tick={{ fill: '#5a5a6e', fontSize: 9 }}
                                        axisLine={false}
                                        tickLine={false}
                                        tickFormatter={(w) => w.split('-')[1]}
                                    />
                                    <YAxis
                                        tick={{ fill: '#5a5a6e', fontSize: 10 }}
                                        axisLine={false}
                                        tickLine={false}
                                    />
                                    <Tooltip content={<CustomTooltip />} />
                                    <Bar
                                        dataKey="score"
                                        fill="url(#barGrad)"
                                        radius={[4, 4, 0, 0]}
                                    />
                                    <defs>
                                        <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor="#f97316" />
                                            <stop offset="100%" stopColor="#a855f7" />
                                        </linearGradient>
                                    </defs>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
