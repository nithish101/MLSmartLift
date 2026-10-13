import { useState } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { api } from '../api';
import { useExerciseData } from '../hooks/useExerciseData';
import { chartTheme, axisProps, CHART_MARGIN, formatShortDate } from '../theme';
import ExercisePicker from './ExercisePicker';
import ChartTooltip from './ChartTooltip';
import { Loading, LoadError } from './StatusMessage';

const loadCoach = (exercise) =>
    Promise.all([
        api.getRecommendation(exercise),
        api.getPredictedVsActual(exercise),
        api.getAlerts(exercise),
    ]).then(([rec, pva, alerts]) => ({ rec, pva, alerts }));

const ALERT_LABELS = { warning: 'Warning', info: 'Info', tip: 'Tip' };

const describeSession = (d) => ({
    title: formatShortDate(d.date),
    lines: [`Predicted ${d.predicted_reps} reps`, `Actual ${d.actual_reps} reps`, `at ${d.weight} lbs`],
});

export default function Recommendation() {
    const [exercise, setExercise] = useState('bench_press');
    const { data, error, loading } = useExerciseData(exercise, loadCoach);

    return (
        <div className="page">
            <div className="page-head">
                <h1 className="page-title">AI Recommendation</h1>
                <ExercisePicker value={exercise} onChange={setExercise} />
            </div>

            {loading && <Loading>Generating recommendation…</Loading>}
            {!loading && error && <LoadError />}
            {!loading && data && <CoachBody {...data} />}
        </div>
    );
}

function CoachBody({ rec, pva, alerts }) {
    const confidence = Math.round(rec.confidence * 100);

    return (
        <div className="panel-grid panel-grid--coach">
            <div className="stack">
                <section className="panel panel--primary" aria-labelledby="next-workout">
                    <h2 id="next-workout" className="label">Next workout</h2>
                    <p className="rec-weight">
                        <span className="num">{rec.recommended_weight}</span>
                        <span className="unit">lbs</span>
                    </p>
                    <dl className="spec">
                        <div className="spec-row">
                            <dt>Expected reps</dt>
                            <dd className="num">{rec.expected_reps}</dd>
                        </div>
                        <div className="spec-row">
                            <dt>Target RIR</dt>
                            <dd className="num">{rec.target_rir}</dd>
                        </div>
                        <div className="spec-row">
                            <dt>Model confidence</dt>
                            <dd className="num">{confidence}%</dd>
                        </div>
                    </dl>
                    <div className="meter-track" role="presentation">
                        <div className="meter-fill meter-fill--ink" style={{ width: `${confidence}%` }} />
                    </div>
                </section>

                {alerts.length > 0 && (
                    <section className="panel" aria-labelledby="alerts-title">
                        <h2 id="alerts-title" className="panel-title">Progression alerts</h2>
                        <ul className="alerts">
                            {alerts.map((alert) => (
                                <li key={alert.message} className={`alert alert--${alert.type}`}>
                                    <span className="alert-type">{ALERT_LABELS[alert.type] ?? 'Info'}</span>
                                    <span>{alert.message}</span>
                                </li>
                            ))}
                        </ul>
                    </section>
                )}
            </div>

            {pva.length > 0 && (
                <section className="panel">
                    <header className="panel-head">
                        <h2 className="panel-title">Predicted vs Actual</h2>
                        <span className="panel-meta">reps per session</span>
                    </header>
                    <div className="legend" aria-hidden="true">
                        <span className="legend-key legend-key--predicted">Predicted</span>
                        <span className="legend-key legend-key--actual">Actual</span>
                    </div>
                    <div className="chart chart--tall">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={pva} barGap={2} margin={CHART_MARGIN}>
                                <CartesianGrid vertical={false} stroke={chartTheme.grid} />
                                <XAxis dataKey="date" {...axisProps} tickFormatter={formatShortDate} minTickGap={16} />
                                <YAxis {...axisProps} allowDecimals={false} width={48} />
                                <Tooltip content={<ChartTooltip describe={describeSession} />} cursor={{ fill: chartTheme.grid }} />
                                <Bar dataKey="predicted_reps" name="Predicted" fill={chartTheme.bar} maxBarSize={14} isAnimationActive={false} />
                                <Bar dataKey="actual_reps" name="Actual" fill={chartTheme.ink} maxBarSize={14} isAnimationActive={false} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <p className="panel-foot">Compare AI predictions to your real performance.</p>
                </section>
            )}
        </div>
    );
}
