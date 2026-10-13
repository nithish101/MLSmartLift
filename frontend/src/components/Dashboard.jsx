import { useState } from 'react';
import {
    LineChart, Line, BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer,
} from 'recharts';
import { api } from '../api';
import { useExerciseData } from '../hooks/useExerciseData';
import { chartTheme, axisProps, CHART_MARGIN, formatShortDate } from '../theme';
import ExercisePicker from './ExercisePicker';
import ChartTooltip from './ChartTooltip';
import { Loading, LoadError } from './StatusMessage';

const loadDashboard = (exercise) =>
    Promise.all([api.getProgress(exercise), api.getVolumeScore(exercise)])
        .then(([progress, volume]) => ({ progress, volume }));

const describeSession = (d) => ({
    title: `${d.estimated_1rm.toFixed(1)} lbs e1RM`,
    lines: [`${d.weight} lbs × ${d.reps} reps, RIR ${d.rir}`, formatShortDate(d.date)],
});

const describeWeek = (d) => ({ title: `Score ${d.score}`, lines: [d.week] });

const VOLUME_TICKS = [0, 25, 50, 75, 100];

export default function Dashboard() {
    const [exercise, setExercise] = useState('bench_press');
    const { data, error, loading } = useExerciseData(exercise, loadDashboard);

    return (
        <div className="page">
            <div className="page-head">
                <h1 className="page-title">Progress Dashboard</h1>
                <ExercisePicker value={exercise} onChange={setExercise} />
            </div>

            {loading && <Loading>Loading dashboard…</Loading>}
            {!loading && error && <LoadError />}
            {!loading && data && <DashboardBody {...data} />}
        </div>
    );
}

function DashboardBody({ progress, volume }) {
    const latest = progress.at(-1);
    const latestE1RM = latest?.estimated_1rm ?? 0;
    const change = latestE1RM - (progress[0]?.estimated_1rm ?? 0);
    const lastIndex = progress.length - 1;
    const weeks = volume?.weekly_scores ?? [];

    return (
        <>
            <dl className="readout">
                <div className="readout-cell">
                    <dt>Est. 1RM</dt>
                    <dd><span className="num">{latestE1RM.toFixed(0)}</span> <span className="unit">lbs</span></dd>
                </div>
                <div className="readout-cell">
                    <dt>Change</dt>
                    <dd className={change >= 0 ? 'is-up' : 'is-down'}>
                        <span className="num">{change >= 0 ? '+' : '−'}{Math.abs(change).toFixed(0)}</span>{' '}
                        <span className="unit">lbs</span>
                    </dd>
                </div>
                <div className="readout-cell">
                    <dt>Volume score</dt>
                    <dd><span className="num">{volume?.current_score ?? 0}</span> <span className="unit">/ 100</span></dd>
                </div>
                <div className="readout-cell">
                    <dt>Sessions</dt>
                    <dd><span className="num">{progress.length}</span></dd>
                </div>
            </dl>

            <div className="panel-grid">
                <section className="panel">
                    <header className="panel-head">
                        <h2 className="panel-title">Estimated 1RM</h2>
                        <span className="panel-meta">lbs, per session</span>
                    </header>
                    {progress.length > 0 ? (
                        <div className="chart">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={progress} margin={CHART_MARGIN}>
                                    <CartesianGrid vertical={false} stroke={chartTheme.grid} />
                                    <XAxis dataKey="date" {...axisProps} tickFormatter={formatShortDate} minTickGap={24} />
                                    <YAxis {...axisProps} domain={['auto', 'auto']} width={48} />
                                    <Tooltip content={<ChartTooltip describe={describeSession} />} cursor={{ stroke: chartTheme.bar }} />
                                    <Line
                                        type="linear"
                                        dataKey="estimated_1rm"
                                        stroke={chartTheme.ink}
                                        strokeWidth={1.5}
                                        isAnimationActive={false}
                                        dot={({ index, cx, cy }) => (
                                            <circle
                                                key={index}
                                                cx={cx}
                                                cy={cy}
                                                r={index === lastIndex ? 4 : 2.5}
                                                fill={index === lastIndex ? chartTheme.signal : chartTheme.ink}
                                            />
                                        )}
                                        activeDot={{ r: 4, fill: chartTheme.ink, stroke: chartTheme.surface, strokeWidth: 2 }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <p className="empty">Log workouts to see progress.</p>
                    )}
                </section>

                {volume && (
                    <section className="panel">
                        <header className="panel-head">
                            <h2 className="panel-title">Weekly Volume</h2>
                            <span className="panel-meta">stress score</span>
                        </header>

                        <div className="meter" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={volume.current_score} aria-label="Volume score">
                            <div className="meter-track">
                                <div className="meter-fill" style={{ width: `${volume.current_score}%` }} />
                            </div>
                            <div className="meter-scale" aria-hidden="true">
                                {VOLUME_TICKS.map((t) => <span key={t}>{t}</span>)}
                            </div>
                        </div>
                        <p className="meter-note">{volume.interpretation}</p>

                        {weeks.length > 0 && (
                            <div className="chart chart--short">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={weeks} margin={CHART_MARGIN}>
                                        <CartesianGrid vertical={false} stroke={chartTheme.grid} />
                                        <XAxis dataKey="week" {...axisProps} tickFormatter={(w) => w.split('-')[1]} />
                                        <YAxis {...axisProps} domain={[0, 100]} ticks={VOLUME_TICKS} width={48} />
                                        <Tooltip content={<ChartTooltip describe={describeWeek} />} cursor={{ fill: chartTheme.grid }} />
                                        <Bar dataKey="score" maxBarSize={28} isAnimationActive={false}>
                                            {weeks.map((w, i) => (
                                                <Cell key={w.week} fill={i === weeks.length - 1 ? chartTheme.signal : chartTheme.bar} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        )}
                    </section>
                )}
            </div>
        </>
    );
}
