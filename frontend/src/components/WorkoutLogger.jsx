import { useState, useCallback } from 'react';
import { api } from '../api';
import ExercisePicker from './ExercisePicker';

const RIR_OPTIONS = [
    { value: -1, label: 'Failure' },
    { value: 0, label: '0' },
    { value: 1, label: '1' },
    { value: 2, label: '2' },
    { value: 3, label: '3' },
    { value: 4, label: '4' },
];

const DEFAULT_SET_COUNT = 4;

const emptySet = () => ({ weight: '', reps: '', rir: 2 });
const emptySets = () => Array.from({ length: DEFAULT_SET_COUNT }, emptySet);
const isComplete = (s) => s.weight && s.reps;

export default function WorkoutLogger({ onWorkoutLogged }) {
    const [exercise, setExercise] = useState('bench_press');
    const [bodyweight, setBodyweight] = useState('180');
    const [sets, setSets] = useState(emptySets);
    const [submitting, setSubmitting] = useState(false);
    const [submitError, setSubmitError] = useState(false);
    const [lastRecovery, setLastRecovery] = useState(null);

    const addSet = () => setSets((prev) => [...prev, emptySet()]);

    const removeSet = (index) =>
        setSets((prev) => (prev.length <= 1 ? prev : prev.filter((_, i) => i !== index)));

    const updateSet = (index, field, value) =>
        setSets((prev) => prev.map((s, i) => (i === index ? { ...s, [field]: value } : s)));

    const handleSubmit = useCallback(async (event) => {
        event.preventDefault();
        const validSets = sets.filter(isComplete);
        if (!validSets.length) return;

        setSubmitting(true);
        setSubmitError(false);
        try {
            const result = await api.createWorkout({
                exercise,
                bodyweight: parseFloat(bodyweight) || 180,
                sets: validSets.map((s) => ({
                    weight: parseFloat(s.weight),
                    reps: parseInt(s.reps, 10),
                    rir: parseInt(s.rir, 10),
                })),
            });

            setLastRecovery(result.recovery);
            setSets(emptySets());
            onWorkoutLogged?.(result);
        } catch (err) {
            console.error('Failed to log workout:', err);
            setSubmitError(true);
        } finally {
            setSubmitting(false);
        }
    }, [exercise, bodyweight, sets, onWorkoutLogged]);

    return (
        <form className="page page--narrow" onSubmit={handleSubmit}>
            <div className="page-head">
                <h1 className="page-title">Log Workout</h1>
                <ExercisePicker value={exercise} onChange={setExercise} labelKey="label" />
            </div>

            <div className="field">
                <label className="label" htmlFor="bodyweight">Bodyweight (lbs)</label>
                <input
                    id="bodyweight"
                    type="number"
                    inputMode="decimal"
                    className="input input--num"
                    value={bodyweight}
                    onChange={(e) => setBodyweight(e.target.value)}
                    placeholder="180"
                />
            </div>

            <section className="panel">
                <header className="panel-head">
                    <h2 className="panel-title">Sets</h2>
                    <button type="button" className="btn btn--quiet btn--sm" onClick={addSet}>
                        Add set
                    </button>
                </header>

                <div className="set-grid" role="table" aria-label="Sets">
                    <div className="set-row set-row--head" role="row">
                        <span role="columnheader" className="label">Set</span>
                        <span role="columnheader" className="label">Weight</span>
                        <span role="columnheader" className="label">Reps</span>
                        <span role="columnheader" className="label">RIR</span>
                        <span role="columnheader"><span className="visually-hidden">Remove</span></span>
                    </div>

                    {sets.map((set, idx) => {
                        const n = idx + 1;
                        return (
                            // Sets have no stable id; index keys are fine since rows are only appended or removed.
                            <div className="set-row" role="row" key={idx}>
                                <span role="cell" className="set-number num">{n}</span>
                                <span role="cell">
                                    <input
                                        type="number"
                                        inputMode="decimal"
                                        className="input input--num"
                                        placeholder="135"
                                        aria-label={`Set ${n} weight`}
                                        value={set.weight}
                                        onChange={(e) => updateSet(idx, 'weight', e.target.value)}
                                    />
                                </span>
                                <span role="cell">
                                    <input
                                        type="number"
                                        inputMode="numeric"
                                        className="input input--num"
                                        placeholder="8"
                                        aria-label={`Set ${n} reps`}
                                        value={set.reps}
                                        onChange={(e) => updateSet(idx, 'reps', e.target.value)}
                                    />
                                </span>
                                <span role="cell">
                                    <select
                                        className="input input--num"
                                        aria-label={`Set ${n} reps in reserve`}
                                        value={set.rir}
                                        onChange={(e) => updateSet(idx, 'rir', parseInt(e.target.value, 10))}
                                    >
                                        {RIR_OPTIONS.map((r) => (
                                            <option key={r.value} value={r.value}>{r.label}</option>
                                        ))}
                                    </select>
                                </span>
                                <span role="cell">
                                    <button
                                        type="button"
                                        className="btn btn--icon"
                                        aria-label={`Remove set ${n}`}
                                        onClick={() => removeSet(idx)}
                                        disabled={sets.length <= 1}
                                    >
                                        ×
                                    </button>
                                </span>
                            </div>
                        );
                    })}
                </div>
            </section>

            {lastRecovery && (
                <div className="notice">
                    <span className="label">Recovery</span>
                    <span>
                        <span className="num notice-value">{lastRecovery.recovery_estimate}</span>
                        <span className="notice-text">estimated before your next session</span>
                    </span>
                </div>
            )}

            {submitError && (
                <p className="status status--error" role="alert">
                    The workout wasn&apos;t saved. Check the API and try again.
                </p>
            )}

            <button
                type="submit"
                className="btn btn--primary btn--block"
                disabled={submitting || !sets.some(isComplete)}
            >
                {submitting ? 'Logging…' : 'Log workout'}
            </button>
        </form>
    );
}
