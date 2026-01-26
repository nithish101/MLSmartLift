import { useState, useCallback } from 'react';
import { api } from '../api';

const EXERCISES = [
    { id: 'bench_press', label: 'Bench Press', emoji: '🏋️' },
    { id: 'squat', label: 'Squat', emoji: '🦵' },
    { id: 'deadlift', label: 'Deadlift', emoji: '💪' },
];

const RIR_OPTIONS = [
    { value: -1, label: 'Failure' },
    { value: 0, label: '0' },
    { value: 1, label: '1' },
    { value: 2, label: '2' },
    { value: 3, label: '3' },
    { value: 4, label: '4' },
];

const emptySet = () => ({ weight: '', reps: '', rir: 2 });

export default function WorkoutLogger({ onWorkoutLogged }) {
    const [exercise, setExercise] = useState('bench_press');
    const [bodyweight, setBodyweight] = useState('180');
    const [sets, setSets] = useState([emptySet(), emptySet(), emptySet(), emptySet()]);
    const [submitting, setSubmitting] = useState(false);
    const [lastRecovery, setLastRecovery] = useState(null);

    const addSet = () => setSets((prev) => [...prev, emptySet()]);

    const removeSet = (index) => {
        if (sets.length <= 1) return;
        setSets((prev) => prev.filter((_, i) => i !== index));
    };

    const updateSet = (index, field, value) => {
        setSets((prev) =>
            prev.map((s, i) => (i === index ? { ...s, [field]: value } : s))
        );
    };

    const handleSubmit = useCallback(async () => {
        const validSets = sets.filter((s) => s.weight && s.reps);
        if (!validSets.length) return;

        setSubmitting(true);
        try {
            const result = await api.createWorkout({
                exercise,
                bodyweight: parseFloat(bodyweight) || 180,
                sets: validSets.map((s) => ({
                    weight: parseFloat(s.weight),
                    reps: parseInt(s.reps),
                    rir: parseInt(s.rir),
                })),
            });

            setLastRecovery(result.recovery);
            setSets([emptySet(), emptySet(), emptySet(), emptySet()]);
            if (onWorkoutLogged) onWorkoutLogged(result);
        } catch (err) {
            console.error('Failed to log workout:', err);
        } finally {
            setSubmitting(false);
        }
    }, [exercise, bodyweight, sets, onWorkoutLogged]);

    return (
        <div className="page">
            <h2 className="page-title">Log Workout</h2>

            {/* Exercise Selection */}
            <div className="exercise-pills">
                {EXERCISES.map((ex) => (
                    <button
                        key={ex.id}
                        className={`exercise-pill ${exercise === ex.id ? 'active' : ''}`}
                        onClick={() => setExercise(ex.id)}
                    >
                        <span>{ex.emoji}</span>
                        <div>{ex.label}</div>
                    </button>
                ))}
            </div>

            {/* Bodyweight */}
            <div className="form-group">
                <label className="form-label">Bodyweight (lbs)</label>
                <input
                    type="number"
                    className="form-input"
                    value={bodyweight}
                    onChange={(e) => setBodyweight(e.target.value)}
                    placeholder="180"
                />
            </div>

            {/* Sets */}
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Sets</span>
                    <button className="btn btn--sm btn--secondary" onClick={addSet}>
                        + Add Set
                    </button>
                </div>

                {/* Headers */}
                <div className="set-row" style={{ paddingBottom: 0 }}>
                    <span></span>
                    <span className="form-label" style={{ textAlign: 'center', marginBottom: 0 }}>Weight</span>
                    <span className="form-label" style={{ textAlign: 'center', marginBottom: 0 }}>Reps</span>
                    <span className="form-label" style={{ textAlign: 'center', marginBottom: 0 }}>RIR</span>
                    <span></span>
                </div>

                {sets.map((set, idx) => (
                    <div className="set-row" key={idx}>
                        <div className="set-number">{idx + 1}</div>
                        <input
                            type="number"
                            className="set-input"
                            placeholder="135"
                            value={set.weight}
                            onChange={(e) => updateSet(idx, 'weight', e.target.value)}
                        />
                        <input
                            type="number"
                            className="set-input"
                            placeholder="8"
                            value={set.reps}
                            onChange={(e) => updateSet(idx, 'reps', e.target.value)}
                        />
                        <select
                            className="set-input"
                            value={set.rir}
                            onChange={(e) => updateSet(idx, 'rir', parseInt(e.target.value))}
                        >
                            {RIR_OPTIONS.map((r) => (
                                <option key={r.value} value={r.value}>
                                    {r.label}
                                </option>
                            ))}
                        </select>
                        <button
                            className="btn btn--icon btn--danger"
                            onClick={() => removeSet(idx)}
                            disabled={sets.length <= 1}
                        >
                            ×
                        </button>
                    </div>
                ))}
            </div>

            {/* Recovery from last workout */}
            {lastRecovery && (
                <div className="recovery-card">
                    <span className="recovery-icon">⏱️</span>
                    <div>
                        <div className="recovery-time">{lastRecovery.recovery_estimate}</div>
                        <div className="recovery-label">Estimated recovery before next session</div>
                    </div>
                </div>
            )}

            {/* Submit */}
            <button
                className="btn btn--primary"
                onClick={handleSubmit}
                disabled={submitting || !sets.some((s) => s.weight && s.reps)}
            >
                {submitting ? 'Logging...' : '🚀 Log Workout'}
            </button>
        </div>
    );
}
