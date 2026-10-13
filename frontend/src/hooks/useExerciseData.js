import { useEffect, useState } from 'react';

/**
 * Loads data for the selected exercise. `load` must be a stable function
 * (define it at module scope) that returns a promise.
 *
 * Loading is derived from whether the stored result belongs to the current
 * exercise, so the effect never sets state synchronously.
 */
export function useExerciseData(exercise, load) {
    const [result, setResult] = useState({ exercise: null, data: null, error: null });

    useEffect(() => {
        let cancelled = false;
        load(exercise).then(
            (data) => { if (!cancelled) setResult({ exercise, data, error: null }); },
            (error) => { if (!cancelled) setResult({ exercise, data: null, error }); },
        );
        return () => { cancelled = true; };
    }, [exercise, load]);

    return { ...result, loading: result.exercise !== exercise };
}
