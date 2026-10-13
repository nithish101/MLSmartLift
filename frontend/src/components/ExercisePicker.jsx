import { EXERCISES } from '../exercises';

export default function ExercisePicker({ value, onChange, labelKey = 'short' }) {
    return (
        <div className="segmented" role="group" aria-label="Exercise">
            {EXERCISES.map((ex) => (
                <button
                    key={ex.id}
                    type="button"
                    className="segmented-option"
                    aria-pressed={value === ex.id}
                    onClick={() => onChange(ex.id)}
                >
                    {ex[labelKey]}
                </button>
            ))}
        </div>
    );
}
