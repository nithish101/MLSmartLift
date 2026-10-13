import { useEffect, useRef, useState } from 'react';

const FEATURES = [
    { name: 'Weight Lifted', pct: 35.7, desc: 'The primary load used in the exercise' },
    { name: 'Bodyweight', pct: 32.7, desc: 'User bodyweight influences relative strength' },
    { name: 'Progression', pct: 11.8, desc: 'Recent trend direction (improving / stalling)' },
    { name: 'Exercise Type', pct: 10.2, desc: 'Bench press, squat, or deadlift' },
    { name: 'Volume Score', pct: 5.4, desc: 'Weekly training stress accumulation' },
    { name: 'RIR', pct: 1.5, desc: 'Reps in reserve — proximity to failure' },
    { name: 'Sets', pct: 1.5, desc: 'Number of working sets per session' },
    { name: 'Frequency', pct: 1.1, desc: 'Training sessions per week' },
];

const MAX_PCT = Math.max(...FEATURES.map((f) => f.pct));

const PERFORMANCE = [
    { label: 'MAE (reps)', value: '1.15' },
    { label: 'R² score', value: '0.38' },
    { label: 'Training samples', value: '5,559' },
    { label: 'Trees (estimators)', value: '200' },
];

export default function ModelExplainer({ isOpen, onClose }) {
    const dialogRef = useRef(null);
    const [expandedFeature, setExpandedFeature] = useState(null);

    useEffect(() => {
        const dialog = dialogRef.current;
        if (isOpen && !dialog.open) dialog.showModal();
        if (!isOpen && dialog.open) dialog.close();
    }, [isOpen]);

    // A click whose target is the <dialog> itself landed on the backdrop.
    const handleClick = (e) => {
        if (e.target === dialogRef.current) dialogRef.current.close();
    };

    return (
        <dialog
            ref={dialogRef}
            className="sheet"
            aria-labelledby="explainer-title"
            onClose={onClose}
            onClick={handleClick}
        >
            <div className="sheet-body">
                <header className="sheet-head">
                    <h2 id="explainer-title" className="sheet-title">How SmartLift works</h2>
                    <button type="button" className="btn btn--quiet btn--sm" onClick={() => dialogRef.current.close()}>
                        Close
                    </button>
                </header>

                <section className="prose">
                    <h3>The Model</h3>
                    <p>
                        SmartLift uses a <strong>Gradient Boosting Regression</strong> model trained on
                        strength progression data. It learns patterns from thousands of training sessions
                        to predict how many reps you can perform at a given weight.
                    </p>
                </section>

                <section className="prose">
                    <h3>Why Gradient Boosting?</h3>
                    <p>
                        Gradient Boosting excels at structured, tabular data — exactly what workout logs are.
                        It builds an ensemble of decision trees sequentially, with each tree correcting errors
                        from the previous ones. This gives us strong predictive accuracy while maintaining
                        interpretable feature importance.
                    </p>
                </section>

                <section className="prose">
                    <h3>Feature Importance</h3>
                    <p>These are the factors the model weighs when making predictions, ranked by importance:</p>
                    <ul className="features">
                        {FEATURES.map((f, i) => {
                            const expanded = expandedFeature === i;
                            return (
                                <li key={f.name}>
                                    <button
                                        type="button"
                                        className="feature"
                                        aria-expanded={expanded}
                                        onClick={() => setExpandedFeature(expanded ? null : i)}
                                    >
                                        <span className="feature-name">{f.name}</span>
                                        <span className="feature-track" aria-hidden="true">
                                            <span className="feature-bar" style={{ width: `${(f.pct / MAX_PCT) * 100}%` }} />
                                        </span>
                                        <span className="feature-pct num">{f.pct}%</span>
                                    </button>
                                    {expanded && <p className="feature-desc">{f.desc}</p>}
                                </li>
                            );
                        })}
                    </ul>
                </section>

                <section className="prose">
                    <h3>Prediction Pipeline</h3>
                    <p>
                        When you log a workout, the model takes your most recent performance data and predicts
                        how many reps you&apos;ll achieve at various weights. The recommendation engine then:
                    </p>
                    <ul>
                        <li>Predicts reps at your current working weight</li>
                        <li>If predicted reps {'>'} 9 → increases weight by 5 lbs</li>
                        <li>If predicted reps {'<'} 5 → decreases weight by 5 lbs</li>
                        <li>Targets the 5–9 rep hypertrophy range</li>
                        <li>Sets target RIR at 2 for productive training stimulus</li>
                    </ul>
                </section>

                <section className="prose">
                    <h3>1RM Estimation</h3>
                    <p>
                        Estimated one-rep max is calculated using the <strong>Epley formula</strong> with
                        an RIR adjustment:
                    </p>
                    <pre className="formula">e1RM = weight × (1 + (reps + RIR) / 30)</pre>
                </section>

                <section className="prose">
                    <h3>Volume Score (0–100)</h3>
                    <p>The volume score measures weekly training stress. It&apos;s calculated from:</p>
                    <ul>
                        <li><strong>Set intensity:</strong> Lower RIR = higher stress per set</li>
                        <li><strong>Training frequency:</strong> More sessions = higher volume</li>
                        <li><strong>Muscle group:</strong> Compound lifts (squat, deadlift) generate more fatigue</li>
                    </ul>
                </section>

                <section className="prose">
                    <h3>Recovery Estimation</h3>
                    <p>
                        After each workout, SmartLift estimates the recovery time needed before your next
                        session. This is based on the muscle group (legs need longer), session intensity
                        (lower RIR = more recovery), and total set volume. All fatigue is designed to be
                        fully recoverable before the next session — no deload weeks needed.
                    </p>
                </section>

                <section className="prose">
                    <h3>Model Performance</h3>
                    <dl className="readout">
                        {PERFORMANCE.map((m) => (
                            <div key={m.label} className="readout-cell">
                                <dt>{m.label}</dt>
                                <dd><span className="num">{m.value}</span></dd>
                            </div>
                        ))}
                    </dl>
                </section>
            </div>
        </dialog>
    );
}
