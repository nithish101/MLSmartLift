/**
 * Recharts tooltip content. `describe(datum)` returns { title, lines }.
 */
export default function ChartTooltip({ active, payload, describe }) {
    if (!active || !payload?.length) return null;
    const { title, lines = [] } = describe(payload[0].payload);
    return (
        <div className="chart-tooltip">
            <div className="chart-tooltip-title">{title}</div>
            {lines.map((line) => (
                <div key={line} className="chart-tooltip-line">{line}</div>
            ))}
        </div>
    );
}
