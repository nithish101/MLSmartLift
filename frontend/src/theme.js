// Chart colors mirror the tokens in index.css; Recharts needs literal values.
export const chartTheme = {
    ink: '#ecebe6',
    muted: '#77756f',
    bar: '#4a4945',
    grid: '#262624',
    signal: '#e96a26',
    surface: '#1c1c1a',
    font: "'IBM Plex Mono', ui-monospace, Menlo, monospace",
};

export const axisProps = {
    tick: { fill: chartTheme.muted, fontSize: 11, fontFamily: chartTheme.font },
    axisLine: false,
    tickLine: false,
};

// Right margin leaves room for the last x-axis label; the negative left
// margin pulls the y-axis labels flush with the panel padding.
export const CHART_MARGIN = { top: 8, right: 20, bottom: 0, left: -16 };

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// "2026-10-09" -> "Oct 9" without going through Date (avoids timezone shifts).
export function formatShortDate(iso) {
    const [, month, day] = iso.split('-');
    return `${MONTHS[Number(month) - 1]} ${Number(day)}`;
}
