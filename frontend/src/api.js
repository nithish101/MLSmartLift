const API_BASE = `http://${window.location.hostname}:8000`;

async function fetchJSON(url, options = {}) {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  // Workouts
  getWorkouts: (exercise) =>
    fetchJSON(exercise ? `/workouts?exercise=${exercise}` : '/workouts'),

  createWorkout: (data) =>
    fetchJSON('/workouts', { method: 'POST', body: JSON.stringify(data) }),

  // AI Recommendation
  getRecommendation: (exercise, bodyweight = 180) =>
    fetchJSON(`/recommendation/${exercise}?bodyweight=${bodyweight}`),

  // Progress
  getProgress: (exercise) =>
    fetchJSON(`/progress/${exercise}`),

  // Volume
  getVolumeScore: (exercise) =>
    fetchJSON(`/volume-score?exercise=${exercise}`),

  // Predicted vs Actual
  getPredictedVsActual: (exercise) =>
    fetchJSON(`/predicted-vs-actual/${exercise}`),

  // Alerts
  getAlerts: (exercise) =>
    fetchJSON(`/alerts/${exercise}`),
};
