import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://127.0.0.1:8000",
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

// These functions intentionally call only endpoints present in the current backend.
export async function checkApiHealth() {
  const response = await api.get("/api/health");
  return response.data;
}

export async function getAnalyticsOverview() {
  const response = await api.get("/api/analytics/overview");
  return response.data;
}

export async function predictFraud(transaction) {
  const response = await api.post("/api/predict", transaction);
  return response.data;
}

export default api;
