
"use strict";

const CHART_MAX = 60;

const chartOpts = {
  responsive: true,
  animation:  { duration: 0 },
  plugins:    { legend: { display: false } },
  scales: {
    x: { display: false },
    y: {
      grid:  { color: "rgba(255,255,255,0.06)" },
      ticks: { color: "rgba(255,255,255,0.4)", font: { size: 10 } },
    },
  },
};

let riskChart, countChart;

function initCharts() {
  const riskCtx = document.getElementById("riskChart");
  const cntCtx  = document.getElementById("countChart");
  if (!riskCtx || !cntCtx) return;

  riskChart = new Chart(riskCtx.getContext("2d"), {
    type: "line",
    data: {
      labels: [], datasets: [{
        data: [], borderColor: "#ff3b5c", backgroundColor: "rgba(255,59,92,0.10)",
        borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4,
      }],
    },
    options: { ...chartOpts, scales: { ...chartOpts.scales, y: { ...chartOpts.scales.y, min: 0, max: 1 } } },
  });

  countChart = new Chart(cntCtx.getContext("2d"), {
    type: "line",
    data: {
      labels: [], datasets: [{
        data: [], borderColor: "#7c5cfc", backgroundColor: "rgba(124,92,252,0.10)",
        borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4,
      }],
    },
    options: chartOpts,
  });
}

function pushPoint(chart, label, value) {
  if (!chart) return;
  chart.data.labels.push(label);
  chart.data.datasets[0].data.push(value);
  if (chart.data.labels.length > CHART_MAX) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update("none");
}

let streaming    = false;
let pollStatus_t = null;
let pollHistory_t = null;
let pollLogs_t   = null;

// ── DOM refs ─────────────────────────────────
const feedSection = () => document.getElementById("feedSection");
const kpi = id    => document.getElementById(id);

// ── Webcam toggle ─────────────────────────────
async function toggleWebcam() {
  if (streaming) { await stopDetection(); return; }
  await startDetection(parseInt(document.getElementById("camIndex").value));
}

async function startDetection(source) {
  const res  = await fetch("/api/start", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ source }),
  });
  const data = await res.json();
  if (data.status === "started" || data.status === "already_running") {
    streaming = true;
    showFeed();
    startPolling();
    document.getElementById("webcamBtn").textContent = "Stop Webcam";
    document.getElementById("webcamBtn").className   = "btn btn-danger";
    document.getElementById("webcamCard").classList.add("active-card");
  }
}

async function stopDetection() {
  await fetch("/api/stop", { method: "POST" });
  streaming = false;
  stopPolling();
  hideFeed();
  document.getElementById("webcamBtn").textContent = "Start Webcam";
  document.getElementById("webcamBtn").className   = "btn btn-primary";
  document.getElementById("webcamCard").classList.remove("active-card");
  document.getElementById("alertBanner").classList.add("hidden");
}

function showFeed() {
  feedSection().style.display = "block";
  initCharts();
  // Force stream refresh
  const ts = Date.now();
  document.getElementById("detectionFeed").src = `/stream/detection?t=${ts}`;
  document.getElementById("zonesFeed").src      = `/stream/zones?t=${ts}`;
}

function hideFeed() {
  feedSection().style.display = "none";
  if (riskChart)  { riskChart.destroy();  riskChart  = null; }
  if (countChart) { countChart.destroy(); countChart = null; }
}

// ── Polling (reduced frequency for performance) ──
function startPolling() {
  stopPolling();
  doStatusPoll();
  doHistoryPoll();
  doLogsPoll();
  pollStatus_t  = setInterval(doStatusPoll,  1500);  // was 900ms
  pollHistory_t = setInterval(doHistoryPoll, 4000);   // was 2000ms
  pollLogs_t    = setInterval(doLogsPoll,    6000);   // was 3500ms
}

function stopPolling() {
  clearInterval(pollStatus_t);
  clearInterval(pollHistory_t);
  clearInterval(pollLogs_t);
}

async function doStatusPoll() {
  try {
    const s = await (await fetch("/api/status")).json();
    updateKPIs(s);
    // Auto-stop UI if backend stopped
    if (!s.running && streaming) {
      streaming = false;
      stopPolling();
    }
  } catch {}
}

async function doHistoryPoll() {
  try {
    const hist = await (await fetch("/api/history")).json();
    if (!riskChart || !countChart) return;
    riskChart.data.labels  = [];
    riskChart.data.datasets[0].data  = [];
    countChart.data.labels = [];
    countChart.data.datasets[0].data = [];
    hist.forEach(h => {
      const lbl = new Date(h.t * 1000).toLocaleTimeString("en", { hour12: false });
      pushPoint(riskChart,  lbl, h.risk);
      pushPoint(countChart, lbl, h.count);
    });
  } catch {}
}

async function doLogsPoll() {
  try {
    const rows = await (await fetch("/api/logs")).json();
    renderLogs(rows);
  } catch {}
}

// ── KPI render ────────────────────────────────
function updateKPIs(s) {
  kpi("kpiCount").textContent   = s.people_count;
  kpi("kpiFps").textContent     = parseFloat(s.fps).toFixed(1);
  kpi("kpiScore").textContent   = parseFloat(s.risk_score).toFixed(3);
  kpi("kpiDensity").textContent = parseFloat(s.density).toFixed(3);
  kpi("kpiSpeed").textContent   = parseFloat(s.avg_speed).toFixed(3);

  const rv = kpi("kpiRisk");
  rv.textContent = s.risk_level;
  rv.className   = "kpi-value risk-" + s.risk_level;

  // Also colour score by risk
  kpi("kpiScore").className = "kpi-value risk-" + s.risk_level;

  // Alert banner
  const banner = document.getElementById("alertBanner");
  if (s.alert_active) banner.classList.remove("hidden");
  else banner.classList.add("hidden");

  // Incident counter
  const icEl = document.getElementById("incidentCount");
  if (icEl) icEl.textContent = `${s.incident_count} incidents`;
}

// ── Log table ─────────────────────────────────
function renderLogs(rows) {
  const tbody = document.getElementById("logBody");
  if (!tbody) return;
  tbody.innerHTML = "";
  rows.forEach(r => {
    const tr  = document.createElement("tr");
    const lvl = (r.risk_level || "LOW");
    tr.innerHTML = `
      <td>${r.timestamp}</td>
      <td>${r.count}</td>
      <td><span class="badge-${lvl}">${lvl}</span></td>
      <td>${Number(r.risk_score).toFixed(3)}</td>
      <td>${Number(r.density).toFixed(3)}</td>
      <td>${Number(r.avg_speed).toFixed(2)}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function clearLogs() {
  if (!confirm("Clear all incident log entries?")) return;
  await fetch("/api/clear_logs", { method: "POST" });
  const tbody = document.getElementById("logBody");
  if (tbody) tbody.innerHTML = "";
}

// ── Init ──────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Auto-restore if server is already running
  fetch("/api/status").then(r => r.json()).then(s => {
    if (s.running) {
      streaming = true;
      showFeed();
      startPolling();
      document.getElementById("webcamBtn").textContent = "Stop Webcam";
      document.getElementById("webcamBtn").className   = "btn btn-danger";
      document.getElementById("webcamCard").classList.add("active-card");
    }
  }).catch(() => {});
});
