const $ = (id) => document.getElementById(id);
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(url, opts) {
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || "Request failed.");
  return data;
}
function showError(id, msg) { const el = $(id); el.textContent = msg || ""; el.hidden = !msg; }

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => {
      const on = t === tab;
      t.classList.toggle("on", on);
      t.setAttribute("aria-selected", on);
    });
    document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== "view-" + tab.dataset.view));
    if (tab.dataset.view === "perf") loadMetrics();
  });
});

/* ---------- single record ---------- */
$("btn-sample").addEventListener("click", async () => {
  showError("single-error");
  try {
    const { record, actual } = await api("/api/sample");
    $("record").value = JSON.stringify(record, null, 2);
    $("actual").textContent = "Recorded label in the dataset: " + (actual || "unknown");
    $("actual").hidden = false;
  } catch (e) { showError("single-error", e.message); }
});

$("btn-analyze").addEventListener("click", async () => {
  showError("single-error");
  let record;
  try { record = JSON.parse($("record").value); }
  catch { return showError("single-error", "The record is not valid JSON."); }

  const btn = $("btn-analyze");
  btn.disabled = true;
  try {
    const r = await api("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(record),
    });
    renderResult(r);
  } catch (e) { showError("single-error", e.message); }
  btn.disabled = false;
});

function renderResult(r) {
  $("empty").hidden = true;
  $("result").hidden = false;

  const v = $("verdict");
  if (!r.suspicious) {
    v.className = "verdict normal";
    v.innerHTML = "<strong>Normal traffic</strong><span>Screened out at stage 1. No attack classification needed.</span>";
  } else if (r.escalated) {
    v.className = "verdict review";
    v.innerHTML = `<strong>Flag for review</strong><span>Suspicious. Closest category is ${esc(r.top[0][0])}, but the model also considers ${esc(r.top[1][0])} likely.</span>`;
  } else {
    v.className = "verdict attack";
    v.innerHTML = `<strong>${esc(r.label)}</strong><span>Suspicious traffic, classified with confidence.</span>`;
  }

  $("m-stage1-fill").style.width = pct(r.p_malicious, 1);
  $("m-stage1-tick").style.left = pct(r.thresholds.binary, 1);
  $("stage1-note").textContent =
    `Malicious probability ${pct(r.p_malicious)}. Records at or above ${pct(r.thresholds.binary)} (marker) go to stage 2.`;

  $("stage2").hidden = !r.suspicious;
  if (!r.suspicious) return;

  const bars = Object.entries(r.probs).map(([name, p], i) => `
    <div class="bar-row ${i === 0 ? "top1" : i === 1 ? "top2" : ""}">
      <span>${esc(name)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${pct(p, 1)}"></div></div>
      <span class="v">${pct(p)}</span>
    </div>`).join("");
  $("class-bars").innerHTML = bars;

  $("m-margin-fill").style.width = pct(Math.min(r.margin, 1), 1);
  $("m-margin-tick").style.left = pct(Math.min(r.thresholds.margin, 1), 1);
  $("margin-note").textContent = r.escalated
    ? `Margin ${r.margin.toFixed(3)} is below the review threshold of ${r.thresholds.margin.toFixed(3)}.`
    : `Margin ${r.margin.toFixed(3)} meets the review threshold of ${r.thresholds.margin.toFixed(3)}.`;
}

/* ---------- batch ---------- */
$("btn-batch").addEventListener("click", async () => {
  showError("batch-error");
  const file = $("file").files[0];
  if (!file) return showError("batch-error", "Choose a CSV file first.");

  const btn = $("btn-batch");
  btn.disabled = true;
  const form = new FormData();
  form.append("file", file);
  try {
    const d = await api("/api/batch", { method: "POST", body: form });
    renderBatch(d);
  } catch (e) { showError("batch-error", e.message); }
  btn.disabled = false;
});

function renderBatch(d) {
  $("batch-out").hidden = false;
  $("batch-stats").innerHTML = [
    ["Records analyzed", d.total.toLocaleString()],
    ["Marked suspicious", d.suspicious.toLocaleString()],
    ["Flagged for review", d.escalated.toLocaleString()],
    ["Review rate", d.suspicious ? pct(d.escalated / d.suspicious) : "0%"],
  ].map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");

  const entries = Object.entries(d.counts).sort((a, b) => b[1] - a[1]);
  const max = entries.length ? entries[0][1] : 1;
  $("batch-bars").innerHTML = entries.map(([name, n], i) => `
    <div class="bar-row ${i === 0 ? "top1" : ""}">
      <span>${esc(name)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${pct(n / max, 1)}"></div></div>
      <span class="v">${n.toLocaleString()}</span>
    </div>`).join("");

  $("batch-table-title").textContent = d.truncated ? `Records (first ${d.rows.length} shown)` : "Records";
  $("batch-table").querySelector("tbody").innerHTML = d.rows.map((r) => `
    <tr>
      <td>${r.row}</td>
      <td>${esc(r.label)}</td>
      <td>${pct(r.p_malicious)}</td>
      <td>${r.margin === null ? "-" : r.margin.toFixed(3)}</td>
      <td>${r.escalated ? '<span class="tag review">Review</span>' : '<span class="tag ok">Confident</span>'}</td>
      <td>${r.actual ? esc(r.actual) : "-"}</td>
    </tr>`).join("");
}

/* ---------- performance ---------- */
let metricsLoaded = false;
async function loadMetrics() {
  if (metricsLoaded) return;
  try {
    const m = await api("/api/metrics");
    metricsLoaded = true;
    $("perf-error").hidden = true;
    $("perf-out").hidden = false;
    const stat = (k, v) => `<div><dt>${k}</dt><dd>${v}</dd></div>`;
    $("perf-stats").innerHTML = [
      stat("Accuracy", pct(m.accuracy)),
      stat("Macro-F1", pct(m.macro_f1)),
      stat("Attack recall", pct(m.binary_recall)),
      stat("Attack precision", pct(m.binary_precision)),
      stat("False-positive rate", pct(m.false_positive_rate, 2)),
      stat("Review rate", pct(m.escalation_rate)),
      stat("Accuracy when confident", m.confident_accuracy == null ? "-" : pct(m.confident_accuracy)),
      stat("Inference per 1,000", m.inference_ms_per_1000.toFixed(0) + " ms"),
    ].join("");

    $("perf-table").querySelector("tbody").innerHTML = Object.entries(m.per_class)
      .sort((a, b) => b[1].support - a[1].support)
      .map(([name, c]) => `<tr><td>${esc(name)}</td><td>${pct(c.precision)}</td><td>${pct(c.recall)}</td><td>${pct(c.f1)}</td><td>${c.support.toLocaleString()}</td></tr>`)
      .join("");

    $("perf-baseline").textContent =
      `Baseline (${m.baseline.name}): accuracy ${pct(m.baseline.accuracy)}, macro-F1 ${pct(m.baseline.macro_f1)}.`;
  } catch (e) {
    $("perf-out").hidden = true;
    showError("perf-error", e.message);
  }
}
