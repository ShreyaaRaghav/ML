/**
 * Cyber Attack Detection — front-end
 *
 * Three panels: Single record · Batch file · Performance
 * Communicates with the Flask REST API on the same origin.
 */

"use strict";

// ── Helpers ───────────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const pct = (x, d = 1) => {
  if (x == null || isNaN(x)) return "—";
  return (x * 100).toFixed(d) + "%";
};

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

const fmt = (n, d = 3) => (n == null ? "—" : Number(n).toFixed(d));

/** Generic fetch wrapper — throws on HTTP errors. */
async function api(url, opts = {}) {
  const res  = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status}).`);
  return data;
}

function showError(id, msg) {
  const el = $(id);
  if (!el) return;
  el.textContent = msg || "";
  el.hidden      = !msg;
}

function showHint(id, msg) {
  const el = $(id);
  if (!el) return;
  el.textContent = msg || "";
  el.hidden      = !msg;
}

// ── Tab navigation ────────────────────────────────────────────────────────────

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => {
      const on = t === tab;
      t.classList.toggle("on", on);
      t.setAttribute("aria-selected", String(on));
    });
    document.querySelectorAll(".view").forEach(
      (v) => (v.hidden = v.id !== "view-" + tab.dataset.view)
    );
    if (tab.dataset.view === "perf") loadMetrics();
  });
});

// ── Status dot (footer) ───────────────────────────────────────────────────────

(async () => {
  const dot = $("status-dot");
  try {
    const s = await api("/api/status");
    if (s.model_loaded) {
      dot.title     = `Model loaded · ${s.features} features · Classes: ${(s.classes || []).join(", ")}`;
      dot.className = "status-dot ok";
    } else {
      dot.title     = "Models not loaded — run train.py first.";
      dot.className = "status-dot err";
    }
  } catch {
    dot.title     = "Could not reach server.";
    dot.className = "status-dot err";
  }
})();

// ── Single record ─────────────────────────────────────────────────────────────

$("btn-sample").addEventListener("click", async () => {
  showError("single-error");
  try {
    const { record, actual } = await api("/api/sample");
    $("record").value = JSON.stringify(record, null, 2);
    const actualEl    = $("actual");
    actualEl.textContent = "Dataset label: " + (actual || "unknown");
    actualEl.hidden      = false;
  } catch (e) {
    showError("single-error", e.message);
  }
});

$("btn-analyze").addEventListener("click", async () => {
  showError("single-error");
  let record;
  try {
    record = JSON.parse($("record").value);
  } catch {
    showError("single-error", "The record is not valid JSON. Check for missing commas or quotes.");
    return;
  }

  const btn    = $("btn-analyze");
  btn.disabled = true;
  try {
    const r = await api("/api/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(record),
    });
    renderResult(r);
  } catch (e) {
    showError("single-error", e.message);
  }
  btn.disabled = false;
});

// ── Render single result ──────────────────────────────────────────────────────

function renderResult(r) {
  $("empty").hidden  = true;
  $("result").hidden = false;

  // Verdict card
  const v = $("verdict");
  if (!r.suspicious) {
    v.className = "verdict normal";
    v.innerHTML =
      `<strong>Normal traffic</strong>` +
      `<span>Screened out at Stage 1. Malicious probability ${pct(r.p_malicious)}.</span>`;
  } else if (r.escalated) {
    const [n1, p1] = r.top[0];
    const [n2, p2] = r.top[1];
    v.className = "verdict review";
    v.innerHTML =
      `<strong>⚑ Flagged for review</strong>` +
      `<span>Suspicious. Top category: <em>${esc(n1)}</em> (${pct(p1)}), ` +
      `but <em>${esc(n2)}</em> (${pct(p2)}) is also considered likely. ` +
      `Confidence margin too low for an automatic decision.</span>`;
  } else {
    const [n1, p1] = r.top[0];
    v.className = "verdict attack";
    v.innerHTML =
      `<strong>${esc(r.label)}</strong>` +
      `<span>Suspicious traffic, classified with confidence (${pct(p1)}).</span>`;
  }

  // Stage 1 meter
  const binPct = pct(r.p_malicious, 1);
  const tBin   = r.thresholds?.binary ?? 0.5;
  $("m-stage1-fill").style.width = binPct;
  $("m-stage1-tick").style.left  = pct(tBin, 1);
  $("stage1-note").textContent   =
    `Malicious probability: ${pct(r.p_malicious)}. ` +
    `Records at or above ${pct(tBin)} (marker ▲) proceed to Stage 2.`;

  // Stage 2 & 3
  $("stage2").hidden = !r.suspicious;
  if (!r.suspicious) return;

  // Class probability bars
  const sorted = Object.entries(r.probs).sort((a, b) => b[1] - a[1]);
  $("class-bars").innerHTML = sorted
    .map(([name, p], i) => `
      <div class="bar-row ${i === 0 ? "top1" : i === 1 ? "top2" : ""}" role="listitem">
        <span>${esc(name)}</span>
        <div class="bar-track">
          <div class="bar-fill" style="width:${pct(p, 2)}"></div>
        </div>
        <span class="v">${pct(p)}</span>
      </div>`)
    .join("");

  // Stage 3 margin meter
  const margin = r.margin ?? 0;
  const tMar   = r.thresholds?.margin ?? 0.2;
  $("m-margin-fill").style.width = pct(Math.min(margin, 1), 1);
  $("m-margin-tick").style.left  = pct(Math.min(tMar,   1), 1);
  $("margin-note").textContent   = r.escalated
    ? `Margin ${fmt(margin)} is below the review threshold ${fmt(tMar)}. → Escalated.`
    : `Margin ${fmt(margin)} meets the confidence threshold ${fmt(tMar)}. → Confident prediction.`;

  // Warn about missing features
  if (r.missing_features?.length) {
    const existing = $("single-missing") || (() => {
      const p   = document.createElement("p");
      p.id      = "single-missing";
      p.className = "hint";
      $("result").appendChild(p);
      return p;
    })();
    existing.textContent =
      `Note: ${r.missing_features.length} features were absent and treated as missing values.`;
  }
}

// ── Batch file ────────────────────────────────────────────────────────────────

// Show chosen filename
$("file").addEventListener("change", (e) => {
  const label = e.target.closest(".file-label").querySelector("span");
  label.textContent = e.target.files[0]?.name ?? "Choose CSV file";
});

$("btn-batch").addEventListener("click", async () => {
  showError("batch-error");
  const file = $("file").files[0];
  if (!file) return showError("batch-error", "Please choose a CSV file first.");

  const btn    = $("btn-batch");
  btn.disabled = true;
  showHint("batch-progress", `Uploading ${file.name} (${(file.size / 1024).toFixed(0)} KB)…`);
  $("batch-out").hidden = true;

  const form = new FormData();
  form.append("file", file);
  try {
    const d = await api("/api/batch", { method: "POST", body: form });
    showHint("batch-progress");
    renderBatch(d);
  } catch (e) {
    showHint("batch-progress");
    showError("batch-error", e.message);
  }
  btn.disabled = false;
});

// ── Render batch result ───────────────────────────────────────────────────────

function renderBatch(d) {
  $("batch-out").hidden = false;

  $("batch-stats").innerHTML = [
    ["Records analyzed",   d.total.toLocaleString()],
    ["Suspicious",         d.suspicious.toLocaleString()],
    ["Flagged for review", d.escalated.toLocaleString()],
    ["Normal",             (d.total - d.suspicious).toLocaleString()],
    ["Review rate",        d.suspicious ? pct(d.escalated / d.suspicious) : "0%"],
    ["Inference time",     d.inference_ms != null ? d.inference_ms.toFixed(0) + " ms" : "—"],
  ]
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`)
    .join("");

  // Label distribution bars
  const entries = Object.entries(d.counts).sort((a, b) => b[1] - a[1]);
  const maxN    = entries.length ? entries[0][1] : 1;
  $("batch-bars").innerHTML = entries
    .map(([name, n], i) => `
      <div class="bar-row ${i === 0 ? "top1" : ""}" role="listitem">
        <span>${esc(name)}</span>
        <div class="bar-track">
          <div class="bar-fill" style="width:${pct(n / maxN, 1)}"></div>
        </div>
        <span class="v">${n.toLocaleString()}</span>
      </div>`)
    .join("");

  // Table
  $("batch-table-title").textContent = d.truncated
    ? `Records (first ${d.rows.length} of ${d.total.toLocaleString()} shown)`
    : `Records (${d.rows.length.toLocaleString()})`;

  $("batch-table").querySelector("tbody").innerHTML = d.rows
    .map(
      (r) => `
      <tr>
        <td>${r.row}</td>
        <td>${esc(r.label)}</td>
        <td class="num">${pct(r.p_malicious)}</td>
        <td class="num">${r.margin == null ? "—" : fmt(r.margin)}</td>
        <td>${
          r.escalated
            ? '<span class="tag review">Review</span>'
            : r.label === "Normal"
            ? '<span class="tag normal">Normal</span>'
            : '<span class="tag ok">Confident</span>'
        }</td>
        <td>${r.actual ? esc(r.actual) : "—"}</td>
      </tr>`
    )
    .join("");
}

// ── Performance metrics ───────────────────────────────────────────────────────

let metricsLoaded = false;

async function loadMetrics() {
  if (metricsLoaded) return;
  showHint("perf-loading", "Loading metrics…");
  showError("perf-error");
  $("perf-out").hidden = true;

  try {
    const m = await api("/api/metrics");
    metricsLoaded    = true;
    $("perf-loading").hidden = true;
    $("perf-out").hidden     = false;

    const stat = (k, v) => `<div><dt>${k}</dt><dd>${v}</dd></div>`;
    $("perf-stats").innerHTML = [
      stat("Accuracy",             pct(m.accuracy)),
      stat("Macro-F1",             pct(m.macro_f1)),
      stat("Attack recall",        pct(m.binary_recall)),
      stat("Attack precision",     pct(m.binary_precision)),
      stat("False-positive rate",  pct(m.false_positive_rate, 2)),
      stat("Escalation rate",      pct(m.escalation_rate)),
      stat("Confident accuracy",   m.confident_accuracy != null ? pct(m.confident_accuracy) : "—"),
      stat("Inference / 1 000",    m.inference_ms_per_1000 != null
                                     ? m.inference_ms_per_1000.toFixed(0) + " ms" : "—"),
    ].join("");

    $("perf-table").querySelector("tbody").innerHTML = Object.entries(m.per_class)
      .sort((a, b) => b[1].support - a[1].support)
      .map(
        ([name, c]) =>
          `<tr>
            <td>${esc(name)}</td>
            <td class="num">${pct(c.precision)}</td>
            <td class="num">${pct(c.recall)}</td>
            <td class="num">${pct(c.f1)}</td>
            <td class="num">${c.support.toLocaleString()}</td>
          </tr>`
      )
      .join("");

    $("perf-thresholds").innerHTML = [
      stat("Binary threshold", fmt(m.thresholds?.binary)),
      stat("Margin threshold", fmt(m.thresholds?.margin)),
      stat("Test records",     m.test_records?.toLocaleString() ?? "—"),
    ].join("");

    $("perf-baseline").textContent =
      `Baseline (${m.baseline.name}): accuracy ${pct(m.baseline.accuracy)}, ` +
      `macro-F1 ${pct(m.baseline.macro_f1)}.`;
  } catch (e) {
    $("perf-loading").hidden = true;
    $("perf-out").hidden     = true;
    showError("perf-error", e.message);
  }
}
