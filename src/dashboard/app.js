// Real-Time Oscilloscope and Dashboard Controller
class Oscilloscope {
  constructor(canvasId, lineColor = "#00d8d6", bufferSize = 400, yMin = -2.0, yMax = 2.0) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");
    this.lineColor = lineColor;
    this.bufferSize = bufferSize;
    this.yMin = yMin;
    this.yMax = yMax;
    this.buffer = new Array(bufferSize).fill(0);
    this.drawGrid();
  }

  pushSample(val) {
    this.buffer.shift();
    this.buffer.push(val);
  }

  drawGrid() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.fillStyle = "#090d15";
    this.ctx.fillRect(0, 0, w, h);

    // Draw subtle medical oscilloscope grid
    this.ctx.strokeStyle = "#141c2c";
    this.ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 30) {
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, h);
      this.ctx.stroke();
    }
    for (let y = 0; y < h; y += 25) {
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(w, y);
      this.ctx.stroke();
    }
  }

  render(multiChannels = null) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.drawGrid();

    if (!multiChannels) {
      this.drawSingleTrace(this.buffer, this.lineColor);
    } else {
      multiChannels.forEach(ch => {
        this.drawSingleTrace(ch.buffer, ch.color);
      });
    }
  }

  drawSingleTrace(buf, color) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();

    const step = w / (buf.length - 1);
    for (let i = 0; i < buf.length; i++) {
      const val = buf[i];
      // Normalize to canvas height
      let norm = (val - this.yMin) / (this.yMax - this.yMin);
      norm = Math.max(0, Math.min(1, norm));
      const y = h - norm * h;
      const x = i * step;

      if (i === 0) this.ctx.moveTo(x, y);
      else this.ctx.lineTo(x, y);
    }
    this.ctx.stroke();
  }
}

// Initialize Oscilloscopes
const ecgScope = new Oscilloscope("ecgCanvas", "#00d8d6", 350, -1.8, 1.8);
const ppgScope = new Oscilloscope("ppgCanvas", "#ff3f34", 350, -1.5, 1.5);

// IMU 3-channel buffers
const imuXBuf = new Array(350).fill(0);
const imuYBuf = new Array(350).fill(0);
const imuZBuf = new Array(350).fill(0);
const imuScope = new Oscilloscope("imuCanvas", "#ffa801", 350, -3.0, 3.0);

// State mapping descriptions
const STATE_DESCRIPTIONS = {
  "RESTING": "Subject is sedentary with normal resting cardiovascular, autonomic, and thermal indices.",
  "ACTIVE_EXERTION": "Physiological tachycardia and elevated motion SMA indicative of intentional physical exertion.",
  "PHYSIOLOGICAL_STRAIN": "Elevated heart rate or suppressed HRV disproportionate to low physical movement (autonomic strain).",
  "MOTION_ARTIFACT": "Transient sensor displacement or lead-off condition. Medical alerts dynamically suppressed."
};

const TASK_ROLES = {
  "Task_SensorIngest": "Hard Real-Time Sensor Ingestion",
  "Task_AlertManager": "Safety Alert & Limits Processor",
  "Task_Preprocessor": "Causal IIR Filter & Window Buffering",
  "Task_FeatureML": "Physiological Feature & State AI",
  "Task_CommBLE": "Local Bluetooth Low Energy GATT",
  "Task_CommWiFi": "Remote Telematics & Cloud Push"
};

// WebSocket connection
let ws = null;

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    document.getElementById("badge-ws").innerText = "LIVE STREAM: CONNECTED";
    document.getElementById("badge-ws").style.color = "var(--accent-green)";
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateDashboard(data);
    } catch (e) {
      console.error("Telemetry parsing error:", e);
    }
  };

  ws.onclose = () => {
    document.getElementById("badge-ws").innerText = "RECONNECTING...";
    document.getElementById("badge-ws").style.color = "var(--accent-amber)";
    setTimeout(connectWebSocket, 1500);
  };
}

function updateDashboard(data) {
  const raw = data.raw || {};
  const tele = data.telemetry || {};
  const feats = tele.features || {};
  const inf = tele.inference || {};
  const rtos = data.rtos || {};
  const alerts = tele.active_alerts || [];

  // Hardware Auto-Detection Status Update
  const hw = data.hardware || {};
  const isHw = hw.is_hardware_connected;
  const hwDot = document.getElementById("hw-dot");
  const hwModeText = document.getElementById("hw-mode-text");
  const hwPort = document.getElementById("hw-port");

  if (isHw) {
    hwDot.className = "hw-dot dot-live";
    hwModeText.innerText = "DATA SOURCE: PHYSICAL HARDWARE (LIVE)";
    hwModeText.style.color = "var(--accent-green)";
    hwPort.innerText = `Port: ${hw.connected_port || 'Wi-Fi HTTP'} | Live Stream`;
  } else {
    hwDot.className = "hw-dot dot-emul";
    hwModeText.innerText = "DATA SOURCE: PhysioNet Dataset Emulation";
    hwModeText.style.color = "#fff";
    hwPort.innerText = "USB/COM & Wi-Fi Ingestion: Listening...";
  }

  const detected = hw.detected_sensors || {};
  function updateHwBadge(id, name, isDetected) {
    const el = document.getElementById(id);
    if (!el) return;
    if (isDetected && isHw) {
      el.className = "sensor-badge active-hw";
      el.innerText = `${name}: LIVE HARDWARE`;
    } else {
      el.className = "sensor-badge";
      el.innerText = `${name}: EMULATED`;
    }
  }

  updateHwBadge("badge-ad8232", "AD8232 [ECG]", detected["AD8232"]);
  updateHwBadge("badge-max30102", "MAX30102 [PPG]", detected["MAX30102"]);
  updateHwBadge("badge-mpu6050", "MPU6050 [IMU]", detected["MPU6050"]);
  updateHwBadge("badge-tmp117", "TMP117 [TEMP]", detected["TMP117"]);


  // 1. Push raw waveform points
  if (raw.ecg_raw !== undefined) {
    // Zero-center normalized representation for display
    const ecgNorm = (raw.ecg_raw - 32500.0) / 1000.0;
    ecgScope.pushSample(ecgNorm);
    ecgScope.render();

    const ppgNorm = (raw.ppg_raw - 72000.0) / 400.0;
    ppgScope.pushSample(ppgNorm);
    ppgScope.render();

    imuXBuf.shift(); imuXBuf.push(raw.a_x || 0);
    imuYBuf.shift(); imuYBuf.push(raw.a_y || 0);
    imuZBuf.shift(); imuZBuf.push(raw.a_z || 0);

    imuScope.render([
      { buffer: imuXBuf, color: "#05c46b" },
      { buffer: imuYBuf, color: "#0088ff" },
      { buffer: imuZBuf, color: "#ffa801" }
    ]);
  }

  // 2. Update Physiological State Badge
  const stateName = inf.state_name || "RESTING";
  const stateBadge = document.getElementById("state-badge");
  stateBadge.innerText = stateName;
  stateBadge.className = "state-badge";
  if (stateName === "RESTING") stateBadge.classList.add("state-resting");
  else if (stateName === "ACTIVE_EXERTION") stateBadge.classList.add("state-exertion");
  else if (stateName === "PHYSIOLOGICAL_STRAIN") stateBadge.classList.add("state-strain");
  else if (stateName === "MOTION_ARTIFACT") stateBadge.classList.add("state-artifact");

  document.getElementById("state-conf").innerText = `${((inf.confidence || 0.98) * 100).toFixed(1)}%`;
  document.getElementById("state-desc").innerText = STATE_DESCRIPTIONS[stateName] || "";

  // Anomaly score meter
  const anom = inf.anomaly_score !== undefined ? inf.anomaly_score : 0.05;
  document.getElementById("anomaly-val").innerText = anom.toFixed(2);
  const anomBar = document.getElementById("anomaly-bar");
  anomBar.style.width = `${Math.min(100, Math.max(2, anom * 100))}%`;
  if (anom > 0.8) anomBar.style.backgroundColor = "var(--accent-red)";
  else if (anom > 0.5) anomBar.style.backgroundColor = "var(--accent-amber)";
  else anomBar.style.backgroundColor = "var(--accent-green)";

  // 3. Update Vitals Cards
  if (feats.hr_bpm) {
    document.getElementById("val-hr").innerHTML = `${Math.round(feats.hr_bpm)} <span class="metric-unit">BPM</span>`;
    document.getElementById("val-pr").innerText = feats.ppg_pulse_rate_bpm ? feats.ppg_pulse_rate_bpm.toFixed(1) : feats.hr_bpm.toFixed(1);
    document.getElementById("val-hrv").innerHTML = `${Math.round(feats.ecg_rr_std_ms || 45)} <span class="metric-unit">ms</span>`;
    document.getElementById("val-rmssd").innerText = Math.round(feats.ecg_rmssd_ms || 38);
    document.getElementById("val-temp").innerHTML = `${(feats.temp_mean_c || 36.6).toFixed(1)} <span class="metric-unit">?C</span>`;
    document.getElementById("val-temptrend").innerText = `${(feats.temp_trend || 0.0) >= 0 ? '+' : ''}${(feats.temp_trend || 0.0).toFixed(2)}`;
    document.getElementById("val-motion").innerHTML = `${(feats.motion_mag_mean || 0.02).toFixed(2)} <span class="metric-unit">g</span>`;
    document.getElementById("val-sma").innerText = (feats.motion_sma || 0.03).toFixed(2);
    document.getElementById("val-jerk").innerText = (feats.motion_jerk_std || 0.01).toFixed(2);
  }

  // 4. Update Alert Banner
  const alertBanner = document.getElementById("alert-banner");
  if (alerts && alerts.length > 0) {
    const topAlert = alerts[0];
    document.getElementById("alert-title").innerText = topAlert.title || topAlert.code;
    document.getElementById("alert-msg").innerText = topAlert.message;
    document.getElementById("alert-sev").innerText = topAlert.severity || "WARNING";
    alertBanner.classList.remove("hidden");
  } else {
    alertBanner.classList.add("hidden");
  }

  // 5. Update ?T-Kernel 3.0 RTOS Task Monitor Table
  if (rtos.tasks) {
    const tbody = document.getElementById("rtos-tasks-body");
    tbody.innerHTML = "";
    rtos.tasks.forEach(t => {
      const tr = document.createElement("tr");
      const role = TASK_ROLES[t.name] || "RTOS Task";
      const stateColor = t.state === "RUNNING" ? "var(--accent-green)" : (t.state === "READY" ? "var(--accent-cyan)" : "var(--text-muted)");
      tr.innerHTML = `
        <td><strong>${t.name}</strong></td>
        <td><span style="color:var(--accent-amber); font-weight:700;">Priority ${t.priority}</span></td>
        <td style="color:var(--text-muted);">${role}</td>
        <td><span style="color:${stateColor}; font-weight:600;">${t.state}</span></td>
        <td>${t.context_switches.toLocaleString()}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Update IPC stats
  if (rtos.mailboxes) {
    const mbxStr = rtos.mailboxes.map(m => `${m.name}: ${m.queue_depth} queued (total: ${m.total_rcv})`).join(" | ");
    document.getElementById("ipc-stats").innerText = `IPC Mailboxes: ${mbxStr}`;
  }

  // 6. Update BLE Hex preview
  if (tele.ble_packet && tele.ble_packet.payload_hex) {
    document.getElementById("ble-hex").innerText = tele.ble_packet.payload_hex.match(/.{1,4}/g).join(" ").toUpperCase();
  }
}

// Scenario control buttons
document.querySelectorAll(".btn-scenario").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".btn-scenario").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    const scenario = btn.getAttribute("data-scenario");
    fetch("/api/scenario", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: scenario, speed: currentSpeed })
    });
  });
});

// Speed control buttons
let currentSpeed = 1.0;
document.querySelectorAll(".btn-speed").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".btn-speed").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentSpeed = parseFloat(btn.getAttribute("data-speed"));
    const activeScenarioBtn = document.querySelector(".btn-scenario.active");
    const scenario = activeScenarioBtn ? activeScenarioBtn.getAttribute("data-scenario") : "sit";
    fetch("/api/scenario", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: scenario, speed: currentSpeed })
    });
  });
});

window.onload = connectWebSocket;
