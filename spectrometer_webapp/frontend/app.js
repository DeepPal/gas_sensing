/* SpectroSense — Frontend Logic
 * Connects to FastAPI backend via WebSocket for live spectra and REST for
 * session management, training, and model deployment.
 */
document.addEventListener("DOMContentLoaded", () => {

    // ── DOM refs ────────────────────────────────────────────────────────────
    // Navigation
    const navItems    = document.querySelectorAll(".nav-item");
    const tabContents = document.querySelectorAll(".tab-content");

    // Tab 1
    const acqGas     = document.getElementById("acq-gas");
    const acqConc    = document.getElementById("acq-conc");
    const btnStart   = document.getElementById("btn-start-log");
    const btnStop    = document.getElementById("btn-stop-log");
    const logStatus  = document.getElementById("log-status");
    const logDot     = document.getElementById("log-dot");
    const logText    = document.getElementById("log-text");
    const mockGas    = document.getElementById("mock-gas");
    const mockConc   = document.getElementById("mock-conc");
    const btnSetMock = document.getElementById("btn-set-mock");

    // Tab 2
    const sessionList         = document.getElementById("session-list");
    const btnRefreshSessions  = document.getElementById("btn-refresh-sessions");
    const btnLoadSummary      = document.getElementById("btn-load-summary");
    const btnTrain            = document.getElementById("btn-train");
    const trainModelType      = document.getElementById("train-model-type");
    const trainTestSize       = document.getElementById("train-test-size");
    const preDenoiseEl        = document.getElementById("pre-denoise");
    const preBaselineEl       = document.getElementById("pre-baseline");
    const preNormEl           = document.getElementById("pre-norm");
    const prePcaEl            = document.getElementById("pre-pca");
    const exploratorySection  = document.getElementById("exploratory-section");
    const trainingResults     = document.getElementById("training-results");
    const resAcc              = document.getElementById("res-acc");
    const resNTrain           = document.getElementById("res-n-train");
    const resNTest            = document.getElementById("res-n-test");
    const resModelName        = document.getElementById("res-model-name");
    const resReportTbody      = document.querySelector("#res-report-table tbody");

    // Tab 3
    const modelSelect         = document.getElementById("model-select");
    const btnRefreshModels    = document.getElementById("btn-refresh-models");
    const btnLoadModel        = document.getElementById("btn-load-model");
    const inferenceOverlay    = document.getElementById("inference-overlay");
    const predGasEl           = document.getElementById("pred-gas");
    const predConfEl          = document.getElementById("pred-conf");
    const currentModelEl      = document.getElementById("current-loaded-model");
    const modelInfoCard       = document.getElementById("model-info-card");
    const probChartSection    = document.getElementById("prob-chart-section");
    const infAcc              = document.getElementById("inf-acc");
    const infAlgo             = document.getElementById("inf-algo");
    const infClasses          = document.getElementById("inf-classes");
    const infDate             = document.getElementById("inf-date");
    const infPreproc          = document.getElementById("inf-preproc");

    // ── Base URLs ────────────────────────────────────────────────────────────
    let host = window.location.host;
    if (!host || window.location.protocol === "file:") host = "127.0.0.1:8080";
    const WS_URL   = `ws://${host}/ws`;
    const HTTP_URL = `http://${host}`;

    // ── Live spectrum chart ──────────────────────────────────────────────────
    const CHART_LAYOUT = {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor:  "rgba(0,0,0,0)",
        margin:  { t: 40, l: 55, r: 20, b: 45 },
        xaxis: {
            title: "Wavelength (nm)", color: "#8b949e",
            gridcolor: "rgba(255,255,255,0.05)", range: [300, 1000]
        },
        yaxis: {
            title: "Intensity (a.u.)", color: "#8b949e",
            gridcolor: "rgba(255,255,255,0.05)"
        },
        title: { text: "Live Spectrometer Feed", font: { color: "#c9d1d9", family: "Inter" } },
        showlegend: false,
    };

    Plotly.newPlot("spectrum-chart", [{
        x: [], y: [], type: "scatter", mode: "lines",
        line: { color: "#58a6ff", width: 2 }
    }], CHART_LAYOUT, { responsive: true, displayModeBar: false });

    // ── Navigation ───────────────────────────────────────────────────────────
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            navItems.forEach(n => n.classList.remove("active"));
            tabContents.forEach(t => { t.classList.remove("active"); t.classList.add("hidden"); });
            item.classList.add("active");
            const target = item.getAttribute("data-target");
            const el = document.getElementById(target);
            el.classList.remove("hidden");
            el.classList.add("active");
            setTimeout(() => Plotly.Plots.resize("spectrum-chart"), 120);

            // Show inference overlay only on Tab 3 when a model is active
            const onInference = target === "tab-inference";
            if (onInference && currentModelEl.textContent !== "No model loaded") {
                inferenceOverlay.classList.remove("hidden");
            } else if (!onInference) {
                inferenceOverlay.classList.add("hidden");
            }
        });
    });

    // ── WebSocket ────────────────────────────────────────────────────────────
    let ws = null;
    let probsInitialized  = false;

    function connectWS() {
        ws = new WebSocket(WS_URL);
        ws.onopen = () => {
            document.getElementById("conn-dot").classList.add("connected");
            document.getElementById("conn-text").textContent = "Connected";
        };

        ws.onmessage = (evt) => {
            const d = JSON.parse(evt.data);

            // Sync sidebar logging status from server state
            if (d.is_logging) {
                logDot.style.background = "#da3633";
                logText.textContent     = "Recording…";
            } else {
                logDot.style.background = "#4d4d4d";
                if (logText.textContent === "Recording…") logText.textContent = "Not Recording";
            }

            // Live chart update
            if (d.wavelengths && d.intensities) {
                const color = d.is_logging ? "#ff5252" : "#58a6ff";
                Plotly.react("spectrum-chart", [{
                    x: d.wavelengths, y: d.intensities,
                    type: "scatter", mode: "lines",
                    line:     { color, width: 2 },
                    fill:     "tozeroy",
                    fillcolor: d.is_logging ? "rgba(255,82,82,0.08)" : "rgba(88,166,255,0.08)"
                }], {
                    ...CHART_LAYOUT,
                    title: {
                        text: d.is_logging ? "🔴 RECORDING" : "Live Spectrometer Feed",
                        font: { color: d.is_logging ? "#ff5252" : "#c9d1d9", family: "Inter" }
                    }
                });
            }

            // Predictions
            if (d.prediction) {
                const GAS_COLORS = {
                    Air: "#7b8898", Acetone: "#ff7b72",
                    Ethanol: "#79c0ff", Methane: "#d2a8ff",
                    IPA: "#56d364", MeOH: "#e3b341"
                };
                predGasEl.textContent   = d.prediction;
                predGasEl.style.color   = GAS_COLORS[d.prediction] || "#fff";
                predConfEl.textContent  = `Confidence: ${(d.confidence * 100).toFixed(1)} %`;

                if (d.probabilities && Object.keys(d.probabilities).length) {
                    updateProbChart(d.probabilities);
                }

                if (document.getElementById("tab-inference").classList.contains("active")) {
                    inferenceOverlay.classList.remove("hidden");
                }
            } else {
                predGasEl.textContent  = "--";
                predConfEl.textContent = "Confidence: --%";
            }
        };

        ws.onclose = () => {
            document.getElementById("conn-dot").classList.remove("connected");
            document.getElementById("conn-text").textContent = "Reconnecting…";
            setTimeout(connectWS, 3000);
        };
    }
    connectWS();

    // ── Probability bar chart (live, Tab 3) ──────────────────────────────────
    function updateProbChart(probs) {
        const gases  = Object.keys(probs);
        const values = Object.values(probs).map(v => parseFloat((v * 100).toFixed(1)));
        const colors = values.map(v => v === Math.max(...values) ? "#58a6ff" : "#30363d");

        if (!probsInitialized) {
            Plotly.newPlot("chart-probs", [{
                type: "bar", orientation: "h",
                x: values, y: gases,
                marker: { color: colors },
                text:  values.map(v => `${v} %`), textposition: "outside",
            }], {
                paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                margin: { t: 5, l: 80, r: 40, b: 30 },
                xaxis: { range: [0, 105], color: "#8b949e",
                         gridcolor: "rgba(255,255,255,0.05)" },
                yaxis: { color: "#c9d1d9" },
                font:  { color: "#c9d1d9", family: "Inter", size: 12 },
            }, { responsive: true, displayModeBar: false });
            probsInitialized = true;
            probChartSection.classList.remove("hidden");
        } else {
            Plotly.react("chart-probs", [{
                type: "bar", orientation: "h",
                x: values, y: gases,
                marker: { color: colors },
                text: values.map(v => `${v} %`), textposition: "outside",
            }], {
                paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                margin: { t: 5, l: 80, r: 40, b: 30 },
                xaxis: { range: [0, 105], color: "#8b949e",
                         gridcolor: "rgba(255,255,255,0.05)" },
                yaxis: { color: "#c9d1d9" },
                font:  { color: "#c9d1d9", family: "Inter", size: 12 },
            });
        }
    }

    // ── TAB 1: Acquisition ───────────────────────────────────────────────────
    function checkAcqParams() {
        btnStart.disabled = !(acqGas.value.trim() && acqConc.value.trim());
    }
    acqGas.addEventListener("input",  checkAcqParams);
    acqConc.addEventListener("input", checkAcqParams);

    btnStart.addEventListener("click", async () => {
        const res = await fetch(`${HTTP_URL}/api/logging/start`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                gas_type:            acqGas.value.trim(),
                concentration:       parseFloat(acqConc.value),
                comments:            document.getElementById("acq-notes").value.trim(),
                integration_time_ms: parseFloat(document.getElementById("acq-int-time").value) || 50,
            })
        });
        if (!res.ok) { alert("Failed to start logging. Is the spectrometer connected?"); return; }
        const data = await res.json();

        btnStart.classList.add("hidden");
        btnStop.classList.remove("hidden");
        acqGas.disabled = acqConc.disabled = true;
        logStatus.textContent = `📁 Saving → ${data.session}`;
        logDot.style.background = "#da3633";
        logText.textContent     = "Recording…";
    });

    btnStop.addEventListener("click", async () => {
        await fetch(`${HTTP_URL}/api/logging/stop`, { method: "POST" });
        btnStart.classList.remove("hidden");
        btnStop.classList.add("hidden");
        acqGas.disabled = acqConc.disabled = false;
        logStatus.textContent = "Session saved.";
        logDot.style.background = "#4d4d4d";
        logText.textContent     = "Not Recording";
        setTimeout(() => { logStatus.textContent = ""; }, 4000);
    });

    btnSetMock.addEventListener("click", async () => {
        await fetch(`${HTTP_URL}/api/environment`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ gas: mockGas.value, conc: parseFloat(mockConc.value) })
        });
    });

    // ── TAB 2: Session list ──────────────────────────────────────────────────
    async function fetchSessions() {
        const res  = await fetch(`${HTTP_URL}/api/sessions`);
        const data = await res.json();
        sessionList.innerHTML = "";
        if (!data.sessions.length) {
            sessionList.innerHTML = '<li class="text-sm text-muted">No sessions yet.</li>';
            return;
        }
        data.sessions.forEach(sess => {
            const li = document.createElement("li");
            li.innerHTML = `<label class="sess-label">
                <input type="checkbox" value="${sess}" class="sess-chk" checked>
                <span class="text-sm">${sess}</span>
            </label>`;
            sessionList.appendChild(li);
        });
    }
    btnRefreshSessions.addEventListener("click", fetchSessions);

    // ── TAB 2: Exploratory chart ─────────────────────────────────────────────
    btnLoadSummary.addEventListener("click", async () => {
        const selected = getSelectedSessions();
        if (!selected.length) { alert("Select at least one session."); return; }

        btnLoadSummary.disabled = true;
        btnLoadSummary.textContent = "Loading…";

        try {
            const res  = await fetch(`${HTTP_URL}/api/sessions/summary`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sessions: selected })
            });
            const data = await res.json();
            renderExploratory(data.groups);
        } catch (e) {
            alert(`Failed to load summary: ${e.message}`);
        } finally {
            btnLoadSummary.disabled = false;
            btnLoadSummary.textContent = "📊 Explore Data";
        }
    });

    function renderExploratory(groups) {
        if (!Object.keys(groups).length) return;
        const PALETTE = ["#58a6ff","#ff7b72","#56d364","#d2a8ff","#e3b341","#79c0ff"];
        const traces  = Object.entries(groups).map(([gas, d], i) => ({
            x: d.wavelengths,
            y: d.mean,
            type: "scatter", mode: "lines",
            name: `${gas} (n=${d.n_spectra})`,
            line: { color: PALETTE[i % PALETTE.length], width: 2 }
        }));
        Plotly.react("chart-exploratory", traces, {
            paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
            margin: { t: 20, l: 55, r: 20, b: 45 },
            xaxis: { title: "Wavelength (nm)", color: "#8b949e",
                     gridcolor: "rgba(255,255,255,0.05)" },
            yaxis: { title: "Mean Intensity",  color: "#8b949e",
                     gridcolor: "rgba(255,255,255,0.05)" },
            legend: { font: { color: "#c9d1d9" }, bgcolor: "rgba(0,0,0,0)" },
            font:   { color: "#c9d1d9", family: "Inter" },
        }, { responsive: true, displayModeBar: false });
        exploratorySection.classList.remove("hidden");
        setTimeout(() => Plotly.Plots.resize("chart-exploratory"), 50);
    }

    // ── TAB 2: Training ──────────────────────────────────────────────────────
    function getSelectedSessions() {
        return Array.from(document.querySelectorAll(".sess-chk:checked")).map(c => c.value);
    }

    function getPreprocessingConfig() {
        return {
            denoising:     preDenoiseEl.value,
            baseline:      preBaselineEl.value,
            normalization: preNormEl.value,
            use_pca:       true,
            pca_components: parseInt(prePcaEl.value, 10),
        };
    }

    btnTrain.addEventListener("click", async () => {
        const selected = getSelectedSessions();
        if (!selected.length) { alert("Select at least one session."); return; }

        btnTrain.disabled    = true;
        btnTrain.textContent = "Training…";
        trainingResults.classList.add("hidden");

        try {
            const res = await fetch(`${HTTP_URL}/api/train`, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    sessions:      selected,
                    model_type:    trainModelType.value,
                    test_size:     parseFloat(trainTestSize.value),
                    preprocessing: getPreprocessingConfig(),
                })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail);
            renderTrainingResults(data);
        } catch (e) {
            alert(`Training failed: ${e.message}`);
        } finally {
            btnTrain.disabled    = false;
            btnTrain.textContent = "🚀 Train Model";
        }
    });

    function renderTrainingResults(data) {
        resAcc.textContent       = `${(data.accuracy * 100).toFixed(2)} %`;
        resNTrain.textContent    = data.metadata?.n_train ?? "--";
        resNTest.textContent     = data.metadata?.n_test  ?? "--";
        resModelName.textContent = data.model_name;

        // Per-class table
        resReportTbody.innerHTML = "";
        const report = data.report || {};
        (data.classes || []).forEach(cls => {
            const row = report[cls] || {};
            const tr  = document.createElement("tr");
            tr.innerHTML = `
                <td>${cls}</td>
                <td>${(row.precision ?? 0).toFixed(3)}</td>
                <td>${(row.recall    ?? 0).toFixed(3)}</td>
                <td>${(row["f1-score"] ?? 0).toFixed(3)}</td>
                <td>${row.support ?? "--"}</td>`;
            resReportTbody.appendChild(tr);
        });

        // Confusion matrix (Plotly annotated heatmap)
        if (data.confusion_matrix && data.classes) {
            const cm      = data.confusion_matrix;
            const classes = data.classes;
            const text    = cm.map(row => row.map(v => String(v)));
            Plotly.react("chart-confusion", [{
                type: "heatmap",
                z:       cm,
                x:       classes,
                y:       classes,
                text:    text,
                texttemplate: "%{text}",
                colorscale: "Blues",
                showscale:  true,
                colorbar: { tickfont: { color: "#c9d1d9" } },
            }], {
                paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                margin: { t: 20, l: 70, r: 20, b: 70 },
                xaxis: { title: "Predicted", color: "#8b949e" },
                yaxis: { title: "Actual",    color: "#8b949e", autorange: "reversed" },
                font:  { color: "#c9d1d9", family: "Inter" },
            }, { responsive: true, displayModeBar: false });
        }

        trainingResults.classList.remove("hidden");
        setTimeout(() => Plotly.Plots.resize("chart-confusion"), 50);

        // Auto-refresh model list so the new model appears in Tab 3
        fetchModels();
    }

    // ── TAB 3: Model loading & inference ─────────────────────────────────────
    async function fetchModels() {
        const res  = await fetch(`${HTTP_URL}/api/models`);
        const data = await res.json();
        modelSelect.innerHTML = '<option value="">-- Select a model --</option>';
        data.models.forEach(m => {
            const opt  = document.createElement("option");
            opt.value  = m;
            opt.textContent = m;
            modelSelect.appendChild(opt);
        });
    }
    btnRefreshModels.addEventListener("click", fetchModels);

    btnLoadModel.addEventListener("click", async () => {
        const selected = modelSelect.value;
        if (!selected) return;

        btnLoadModel.disabled    = true;
        btnLoadModel.textContent = "Loading…";
        try {
            const res  = await fetch(`${HTTP_URL}/api/models/load/${selected}`, { method: "POST" });
            if (!res.ok) throw new Error("Failed to load model.");
            const data = await res.json();

            currentModelEl.textContent = `Active: ${selected}`;
            inferenceOverlay.classList.remove("hidden");
            renderModelInfo(data.metadata || {}, selected);
            // Reset probability chart for the new model's classes
            probsInitialized = false;
            probChartSection.classList.add("hidden");

        } catch (e) {
            alert(e.message);
        } finally {
            btnLoadModel.disabled    = false;
            btnLoadModel.textContent = "Load & Activate";
        }
    });

    function renderModelInfo(meta, modelName) {
        infAcc.textContent     = meta.accuracy ? `${(meta.accuracy * 100).toFixed(2)} %` : "--";
        infAlgo.textContent    = meta.model_type  || modelName.split("_")[0];
        infClasses.textContent = (meta.classes   || []).join(", ") || "--";
        infDate.textContent    = meta.trained_at  || "--";

        // Preprocessing summary
        const pre = meta.preprocessing || {};
        infPreproc.textContent = [
            pre.denoising    ? `Denoising: ${pre.denoising}`       : null,
            pre.baseline     ? `Baseline: ${pre.baseline}`          : null,
            pre.normalization ? `Norm: ${pre.normalization}`        : null,
            pre.use_pca      ? `PCA: ${pre.pca_components} comps`  : null,
            pre.use_lspr_features ? "LSPR Δλ features: ON"         : null,
        ].filter(Boolean).join("  ·  ");

        modelInfoCard.classList.remove("hidden");
    }

    // ── Initial load ─────────────────────────────────────────────────────────
    fetchSessions();
    fetchModels();

    // Show server-side data directory path in the acquisition tab
    fetch(`${HTTP_URL}/api/config`)
        .then(r => r.json())
        .then(cfg => {
            const el = document.getElementById("data-dir-display");
            if (el) el.textContent = `Data directory: ${cfg.data_dir}`;
            const mockBadge = document.getElementById("mock-badge");
            if (mockBadge && cfg.use_mock) mockBadge.classList.remove("hidden");
        })
        .catch(() => {});
});
