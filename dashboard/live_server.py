"""
dashboard.live_server
=====================
FastAPI + WebSocket live-spectrum server for the CCS200 spectrometer.

Architecture
------------
Runs as a **daemon thread inside the same process as Streamlit**, sharing
``LiveDataStore`` directly — no IPC, no serialisation overhead beyond the
WebSocket frame itself.

Two WebSocket push endpoints:

``/ws/spectrum``
    Pushes ``{wl, i}`` the instant a new frame arrives in ``LiveDataStore``.

``/ws/trend``
    Pushes rolling trend data (shift, concentration, timestamps) every 200 ms.

Static assets (uPlot JS/CSS) are served locally from ``dashboard/static/``
— no CDN or internet dependency.

Usage
-----
::

    from dashboard.live_server import start_live_server
    start_live_server(port=5006)

Navigate to http://localhost:5006 for the standalone live view.
Embed in Streamlit with ``st.components.v1.iframe("http://localhost:5006", height=700)``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.inference.live_state import LiveDataStore

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Live Spectrum Server", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# HTML page (uPlot loaded from local /static/ — no CDN dependency)
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Live Spectrum</title>
<link href="/static/uPlot.min.css" rel="stylesheet"/>
<script src="/static/uPlot.iife.min.js"></script>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,sans-serif;background:#0e1117;color:#fafafa;padding:10px}
  .topbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:8px}
  #status{padding:3px 10px;border-radius:4px;font-size:.8rem;font-weight:700}
  .acq{background:#1a4a1a;color:#6ee06e}.stop{background:#4a1a1a;color:#e06e6e}
  .wait{background:#3a3a1a;color:#e0d06e}
  .metric{display:inline-block}
  .metric .label{font-size:.65rem;color:#888;text-transform:uppercase;letter-spacing:.05em}
  .metric .value{font-size:1.2rem;font-weight:700}
  .fps{margin-left:auto;font-size:.7rem;color:#555}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
  .card{background:#1a1d27;border-radius:6px;padding:8px}
  .card.full{grid-column:1/-1}
  .card h3{font-size:.75rem;font-weight:600;color:#a8c8f8;margin-bottom:6px;text-transform:uppercase;letter-spacing:.06em}
  .uplot{width:100%!important}
  .uplot canvas{width:100%!important}
  #err{display:none;background:#4a1a1a;color:#e06e6e;padding:8px;border-radius:4px;margin-top:6px;font-size:.8rem}
</style>
</head>
<body>
<div class="topbar">
  <span id="status" class="wait">WAITING</span>
  <span class="metric"><span class="label">Samples</span><br><span class="value" id="m-count">—</span></span>
  <span class="metric"><span class="label">Peak&nbsp;λ&nbsp;(nm)</span><br><span class="value" id="m-peak">—</span></span>
  <span class="metric"><span class="label">Δλ&nbsp;(nm)</span><br><span class="value" id="m-shift">—</span></span>
  <span class="metric"><span class="label">Conc&nbsp;(ppm)</span><br><span class="value" id="m-conc">—</span></span>
  <span class="metric"><span class="label">SNR</span><br><span class="value" id="m-snr">—</span></span>
  <span class="fps" id="m-fps"></span>
</div>
<div id="err">WebSocket disconnected — reconnecting…</div>

<div class="grid">
  <div class="card full">
    <h3>Live Spectrum</h3>
    <div id="ch-spectrum"></div>
  </div>
  <div class="card">
    <h3>Wavelength Shift Δλ (nm)</h3>
    <div id="ch-shift"></div>
  </div>
  <div class="card">
    <h3>Concentration (ppm)</h3>
    <div id="ch-conc"></div>
  </div>
</div>

<script>
// ── Chart factory ─────────────────────────────────────────────────────────────
const GRID = {stroke:"#2a2d3a", width:1};
const AX   = {stroke:"#888", ticks:{stroke:"#888"}};

function mkSpectrum(el) {
  return new uPlot({
    width: el.clientWidth || 860, height: 260,
    cursor:{show:true}, legend:{show:false},
    axes:[
      {...AX, grid:GRID, label:"Wavelength (nm)", labelFont:"11px system-ui", font:"11px system-ui"},
      {...AX, grid:GRID, label:"Intensity (counts)", labelFont:"11px system-ui", font:"11px system-ui"},
    ],
    series:[
      {},
      {stroke:"#5a9fd4", width:1.5, label:"Intensity"},
      {stroke:"#e06060", width:1, dash:[4,3], label:"Reference", show:false},
    ],
    scales:{x:{time:false}},
  }, [[],[],[]], el);
}

function mkTrend(el, label, color, yLabel) {
  return new uPlot({
    width: el.clientWidth || 420, height: 200,
    cursor:{show:true}, legend:{show:false},
    axes:[
      {...AX, grid:GRID, label:"Time (s)", labelFont:"11px system-ui", font:"11px system-ui"},
      {...AX, grid:GRID, label:yLabel||label, labelFont:"11px system-ui", font:"11px system-ui"},
    ],
    series:[{},{stroke:color, width:2, label:label}],
    scales:{x:{time:false}},
  }, [[],[]], el);
}

const uSpec  = mkSpectrum(document.getElementById("ch-spectrum"));
const uShift = mkTrend(document.getElementById("ch-shift"), "Δλ (nm)", "#9b59b6", "nm");
const uConc  = mkTrend(document.getElementById("ch-conc"),  "ppm",     "#2ecc71", "ppm");

// Responsive resize
const ro = new ResizeObserver(()=>{
  const w1 = document.getElementById("ch-spectrum").clientWidth;
  const w2 = document.getElementById("ch-shift").clientWidth;
  if(w1>0) uSpec.setSize({width:w1, height:260});
  if(w2>0){ uShift.setSize({width:w2,height:200}); uConc.setSize({width:w2,height:200}); }
});
ro.observe(document.body);

// ── Status helpers ────────────────────────────────────────────────────────────
function setStatus(running, count) {
  const el = document.getElementById("status");
  if(running){ el.textContent="ACQUIRING"; el.className="acq"; }
  else if(count>0){ el.textContent="STOPPED"; el.className="stop"; }
  else { el.textContent="WAITING"; el.className="wait"; }
}
function setMetric(id, val, decimals) {
  document.getElementById(id).textContent = (val!=null && isFinite(val)) ? val.toFixed(decimals) : "—";
}

// ── Spectrum WebSocket ────────────────────────────────────────────────────────
let specFrames=0, specT0=performance.now();
let refWl=null, refI=null;

function connectSpectrum(){
  const errEl = document.getElementById("err");
  const ws = new WebSocket(`ws://${location.host}/ws/spectrum`);

  ws.onopen  = ()=>{ errEl.style.display="none"; };
  ws.onclose = ()=>{ errEl.style.display="block"; setTimeout(connectSpectrum, 1500); };
  ws.onerror = ()=>{ ws.close(); };

  ws.onmessage = ev=>{
    let msg;
    try{ msg=JSON.parse(ev.data); }catch(e){ return; }
    if(!msg.wl || !msg.i || msg.wl.length===0) return;

    // First frame sets reference for diff overlay
    if(!refWl){ refWl=msg.wl; refI=msg.i.slice(); }

    const ref = refWl ? refI : new Array(msg.wl.length).fill(null);
    uSpec.setData([msg.wl, msg.i, ref]);

    // FPS
    specFrames++;
    const dt=(performance.now()-specT0)/1000;
    if(dt>2){ document.getElementById("m-fps").textContent=(specFrames/dt).toFixed(1)+" fps"; specFrames=0; specT0=performance.now(); }
  };
}

// ── Trend WebSocket ───────────────────────────────────────────────────────────
function connectTrend(){
  const ws = new WebSocket(`ws://${location.host}/ws/trend`);

  ws.onclose = ()=>{ setTimeout(connectTrend, 1500); };
  ws.onerror = ()=>{ ws.close(); };

  ws.onmessage = ev=>{
    let d;
    try{ d=JSON.parse(ev.data); }catch(e){ return; }

    setStatus(d.running, d.count||0);
    document.getElementById("m-count").textContent = d.count!=null ? d.count.toLocaleString() : "—";
    setMetric("m-peak",  d.peak,  3);
    setMetric("m-shift", d.shift, 4);
    setMetric("m-conc",  d.conc,  4);
    setMetric("m-snr",   d.snr,   1);

    if(d.ts && d.shifts){ uShift.setData([d.ts, d.shifts]); }
    if(d.ts && d.concs) { uConc.setData ([d.ts, d.concs]);  }
  };
}

connectSpectrum();
connectTrend();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _HTML


# ---------------------------------------------------------------------------
# WebSocket helpers
# ---------------------------------------------------------------------------


class _Broadcaster:
    """Thread-safe set of connected WebSocket clients."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    def add(self, ws: WebSocket) -> None:
        self._clients.add(ws)

    def discard(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def send(self, msg: str) -> None:
        """Broadcast to all clients; silently drop dead connections."""
        dead: set[WebSocket] = set()
        for ws in list(self._clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self._clients -= dead


_spec_bc  = _Broadcaster()
_trend_bc = _Broadcaster()


async def _ws_handler(ws: WebSocket, bc: _Broadcaster) -> None:
    """Accept connection, add to broadcaster, wait for disconnect."""
    await ws.accept()
    bc.add(ws)
    try:
        while True:
            # Keepalive — client doesn't need to send anything; we just
            # need to detect disconnect (receive raises WebSocketDisconnect).
            await ws.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        bc.discard(ws)


@app.websocket("/ws/spectrum")
async def ws_spectrum(ws: WebSocket) -> None:
    await _ws_handler(ws, _spec_bc)


@app.websocket("/ws/trend")
async def ws_trend(ws: WebSocket) -> None:
    await _ws_handler(ws, _trend_bc)


# ---------------------------------------------------------------------------
# Broadcast loops (started as asyncio tasks on server startup)
# ---------------------------------------------------------------------------


async def _spectrum_loop() -> None:
    """Push new spectrum frame the instant LiveDataStore gets one."""
    last_count = -1
    while True:
        count = LiveDataStore.get_sample_count()
        if count != last_count:
            last_count = count
            spectrum = LiveDataStore.get_latest_spectrum()
            if spectrum is not None and _spec_bc._clients:
                wl, intensity = spectrum
                msg = json.dumps({"wl": wl.tolist(), "i": intensity.tolist()})
                await _spec_bc.send(msg)
        await asyncio.sleep(0.05)  # poll at 20 Hz; CCS200 delivers at ~2.4 Hz


async def _trend_loop() -> None:
    """Push rolling trend data every 200 ms."""
    last_count = -1
    t0: float | None = None  # epoch of first frame (seconds)

    while True:
        await asyncio.sleep(0.2)
        count = LiveDataStore.get_sample_count()
        if count == last_count or not _trend_bc._clients:
            continue
        last_count = count

        recent = LiveDataStore.get_latest(300)
        last   = LiveDataStore.get_last_result()

        # Build time axis in seconds from first frame
        ts_list: list[float] = []
        for r in recent:
            raw = r.get("timestamp")
            if raw is None:
                continue
            # timestamp may be a datetime object or a unix float
            try:
                from datetime import datetime
                epoch = raw.timestamp() if isinstance(raw, datetime) else float(raw)
            except Exception:
                continue
            ts_list.append(epoch)

        if ts_list:
            if t0 is None:
                t0 = ts_list[0]
            rel = [t - t0 for t in ts_list]
        else:
            rel = list(range(len(recent)))

        shifts = [r.get("wavelength_shift") for r in recent[-len(rel):]]
        concs  = [r.get("concentration_ppm") for r in recent[-len(rel):]]

        payload: dict[str, Any] = {
            "running": LiveDataStore.is_running(),
            "count":   count,
            "ts":      rel,
            "shifts":  shifts,
            "concs":   concs,
        }
        if last:
            payload["peak"]  = last.get("peak_wavelength")
            payload["shift"] = last.get("wavelength_shift")
            payload["conc"]  = last.get("concentration_ppm")
            payload["snr"]   = last.get("snr")

        await _trend_bc.send(json.dumps(payload))


@app.on_event("startup")
async def _on_startup() -> None:
    asyncio.create_task(_spectrum_loop())
    asyncio.create_task(_trend_loop())


# ---------------------------------------------------------------------------
# Public launcher
# ---------------------------------------------------------------------------

_server_started = False
_server_lock    = threading.Lock()


def start_live_server(port: int = 5006) -> None:
    """Start the uvicorn server in a background daemon thread.

    Safe to call multiple times — only starts once.  Uses SelectorEventLoop
    on Windows for maximum asyncio compatibility with uvicorn.
    """
    global _server_started
    with _server_lock:
        if _server_started:
            return
        _server_started = True

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _run() -> None:
        # Windows: SelectorEventLoop is more compatible with uvicorn than the
        # default ProactorEventLoop (which doesn't support all asyncio features).
        if sys.platform == "win32":
            loop: asyncio.AbstractEventLoop = asyncio.SelectorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        except Exception:
            log.exception("Live server crashed — live view will be unavailable")
        finally:
            loop.close()

    thread = threading.Thread(target=_run, name="live-server", daemon=True)
    thread.start()
    log.info("Live server starting on http://0.0.0.0:%d", port)
