import os
import asyncio
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from hardware import get_spectrometer, Ccs200Hardware
from data_manager import DataManager
from ml_pipeline import MLPipeline

# ── Core services ─────────────────────────────────────────────────────────────
# Toggle via env: SPECTROMETER_USE_MOCK=false when the real Thorlabs CCS200 is connected
spectrometer  = get_spectrometer(use_mock=os.getenv("SPECTROMETER_USE_MOCK", "true").lower() == "true")
data_manager  = DataManager(os.path.join(os.path.dirname(__file__), "..", "data", "acquisitions"))
ml_pipeline   = MLPipeline(os.path.join(os.path.dirname(__file__), "..", "data", "models"))

app = FastAPI(title="Spectrometer Virtual Sensor Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8501",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
os.makedirs(frontend_path, exist_ok=True)

spectrometer.connect()

# ── Shared state ───────────────────────────────────────────────────────────────
clients:         set  = set()
logging_active:  bool = False
latest_spectrum: dict = {"wl": None, "intys": None}   # written by acq loop


# ── Background acquisition loop ────────────────────────────────────────────────
async def acquisition_loop() -> None:
    """
    Single source-of-truth for spectrum data, running at 10 Hz.

    Responsibilities:
      • Poll the spectrometer hardware every 100 ms.
      • Cache the latest (wl, intys) pair so WebSocket clients can read it.
      • Save each frame to disk when logging_active is True.

    Decoupled from WebSocket clients: data is saved even if no browser tab
    is open, and the hardware is polled exactly once per frame (no duplicate
    reads from concurrent WS connections).
    """
    global latest_spectrum
    while True:
        try:
            wls, intys = spectrometer.get_spectrum()
            if wls and intys:
                latest_spectrum = {"wl": wls, "intys": intys}
                if logging_active:
                    data_manager.save_spectrum(wls, intys)
        except Exception:
            pass
        await asyncio.sleep(0.1)   # 10 Hz


@app.on_event("startup")
async def startup_event() -> None:
    asyncio.create_task(acquisition_loop())


# ── Request models ────────────────────────────────────────────────────────────
class EnvData(BaseModel):
    gas:  str
    conc: float

class StartLoggingReq(BaseModel):
    gas_type:             str
    concentration:        float
    comments:             Optional[str]   = ""
    integration_time_ms:  Optional[float] = 50.0

class TrainReq(BaseModel):
    sessions:      List[str]
    model_type:    str = "RandomForest"
    test_size:     float = 0.2
    preprocessing: Optional[Dict[str, Any]] = None

class ReferenceReq(BaseModel):
    wavelengths: List[float]
    intensities: List[float]


# ═══════════════════════════════════════════════════════════════════════════════
# Acquisition
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/environment")
async def set_environment(data: EnvData):
    """(Simulation only) Set the physical gas environment around the mock sensor."""
    spectrometer.set_environment(data.gas, data.conc)
    return {"status": "Environment updated"}


@app.post("/api/logging/start")
async def start_logging(req: StartLoggingReq):
    global logging_active
    if not spectrometer.connected:
        raise HTTPException(status_code=400, detail="Spectrometer not connected")
    int_time = max(1.0, float(req.integration_time_ms or 50.0))
    spectrometer.set_integration_time(int_time)
    metadata = {
        "gas_type":             req.gas_type,
        "concentration":        req.concentration,
        "comments":             req.comments,
        "integration_time_ms":  int_time,
    }
    data_manager.start_logging_session(metadata)
    logging_active = True
    spectrometer.start_acquisition()
    return {
        "status":   "Logging started",
        "session":  data_manager.current_session.name,
        "data_dir": str(data_manager.data_dir.resolve()),
    }


@app.post("/api/logging/stop")
async def stop_logging():
    global logging_active
    logging_active = False
    session = data_manager.current_session
    n_spectra = len(list(session.glob("*.csv"))) if session else 0
    data_manager.stop_logging_session()
    return {"status": "Logging stopped", "spectra_saved": n_spectra}


@app.get("/api/sessions")
async def get_sessions():
    return {"sessions": data_manager.get_sessions()}


@app.get("/api/config")
async def get_config():
    """Return server-side paths and hardware mode to the UI."""
    return {
        "data_dir":   str(data_manager.data_dir.resolve()),
        "models_dir": str(ml_pipeline.models_dir.resolve()),
        "use_mock":   not isinstance(spectrometer, Ccs200Hardware),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Exploratory Data (Analysis Tab)
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/sessions/summary")
async def sessions_summary(body: dict):
    """Return per-gas-type mean spectra for the exploratory overlay chart."""
    groups: dict = {}

    for sess in body.get("sessions", []):
        X_sess, wl_sess, meta = data_manager.load_session_data(sess)
        if X_sess is None or len(X_sess) == 0:
            continue
        gas = meta.get("gas_type", "Unknown")
        if gas not in groups:
            groups[gas] = {"wavelengths": wl_sess.tolist(), "spectra": []}
        for row in X_sess:
            aligned = np.interp(groups[gas]["wavelengths"], wl_sess, row)
            groups[gas]["spectra"].append(aligned.tolist())

    result = {}
    for gas, d in groups.items():
        result[gas] = {
            "wavelengths": d["wavelengths"],
            "mean":        np.mean(d["spectra"], axis=0).tolist(),
            "n_spectra":   len(d["spectra"]),
        }

    return {"groups": result}


# ═══════════════════════════════════════════════════════════════════════════════
# Reference spectrum (LSPR Δλ mode)
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/reference")
async def set_reference(req: ReferenceReq):
    """Upload a blank/air reference spectrum for LSPR differential signal."""
    ml_pipeline.set_reference(req.wavelengths, req.intensities)
    return {
        "status":   "Reference set",
        "peak_wl":  ml_pipeline._ref_peak_wl,
        "n_points": len(req.wavelengths),
    }


@app.delete("/api/reference")
async def clear_reference():
    ml_pipeline.clear_reference()
    return {"status": "Reference cleared"}


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/train")
async def train_model(req: TrainReq):
    X_all, y_all, wl_common = [], [], None

    for sess in req.sessions:
        X_sess, wl_sess, meta = data_manager.load_session_data(sess)
        if X_sess is None or len(X_sess) == 0:
            continue
        gas = meta.get("gas_type", "Unknown")
        if wl_common is None:
            wl_common = wl_sess
        X_all.extend(X_sess.tolist())
        y_all.extend([gas] * len(X_sess))

    if len(set(y_all)) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need recordings from at least 2 different gas types.")

    result = ml_pipeline.train_model(
        X_raw       = np.array(X_all),
        y_labels    = np.array(y_all),
        wavelengths = wl_common,
        model_type  = req.model_type,
        config      = req.preprocessing,
        test_size   = req.test_size,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Model management
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/models")
async def get_models():
    return {"models": ml_pipeline.get_available_models()}


@app.get("/api/models/{model_name}/info")
async def get_model_info(model_name: str):
    info = ml_pipeline.get_model_info(model_name)
    if not info:
        raise HTTPException(status_code=404, detail="Model not found")
    return info


@app.post("/api/models/load/{model_name}")
async def load_model(model_name: str):
    success = ml_pipeline.load_model(model_name)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {
        "status":   "Model loaded",
        "model":    model_name,
        "metadata": ml_pipeline.current_metadata,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket — live spectrum + real-time inference at 10 Hz
#
# Reads from `latest_spectrum` (written by acquisition_loop) — no direct
# hardware calls here, so concurrent WS clients don't cause duplicate reads.
# ═══════════════════════════════════════════════════════════════════════════════
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            wls   = latest_spectrum.get("wl")
            intys = latest_spectrum.get("intys")

            payload: dict = {
                "wavelengths":   wls,
                "intensities":   intys,
                "is_logging":    logging_active,
                "prediction":    None,
                "confidence":    None,
                "probabilities": {},
                "model_name":    None,
            }

            if wls and intys and ml_pipeline.current_model:
                try:
                    pred = ml_pipeline.predict(wls, intys)
                    if "error" not in pred:
                        payload["prediction"]    = pred["prediction"]
                        payload["confidence"]    = pred["confidence"]
                        payload["probabilities"] = pred.get("probabilities", {})
                        payload["model_name"]    = pred["model"]
                except Exception:
                    pass

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.1)   # 10 Hz

    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception:
        clients.discard(websocket)


# Mount static frontend AFTER all API routes
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
