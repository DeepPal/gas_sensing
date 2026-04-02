// API client — all HTTP + WebSocket helpers

const BASE = ''  // same origin; vite dev proxy handles /api → :8765

export interface HealthResponse {
  status: string
  version: string
  hardware: string
  simulate: boolean
  physics_plugin: string
  integration_time_ms?: number
  quality_settings: { saturation_threshold: number; snr_warn_threshold: number }
  drift_settings: { drift_threshold_nm_per_min: number; window_frames: number }
}

export interface AcquisitionConfig {
  integration_time_ms: number
  gas_label: string
  target_concentration: number | null
}

export interface CalibrationPoint {
  concentration: number
  delta_lambda: number
}

export interface SessionMeta {
  session_id: string
  started_at: string
  stopped_at: string | null
  frame_count: number
  gas_label: string
}

export interface SessionDetail extends SessionMeta {
  agent_events: AgentEvent[]
}

export interface AgentEvent {
  source: string
  level: string
  type: string
  text: string
  data?: Record<string, unknown>
  timestamp?: string
}

export interface SpectrumFrame {
  wl: number[]
  i: number[]
  frame: number
  concentration_ppm?: number
  ci_low?: number
  ci_high?: number
  peak_shift_nm?: number
  peak_wavelength?: number
  snr?: number
  gas_type?: string
  confidence_score?: number
}

export interface QualitySettings {
  saturation_threshold?: number
  snr_warn_threshold?: number
}

export interface DriftSettings {
  drift_threshold_nm_per_min?: number
  window_frames?: number
}

// Multi-analyte types
export interface AnalyteInfo {
  analytes: string[]
  n_peaks: number
  peak_wavelengths_nm: number[]
  S_matrix: number[][] | null
}

export interface MixtureInferenceResult {
  concentrations_ppm: Record<string, number>
  residual_nm: number
  solver: string
  success: boolean
  predicted_shifts_nm: number[]
}

export interface SimCalibrationSummaryRow {
  concentration_ppm: number
  mean_shift_nm: number
  std_shift_nm: number
  n: number
}

export interface QualificationExportResponse {
  status: string
  session_id: string
  artifact: string
  paths: Record<string, string>
  signature: { algorithm: string; payload_sha256: string; signed: boolean; signature?: string }
}

export interface QualificationPackageResponse {
  status: string
  session_id: string
  package_path: string
  included: string[]
  signature: { algorithm: string; payload_sha256: string; signed: boolean; signature?: string }
}

export const api = {
  async health(): Promise<HealthResponse> {
    const r = await fetch(`${BASE}/api/health`)
    return r.json()
  },

  async configAcquisition(cfg: AcquisitionConfig) {
    const r = await fetch(`${BASE}/api/acquisition/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    })
    return r.json()
  },

  async startSession() {
    const r = await fetch(`${BASE}/api/acquisition/start`, { method: 'POST' })
    return r.json()
  },

  async stopSession() {
    const r = await fetch(`${BASE}/api/acquisition/stop`, { method: 'POST' })
    return r.json()
  },

  async captureReference(): Promise<{
    status: string
    peak_wavelength: number | null
    peak_wavelengths: number[]
    n_peaks: number
    fwhm_nm: number | null
    error?: string
  }> {
    const r = await fetch(`${BASE}/api/acquisition/reference`, { method: 'POST' })
    return r.json()
  },

  async addCalibrationPoint(point: CalibrationPoint) {
    const r = await fetch(`${BASE}/api/calibration/add-point`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(point),
    })
    return r.json()
  },

  async suggestConcentration() {
    const r = await fetch(`${BASE}/api/calibration/suggest`, { method: 'POST' })
    return r.json()
  },

  async listSessions(): Promise<SessionMeta[]> {
    const r = await fetch(`${BASE}/api/sessions`)
    return r.json()
  },

  async getSession(sessionId: string): Promise<SessionDetail> {
    const r = await fetch(`${BASE}/api/sessions/${sessionId}`)
    return r.json()
  },

  async generateReport(sessionId: string): Promise<{ report: string; session_id: string }> {
    const r = await fetch(`${BASE}/api/reports/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    })
    if (!r.ok) {
      let detail = `HTTP ${r.status}`
      try { const body = await r.json(); detail = body.detail ?? detail } catch { /* ignore */ }
      throw new Error(detail)
    }
    return r.json()
  },

  async exportQualificationDossier(
    sessionId: string | null,
    artifact: 'json' | 'html' | 'both' = 'both',
  ): Promise<QualificationExportResponse> {
    const params = new URLSearchParams({ artifact })
    if (sessionId) params.set('session_id', sessionId)
    const r = await fetch(`${BASE}/api/qualification/dossier/export?${params.toString()}`, {
      method: 'POST',
    })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    return r.json()
  },

  async createResearchPackage(sessionId: string | null): Promise<QualificationPackageResponse> {
    const params = new URLSearchParams()
    if (sessionId) params.set('session_id', sessionId)
    const q = params.toString()
    const r = await fetch(`${BASE}/api/qualification/package${q ? `?${q}` : ''}`, {
      method: 'POST',
    })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    return r.json()
  },

  async setAutoExplain(enabled: boolean) {
    const r = await fetch(`${BASE}/api/agents/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ auto_explain: enabled }),
    })
    return r.json()
  },

  async setQualitySettings(settings: QualitySettings) {
    const r = await fetch(`${BASE}/api/agents/quality-settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    })
    return r.json()
  },

  async setDriftSettings(settings: DriftSettings) {
    const r = await fetch(`${BASE}/api/agents/drift-settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    })
    return r.json()
  },

  // ── Multi-analyte API ──────────────────────────────────────────────────

  async listAnalytes(): Promise<AnalyteInfo> {
    const r = await fetch(`${BASE}/api/analytes`)
    return r.json()
  },

  async inferMixture(payload: {
    delta_lambda: number[]
    analytes: string[]
    S_matrix: number[][]
    Kd_matrix?: number[][] | null
    use_nonlinear?: boolean
  }): Promise<MixtureInferenceResult> {
    const r = await fetch(`${BASE}/api/inference/mixture`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    return r.json()
  },

  async fitSensitivityMatrix(payload: {
    analytes: string[]
    n_peaks: number
    calibration_data: Array<{
      analyte: string
      peak_idx: number
      conc_ppm: number[]
      shifts_nm: number[]
    }>
  }) {
    const r = await fetch(`${BASE}/api/calibration/sensitivity-matrix/fit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    return r.json()
  },

  async generateSimulation(payload: {
    peak_nm?: number
    fwhm_nm?: number
    analyte_name?: string
    sensitivity_nm_per_ppm?: number
    tau_s?: number
    kd_ppm?: number
    concentrations?: number[]
    n_sessions?: number
    random_seed?: number
  }): Promise<{ status: string; analyte: string; calibration_summary: SimCalibrationSummaryRow[] }> {
    const r = await fetch(`${BASE}/api/simulation/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    return r.json()
  },

  /** Stream Claude answer via SSE; calls onChunk with each text chunk, resolves when done. */
  async ask(query: string, onChunk: (text: string) => void): Promise<void> {
    const r = await fetch(`${BASE}/api/agents/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    })
    const reader = r.body!.getReader()
    const dec = new TextDecoder()
    let buf = ''
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop()!
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const payload = JSON.parse(line.slice(6))
          if (payload.text) onChunk(payload.text)
          if (payload.done) return
        } catch (err) { console.warn('[sse/ask] malformed chunk:', err) }
      }
    }
  },
}

export function connectSpectrum(onFrame: (f: SpectrumFrame) => void): WebSocket {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${protocol}://${location.host}/ws/spectrum`)
  ws.onmessage = (e) => {
    try { onFrame(JSON.parse(e.data)) }
    catch (err) { console.warn('[ws/spectrum] malformed frame:', err) }
  }
  return ws
}

export function connectAgentEvents(onEvent: (e: AgentEvent) => void): WebSocket {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${protocol}://${location.host}/ws/agent-events`)
  ws.onmessage = (e) => {
    try { onEvent(JSON.parse(e.data)) }
    catch (err) { console.warn('[ws/agent-events] malformed event:', err) }
  }
  return ws
}
