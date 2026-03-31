// API client — all HTTP + WebSocket helpers

const BASE = ''  // same origin; vite dev proxy handles /api → :8765

export interface HealthResponse {
  status: string
  version: string
  hardware: string
  simulate: boolean
  physics_plugin: string
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

  async captureReference() {
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
    if (!r.ok) throw new Error(`Report generation failed: ${r.status}`)
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
        } catch {/* ignore malformed */}
      }
    }
  },
}

export function connectSpectrum(onFrame: (f: SpectrumFrame) => void): WebSocket {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${protocol}://${location.host}/ws/spectrum`)
  ws.onmessage = (e) => {
    try { onFrame(JSON.parse(e.data)) } catch {/* ignore */}
  }
  return ws
}

export function connectAgentEvents(onEvent: (e: AgentEvent) => void): WebSocket {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${protocol}://${location.host}/ws/agent-events`)
  ws.onmessage = (e) => {
    try { onEvent(JSON.parse(e.data)) } catch {/* ignore */}
  }
  return ws
}
