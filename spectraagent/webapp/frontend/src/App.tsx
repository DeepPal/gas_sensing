import { Suspense, lazy, useEffect, useRef, useState, useCallback, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import {
  Activity, Wifi, WifiOff, Play, Square, Camera, Plus,
  MessageSquare, ChevronDown, ChevronUp, Settings, Zap,
  AlertTriangle, Info, FlaskConical, FileText, History,
  X, TrendingUp, Sliders, CheckCircle2, Download, ClipboardList,
  Circle, CheckCircle, Loader,
} from 'lucide-react'
import { api, connectSpectrum, connectAgentEvents } from './api'
import type {
  SpectrumFrame, AgentEvent, SessionMeta, SessionDetail, HealthResponse,
  AnalyteInfo, MixtureInferenceResult, ResearchFlowResponse,
  QualificationDossierResponse, QualificationExportResponse, QualificationPackageResponse,
} from './api'
import './App.css'

const LazyReactMarkdown = lazy(() => import('react-markdown'))

// ─── Helpers ─────────────────────────────────────────────────────────────────

function levelIcon(level: string) {
  if (level === 'error') return <AlertTriangle size={13} />
  if (level === 'warn' || level === 'warning') return <AlertTriangle size={13} />
  if (level === 'info') return <Info size={13} />
  if (level === 'ok') return <Activity size={13} />
  if (level === 'claude') return <MessageSquare size={13} />
  return <Zap size={13} />
}

interface TrendPoint {
  frame: number
  shift: number
  conc?: number
  ciLow?: number
  ciHigh?: number
}

// Agent events stamped with a client-side ID so we don't use array index as key
type StampedEvent = AgentEvent & { _id: number }

interface SessionAnalysis {
  lod_ppm?: number
  loq_ppm?: number
  r_squared?: number
  drift_rate_nm_per_min?: number
  mean_snr?: number
  frame_count?: number
}

// ─── Modal overlay ─────────────────────────────────────────────────────────────

function Modal({ title, onClose, children }: {
  title: string; onClose: () => void; children: React.ReactNode
}) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <span className="modal-title">{title}</span>
          <button type="button" className="modal-close" onClick={onClose} title="Close">
            <X size={14} />
          </button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  )
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [wsConnected, setWsConnected] = useState(false)
  const [wsReconnecting, setWsReconnecting] = useState(false)
  const [dashboardAvailable, setDashboardAvailable] = useState(false)
  const [spectrum, setSpectrum] = useState<{ wl: number[]; i: number[] } | null>(null)
  const [frameNum, setFrameNum] = useState(0)
  const [latestResult, setLatestResult] = useState<Partial<SpectrumFrame>>({})
  const [trend, setTrend] = useState<TrendPoint[]>([])
  const [events, setEvents] = useState<StampedEvent[]>([])
  const [sessionRunning, setSessionRunning] = useState(false)
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [gasLabel, setGasLabel] = useState('Ethanol')
  const [targetConc, setTargetConc] = useState('')
  const [integrationMs, setIntegrationMs] = useState(50)

  // Calibration
  const [calConc, setCalConc] = useState('')
  const [calDelta, setCalDelta] = useState('')
  const [calPoints, setCalPoints] = useState<{ c: number; d: number }[]>([])
  const [suggestedConc, setSuggestedConc] = useState<number | null>(null)

  // Claude ask
  const [askQuery, setAskQuery] = useState('')
  const [askAnswer, setAskAnswer] = useState('')
  const [askStreaming, setAskStreaming] = useState(false)
  const [autoExplain, setAutoExplain] = useState(false)

  // Panel visibility
  const [showAsk, setShowAsk] = useState(false)
  const [showCal, setShowCal] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(true)

  // Quality / drift settings (initialised from health)
  const [satThreshold, setSatThreshold] = useState(60000)
  const [snrThreshold, setSnrThreshold] = useState(3.0)
  const [driftThreshold, setDriftThreshold] = useState(0.05)
  const [driftWindow, setDriftWindow] = useState(60)

  // Session history
  const [sessions, setSessions] = useState<SessionMeta[]>([])
  const [sessionDetail, setSessionDetail] = useState<SessionDetail | null>(null)

  // Session analysis (from session_complete event)
  const [sessionAnalysis, setSessionAnalysis] = useState<SessionAnalysis | null>(null)

  // Report modal
  const [reportContent, setReportContent] = useState<string | null>(null)
  const [reportGenerating, setReportGenerating] = useState(false)
  const [showReport, setShowReport] = useState(false)
  const [qualBusy, setQualBusy] = useState(false)
  const [qualNotice, setQualNotice] = useState<string | null>(null)
  const [researchFlow, setResearchFlow] = useState<ResearchFlowResponse | null>(null)
  const [qualificationDossier, setQualificationDossier] = useState<QualificationDossierResponse | null>(null)
  const [lastExport, setLastExport] = useState<QualificationExportResponse | null>(null)
  const [lastPackage, setLastPackage] = useState<QualificationPackageResponse | null>(null)

  // Anomaly / Claude explanation modal
  const [anomalyEvent, setAnomalyEvent] = useState<AgentEvent | null>(null)

  // Reference peak positions — discovered at runtime from reference capture.
  // Supports multi-peak sensors: one entry per detected spectral peak.
  const [refPeakWls, setRefPeakWls] = useState<number[]>([])

  // Planner suggestions from events
  const [plannerSuggestions, setPlannerSuggestions] = useState<string[]>([])

  // Multi-analyte sensor info (polled from /api/analytes)
  const [analyteInfo, setAnalyteInfo] = useState<AnalyteInfo | null>(null)
  // Latest mixture inference result
  const [mixtureResult, setMixtureResult] = useState<MixtureInferenceResult | null>(null)
  // Kinetic phase per peak: 'association' | 'equilibrium' | 'dissociation'
  const [kineticPhase, setKineticPhase] = useState<string>('waiting')

  const specWs = useRef<WebSocket | null>(null)
  const agentWs = useRef<WebSocket | null>(null)
  const eventsListRef = useRef<HTMLDivElement>(null)
  const userScrolledRef = useRef(false)
  const settingsSeededRef = useRef(false)
  const eventIdRef = useRef(0)
  const healthReachableRef = useRef(false)

  // ── Health poll (seeds settings sliders on first successful response only) ─
  useEffect(() => {
    const poll = async () => {
      try {
        const h = await api.health()
        setHealth(h)
        healthReachableRef.current = true
        if (!settingsSeededRef.current) {
          settingsSeededRef.current = true
          setSatThreshold(h.quality_settings.saturation_threshold)
          setSnrThreshold(h.quality_settings.snr_warn_threshold)
          setDriftThreshold(h.drift_settings.drift_threshold_nm_per_min)
          setDriftWindow(h.drift_settings.window_frames)
          if (h.integration_time_ms) setIntegrationMs(h.integration_time_ms)
        }
      } catch (err) {
        // Suppress during startup warmup; log if server was previously reachable
        if (healthReachableRef.current) console.warn('[health poll] server unreachable:', err)
      }
    }
    poll()
    const t = setInterval(poll, 10000)
    return () => clearInterval(t)
  }, [])

  // ── Readiness / qualification poll ──────────────────────────────────────
  useEffect(() => {
    const poll = async () => {
      try {
        const [flow, dossier] = await Promise.all([
          api.getResearchFlow(),
          api.getQualificationDossier(currentSessionId),
        ])
        setResearchFlow(flow)
        setQualificationDossier(dossier)
      } catch (err) {
        console.warn('[readiness poll]', err)
      }
    }
    poll()
    const t = setInterval(poll, 15000)
    return () => clearInterval(t)
  }, [currentSessionId])

  // ── Analyte info poll (once on mount, refreshed when session starts) ────
  useEffect(() => {
    const poll = async () => {
      try {
        const info = await api.listAnalytes()
        if (info.analytes.length > 0) setAnalyteInfo(info)
      } catch { /* suppress */ }
    }
    poll()
    const t = setInterval(poll, 30000)
    return () => clearInterval(t)
  }, [])

  // ── Dashboard availability check ─────────────────────────────────────────
  useEffect(() => {
    const checkDashboard = async () => {
      try {
        const resp = await fetch(`${location.protocol}//${location.hostname}:8501/_stcore/health`)
        setDashboardAvailable(resp.ok)
      } catch {
        setDashboardAvailable(false)
      }
    }
    checkDashboard()
    const t = setInterval(checkDashboard, 10000) // Check every 10 seconds
    return () => clearInterval(t)
  }, [])

  // ── Mixture inference — run when latestResult has multi-peak shifts ──────
  useEffect(() => {
    if (!analyteInfo || !analyteInfo.S_matrix || analyteInfo.analytes.length < 2) return
    // Collect peak shifts from latestResult if available
    const shift = latestResult.peak_shift_nm
    if (shift === undefined) return
    // Build shift vector (single shift for now; extend when multi-peak WS is added)
    const deltaLambda = [shift]
    if (deltaLambda.length !== analyteInfo.n_peaks) return
    api.inferMixture({
      delta_lambda: deltaLambda,
      analytes: analyteInfo.analytes,
      S_matrix: analyteInfo.S_matrix,
      use_nonlinear: false,
    }).then(r => setMixtureResult(r)).catch(() => {/* suppress */})
  }, [latestResult.peak_shift_nm, analyteInfo])

  // ── Kinetic phase from trend data ────────────────────────────────────────
  useEffect(() => {
    if (trend.length < 3) { setKineticPhase('waiting'); return }
    const recent = trend.slice(-10)
    const shifts = recent.map((d: TrendPoint) => Math.abs(d.shift ?? 0))
    if (shifts.every((s: number) => s < 0.02)) { setKineticPhase('baseline'); return }
    const last3 = shifts.slice(-3)
    const delta = last3[2] - last3[0]
    if (delta > 0.005) setKineticPhase('association')
    else if (delta < -0.005) setKineticPhase('dissociation')
    else setKineticPhase('equilibrium')
  }, [trend])

  // ── Spectrum WebSocket ───────────────────────────────────────────────────
  useEffect(() => {
    let unmounted = false
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    const connect = () => {
      const ws = connectSpectrum((frame) => {
        if (unmounted) return
        setSpectrum({ wl: frame.wl, i: frame.i })
        setFrameNum(frame.frame)
        setLatestResult(prev => ({
          concentration_ppm: frame.concentration_ppm,
          ci_low: frame.ci_low,
          ci_high: frame.ci_high,
          peak_shift_nm: frame.peak_shift_nm,
          peak_wavelength: frame.peak_wavelength,
          snr: frame.snr,
          // Keep last known classification result (not every frame has it)
          gas_type: frame.gas_type ?? prev.gas_type,
          confidence_score: frame.confidence_score ?? prev.confidence_score,
        }))
        if (frame.peak_shift_nm !== undefined || frame.concentration_ppm !== undefined) {
          setTrend(prev => [...prev, {
            frame: frame.frame,
            shift: frame.peak_shift_nm ?? 0,
            conc: frame.concentration_ppm,
            ciLow: frame.ci_low,
            ciHigh: frame.ci_high,
          }].slice(-300))
        }
        setWsConnected(true)
        setWsReconnecting(false)
      })
      ws.onclose = () => {
        if (unmounted) return
        setWsConnected(false)
        setWsReconnecting(true)
        reconnectTimer = setTimeout(connect, 2000)
      }
      ws.onerror = () => ws.close()
      specWs.current = ws
    }
    connect()
    return () => {
      unmounted = true
      if (reconnectTimer !== null) clearTimeout(reconnectTimer)
      specWs.current?.close()
    }
  }, [])

  // ── Agent events WebSocket ───────────────────────────────────────────────
  useEffect(() => {
    let unmounted = false
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    const connect = () => {
      const ws = connectAgentEvents((ev) => {
        if (unmounted) return
        setEvents(prev => [{ ...ev, _id: ++eventIdRef.current }, ...prev].slice(0, 100))

        // Session complete → extract analysis metrics
        if (ev.type === 'session_complete' && ev.data) {
          const d = ev.data as Record<string, unknown>
          setSessionAnalysis({
            lod_ppm: d.lod_ppm as number | undefined,
            loq_ppm: d.loq_ppm as number | undefined,
            r_squared: d.r_squared as number | undefined,
            drift_rate_nm_per_min: d.drift_rate_nm_per_min as number | undefined,
            mean_snr: d.mean_snr as number | undefined,
            frame_count: d.frame_count as number | undefined,
          })
        }

        // Claude narrative / anomaly events → surface as modal
        if (
          (ev.level === 'claude' ||
            ev.source.toLowerCase().includes('explainer') ||
            ev.source.toLowerCase().includes('narrator')) &&
          ev.text.length > 80
        ) {
          setAnomalyEvent(ev)
        }

        // Experiment planner suggestions
        if (ev.type === 'experiment_suggestion' || ev.type === 'suggestion') {
          setPlannerSuggestions(prev => [ev.text, ...prev].slice(0, 5))
        }
      })
      ws.onclose = () => {
        if (unmounted) return
        setWsReconnecting(true)
        reconnectTimer = setTimeout(connect, 2000)
      }
      ws.onerror = () => ws.close()
      agentWs.current = ws
    }
    connect()
    return () => {
      unmounted = true
      if (reconnectTimer !== null) clearTimeout(reconnectTimer)
      agentWs.current?.close()
    }
  }, [])

  // Auto-scroll events list to top when new events arrive, unless user has manually scrolled down
  useEffect(() => {
    const el = eventsListRef.current
    if (!el || userScrolledRef.current) return
    el.scrollTop = 0
  }, [events])

  // ── Shared error helper ───────────────────────────────────────────────────
  const pushError = useCallback((source: string, err: unknown) => {
    setEvents(prev => [{
      _id: ++eventIdRef.current,
      source, level: 'error', type: 'ui_error',
      text: err instanceof Error ? err.message : String(err),
    }, ...prev].slice(0, 100))
  }, [])

  // ── Session controls ─────────────────────────────────────────────────────
  const startSession = async () => {
    try {
      await api.configAcquisition({
        integration_time_ms: integrationMs,
        gas_label: gasLabel,
        target_concentration: targetConc ? parseFloat(targetConc) : null,
      })
      const r = await api.startSession()
      setSessionRunning(true)
      setTrend([])
      setSessionAnalysis(null)
      if (r.session_id) setCurrentSessionId(r.session_id as string)
    } catch (err) {
      pushError('UI/Session', err)
    }
  }

  const stopSession = async () => {
    try {
      await api.stopSession()
      setSessionRunning(false)
      const list = await api.listSessions()
      setSessions(list)
    } catch (err) {
      setSessionRunning(false)
      pushError('UI/Session', err)
    }
  }

  const captureRef = async () => {
    try {
      const r = await api.captureReference()
      // Store all detected peak positions — works for single and multi-peak sensors
      if (r.peak_wavelengths?.length) {
        setRefPeakWls(r.peak_wavelengths)
      } else if (r.peak_wavelength != null) {
        setRefPeakWls([r.peak_wavelength])
      }
      const peakInfo = r.peak_wavelengths?.length
        ? `${r.peak_wavelengths.length} peak(s) at ${r.peak_wavelengths.map((w: number) => w.toFixed(2)).join(', ')} nm`
        : r.peak_wavelength != null ? `peak at ${r.peak_wavelength.toFixed(2)} nm` : 'no peak detected'
      setEvents(prev => [{
        _id: ++eventIdRef.current,
        source: 'UI', level: 'info', type: 'reference_captured',
        text: r.error ?? `Reference captured — ${peakInfo}`,
      }, ...prev])
    } catch (err) {
      pushError('UI/Reference', err)
    }
  }

  // ── Session history ──────────────────────────────────────────────────────
  const loadSessions = async () => {
    try { setSessions(await api.listSessions()) }
    catch (err) { console.warn('[sessions/list]', err) }
  }

  const openSession = async (id: string) => {
    try { setSessionDetail(await api.getSession(id)) }
    catch (err) { console.warn('[sessions/get]', err) }
  }

  // ── Report generation ────────────────────────────────────────────────────
  const generateReport = async () => {
    if (!currentSessionId) return
    setReportGenerating(true)
    try {
      const result = await api.generateReport(currentSessionId)
      setReportContent(result.report)
      setShowReport(true)
    } catch (err) {
      setEvents(prev => [{
        _id: ++eventIdRef.current,
        source: 'UI', level: 'error', type: 'report_error',
        text: String(err),
      }, ...prev])
    } finally {
      setReportGenerating(false)
    }
  }

  const exportDossier = async () => {
    setQualBusy(true)
    setQualNotice(null)
    try {
      const r = await api.exportQualificationDossier(currentSessionId, 'both')
      setLastExport(r)
      const signed = r.signature?.signed ? 'signed' : 'unsigned'
      setQualNotice(`Dossier exported (${signed}). JSON: ${r.paths.json ?? 'n/a'}`)
      setEvents(prev => [{
        _id: ++eventIdRef.current,
        source: 'UI', level: 'info', type: 'qualification_exported',
        text: `Qualification dossier exported (${signed}) for ${r.session_id}`,
      }, ...prev])
    } catch (err) {
      setQualNotice(`Export failed: ${String(err)}`)
      pushError('UI/QualificationExport', err)
    } finally {
      setQualBusy(false)
    }
  }

  const buildResearchPackage = async () => {
    setQualBusy(true)
    setQualNotice(null)
    try {
      const r = await api.createResearchPackage(currentSessionId)
      setLastPackage(r)
      const signed = r.signature?.signed ? 'signed' : 'unsigned'
      setQualNotice(`Research package created (${signed}): ${r.package_path}`)
      setEvents(prev => [{
        _id: ++eventIdRef.current,
        source: 'UI', level: 'info', type: 'research_package_created',
        text: `Research package created with ${r.included.length} artifacts`,
      }, ...prev])
    } catch (err) {
      setQualNotice(`Package build failed: ${String(err)}`)
      pushError('UI/QualificationPackage', err)
    } finally {
      setQualBusy(false)
    }
  }

  // ── Settings ─────────────────────────────────────────────────────────────
  const saveQualitySettings = async () => {
    try {
      await api.setQualitySettings({ saturation_threshold: satThreshold, snr_warn_threshold: snrThreshold })
      setEvents(prev => [{
        _id: ++eventIdRef.current,
        source: 'UI', level: 'info', type: 'settings_updated',
        text: `Quality settings updated: saturation=${satThreshold}, SNR warn=${snrThreshold}`,
      }, ...prev])
    } catch (err) { pushError('UI/Settings', err) }
  }

  const saveDriftSettings = async () => {
    try {
      await api.setDriftSettings({ drift_threshold_nm_per_min: driftThreshold, window_frames: driftWindow })
      setEvents(prev => [{
        _id: ++eventIdRef.current,
        source: 'UI', level: 'info', type: 'settings_updated',
        text: `Drift settings updated: threshold=${driftThreshold} nm/min, window=${driftWindow} frames`,
      }, ...prev])
    } catch (err) { pushError('UI/Settings', err) }
  }

  // ── Calibration ──────────────────────────────────────────────────────────
  const addCalPoint = async () => {
    if (!calConc || !calDelta) return
    const c = parseFloat(calConc), d = parseFloat(calDelta)
    try {
      await api.addCalibrationPoint({ concentration: c, delta_lambda: d })
      setCalPoints(prev => [...prev, { c, d }])
      setCalConc('')
      setCalDelta('')
    } catch (err) { pushError('UI/Calibration', err) }
  }

  const suggestNext = async () => {
    try {
      const r = await api.suggestConcentration()
      setSuggestedConc(r.suggestion ?? null)
    } catch (err) { pushError('UI/Calibration', err) }
  }

  // ── Claude ask ───────────────────────────────────────────────────────────
  const submitAsk = useCallback(async () => {
    if (!askQuery.trim() || askStreaming) return
    setAskAnswer('')
    setAskStreaming(true)
    try {
      await api.ask(askQuery, (chunk) => setAskAnswer(prev => prev + chunk))
    } catch (err) {
      setAskAnswer(`Error: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setAskStreaming(false)
    }
  }, [askQuery, askStreaming])

  const toggleAutoExplain = async () => {
    const next = !autoExplain
    setAutoExplain(next)
    try {
      await api.setAutoExplain(next)
    } catch (err) {
      setAutoExplain(!next) // revert optimistic update
      pushError('UI/Settings', err)
    }
  }

  // ── Chart data ───────────────────────────────────────────────────────────
  const specData = useMemo(() =>
    spectrum
      ? spectrum.wl
          .map((w, i) => ({ wl: parseFloat(w.toFixed(1)), intensity: parseFloat(spectrum.i[i].toFixed(4)) }))
          .filter((_, i) => i % 4 === 0)
      : [],
    [spectrum]
  )

  const hasCIBands = useMemo(() => trend.some(t => t.ciLow !== undefined), [trend])
  const qualificationChecks = qualificationDossier?.checks ?? []
  const passedQualificationChecks = qualificationChecks.filter(check => check.pass).length
  const failedCriticalChecks = qualificationChecks.filter(check => check.critical && !check.pass)
  const reproducibility = qualificationDossier?.reproducibility
  const latestArtifactCount =
    (lastExport?.paths ? Object.keys(lastExport.paths).length : 0) +
    (lastPackage?.package_path ? 1 : 0)

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <FlaskConical size={22} className="logo-icon" />
          <span className="logo-text">SpectraAgent</span>
          {health && <span className="version-badge">v{health.version}</span>}
          {health && <span className="physics-badge">{health.physics_plugin}</span>}
        </div>
        <div className="header-right">
          {health && (
            <span className={`hw-badge ${(health.simulate || health.hardware === 'Simulation') ? 'sim' : 'live'}`}>
              {(health.simulate || health.hardware === 'Simulation') ? '⚡ Simulation' : `🔬 ${health.hardware}`}
            </span>
          )}
          <a
            href={dashboardAvailable ? `${location.protocol}//${location.hostname}:8501` : undefined}
            target="_blank"
            rel="noopener noreferrer"
            className={`workbench-link ${dashboardAvailable ? '' : 'disabled'}`}
            title={dashboardAvailable
              ? "Open Analysis Workbench — GPR/PLS training, LOD/LOQ, publication figures"
              : "Analysis Workbench not available — start with 'spectraagent start-all --simulate'"
            }
            onClick={!dashboardAvailable ? (e) => { e.preventDefault(); alert('Analysis Workbench is not running. Start it with: spectraagent start-all --simulate'); } : undefined}
          >
            <FlaskConical size={13} />
            Analysis Workbench {!dashboardAvailable && '(Offline)'}
          </a>
          <span className={`ws-badge ${wsConnected ? 'on' : wsReconnecting ? 'reconnecting' : 'off'}`}>
            {wsConnected ? <Wifi size={13} /> : <WifiOff size={13} />}
            {wsConnected ? 'Live' : wsReconnecting ? 'Reconnecting…' : 'Connecting…'}
          </span>
        </div>
      </header>

      <div className="layout">
        {/* ── Sidebar ── */}
        <aside className="sidebar">

          {/* Session config */}
          <section className="card">
            <h2><Settings size={15} /> Session</h2>
            <label>Gas / analyte
              <input value={gasLabel} onChange={e => setGasLabel(e.target.value)} />
            </label>
            <label>Target concentration (ppm)
              <input type="number" placeholder="optional" value={targetConc}
                onChange={e => setTargetConc(e.target.value)} />
            </label>
            <label>Integration time (ms)
              <input type="number" min={10} max={10000} value={integrationMs}
                onChange={e => setIntegrationMs(Number(e.target.value))} />
            </label>
            <div className="btn-row">
              {!sessionRunning
                ? <button type="button" className="btn-primary" onClick={startSession}><Play size={14} />Start</button>
                : <button type="button" className="btn-danger" onClick={stopSession}><Square size={14} />Stop</button>}
              <button type="button" className="btn-secondary" onClick={captureRef}><Camera size={14} />Reference</button>
            </div>
            {sessionRunning && (
              <div className="session-pill">
                <span className="pulse" /> Recording · frame {frameNum}
              </div>
            )}
            {!sessionRunning && currentSessionId && (
              <div className="post-session-actions">
                <button
                  type="button"
                  className="btn-secondary full-width"
                  onClick={generateReport}
                  disabled={reportGenerating}
                >
                  <FileText size={13} />
                  {reportGenerating ? 'Generating…' : 'Generate Report'}
                </button>
                <div className="session-id-badge">Qualification Center unlocked for this session</div>
                {qualNotice && <div className="session-id-badge">{qualNotice}</div>}
                <div className="session-id-badge">ID: {currentSessionId.slice(0, 18)}…</div>
              </div>
            )}
          </section>

          {/* Session analysis (populated from session_complete agent event) */}
          {sessionAnalysis && (
            <section className="card analysis-card">
              <h2><TrendingUp size={15} /> Session Analysis</h2>
              <div className="analysis-grid">
                {sessionAnalysis.lod_ppm !== undefined && (
                  <div className="an-metric">
                    <span className="an-label">LOD</span>
                    <span className="an-val">{sessionAnalysis.lod_ppm.toExponential(2)} ppm</span>
                  </div>
                )}
                {sessionAnalysis.loq_ppm !== undefined && (
                  <div className="an-metric">
                    <span className="an-label">LOQ</span>
                    <span className="an-val">{sessionAnalysis.loq_ppm.toExponential(2)} ppm</span>
                  </div>
                )}
                {sessionAnalysis.r_squared !== undefined && (
                  <div className="an-metric">
                    <span className="an-label">R²</span>
                    <span className={`an-val ${sessionAnalysis.r_squared < 0.9 ? 'warn' : 'good'}`}>
                      {sessionAnalysis.r_squared.toFixed(4)}
                    </span>
                  </div>
                )}
                {sessionAnalysis.mean_snr !== undefined && (
                  <div className="an-metric">
                    <span className="an-label">Mean SNR</span>
                    <span className="an-val">{sessionAnalysis.mean_snr.toFixed(1)}</span>
                  </div>
                )}
                {sessionAnalysis.drift_rate_nm_per_min !== undefined && (
                  <div className="an-metric an-full">
                    <span className="an-label">Drift rate</span>
                    <span className={`an-val ${Math.abs(sessionAnalysis.drift_rate_nm_per_min) > 0.05 ? 'warn' : 'good'}`}>
                      {sessionAnalysis.drift_rate_nm_per_min.toFixed(4)} nm/min
                    </span>
                  </div>
                )}
              </div>
            </section>
          )}

          {/* Live result */}
          {(latestResult.concentration_ppm !== undefined || latestResult.peak_shift_nm !== undefined || latestResult.peak_wavelength !== undefined) && (
            <section className="card result-card">
              <h2><Activity size={15} /> Live result</h2>
              {(latestResult.gas_type || latestResult.confidence_score !== undefined) && (
                <div className="gas-badge-row">
                  {latestResult.gas_type && (
                    <span className="gas-badge">{latestResult.gas_type}</span>
                  )}
                  {latestResult.confidence_score !== undefined && (
                    <span className="conf-badge">
                      {(latestResult.confidence_score * 100).toFixed(0)}% conf
                    </span>
                  )}
                </div>
              )}
              {latestResult.concentration_ppm !== undefined && (
                <div className="metric">
                  <span className="mlabel">Concentration</span>
                  <span className="mval">{latestResult.concentration_ppm.toFixed(3)} ppm</span>
                  {latestResult.ci_low !== undefined && (
                    <span className="mci">
                      95% CI [{latestResult.ci_low.toFixed(3)}, {latestResult.ci_high!.toFixed(3)}]
                    </span>
                  )}
                </div>
              )}
              {latestResult.peak_shift_nm !== undefined && (
                <div className="metric">
                  <span className="mlabel">Δλ peak shift</span>
                  <span className="mval">{latestResult.peak_shift_nm.toFixed(4)} nm</span>
                </div>
              )}
              {latestResult.peak_wavelength !== undefined && (
                <div className="metric">
                  <span className="mlabel">Peak λ</span>
                  <span className="mval-sm">{latestResult.peak_wavelength.toFixed(3)} nm</span>
                </div>
              )}
              {latestResult.snr !== undefined && (
                <div className="metric">
                  <span className="mlabel">SNR</span>
                  <span className={`mval-sm ${latestResult.snr < 3 ? 'warn' : ''}`}>
                    {latestResult.snr.toFixed(1)}
                  </span>
                </div>
              )}
            </section>
          )}

          {/* Multi-analyte panel */}
          {analyteInfo && analyteInfo.analytes.length > 0 && (
            <section className="card">
              <h2><FlaskConical size={15} /> Multi-Analyte</h2>
              <div className="multi-analyte-header">
                <span className="phase-badge" data-phase={kineticPhase}>{kineticPhase}</span>
                {analyteInfo.n_peaks > 1 && (
                  <span className="peak-badge">{analyteInfo.n_peaks} peaks</span>
                )}
              </div>
              {/* Concentration bars per analyte */}
              <div className="analyte-bars">
                {mixtureResult
                  ? analyteInfo.analytes.map(name => {
                      const c = mixtureResult.concentrations_ppm[name] ?? 0
                      const maxBar = 5.0
                      const pct = Math.min(100, (c / maxBar) * 100)
                      return (
                        <div key={name} className="analyte-row">
                          <span className="analyte-name">{name}</span>
                          <div className="analyte-bar-bg">
                            <div className="analyte-bar-fill" {...{ style: { '--bar-pct': `${pct}%` } as React.CSSProperties }} />
                          </div>
                          <span className="analyte-conc">{c.toFixed(3)} ppm</span>
                        </div>
                      )
                    })
                  : analyteInfo.analytes.map(name => (
                      <div key={name} className="analyte-row">
                        <span className="analyte-name">{name}</span>
                        <div className="analyte-bar-bg">
                          <div className="analyte-bar-fill" {...{ style: { '--bar-pct': '0%' } as React.CSSProperties }} />
                        </div>
                        <span className="analyte-conc">— ppm</span>
                      </div>
                    ))
                }
              </div>
              {/* S-matrix heatmap (text representation) */}
              {analyteInfo.S_matrix && (
                <details className="s-matrix-details">
                  <summary>Sensitivity matrix S</summary>
                  <table className="s-matrix-table">
                    <thead>
                      <tr>
                        <th>Analyte \ Peak</th>
                        {analyteInfo.peak_wavelengths_nm.map((wl, j) => (
                          <th key={j}>{wl.toFixed(0)} nm</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {analyteInfo.analytes.map((name, i) => (
                        <tr key={name}>
                          <td>{name}</td>
                          {(analyteInfo.S_matrix![i] ?? []).map((v, j) => (
                            <td key={j} className={v < 0 ? 'shift-neg' : 'shift-pos'}>
                              {v.toFixed(4)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </details>
              )}
              {mixtureResult && (
                <div className="residual-row">
                  <span className="mlabel">Fit residual</span>
                  <span className="mval-sm">{mixtureResult.residual_nm.toFixed(4)} nm</span>
                  <span className="solver-badge">{mixtureResult.solver}</span>
                </div>
              )}
            </section>
          )}

          {/* Planner suggestions */}
          {plannerSuggestions.length > 0 && (
            <section className="card">
              <button type="button" className="collapse-hdr" onClick={() => setShowSuggestions(v => !v)}>
                <h2><Zap size={15} /> Planner Suggestions</h2>
                {showSuggestions ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>
              {showSuggestions && (
                <div className="panel-body">
                  {plannerSuggestions.map((s, i) => (
                    <div key={i} className="suggestion-item">{s}</div>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* Calibration */}
          <section className="card">
            <button type="button" className="collapse-hdr" onClick={() => setShowCal(v => !v)}>
              <h2><Plus size={15} /> Calibration</h2>
              {showCal ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            {showCal && (
              <div className="panel-body">
                <label>Concentration (ppm)
                  <input type="number" value={calConc} onChange={e => setCalConc(e.target.value)} />
                </label>
                <label>Δλ (nm)
                  <input type="number" step="0.001" value={calDelta} onChange={e => setCalDelta(e.target.value)} />
                </label>
                <div className="btn-row">
                  <button type="button" className="btn-secondary" onClick={addCalPoint}>
                    <Plus size={13} />Add
                  </button>
                  <button type="button" className="btn-secondary" onClick={suggestNext}>
                    Suggest next
                  </button>
                </div>
                {suggestedConc !== null && (
                  <div className="suggestion">Next: <strong>{suggestedConc.toFixed(2)} ppm</strong></div>
                )}
                {calPoints.length > 0 && (
                  <div className="cal-chips">
                    {calPoints.map((p, i) => (
                      <span key={i} className="chip">{p.c} ppm / {p.d} nm</span>
                    ))}
                  </div>
                )}
                {/* Mini calibration curve */}
                {calPoints.length >= 2 && (
                  <div className="cal-curve-mini">
                    <div className="cal-curve-label">Calibration curve (Δλ vs [C])</div>
                    <ResponsiveContainer width="100%" height={100}>
                      <LineChart
                        data={[...calPoints].sort((a, b) => a.c - b.c).map(p => ({ c: p.c, d: p.d }))}
                        margin={{ top: 4, right: 4, bottom: 18, left: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="c" tick={{ fontSize: 9, fill: '#64748b' }}
                          label={{ value: 'ppm', position: 'insideBottom', offset: -12, fill: '#64748b', fontSize: 9 }} />
                        <YAxis tick={{ fontSize: 9, fill: '#64748b' }} width={32} />
                        <Line type="monotone" dataKey="d" stroke="#f59e0b"
                          dot={{ r: 3, fill: '#f59e0b' }} strokeWidth={1.5}
                          isAnimationActive={false} name="Δλ (nm)" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            )}
          </section>

          {/* Ask Claude */}
          <section className="card">
            <button type="button" className="collapse-hdr" onClick={() => setShowAsk(v => !v)}>
              <h2><MessageSquare size={15} /> Ask Claude</h2>
              {showAsk ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            {showAsk && (
              <div className="panel-body">
                <label className="toggle-row">
                  Auto-explain anomalies
                  <input type="checkbox" checked={autoExplain} onChange={toggleAutoExplain} />
                </label>
                <textarea rows={3} className="ask-ta"
                  placeholder="Ask about drift, LOD, anomalies… (Ctrl+Enter)"
                  value={askQuery}
                  onChange={e => setAskQuery(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && e.ctrlKey) submitAsk() }}
                />
                <button type="button" className="btn-primary" onClick={submitAsk} disabled={askStreaming}>
                  {askStreaming ? 'Streaming…' : 'Ask'}
                </button>
                {askAnswer && (
                  <div className="ask-answer">
                    <Suspense fallback={<div>Loading formatted response…</div>}>
                      <LazyReactMarkdown>{askAnswer}</LazyReactMarkdown>
                    </Suspense>
                    {askStreaming && <span className="blink">▍</span>}
                  </div>
                )}
              </div>
            )}
          </section>

          {/* Detection settings */}
          <section className="card">
            <button type="button" className="collapse-hdr" onClick={() => setShowSettings(v => !v)}>
              <h2><Sliders size={15} /> Detection Settings</h2>
              {showSettings ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            {showSettings && (
              <div className="panel-body">
                <div className="settings-group-label">Quality Gate</div>
                <label>Saturation threshold (counts)
                  <input type="number" value={satThreshold}
                    onChange={e => setSatThreshold(Number(e.target.value))} />
                </label>
                <label>SNR warn threshold
                  <input type="number" step="0.1" value={snrThreshold}
                    onChange={e => setSnrThreshold(Number(e.target.value))} />
                </label>
                <button type="button" className="btn-secondary btn-mb"
                  onClick={saveQualitySettings}>
                  Apply quality settings
                </button>
                <div className="settings-group-label">Drift Monitor</div>
                <label>Drift threshold (nm/min)
                  <input type="number" step="0.001" value={driftThreshold}
                    onChange={e => setDriftThreshold(Number(e.target.value))} />
                </label>
                <label>Window (frames)
                  <input type="number" min={10} value={driftWindow}
                    onChange={e => setDriftWindow(Number(e.target.value))} />
                </label>
                <button type="button" className="btn-secondary" onClick={saveDriftSettings}>
                  Apply drift settings
                </button>
              </div>
            )}
          </section>

          {/* Session history */}
          <section className="card">
            <button
              type="button"
              className="collapse-hdr"
              onClick={() => {
                const next = !showHistory
                setShowHistory(next)
                if (next) loadSessions()
              }}
            >
              <h2><History size={15} /> Session History</h2>
              {showHistory ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            {showHistory && (
              <div className="panel-body">
                {sessions.length === 0 && (
                  <div className="ev-empty">No past sessions found.</div>
                )}
                {sessions.map(s => (
                  <div
                    key={s.session_id}
                    className="session-row"
                    onClick={() => openSession(s.session_id)}
                  >
                    <div className="ses-id">{s.session_id.slice(0, 14)}…</div>
                    <div className="ses-meta">{s.gas_label} · {s.frame_count} frames</div>
                    <div className="ses-date">{new Date(s.started_at).toLocaleString()}</div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </aside>

        {/* ── Main ── */}
        <main className="main">

          {/* Live Spectrum */}
          <section className="card chart-card">
            <h2><Activity size={15} /> Live Spectrum · frame {frameNum}</h2>
            {specData.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={specData} margin={{ top: 4, right: 16, bottom: 20, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="wl" domain={['dataMin', 'dataMax']} tickCount={8}
                    label={{ value: 'Wavelength (nm)', position: 'insideBottom', offset: -12, fill: '#64748b', fontSize: 11 }}
                    tick={{ fontSize: 10, fill: '#64748b' }} />
                  <YAxis width={50} tick={{ fontSize: 10, fill: '#64748b' }} />
                  <Tooltip
                    contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 6, fontSize: 11 }}
                    formatter={(v) => [Number(v).toFixed(4), 'Intensity']}
                    labelFormatter={(l) => `λ = ${l} nm`}
                  />
                  {refPeakWls.map((wl, i) => (
                    <ReferenceLine key={wl} x={wl} stroke="#f59e0b" strokeDasharray="4 2"
                      label={{ value: refPeakWls.length === 1 ? 'λ_ref' : `λ_ref${i + 1}`, fill: '#f59e0b', fontSize: 10, position: 'top' }} />
                  ))}
                  <Line type="monotone" dataKey="intensity" stroke="#38bdf8"
                    dot={false} strokeWidth={1.5} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="chart-empty">Waiting for first spectrum frame…</div>
            )}
          </section>

          {/* Concentration + Shift Trend (with CI bands) */}
          <section className="card chart-card">
            <h2>
              <Activity size={15} /> Concentration &amp; Shift Trend
              {hasCIBands && <span className="chart-legend-badge ci">± 95% CI</span>}
            </h2>
            {trend.length > 1 ? (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={trend} margin={{ top: 4, right: 16, bottom: 20, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="frame"
                    label={{ value: 'Frame', position: 'insideBottom', offset: -12, fill: '#64748b', fontSize: 11 }}
                    tick={{ fontSize: 10, fill: '#64748b' }} />
                  <YAxis width={56} tick={{ fontSize: 10, fill: '#64748b' }}
                    label={{ value: 'ppm / Δλ nm', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 6, fontSize: 11 }}
                  />
                  {/* CI band — dashed upper/lower bounds */}
                  {hasCIBands && (
                    <>
                      <Line type="monotone" dataKey="ciHigh" stroke="#22c55e"
                        strokeDasharray="3 2" strokeWidth={1} dot={false}
                        opacity={0.45} name="CI high" isAnimationActive={false} />
                      <Line type="monotone" dataKey="ciLow" stroke="#22c55e"
                        strokeDasharray="3 2" strokeWidth={1} dot={false}
                        opacity={0.45} name="CI low" isAnimationActive={false} />
                    </>
                  )}
                  {trend.some(t => t.conc !== undefined) && (
                    <Line type="monotone" dataKey="conc" stroke="#22c55e"
                      dot={false} strokeWidth={2} name="Concentration (ppm)" isAnimationActive={false} />
                  )}
                  <Line type="monotone" dataKey="shift" stroke="#a78bfa"
                    dot={false} strokeWidth={1.5} name="Δλ (nm)" isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="chart-empty">Start a session to see the concentration trend…</div>
            )}
          </section>

          {/* ── Measurement Protocol Stepper ── */}
          {(() => {
            const workflowSteps: { id: string; label: string; hint: string; done: boolean; active: boolean }[] = [
              {
                id: 'baseline',
                label: 'Hardware ready',
                hint: 'Spectrometer connected & health OK',
                done: health?.status === 'ok',
                active: health === null || health.status !== 'ok',
              },
              {
                id: 'reference',
                label: 'Capture reference spectrum',
                hint: 'Press Reference to record λ_ref baseline',
                done: refPeakWls.length > 0,
                active: health?.status === 'ok' && refPeakWls.length === 0,
              },
              {
                id: 'calibrate',
                label: 'Add ≥3 calibration points',
                hint: 'Cover low / mid / high concentration range',
                done: calPoints.length >= 3,
                active: refPeakWls.length > 0 && calPoints.length < 3,
              },
              {
                id: 'session',
                label: 'Run measurement session',
                hint: 'Click Start, expose sensor to analyte',
                done: !!currentSessionId,
                active: calPoints.length >= 3 && !currentSessionId,
              },
              {
                id: 'analyze',
                label: 'Wait for session analysis',
                hint: 'Stop session — AI pipeline auto-runs',
                done: sessionAnalysis !== null,
                active: sessionRunning || (!!currentSessionId && sessionAnalysis === null),
              },
              {
                id: 'export',
                label: 'Export qualification artifacts',
                hint: 'Download dossier, export JSON package',
                done: !!(lastExport || lastPackage),
                active: sessionAnalysis !== null && !(lastExport || lastPackage),
              },
            ]
            const allDone = workflowSteps.every(s => s.done)
            const currentIdx = workflowSteps.findIndex(s => s.active)
            return (
              <section className="card workflow-card">
                <h2><ClipboardList size={15} /> Measurement Protocol</h2>
                <div className="workflow-stepper">
                  {workflowSteps.map((step, i) => {
                    const state = step.done ? 'done' : step.active ? 'active' : 'pending'
                    return (
                      <div key={step.id} className={`workflow-step ${state}`}>
                        <span className="workflow-step-num">
                          {step.done
                            ? <CheckCircle size={15} />
                            : step.active
                              ? <Loader size={15} className="spin" />
                              : <Circle size={15} />}
                        </span>
                        <div className="workflow-step-body">
                          <span className="workflow-step-label">{step.label}</span>
                          {state !== 'done' && (
                            <span className="workflow-step-hint">{step.hint}</span>
                          )}
                        </div>
                        <span className="workflow-step-index">{i + 1}/{workflowSteps.length}</span>
                      </div>
                    )
                  })}
                </div>
                {allDone && (
                  <div className="workflow-complete-banner">
                    <CheckCircle2 size={14} /> Full measurement cycle complete — ready for publication!
                  </div>
                )}
                {currentIdx >= 0 && !allDone && (
                  <div className="workflow-progress-bar">
                    <div
                      className="workflow-progress-fill"
                      style={{ width: `${Math.round((currentIdx / workflowSteps.length) * 100)}%` }}
                    />
                  </div>
                )}
              </section>
            )
          })()}

          {/* Session detail inline panel (shown when a history session is clicked) */}
          {sessionDetail && (
            <section className="card">
              <div className="session-detail-header">
                <h2 className="session-detail-title">
                  <History size={15} /> {sessionDetail.gas_label} — {sessionDetail.session_id.slice(0, 16)}…
                </h2>
                <button type="button" className="icon-btn" onClick={() => setSessionDetail(null)} title="Close session detail">
                  <X size={14} />
                </button>
              </div>
              <div className="session-detail-meta">
                <span>{sessionDetail.frame_count} frames</span>
                <span>{new Date(sessionDetail.started_at).toLocaleString()}</span>
                {sessionDetail.stopped_at && (
                  <span>→ {new Date(sessionDetail.stopped_at).toLocaleString()}</span>
                )}
              </div>
              <div className="events-list session-detail-events">
                {sessionDetail.agent_events.slice(0, 30).map((ev, i) => (
                  <div key={i} className={`ev-row ev-${ev.level}`}>
                    <span className="ev-icon">{levelIcon(ev.level)}</span>
                    <div className="ev-body">
                      <span className="ev-source">{ev.source}</span>
                      <span className="ev-type">{ev.type}</span>
                      <p className="ev-text">{ev.text}</p>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Qualification Center */}
          {(researchFlow || qualificationDossier) && (
            <section className="card readiness-card qualification-center">
              <h2><CheckCircle2 size={15} /> Qualification Center</h2>

              <div className="qualification-hero">
                {researchFlow && (
                  <div className="qualification-hero-block">
                    <span className="an-label">Research readiness</span>
                    <div className="readiness-topline">
                      <div className="readiness-score-block">
                        <span className="readiness-score">{researchFlow.readiness_score}</span>
                        <span className="readiness-caption">out of 100</span>
                      </div>
                      <div className={`coach-badge ${researchFlow.commercialization_signal}`}>
                        {researchFlow.commercialization_signal === 'strong' ? 'Pilot-ready signal' : 'Developing signal'}
                      </div>
                    </div>
                    <div className="qualification-subtext">
                      {researchFlow.session_running
                        ? 'Session is active. Let the run finish before packaging artifacts.'
                        : 'Readiness combines hardware, reference capture, calibration quality, and AI availability.'}
                    </div>
                  </div>
                )}

                {qualificationDossier && (
                  <div className="qualification-hero-block qualification-hero-block-emphasis">
                    <span className="an-label">Buyer qualification</span>
                    <div className="qualification-score-row">
                      <span className={`coach-pass ${qualificationDossier.overall_pass ? 'pass' : 'fail'}`}>
                        {qualificationDossier.overall_pass ? 'Qualified' : 'Action required'}
                      </span>
                      {qualificationDossier.qualification_tier && (
                        <span className="coach-tier">Tier: {qualificationDossier.qualification_tier}</span>
                      )}
                      {qualificationDossier.score !== undefined && (
                        <span className="coach-tier">Score: {qualificationDossier.score}</span>
                      )}
                    </div>
                    <div className="qualification-summary-metrics">
                      <div className="qualification-metric-chip">
                        <span className="qualification-metric-label">Checks passed</span>
                        <span className="qualification-metric-value">
                          {passedQualificationChecks}/{qualificationChecks.length || 0}
                        </span>
                      </div>
                      <div className="qualification-metric-chip">
                        <span className="qualification-metric-label">Critical blockers</span>
                        <span className={`qualification-metric-value ${failedCriticalChecks.length > 0 ? 'warn' : 'good'}`}>
                          {failedCriticalChecks.length}
                        </span>
                      </div>
                      <div className="qualification-metric-chip">
                        <span className="qualification-metric-label">Artifacts ready</span>
                        <span className="qualification-metric-value">{latestArtifactCount}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="qualification-action-row">
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={generateReport}
                  disabled={!currentSessionId || reportGenerating}
                >
                  <FileText size={13} />
                  {reportGenerating ? 'Generating…' : 'Generate report'}
                </button>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={exportDossier}
                  disabled={qualBusy || !currentSessionId}
                >
                  <FileText size={13} />
                  {qualBusy ? 'Working…' : 'Export dossier'}
                </button>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={buildResearchPackage}
                  disabled={qualBusy || !currentSessionId}
                >
                  <History size={13} />
                  {qualBusy ? 'Working…' : 'Build package zip'}
                </button>
              </div>

              {currentSessionId && (
                <div className="qualification-session-row">
                  <span className="coach-tier">Session: {currentSessionId.slice(0, 18)}…</span>
                  {lastExport?.signature && (
                    <span className="coach-tier">
                      Signing: {lastExport.signature.signed ? lastExport.signature.algorithm : 'unsigned'}
                    </span>
                  )}
                </div>
              )}

              {qualNotice && <div className="qualification-notice">{qualNotice}</div>}

              {researchFlow && (
                <div className="qualification-section">
                  <div className="coach-list-title">Readiness checkpoints</div>
                  <div className="coach-grid">
                    {researchFlow.checkpoints.map(cp => (
                      <div key={cp.id} className={`coach-check ${cp.done ? 'done' : 'todo'}`}>
                        <div className="coach-check-title">{cp.title}</div>
                        <div className="coach-check-meta">{cp.done ? 'done' : 'pending'} · {cp.impact} impact</div>
                        {(cp.value !== undefined || cp.target !== undefined) && (
                          <div className="coach-check-detail">
                            {cp.value ?? 'n/a'} / {cp.target ?? 'n/a'} target
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {qualificationDossier && qualificationDossier.status === 'ok' && (
                <div className="qualification-columns">
                  <div className="qualification-section">
                    <div className="coach-list-title">Qualification checks</div>
                    <div className="qualification-check-list">
                      {qualificationChecks.map(check => (
                        <div key={check.id} className={`qualification-check-row ${check.pass ? 'pass' : 'fail'}`}>
                          <div className="qualification-check-main">
                            <div className="qualification-check-title-row">
                              <span className="qualification-check-title">{check.title}</span>
                              <span className={`qualification-check-badge ${check.pass ? 'pass' : 'fail'}`}>
                                {check.pass ? 'pass' : check.critical ? 'critical' : 'warn'}
                              </span>
                            </div>
                            <div className="qualification-check-detail">
                              Value: {check.value ?? 'n/a'} · Target: {check.target}
                            </div>
                            <div className="qualification-check-recommendation">{check.recommendation}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="qualification-section">
                    {researchFlow && researchFlow.next_steps.length > 0 && (
                      <div className="coach-list-block qualification-block-tight">
                        <div className="coach-list-title">Research next steps</div>
                        {researchFlow.next_steps.slice(0, 5).map((step, idx) => (
                          <div key={idx} className="coach-list-item">{step}</div>
                        ))}
                      </div>
                    )}

                    {qualificationDossier.next_actions.length > 0 && (
                      <div className="coach-list-block qualification-block-tight">
                        <div className="coach-list-title">Qualification actions</div>
                        {qualificationDossier.next_actions.map((step, idx) => (
                          <div key={idx} className="coach-list-item">{step}</div>
                        ))}
                      </div>
                    )}

                    {reproducibility && (
                      <div className="coach-list-block qualification-block-tight">
                        <div className="coach-list-title">Cross-session reproducibility</div>
                        {reproducibility.available ? (
                          <div className="repro-summary-grid">
                            <div className="repro-metric-card">
                              <span className="repro-metric-label">Completed sessions</span>
                              <span className="repro-metric-value">{reproducibility.session_count ?? 'n/a'}</span>
                            </div>
                            <div className="repro-metric-card">
                              <span className="repro-metric-label">LOD RSD</span>
                              <span className="repro-metric-value">
                                {reproducibility.lod_rsd_pct !== null && reproducibility.lod_rsd_pct !== undefined
                                  ? `${reproducibility.lod_rsd_pct.toFixed(1)}%`
                                  : 'n/a'}
                              </span>
                            </div>
                            <div className="repro-metric-card">
                              <span className="repro-metric-label">LOQ RSD</span>
                              <span className="repro-metric-value">
                                {reproducibility.loq_rsd_pct !== null && reproducibility.loq_rsd_pct !== undefined
                                  ? `${reproducibility.loq_rsd_pct.toFixed(1)}%`
                                  : 'n/a'}
                              </span>
                            </div>
                            <div className="repro-metric-card">
                              <span className="repro-metric-label">R² min</span>
                              <span className="repro-metric-value">
                                {reproducibility.r2_min !== null && reproducibility.r2_min !== undefined
                                  ? reproducibility.r2_min.toFixed(4)
                                  : 'n/a'}
                              </span>
                            </div>
                            <div className="repro-metric-card repro-metric-card-status">
                              <span className="repro-metric-label">Batch readiness</span>
                              <span className={`repro-status ${reproducibility.batch_ready === true ? 'pass' : reproducibility.batch_ready === false ? 'fail' : 'pending'}`}>
                                {reproducibility.batch_ready === true
                                  ? 'ready'
                                  : reproducibility.batch_ready === false
                                    ? 'not ready'
                                    : 'pending'}
                              </span>
                            </div>
                          </div>
                        ) : (
                          <div className="qualification-empty-state">
                            Reproducibility summary unavailable: {reproducibility.reason ?? 'insufficient data'}.
                          </div>
                        )}
                        {reproducibility.notes && reproducibility.notes.length > 0 && (
                          <div className="repro-notes-list">
                            {reproducibility.notes.map((note, idx) => (
                              <div key={idx} className="coach-list-item">{note}</div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    <div className="coach-list-block qualification-block-tight">
                      <div className="coach-list-title">Artifact vault</div>
                      {(lastExport || lastPackage) ? (
                        <div className="artifact-links">
                          {lastExport?.paths?.json && (
                            <a className="artifact-link" href={api.artifactDownloadUrl(lastExport.paths.json)} target="_blank" rel="noopener noreferrer">
                              <Download size={12} /> Dossier JSON
                            </a>
                          )}
                          {lastExport?.paths?.html && (
                            <a className="artifact-link" href={api.artifactDownloadUrl(lastExport.paths.html)} target="_blank" rel="noopener noreferrer">
                              <Download size={12} /> Dossier HTML
                            </a>
                          )}
                          {lastExport?.paths?.signature && (
                            <a className="artifact-link" href={api.artifactDownloadUrl(lastExport.paths.signature)} target="_blank" rel="noopener noreferrer">
                              <Download size={12} /> Signature JSON
                            </a>
                          )}
                          {lastPackage?.package_path && (
                            <a className="artifact-link" href={api.artifactDownloadUrl(lastPackage.package_path)} target="_blank" rel="noopener noreferrer">
                              <Download size={12} /> Research Package ZIP
                            </a>
                          )}
                        </div>
                      ) : (
                        <div className="qualification-empty-state">
                          Export the dossier or build a package to populate downloadable buyer artifacts.
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </section>
          )}

          {/* Agent Events */}
          <section className="card events-card">
            <h2>
              <Zap size={15} /> Agent Events
              <span className="ev-count">{events.length}</span>
            </h2>
            <div
              className="events-list"
              ref={eventsListRef}
              onScroll={() => {
                const el = eventsListRef.current
                if (!el) return
                // If user scrolled away from top, disable auto-scroll
                userScrolledRef.current = el.scrollTop > 40
              }}
            >
              {events.length === 0 && (
                <div className="ev-empty">No events yet — agents are listening…</div>
              )}
              {events.map((ev) => (
                <div key={ev._id} className={`ev-row ev-${ev.level}`}>
                  <span className="ev-icon">{levelIcon(ev.level)}</span>
                  <div className="ev-body">
                    <span className="ev-source">{ev.source}</span>
                    <span className="ev-type">{ev.type}</span>
                    <p className="ev-text">{ev.text}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </main>
      </div>

      {/* ── Modals ── */}

      {showReport && reportContent && (
        <Modal title="Session Report" onClose={() => setShowReport(false)}>
          <div className="report-actions">
            <button
              type="button"
              className="btn-secondary btn-sm"
              onClick={() => navigator.clipboard.writeText(reportContent)}
            >
              Copy
            </button>
          </div>
          <div className="report-md">
            <Suspense fallback={<div>Loading formatted report…</div>}>
              <LazyReactMarkdown>{reportContent}</LazyReactMarkdown>
            </Suspense>
          </div>
        </Modal>
      )}

      {anomalyEvent && (
        <Modal
          title={`${anomalyEvent.source} — ${anomalyEvent.type}`}
          onClose={() => setAnomalyEvent(null)}
        >
          <div className="anomaly-md">
            <Suspense fallback={<div>Loading anomaly details…</div>}>
              <LazyReactMarkdown>{anomalyEvent.text}</LazyReactMarkdown>
            </Suspense>
          </div>
          {anomalyEvent.data && Object.keys(anomalyEvent.data).length > 0 && (
            <pre className="anomaly-data">{JSON.stringify(anomalyEvent.data, null, 2)}</pre>
          )}
        </Modal>
      )}
    </div>
  )
}
