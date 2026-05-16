import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const mockApi = vi.hoisted(() => ({
  health: vi.fn(),
  getResearchFlow: vi.fn(),
  getQualificationDossier: vi.fn(),
  listAnalytes: vi.fn(),
  inferMixture: vi.fn(),
  configAcquisition: vi.fn(),
  startSession: vi.fn(),
  stopSession: vi.fn(),
  captureReference: vi.fn(),
  listSessions: vi.fn(),
  getSession: vi.fn(),
  generateReport: vi.fn(),
  exportQualificationDossier: vi.fn(),
  createResearchPackage: vi.fn(),
  artifactDownloadUrl: vi.fn((path: string) => `/api/artifacts/download?path=${encodeURIComponent(path)}`),
  setQualitySettings: vi.fn(),
  setDriftSettings: vi.fn(),
  addCalibrationPoint: vi.fn(),
  suggestConcentration: vi.fn(),
  ask: vi.fn(),
  setAutoExplain: vi.fn(),
}))

vi.mock('./api', () => ({
  api: mockApi,
  connectSpectrum: vi.fn(() => ({ close: vi.fn() })),
  connectAgentEvents: vi.fn(() => ({ close: vi.fn() })),
}))

describe('Qualification Center contract', () => {
  beforeEach(() => {
    vi.clearAllMocks()

    mockApi.health.mockResolvedValue({
      status: 'ok',
      version: 'test',
      hardware: 'sim',
      simulate: true,
      physics_plugin: 'test',
      integration_time_ms: 50,
      quality_settings: { saturation_threshold: 60000, snr_warn_threshold: 3.0 },
      drift_settings: { drift_threshold_nm_per_min: 0.05, window_frames: 60 },
    })

    mockApi.getResearchFlow.mockResolvedValue({
      readiness_score: 91,
      session_running: false,
      commercialization_signal: 'strong',
      checkpoints: [
        { id: 'cp1', title: 'LOD', done: true, impact: 'high', value: '0.8 ppm', target: '<= 1.0 ppm' },
      ],
      next_steps: ['Run blinded validation batch'],
    })

    mockApi.getQualificationDossier.mockResolvedValue({
      status: 'ok',
      session_id: 'session-test-001',
      overall_pass: true,
      qualification_tier: 'gold',
      score: 96,
      checks: [
        {
          id: 'q1',
          title: 'Cross-session stability',
          value: 'pass',
          target: 'pass',
          pass: true,
          critical: true,
          recommendation: 'Maintain current protocol.',
        },
      ],
      next_actions: ['Prepare external review package'],
      reproducibility: {
        available: true,
        session_count: 8,
        lod_rsd_pct: 12.5,
        loq_rsd_pct: 14.2,
        r2_min: 0.9821,
        batch_ready: true,
        notes: ['All key thresholds satisfy release criteria.'],
      },
    })

    mockApi.listAnalytes.mockResolvedValue({
      analytes: [],
      n_peaks: 0,
      peak_wavelengths_nm: [],
      S_matrix: null,
    })

    mockApi.listSessions.mockResolvedValue([])
  })

  it('renders reproducibility summary from dossier payload', async () => {
    render(<App />)

    await waitFor(() => {
      expect(screen.getByText('Cross-session reproducibility')).toBeInTheDocument()
    })

    expect(screen.getByText('12.5%')).toBeInTheDocument()
    expect(screen.getByText('14.2%')).toBeInTheDocument()
    expect(screen.getByText('0.9821')).toBeInTheDocument()
    expect(screen.getByText('ready')).toBeInTheDocument()
    expect(screen.getByText('All key thresholds satisfy release criteria.')).toBeInTheDocument()

    expect(mockApi.getQualificationDossier).toHaveBeenCalled()
    expect(mockApi.getResearchFlow).toHaveBeenCalled()
  })
})
