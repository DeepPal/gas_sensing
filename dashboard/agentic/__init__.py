"""Agentic pipeline tab — split into focused modules.

Sub-modules:
  tab.py            -- Streamlit UI wiring; defines render_agentic_pipeline_tab()
  steps.py          -- pipeline step business logic functions
  visualizations.py -- Plotly chart builders (return figures)
  lspr_physics.py   -- Lorentzian fit, FOM, WLS correction, FWHM helpers
"""
from .tab import render_agentic_pipeline_tab

# Legacy alias: app.py imports `render` from dashboard.agentic_pipeline_tab
render = render_agentic_pipeline_tab

__all__ = ["render_agentic_pipeline_tab", "render"]
