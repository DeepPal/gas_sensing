"""
run.py is retired.

Use one of the following instead:

    python launcher.py                    # start both services (recommended)
    spectraagent start --simulate         # SpectraAgent only (no hardware)
    spectraagent start --hw               # SpectraAgent with real spectrometer
    streamlit run dashboard/app.py        # Streamlit dashboard only
    docker compose up                     # everything in containers

See docs/quickstart/research.md for a full getting-started guide.
"""
import sys

print(__doc__)
sys.exit(1)
