#!/bin/bash
# run_dashboard_secure.sh - Launch the dashboard with authentication
# 
# Usage:
#   bash run_dashboard_secure.sh                 # Uses env var or hashed password file
#   DASHBOARD_PASSWORD="my-lab-pwd" bash run_dashboard_secure.sh
#
# For persistent password:
#   python -m dashboard.auth --set-password

set -e

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if needed
if [ -f ".venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Print startup info
echo ""
echo "=========================================="
echo "🔬 SpectraAgent — Spectrometer-Based Sensing Platform"
echo "=========================================="
echo "Dashboard URL: http://localhost:8501"
echo "---"

# Show password status / fail closed when no secret is configured.
if [ -n "$DASHBOARD_PASSWORD" ]; then
    echo "✓ Using DASHBOARD_PASSWORD from environment"
elif [ -n "$DASHBOARD_PASSWORD_HASH" ]; then
    echo "✓ Using DASHBOARD_PASSWORD_HASH from environment"
elif [ -f "$HOME/.streamlit_au_mip_password_hash" ]; then
    echo "✓ Using password hash from ~/.streamlit_au_mip_password_hash"
elif [ -f "$HOME/.streamlit_au_mip_password" ]; then
    echo "⚠️  Using legacy plaintext password file ~/.streamlit_au_mip_password"
else
    echo "✗ No dashboard password configured"
    echo "  Run: python -m dashboard.auth --set-password"
    echo "  Or set DASHBOARD_PASSWORD for this shell session"
    exit 1
fi

echo "---"
echo "Health check: python -m dashboard.health"
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Try to generate self-signed certs (non-fatal if OpenSSL is unavailable)
python -m dashboard.security >/dev/null 2>&1 || true

CERT_FILE=".streamlit/certs/server.crt"
KEY_FILE=".streamlit/certs/server.key"
SSL_ARGS=()
if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
    echo "HTTPS mode: enabled (self-signed cert)"
    echo "Dashboard URL: https://localhost:8501"
    SSL_ARGS=(--server.sslCertFile="$CERT_FILE" --server.sslKeyFile="$KEY_FILE")
else
    echo "HTTPS mode: unavailable (running HTTP)"
fi
echo ""

# Launch Streamlit with secure settings
streamlit run dashboard/app.py \
    --logger.level=info \
    --client.showErrorDetails=false \
    --server.headless=true \
    --server.enableXsrfProtection=true \
    --server.enableCORS=false \
    "${SSL_ARGS[@]}"
