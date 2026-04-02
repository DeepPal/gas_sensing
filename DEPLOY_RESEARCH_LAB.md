# Research Lab Deployment Guide

## 🎯 Single-Machine Lab Deployment

This guide covers deploying the LSPR platform to a single research lab machine with basic security and monitoring.

---

## Prerequisites

- **Machine:** Windows, macOS, or Linux
- **Python:** 3.9+ (or use conda environment)
- **Hardware:** ThorLabs CCS200 spectrometer (optional; platform falls back to simulation)
- **Network:** Lab network (device does NOT need public internet)

---

## Installation (One-Time Setup)

### Step 1: Clone and Install

```bash
cd ~/Research  # or your preferred location
git clone https://github.com/DeepPal/gas_sensing.git
cd gas_sensing

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate.bat

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Step 2: Configure the Dashboard Password

```bash
# Option A: ephemeral password for this shell session
set DASHBOARD_PASSWORD=my-lab-password        # Windows
export DASHBOARD_PASSWORD=my-lab-password     # macOS/Linux

# Option B: persistent hashed password file (recommended)
python -m dashboard.auth --set-password
```

The recommended path writes a PBKDF2-SHA256 verifier to:

- Windows: `%USERPROFILE%\.streamlit_au_mip_password_hash`
- macOS/Linux: `~/.streamlit_au_mip_password_hash`

---

## Running the Dashboard

### Option 1: Secure Dashboard (Recommended)

```bash
# Windows
run_dashboard_secure.bat

# macOS/Linux
bash run_dashboard_secure.sh
```

This will:
- ✅ Prompt for password on first access
- ✅ Show health check status in sidebar
- ✅ Disable CSRF attacks and use HTTPS automatically when local certs are available
- ✅ Log all interactions to `logs/dashboard.log`

**Access:**
```
http://localhost:8501  (or your machine's IP address)
```

### Option 2: Quick Launch (Development Only)

```bash
streamlit run dashboard/app.py
```

Authentication is still enforced by `dashboard/app.py`; this bypasses only the secure launcher conveniences.

---

## Health Check & Monitoring

The dashboard now includes a **System Status** panel in the sidebar showing:
- ✅/❌ Overall health
- 📊 Disk space available
- 📡 Spectrometer & live server status
- 🔴/🟢 Hardware connectivity

### Manual Health Check

```bash
python -m dashboard.health
```

Output:
```
======================================================================
LSPR HEALTH CHECK REPORT
======================================================================
Timestamp:  2026-03-28T10:45:32.123456
Hostname:   lab-machine-01

Application: LSPR Gas Sensing Platform v1.0.0

📦 Disk Space:
  Available: 45.32 GB / 100.00 GB
  Status:    OK

📝 Logs:
  File:  /home/lab/gas_sensing/logs/dashboard.log
  Status: OK

⚙️  Hardware:
  Spectrometer: Importable ✓
  Live Server:  Connected ✓

✓ OVERALL HEALTHY
======================================================================
```

---

## Troubleshooting

### "No dashboard password configured"

- **Check:** `DASHBOARD_PASSWORD`, `DASHBOARD_PASSWORD_HASH`, or the hashed password file exists
- **Action:** Run `python -m dashboard.auth --set-password`

### "Incorrect password" on first launch

- **Check:** Environment variable or password file is set correctly
- **Action:** Restart the dashboard and re-enter password
- **Recommendation:** Prefer the hashed password file over plaintext env vars for persistent deployments

### "Disk space: LOW" warning

- **Check:** Free space on the machine
- **Action:** Archive old session data to external storage or delete old sessions
  ```bash
  rm output/sessions/*.json  # Delete session metadata
  ```

### "Live Server: NOT CONNECTED"

- **Check:** Is the live data server running?
- **Action:** This is non-critical; the main dashboard will still work in standard mode

### "Spectrometer: NOT AVAILABLE"

- **Check:** Is the hardware plugged in and VISA drivers installed?
- **Action:** Platform automatically falls back to simulation mode; you can still test workflows

---

## Lab Network Deployment

To make the dashboard accessible from other machines in the lab:

### Option A: Share via Lab Network (Recommended)

1. **Find your machine's IP address:**
   ```bash
   # Windows PowerShell
   ipconfig | findstr IPv4
   
   # macOS/Linux
   ifconfig | grep "inet "
   ```

2. **Share the URL:**
   ```
   http://<YOUR_IP_ADDRESS>:8501
   ```

3. **Colleagues access from their machines:**
   - Open browser on their machine
   - Enter the URL
   - Enter the lab password

### Option B: SSH Tunnel (More Secure)

```bash
# From colleague's machine
ssh -L 8501:localhost:8501 lab_user@lab_machine
# Then open: http://localhost:8501
```

---

## Security Notes

### ✅ What's Secured

- Password-protected dashboard (prevents unauthorized access)
- PBKDF2-SHA256 hashed password file support for persistent deployments
- CSRF protection enabled (prevents cross-site attacks)
- CORS disabled (only same-machine access by default)
- Session-based authentication (logged out after inactivity)
- Logs stored locally with timestamps

### ⚠️ Limitations

- Passwords supplied through `DASHBOARD_PASSWORD` remain plaintext process secrets
- Self-signed HTTPS still requires operators to trust the local certificate
- File upload accepts CSV files (validate filenames in prod)

---

## Docker Deployment

Use Docker Compose when you want the Streamlit dashboard and the live SpectraAgent runtime together:

```bash
docker compose up --build
```

This starts:

- `spectraagent` on `http://localhost:8765`
- `dashboard` on `http://localhost:8501`

Optional environment variables:

```bash
export DASHBOARD_PASSWORD=my-lab-password
export ANTHROPIC_API_KEY=...
export SPECTRAAGENT_BASE_URL=http://localhost:8765
```

### 🔐 Recommendations

1. **Use a strong lab password** (≥ 12 characters)
2. **Change password regularly** (quarterly)
3. **Monitor logs** for suspicious activity:
   ```bash
   tail -f logs/dashboard.log
   ```
4. **Backup data** regularly:
   ```bash
   tar -czf backup_$(date +%Y%m%d).tar.gz output/ data/
   ```

---

## Monitoring & Maintenance

### Log Rotation

Logs are automatically rotated at 5 MB with 3 backups kept. View latest:

```bash
tail -100 logs/dashboard.log
```

### Session Management

Sessions are stored in `output/sessions/`. To clear old sessions:

```bash
find output/sessions -mtime +7 -delete  # Delete sessions older than 7 days
```

### Configuration

Edit lab-specific settings in `config/config.yaml`:

```yaml
environment:
  enabled: true
  reference:
    temperature: 25.0    # Lab ambient temp
    humidity: 50.0       # Lab RH%

calibration:
  data_dir: "data/JOY_Data"
  models_dir: "models/registry"
```

---

## What's NOT Yet Implemented (For Future Upgrades)

| Feature | Why needed | Priority |
|---------|-----------|----------|
| **HTTPS/SSL** | Encrypt data in transit | 🟡 Medium (1-2 months) |
| **Database backing** | Replace CSV sessions | 🟡 Medium (grow to >1000 sessions) |
| **Multi-user auth** | Lab-wide role management | 🟡 Medium (scale to 10+ users) |
| **Mobile app** | On-the-go monitoring | 🔴 Low (research lab focused) |

For now, this deployment is **perfect for a research lab with 1-5 concurrent users**.

---

## Support

- **Email:** research@example.com
- **Issues:** Create a GitHub issue with logs
- **Logs location:** `logs/dashboard.log`

---

## Quickstart Cheat Sheet

```bash
# First-time setup (one machine, one person)
1. git clone & pip install -e .
2. echo "my-password" > ~/.streamlit_au_mip_password
3. chmod 600 ~/.streamlit_au_mip_password
4. run_dashboard_secure.bat  (or .sh)
5. Open http://localhost:8501
6. Enter password
7. Start analyzing!

# Day-to-day
run_dashboard_secure.bat & python -m dashboard.health

# Troubleshooting
tail logs/dashboard.log
python -m dashboard.health
```

---

**Version:** 1.0.0 | **Last Updated:** March 2026 | **Maturity:** Research-Grade ✓
