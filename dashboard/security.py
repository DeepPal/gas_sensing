"""
dashboard/security.py - HTTPS certificate generation & management
==================================================================

Generates self-signed SSL certificates for secure local deployment.
Suitable for research lab networks where all users are trusted.

For lab networks, self-signed certificates are acceptable because:
1. All users are on the same internal network (not internet-facing)
2. Data is sensitive but not subject to eavesdropping by external actors
3. Authentication is via password (not SSL alone)
4. Audit logs track all access
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import subprocess

log = logging.getLogger(__name__)


def generate_self_signed_cert(
    app_root: Path | None = None,
    validity_days: int = 365,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """
    Generate a self-signed SSL certificate for local HTTPS.

    Parameters
    ----------
    app_root : Path
        Root of the application (where certs will be stored)
    validity_days : int
        Number of days certificate is valid (default: 365 = 1 year)
    overwrite : bool
        If True, regenerate even if cert exists

    Returns
    -------
    tuple[Path, Path]
        (cert_path, key_path) — paths to the generated certificate and key files

    Raises
    ------
    RuntimeError
        If OpenSSL is not available or certificate generation fails
    """
    app_root = app_root or Path(__file__).resolve().parents[1]
    certs_dir = app_root / ".streamlit" / "certs"
    certs_dir.mkdir(parents=True, exist_ok=True)

    cert_file = certs_dir / "server.crt"
    key_file = certs_dir / "server.key"

    # Skip generation if cert already exists and is valid
    if not overwrite and cert_file.exists() and key_file.exists():
        if _is_cert_valid(cert_file):
            log.info("✓ Using existing valid certificate: %s", cert_file)
            return cert_file, key_file
        else:
            log.warning("Certificate expired; regenerating...")

    # Check for OpenSSL
    try:
        subprocess.run(["openssl", "version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "OpenSSL is not installed. Required for certificate generation.\n"
            "Install via: conda install openssl  (or apt/brew)"
        ) from None

    log.info("🔐 Generating self-signed SSL certificate (valid for %d days)...", validity_days)

    # Generate private key + self-signed certificate
    cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:4096",
        "-keyout",
        str(key_file),
        "-out",
        str(cert_file),
        "-days",
        str(validity_days),
        "-nodes",
        "-subj",
        "/C=US/ST=Lab/L=Lab/O=Research/CN=localhost",
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        log.info("✓ Certificate generated: %s", cert_file)
        log.info("✓ Private key generated: %s", key_file)
        return cert_file, key_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Certificate generation failed: {e.stderr.decode()}") from e


def _is_cert_valid(cert_path: Path, days_remaining: int = 7) -> bool:
    """
    Check if a certificate is valid and won't expire soon.

    Parameters
    ----------
    cert_path : Path
        Path to the certificate file
    days_remaining : int
        Minimum days until expiration (default: 7)

    Returns
    -------
    bool
        True if certificate is valid for at least days_remaining days
    """
    try:
        result = subprocess.run(
            ["openssl", "x509", "-in", str(cert_path), "-noout", "-dates"],
            capture_output=True,
            check=True,
            text=True,
        )

        # Parse expiration date from output
        # Output format: "notAfter=Apr 28 10:45:32 2027 GMT"
        for line in result.stdout.split("\n"):
            if "notAfter=" in line:
                # Extract date string
                date_str = line.split("notAfter=")[1].strip()
                # Parse (format: "Apr 28 10:45:32 2027 GMT")
                expiry = datetime.strptime(date_str, "%b %d %H:%M:%S %Y %Z").replace(
                    tzinfo=timezone.utc
                )
                days_until_expiry = (expiry - datetime.now(timezone.utc)).days
                return days_until_expiry >= days_remaining

    except Exception as e:
        log.warning("Could not verify certificate: %s", e)
        return False

    return False


def setup_https(app_root: Path | None = None) -> dict[str, Path]:
    """
    Complete HTTPS setup: generate cert and return Streamlit config.

    Parameters
    ----------
    app_root : Path
        Root of the application

    Returns
    -------
    dict[str, Path]
        Dictionary with 'certfile' and 'keyfile' paths
    """
    cert_path, key_path = generate_self_signed_cert(app_root)

    return {
        "certfile": cert_path,
        "keyfile": key_path,
    }


if __name__ == "__main__":
    # CLI usage: python -m dashboard.security
    import sys

    logging.basicConfig(level=logging.INFO)

    try:
        cert_path, key_path = generate_self_signed_cert(overwrite=("--force" in sys.argv))
        print("\n✓ Certificates ready for deployment")
        print(f"  Certificate: {cert_path}")
        print(f"  Private Key: {key_path}")
        print("\nEnable in .streamlit/config.toml:")
        print("  [server]")
        print(f"  sslCertFile = \"{cert_path}\"")
        print(f"  sslKeyFile = \"{key_path}\"")
    except RuntimeError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
