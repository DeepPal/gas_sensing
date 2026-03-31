"""
dashboard/auth.py - Simple authentication for single-machine lab deployment
============================================================================

Provides password-based authentication for the Streamlit dashboard.
Suitable for research lab networks where all users are trusted.

Usage
-----
In dashboard/app.py, add at the top (before any other st calls):

    from dashboard.auth import check_password
    if not check_password():
        st.stop()
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from pathlib import Path

import streamlit as st

log = logging.getLogger(__name__)


def _get_stored_password() -> str:
    """
    Retrieve the dashboard password from environment or config file.
    
    Priority order:
    1. Environment variable: DASHBOARD_PASSWORD
    2. Config file: ~/.streamlit_au_mip_password
    3. Fallback: "research-lab-default" (OVERRIDE THIS!)
    
    Returns
    -------
    str
        The stored password hash (for comparison)
    """
    import os

    # Check environment first
    env_pwd = os.environ.get("DASHBOARD_PASSWORD")
    if env_pwd:
        return hashlib.sha256(env_pwd.encode()).hexdigest()

    # Check config file
    config_path = Path.home() / ".streamlit_au_mip_password"
    if config_path.exists():
        pwd = config_path.read_text().strip()
        return hashlib.sha256(pwd.encode()).hexdigest()

    # Fallback (NOT SECURE — operator must override)
    log.warning(
        "No password configured. Using fallback. Override via DASHBOARD_PASSWORD env var."
    )
    return hashlib.sha256("research-lab-default".encode()).hexdigest()


def check_password() -> bool:
    """
    Display password input and validate against stored password.
    
    Returns
    -------
    bool
        True if password is correct (or already authenticated in session),
        False otherwise.
    
    Side effects
    -----------
    - Displays password input form if not authenticated
    - Sets st.session_state.password_correct on successful auth
    - Displays error message on failed auth
    """
    # Check if already authenticated in this session
    if st.session_state.get("password_correct", False):
        return True

    st.set_page_config(
        page_title="Au-MIP LSPR — Login",
        page_icon="🔐",
        layout="centered",
    )

    st.markdown("## 🔬 Au-MIP LSPR Gas Analysis Platform")
    st.markdown("**Research Lab Dashboard**")
    st.markdown("---")

    # Password input
    password = st.text_input(
        "Lab Password",
        type="password",
        placeholder="Enter the dashboard password from your lab notes",
    )

    if st.button("Login", use_container_width=True):
        stored_hash = _get_stored_password()
        input_hash = hashlib.sha256(password.encode()).hexdigest()

        if hmac.compare_digest(input_hash, stored_hash):
            st.session_state.password_correct = True
            st.success("✓ Authentication successful. Reloading dashboard...")
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
            log.warning("Failed password attempt from client")

    st.info(
        """
        **First time?**
        - Default password: `research-lab-default`
        - To customize: set `DASHBOARD_PASSWORD` environment variable
        - For production: override via `~/.streamlit_au_mip_password` file
        """
    )

    return False


def set_password(new_password: str) -> None:
    """
    Programmatically set the password (for lab admin use).
    
    Parameters
    ----------
    new_password : str
        The new plaintext password to set
    
    Examples
    --------
    >>> import subprocess
    >>> set_password("my-new-lab-password")
    """
    import os

    # Set environment variable for this session
    os.environ["DASHBOARD_PASSWORD"] = new_password

    # Optionally write to config file (requires admin access)
    config_path = Path.home() / ".streamlit_au_mip_password"
    try:
        config_path.write_text(new_password)
        # Try to set permissive mode (owner read/write only) on Unix-like systems
        try:
            config_path.chmod(0o600)
        except (AttributeError, NotImplementedError):
            pass  # chmod not available on all platforms
        log.info("Password updated in config file: %s", config_path)
    except (PermissionError, IOError) as e:
        log.warning("Could not write password to config file: %s", e)
