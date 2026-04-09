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

import argparse
from contextlib import suppress
import getpass
import hashlib
import hmac
import logging
from pathlib import Path
from typing import Final

log = logging.getLogger(__name__)

PBKDF2_PREFIX: Final[str] = "pbkdf2_sha256"
PBKDF2_ITERATIONS: Final[int] = 100_000
PASSWORD_HASH_FILE: Final[str] = ".streamlit_au_mip_password_hash"
LEGACY_PASSWORD_FILE: Final[str] = ".streamlit_au_mip_password"


def _pbkdf2_hash_password(
    password: str,
    *,
    salt: bytes | None = None,
    iterations: int = PBKDF2_ITERATIONS,
) -> str:
    """Return a self-contained PBKDF2-SHA256 verifier string."""
    if not password:
        raise ValueError("Password must not be empty.")

    salt_bytes = salt if salt is not None else hashlib.sha256(password.encode("utf-8")).digest()[:16]
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, iterations)
    return f"{PBKDF2_PREFIX}${iterations}${salt_bytes.hex()}${derived.hex()}"


def _is_pbkdf2_verifier(value: str) -> bool:
    return value.startswith(f"{PBKDF2_PREFIX}$")


def _verify_password(password: str, stored_secret: str) -> bool:
    """Verify user input against a PBKDF2 verifier or compatibility plaintext secret."""
    if _is_pbkdf2_verifier(stored_secret):
        try:
            _, iterations_text, salt_hex, expected_hex = stored_secret.split("$", 3)
            candidate = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                bytes.fromhex(salt_hex),
                int(iterations_text),
            ).hex()
        except (TypeError, ValueError):
            log.error("Invalid dashboard password hash format.")
            return False
        return hmac.compare_digest(candidate, expected_hex)

    return hmac.compare_digest(password, stored_secret)


def _get_stored_password() -> str | None:
    """
    Retrieve the dashboard password verifier or compatibility secret.

    Priority order:
    1. Environment variable: DASHBOARD_PASSWORD_HASH
    2. Environment variable: DASHBOARD_PASSWORD
    3. Config file: ~/.streamlit_au_mip_password_hash
    4. Legacy plaintext file: ~/.streamlit_au_mip_password

    Returns
    -------
    str | None
        Stored verifier or plaintext password. Returns None if not configured.
    """
    import os

    env_hash = os.environ.get("DASHBOARD_PASSWORD_HASH")
    if env_hash:
        return env_hash.strip()

    env_pwd = os.environ.get("DASHBOARD_PASSWORD")
    if env_pwd:
        return env_pwd

    hash_path = Path.home() / PASSWORD_HASH_FILE
    if hash_path.exists():
        return hash_path.read_text(encoding="utf-8").strip()

    legacy_path = Path.home() / LEGACY_PASSWORD_FILE
    if legacy_path.exists():
        log.warning(
            "Using legacy plaintext password file at %s. Migrate to %s for deployable setups.",
            legacy_path,
            hash_path,
        )
        return legacy_path.read_text(encoding="utf-8").strip()

    return None


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
    try:
        import streamlit as st
    except ModuleNotFoundError:
        raise RuntimeError("Streamlit is required to run dashboard authentication UI.") from None

    # Check if already authenticated in this session
    if st.session_state.get("password_correct", False):
        return True

    stored_secret = _get_stored_password()
    if not stored_secret:
        st.set_page_config(
            page_title="SpectraAgent - Setup Required",
            page_icon="🔐",
            layout="centered",
        )
        st.error("No dashboard password is configured.")
        st.info(
            "Set DASHBOARD_PASSWORD for an ephemeral secret, or run "
            "`python -m dashboard.auth --set-password` to create a hashed password file."
        )
        return False

    st.set_page_config(
        page_title="the sensor LSPR — Login",
        page_icon="🔐",
        layout="centered",
    )

    st.markdown("## 🔬 SpectraAgent — Spectrometer-Based Sensing Platform")
    st.markdown("**Research Lab Dashboard**")
    st.markdown("---")

    # Password input
    password = st.text_input(
        "Lab Password",
        type="password",
        placeholder="Enter the dashboard password from your lab notes",
    )

    if st.button("Login", use_container_width=True):
        if _verify_password(password, stored_secret):
            st.session_state.password_correct = True
            st.success("✓ Authentication successful. Reloading dashboard...")
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
            log.warning("Failed password attempt from client")

    st.info(
        """
        **Authentication sources**
        - `DASHBOARD_PASSWORD_HASH` for a PBKDF2 verifier
        - `DASHBOARD_PASSWORD` for a session-only password
        - `~/.streamlit_au_mip_password_hash` for persistent deployable setups
        """
    )

    return False


def set_password(new_password: str) -> None:
    """
    Programmatically set the password hash (for lab admin use).

    Parameters
    ----------
    new_password : str
        The new plaintext password to set

    Examples
    --------
    >>> set_password("my-new-lab-password")
    """
    import os

    password_hash = _pbkdf2_hash_password(new_password)
    os.environ["DASHBOARD_PASSWORD_HASH"] = password_hash

    config_path = Path.home() / PASSWORD_HASH_FILE
    try:
        config_path.write_text(password_hash, encoding="utf-8")
        with suppress(AttributeError, NotImplementedError):
            config_path.chmod(0o600)
        log.info("Password updated in config file: %s", config_path)
    except OSError as e:
        log.warning("Could not write password to config file: %s", e)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage SpectraAgent dashboard passwords.")
    parser.add_argument("--set-password", action="store_true", help="Create or overwrite the hashed password file.")
    parser.add_argument("--password", help="Password to hash. If omitted, prompt securely.")
    parser.add_argument("--print-hash", metavar="PASSWORD", help="Print a PBKDF2 verifier for the supplied password.")
    parser.add_argument(
        "--verify",
        nargs="?",
        const="__PROMPT__",
        metavar="PASSWORD",
        help="Verify a password against the currently configured secret. Prompts if omitted.",
    )
    return parser.parse_args()


def _prompt_for_password() -> str:
    password = getpass.getpass("Dashboard password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise ValueError("Passwords did not match.")
    return password


def main() -> int:
    args = _parse_args()

    if args.print_hash:
        print(_pbkdf2_hash_password(args.print_hash))
        return 0

    if args.set_password:
        password = args.password or _prompt_for_password()
        set_password(password)
        print(f"Wrote hashed password file to {Path.home() / PASSWORD_HASH_FILE}")
        return 0

    if args.verify is not None:
        configured = _get_stored_password()
        if not configured:
            print("Dashboard password is not configured.")
            return 1

        password = getpass.getpass("Dashboard password to verify: ") if args.verify == "__PROMPT__" else args.verify
        if _verify_password(password, configured):
            print("Password verification: OK")
            return 0

        print("Password verification: FAILED")
        return 2

    configured = _get_stored_password()
    if configured:
        source = "PBKDF2 hash" if _is_pbkdf2_verifier(configured) else "plaintext secret"
        print(f"Dashboard password is configured via {source}.")
        return 0

    print("Dashboard password is not configured.")
    print("Use --set-password to create a hashed password file.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
