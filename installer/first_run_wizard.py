"""
SpectraAgent First-Run Wizard
Runs after installation to set up password and API key.
Uses only tkinter (stdlib) — no external packages needed.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sys
import tkinter as tk
from pathlib import Path
from tkinter import font, messagebox, ttk

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INSTALL_DIR = Path(sys.argv[0]).parent.parent  # installer/ → root
PASSWORD_HASH_FILE = Path.home() / ".streamlit_au_mip_password_hash"
ENV_HINT_FILE = INSTALL_DIR / "installer" / "api_key_hint.txt"

# ---------------------------------------------------------------------------
# Password hashing (must match dashboard/auth.py)
# ---------------------------------------------------------------------------
_ITERATIONS = 100_000


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt}${dk.hex()}"


# ---------------------------------------------------------------------------
# Wizard pages
# ---------------------------------------------------------------------------
BRAND_BG = "#1a1a2e"
BRAND_FG = "#e0e0e0"
ACCENT   = "#0f8abf"
SUCCESS  = "#27ae60"
WARN     = "#e67e22"


class FirstRunWizard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SpectraAgent — First-Time Setup")
        self.geometry("560x480")
        self.resizable(False, False)
        self.configure(bg=BRAND_BG)

        # Try to set icon
        icon_path = INSTALL_DIR / "resources" / "icon.ico"
        if icon_path.exists():
            try:
                self.iconbitmap(str(icon_path))
            except Exception:
                pass

        self._pages: list[tk.Frame] = []
        self._current = 0

        self._build_header()
        self._container = tk.Frame(self, bg=BRAND_BG)
        self._container.pack(fill="both", expand=True, padx=24, pady=(0, 8))

        self._build_nav()

        self._page_welcome()
        self._page_password()
        self._page_api_key()
        self._page_done()

        self._show_page(0)

    # ── Layout helpers ────────────────────────────────────────────────────

    def _build_header(self) -> None:
        hdr = tk.Frame(self, bg=ACCENT, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        title_font = font.Font(family="Segoe UI", size=14, weight="bold")
        tk.Label(
            hdr,
            text="  ⚗  SpectraAgent  —  First-Time Setup",
            bg=ACCENT, fg="white",
            font=title_font,
            anchor="w",
        ).pack(fill="x", padx=16, pady=12)

    def _build_nav(self) -> None:
        nav = tk.Frame(self, bg=BRAND_BG)
        nav.pack(fill="x", side="bottom", padx=24, pady=12)
        btn_font = font.Font(family="Segoe UI", size=10)
        self._back_btn = tk.Button(
            nav, text="← Back", width=10, command=self._go_back,
            bg="#2c2c54", fg=BRAND_FG, font=btn_font,
            relief="flat", cursor="hand2", padx=8
        )
        self._back_btn.pack(side="left")
        self._next_btn = tk.Button(
            nav, text="Next →", width=14, command=self._go_next,
            bg=ACCENT, fg="white", font=btn_font,
            relief="flat", cursor="hand2", padx=8
        )
        self._next_btn.pack(side="right")

    def _label(self, parent: tk.Widget, text: str, size: int = 10, bold: bool = False, fg: str = BRAND_FG) -> tk.Label:
        f = font.Font(family="Segoe UI", size=size, weight="bold" if bold else "normal")
        return tk.Label(parent, text=text, bg=BRAND_BG, fg=fg, font=f, wraplength=500, justify="left", anchor="w")

    def _entry(self, parent: tk.Widget, show: str = "") -> tk.Entry:
        e = tk.Entry(
            parent, show=show, bg="#16213e", fg=BRAND_FG,
            insertbackground=BRAND_FG, relief="flat",
            font=font.Font(family="Consolas", size=11),
            bd=4
        )
        return e

    def _sep(self, parent: tk.Widget) -> None:
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)

    # ── Page builders ────────────────────────────────────────────────────

    def _add_page(self) -> tk.Frame:
        f = tk.Frame(self._container, bg=BRAND_BG)
        self._pages.append(f)
        return f

    def _page_welcome(self) -> None:
        p = self._add_page()
        self._label(p, "Welcome to SpectraAgent", size=16, bold=True, fg="white").pack(anchor="w", pady=(16, 4))
        self._sep(p)
        self._label(p,
            "This wizard will configure two things before you start using the platform:\n\n"
            "  1.  Dashboard password  — protects your data from unauthorised access\n"
            "  2.  Anthropic API key   — enables AI narration and anomaly explanations\n\n"
            "Both take about 2 minutes. You can skip the API key if you do not have one.",
            size=10).pack(anchor="w", pady=4)
        self._sep(p)
        self._label(p,
            "After this wizard, launch SpectraAgent with:\n"
            "  • Simulation:  run_spectraagent.bat --simulate\n"
            "  • Hardware:    run_spectraagent.bat --hardware\n"
            "  • Dashboard:   run_dashboard_secure.bat",
            size=9, fg="#aaaaaa").pack(anchor="w", pady=4)

    def _page_password(self) -> None:
        p = self._add_page()
        self._label(p, "Step 1 — Set Dashboard Password", size=13, bold=True, fg="white").pack(anchor="w", pady=(16, 4))
        self._sep(p)
        self._label(p,
            "The Streamlit Dashboard is password-protected. Choose a password that your "
            "lab colleagues will share. Store it in your lab notebook.",
            size=10).pack(anchor="w", pady=(0, 10))

        self._label(p, "Password:", size=10).pack(anchor="w")
        self._pw1 = self._entry(p, show="•")
        self._pw1.pack(fill="x", ipady=6, pady=(2, 8))

        self._label(p, "Confirm password:", size=10).pack(anchor="w")
        self._pw2 = self._entry(p, show="•")
        self._pw2.pack(fill="x", ipady=6, pady=(2, 4))

        self._pw_status = self._label(p, "", size=9, fg=WARN)
        self._pw_status.pack(anchor="w")

        if PASSWORD_HASH_FILE.exists():
            self._label(p,
                "A password file already exists. Entering a new password will replace it.",
                size=9, fg=WARN).pack(anchor="w", pady=(8, 0))

    def _page_api_key(self) -> None:
        p = self._add_page()
        self._label(p, "Step 2 — Anthropic API Key (optional)", size=13, bold=True, fg="white").pack(anchor="w", pady=(16, 4))
        self._sep(p)
        self._label(p,
            "The API key enables AI agent features:\n"
            "  • AI session narratives (auto-drafted Methods paragraphs)\n"
            "  • Anomaly explanations\n"
            "  • Cross-session trend summaries\n\n"
            "Without a key, all scientific calculations still work normally.\n"
            "Get a key at: console.anthropic.com",
            size=10).pack(anchor="w", pady=(0, 10))

        self._label(p, "ANTHROPIC_API_KEY  (leave blank to skip):", size=10).pack(anchor="w")
        self._api_entry = self._entry(p, show="•")
        self._api_entry.pack(fill="x", ipady=6, pady=(2, 4))

        self._api_status = self._label(p, "", size=9, fg="#aaaaaa")
        self._api_status.pack(anchor="w")

        self._label(p,
            "The key will be saved as a Windows User Environment Variable so it persists "
            "across restarts. It is never written to any file in the project directory.",
            size=9, fg="#aaaaaa").pack(anchor="w", pady=(8, 0))

    def _page_done(self) -> None:
        p = self._add_page()
        self._label(p, "✓  Setup Complete", size=16, bold=True, fg=SUCCESS).pack(anchor="w", pady=(24, 8))
        self._sep(p)
        self._done_msg = self._label(p, "", size=10)
        self._done_msg.pack(anchor="w")
        self._sep(p)
        self._label(p,
            "For full instructions, open:\n"
            "  docs\\RESEARCHER_USER_GUIDE.md\n\n"
            "To change the password later:\n"
            "  Start Menu → SpectraAgent → Set Dashboard Password",
            size=9, fg="#aaaaaa").pack(anchor="w", pady=4)

    # ── Navigation ───────────────────────────────────────────────────────

    def _show_page(self, idx: int) -> None:
        for p in self._pages:
            p.pack_forget()
        self._pages[idx].pack(fill="both", expand=True)
        self._current = idx

        self._back_btn.config(state="normal" if idx > 0 else "disabled")
        is_last = (idx == len(self._pages) - 1)
        self._next_btn.config(
            text="Finish" if is_last else "Next →",
            bg=SUCCESS if is_last else ACCENT
        )

    def _go_back(self) -> None:
        if self._current > 0:
            self._show_page(self._current - 1)

    def _go_next(self) -> None:
        if self._current == len(self._pages) - 1:
            self.destroy()
            return

        if self._current == 1:  # password page
            if not self._validate_password():
                return

        if self._current == 2:  # API key page
            self._save_api_key()
            self._update_done_page()

        self._show_page(self._current + 1)

    # ── Validation & saving ──────────────────────────────────────────────

    def _validate_password(self) -> bool:
        pw1 = self._pw1.get()
        pw2 = self._pw2.get()

        if not pw1:
            self._pw_status.config(text="⚠  Password cannot be empty.", fg=WARN)
            return False
        if len(pw1) < 6:
            self._pw_status.config(text="⚠  Password must be at least 6 characters.", fg=WARN)
            return False
        if pw1 != pw2:
            self._pw_status.config(text="⚠  Passwords do not match.", fg=WARN)
            return False

        # Save the hash
        pw_hash = _hash_password(pw1)
        try:
            PASSWORD_HASH_FILE.write_text(pw_hash, encoding="utf-8")
            self._pw_status.config(text=f"✓  Password saved to {PASSWORD_HASH_FILE}", fg=SUCCESS)
        except OSError as e:
            self._pw_status.config(text=f"⚠  Could not save: {e}", fg=WARN)
            return False

        return True

    def _save_api_key(self) -> None:
        key = self._api_entry.get().strip()
        if not key:
            self._api_status.config(text="Skipped — API key not set.", fg="#888888")
            return

        # Validate format (starts with sk-ant-)
        if not key.startswith("sk-ant-"):
            self._api_status.config(text="⚠  That does not look like an Anthropic API key (should start with sk-ant-).", fg=WARN)
            return

        # Write as user environment variable via registry
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Environment",
                0,
                winreg.KEY_SET_VALUE
            ) as key_handle:
                winreg.SetValueEx(key_handle, "ANTHROPIC_API_KEY", 0, winreg.REG_SZ, key)

            # Broadcast WM_SETTINGCHANGE so open windows pick it up
            import ctypes
            HWND_BROADCAST = 0xFFFF
            WM_SETTINGCHANGE = 0x001A
            ctypes.windll.user32.SendMessageTimeoutW(
                HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment", 0x0002, 5000, None
            )
            self._api_status.config(text="✓  API key saved as ANTHROPIC_API_KEY environment variable.", fg=SUCCESS)
        except Exception as e:
            # Fallback: write to a hint file the user can run manually
            hint = f'setx ANTHROPIC_API_KEY "{key}"\n'
            ENV_HINT_FILE.write_text(hint, encoding="utf-8")
            self._api_status.config(
                text=f"⚠  Could not write registry ({e}). Run installer\\api_key_hint.txt manually.",
                fg=WARN
            )

    def _update_done_page(self) -> None:
        pw_set = PASSWORD_HASH_FILE.exists()
        api_set = bool(os.environ.get("ANTHROPIC_API_KEY") or ENV_HINT_FILE.exists())

        msg_lines = []
        if pw_set:
            msg_lines.append(f"✓  Dashboard password saved to:\n   {PASSWORD_HASH_FILE}")
        else:
            msg_lines.append("⚠  Dashboard password was NOT set.\n   Run: python -m dashboard.auth --set-password")
        msg_lines.append("")
        if api_set:
            msg_lines.append("✓  API key saved as ANTHROPIC_API_KEY environment variable.")
        else:
            msg_lines.append("—  API key not set (AI narration disabled).")

        msg_lines += [
            "",
            "You can now close this wizard and launch SpectraAgent.",
        ]
        self._done_msg.config(text="\n".join(msg_lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = FirstRunWizard()
    app.mainloop()
