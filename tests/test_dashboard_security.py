from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import dashboard.security as security


class _Completed:
    def __init__(self, stdout: str = "") -> None:
        self.stdout = stdout


def test_is_cert_valid_true_with_future_not_after(monkeypatch) -> None:
    future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%b %d %H:%M:%S %Y GMT")

    def _fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        return _Completed(stdout=f"notBefore=Jan 01 00:00:00 2020 GMT\nnotAfter={future}\n")

    monkeypatch.setattr(security.subprocess, "run", _fake_run)
    assert security._is_cert_valid(Path("dummy.crt"), days_remaining=7) is True


def test_is_cert_valid_false_when_subprocess_fails(monkeypatch) -> None:
    def _raise(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("openssl failure")

    monkeypatch.setattr(security.subprocess, "run", _raise)
    assert security._is_cert_valid(Path("dummy.crt")) is False


def test_generate_self_signed_cert_reuses_existing_valid_cert(tmp_path: Path, monkeypatch) -> None:
    cert_dir = tmp_path / ".streamlit" / "certs"
    cert_dir.mkdir(parents=True)
    cert_file = cert_dir / "server.crt"
    key_file = cert_dir / "server.key"
    cert_file.write_text("cert", encoding="utf-8")
    key_file.write_text("key", encoding="utf-8")

    monkeypatch.setattr(security, "_is_cert_valid", lambda *_args, **_kwargs: True)

    cert_path, key_path = security.generate_self_signed_cert(app_root=tmp_path, overwrite=False)

    assert cert_path == cert_file
    assert key_path == key_file


def test_setup_https_returns_cert_and_key_paths(tmp_path: Path, monkeypatch) -> None:
    expected_cert = tmp_path / "cert.crt"
    expected_key = tmp_path / "key.key"

    monkeypatch.setattr(
        security,
        "generate_self_signed_cert",
        lambda app_root=None: (expected_cert, expected_key),
    )

    result = security.setup_https(app_root=tmp_path)

    assert result["certfile"] == expected_cert
    assert result["keyfile"] == expected_key
