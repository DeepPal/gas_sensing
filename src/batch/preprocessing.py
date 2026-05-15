"""Batch preprocessing utilities — file sorting and frame ordering.

All functions are CONFIG-free: every parameter is passed explicitly.
"""
from __future__ import annotations

from collections.abc import Sequence
import logging
import math
import os
import re

log = logging.getLogger(__name__)


def sort_frame_paths(paths: Sequence[str]) -> list[str]:
    def _key(p: str) -> tuple[int, float, float]:
        name = os.path.basename(p)

        # Prefer explicit timestamp in names like: t1_20241029_11h25m41s826ms
        # trial = t1, date = 20241029, time = 11:25:41.826
        ts_match = re.search(
            r"t(?P<trial>\d+)[^0-9]*(?P<date>\d{8})_(?P<hour>\d{1,2})h(?P<minute>\d{1,2})m(?P<second>\d{1,2})s(?P<msec>\d{1,3})ms",
            name,
        )
        if ts_match:
            try:
                date_str = ts_match.group("date")  # YYYYMMDD
                hour = int(ts_match.group("hour"))
                minute = int(ts_match.group("minute"))
                second = int(ts_match.group("second"))
                msec = int(ts_match.group("msec"))

                date_int = int(date_str)
                time_key = ((hour * 3600 + minute * 60 + second) * 1000.0) + float(msec)
                try:
                    mtime = os.path.getmtime(p)
                except OSError:
                    mtime = float("inf")
                return date_int, time_key, mtime
            except Exception:
                # Fall back to generic numeric+mtime ordering
                pass

        # Fallback: original behavior based on last numeric token and modification time
        digits = re.findall(r"(\d+)", name)
        if digits:
            try:
                idx = int(digits[-1])
            except ValueError:
                idx = math.inf  # type: ignore[assignment]
        else:
            idx = math.inf  # type: ignore[assignment]
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            log.warning("sort_frame_paths: cannot stat %r — placing at end of sort order.", p)
            mtime = float("inf")
        int_idx = int(idx) if math.isfinite(idx) else 2**31 - 1
        float_idx = float(idx) if math.isfinite(idx) else float("inf")
        return (int_idx, float_idx, mtime)

    return sorted(paths, key=_key)
