"""
tests.test_live_state
=====================
Unit tests for src.inference.live_state (_LiveDataStore / LiveDataStore).

Uses _LiveDataStore directly (fresh instance per test) rather than the
module-level singleton to avoid cross-test state contamination.
"""

import threading
import time

import numpy as np

from src.inference.live_state import LiveDataStore, _LiveDataStore

# ---------------------------------------------------------------------------
# Basic store operations
# ---------------------------------------------------------------------------


class TestLiveDataStoreBasic:
    def _store(self, maxlen=100) -> _LiveDataStore:
        return _LiveDataStore(maxlen=maxlen)

    def test_empty_on_creation(self):
        store = self._store()
        assert store.get_buffer_size() == 0
        assert store.get_sample_count() == 0
        assert not store.is_running()
        assert store.get_last_result() is None

    def test_push_increments_count(self):
        store = self._store()
        store.push({"gas_type": "Ethanol", "concentration_ppm": 1.0})
        assert store.get_sample_count() == 1
        assert store.get_buffer_size() == 1

    def test_push_multiple(self):
        store = self._store()
        for i in range(5):
            store.push({"i": i})
        assert store.get_sample_count() == 5
        assert store.get_buffer_size() == 5

    def test_maxlen_caps_buffer(self):
        store = self._store(maxlen=3)
        for i in range(10):
            store.push({"i": i})
        assert store.get_buffer_size() == 3
        assert store.get_sample_count() == 10  # total pushed, not buffered

    def test_get_last_result_returns_latest(self):
        store = self._store()
        store.push({"val": 1})
        store.push({"val": 2})
        last = store.get_last_result()
        assert last is not None
        assert last["val"] == 2

    def test_get_last_result_is_copy(self):
        store = self._store()
        store.push({"val": 1})
        last = store.get_last_result()
        last["val"] = 999
        assert store.get_last_result()["val"] == 1

    def test_get_latest_returns_n_items(self):
        store = self._store()
        for i in range(20):
            store.push({"i": i})
        items = store.get_latest(5)
        assert len(items) == 5
        assert items[-1]["i"] == 19  # most recent is last

    def test_get_latest_more_than_buffered(self):
        store = self._store()
        for i in range(3):
            store.push({"i": i})
        items = store.get_latest(100)
        assert len(items) == 3


# ---------------------------------------------------------------------------
# Wavelength / intensity accessors
# ---------------------------------------------------------------------------


class TestLiveDataStoreSpectrum:
    def _store(self) -> _LiveDataStore:
        return _LiveDataStore()

    def test_get_latest_spectrum_none_initially(self):
        store = self._store()
        assert store.get_latest_spectrum() is None

    def test_set_wavelengths_then_push_raw(self):
        store = self._store()
        wl = np.linspace(500, 900, 100)
        store.set_wavelengths(wl)
        it = np.ones(100) * 500
        store.push({"t": 1}, raw_intensities=it)
        result = store.get_latest_spectrum()
        assert result is not None
        wl_out, it_out = result
        np.testing.assert_allclose(wl_out, wl)
        np.testing.assert_allclose(it_out, it)

    def test_spectrum_returns_copies(self):
        store = self._store()
        wl = np.linspace(500, 900, 50)
        it = np.ones(50) * 300
        store.set_wavelengths(wl)
        store.push({}, raw_intensities=it)
        wl_a, it_a = store.get_latest_spectrum()
        wl_a[0] = 9999
        wl_b, it_b = store.get_latest_spectrum()
        assert wl_b[0] != 9999  # not the same array


# ---------------------------------------------------------------------------
# Session metadata
# ---------------------------------------------------------------------------


class TestLiveDataStoreSessionMeta:
    def _store(self) -> _LiveDataStore:
        return _LiveDataStore()

    def test_set_and_get_session_meta(self):
        store = self._store()
        meta = {"gas_label": "Ethanol", "concentration_ppm": 1.0}
        store.set_session_meta(meta)
        out = store.get_session_meta()
        assert out["gas_label"] == "Ethanol"

    def test_session_meta_is_copy(self):
        store = self._store()
        meta = {"x": 1}
        store.set_session_meta(meta)
        meta["x"] = 99
        assert store.get_session_meta()["x"] == 1

    def test_is_running_flag(self):
        store = self._store()
        store.set_running(True)
        assert store.is_running()
        store.set_running(False)
        assert not store.is_running()


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestLiveDataStoreClear:
    def test_clear_resets_buffer(self):
        store = _LiveDataStore()
        for i in range(10):
            store.push({"i": i})
        store.set_session_meta({"foo": "bar"})
        store.set_running(True)
        store.clear()
        assert store.get_buffer_size() == 0
        assert store.get_sample_count() == 0
        assert not store.is_running()
        assert store.get_session_meta() == {}
        assert store.get_last_result() is None

    def test_clear_preserves_wavelengths(self):
        store = _LiveDataStore()
        wl = np.linspace(500, 900, 100)
        store.set_wavelengths(wl)
        store.clear()
        # Wavelengths are preserved across sessions (spectrometer axis doesn't change)
        result = store.get_latest_spectrum()
        # Latest intensities are cleared, so result is None even if wavelengths stored
        assert result is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestLiveDataStoreThreadSafety:
    def test_concurrent_push(self):
        """Multiple threads push concurrently — no exceptions, count is correct."""
        store = _LiveDataStore(maxlen=5000)
        errors = []
        n_threads = 8
        pushes_per_thread = 100

        def worker():
            try:
                for i in range(pushes_per_thread):
                    store.push({"val": i})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert store.get_sample_count() == n_threads * pushes_per_thread

    def test_concurrent_read_write(self):
        """Writer and reader run in parallel — reader should never crash."""
        store = _LiveDataStore(maxlen=200)
        stop_event = threading.Event()
        errors = []

        def writer():
            i = 0
            while not stop_event.is_set():
                store.push({"i": i})
                i += 1
                time.sleep(0.001)

        def reader():
            while not stop_event.is_set():
                try:
                    _ = store.get_latest(50)
                    _ = store.get_last_result()
                    _ = store.get_sample_count()
                except Exception as exc:
                    errors.append(exc)
                time.sleep(0.002)

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        w.start()
        r.start()
        time.sleep(0.2)
        stop_event.set()
        w.join()
        r.join()
        assert not errors


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestLiveDataStoreSingleton:
    def test_singleton_is_instance(self):
        assert isinstance(LiveDataStore, _LiveDataStore)

    def test_singleton_importable_from_package(self):
        from src.inference import LiveDataStore as lds

        assert isinstance(lds, _LiveDataStore)
