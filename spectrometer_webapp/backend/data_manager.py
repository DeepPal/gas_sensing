import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

class DataManager:
    def __init__(self, data_dir: str = "../data/acquisitions"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = None

    def start_logging_session(self, metadata: dict):
        """Start a new logging session folder."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        gas = metadata.get("gas_type", "unknown").replace(" ", "_")
        session_name = f"{timestamp}_{gas}"
        self.current_session = self.data_dir / session_name
        self.current_session.mkdir(parents=True, exist_ok=True)
        
        with open(self.current_session / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        return self.current_session

    def save_spectrum(self, wavelengths, intensities):
        """Save a single spectrum measurement as CSV."""
        if not self.current_session:
            return
        
        idx = len(list(self.current_session.glob("*.csv")))
        filepath = self.current_session / f"spectrum_{idx:05d}.csv"
        
        df = pd.DataFrame({"wavelength": wavelengths, "intensity": intensities})
        df.to_csv(filepath, index=False)

    def stop_logging_session(self):
        self.current_session = None

    def get_sessions(self):
        """Return list of available session folders."""
        if not self.data_dir.exists():
            return []
        sessions = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        return sorted(sessions, reverse=True)

    def load_session_data(self, session_name: str):
        """Load all spectra and metadata for a given session."""
        session_path = self.data_dir / session_name
        if not session_path.exists():
            return None, None, None
        
        metadata_path = session_path / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        csv_files = sorted(session_path.glob("*.csv"))
        if not csv_files:
            return np.array([]), np.array([]), metadata

        spectra = []
        for f in csv_files:
            df = pd.read_csv(f)
            spectra.append(df['intensity'].values)

        wavelengths = pd.read_csv(csv_files[0])['wavelength'].values
        return np.array(spectra), wavelengths, metadata
