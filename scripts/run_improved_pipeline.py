"""
UNIFIED GAS SENSING PIPELINE
==========================

Single professional pipeline that combines real-time acquisition with advanced analysis.
This replaces fragmented approach with one cohesive system.

Features:
- Real-time spectrometer integration
- Advanced gas analysis algorithms
- Professional calibration management
- Live monitoring dashboard
- Publication-ready outputs
"""

from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

# Acquisition service lives in gas_analysis.acquisition (CCS200 driver)
try:
    from gas_analysis.acquisition.ccs200_realtime import RealtimeAcquisitionService
except ImportError:
    RealtimeAcquisitionService = None  # type: ignore[assignment, misc]

from gas_analysis.utils.generate_summary_table import generate_results_summary


class UnifiedGasSensingPipeline:
    """Unified pipeline combining real-time acquisition with analysis."""

    def __init__(self):
        self.is_running = False
        self.acquisition_service = None
        self.results_buffer = []

    def discover_spectrometers(self):
        """Discover connected spectrometers."""
        print("Discovering Spectrometers...")

        try:
            from gas_analysis.acquisition.device_discovery import discover_ccs200_resources

            resources, warnings = discover_ccs200_resources()

            if resources:
                print(f"Found {len(resources)} spectrometer(s):")
                for i, resource in enumerate(resources, 1):
                    print(f"  {i}. {resource}")
                return resources
            else:
                print("No spectrometers found")
                print("  - Check USB connections")
                print("  - Install ThorLabs drivers")
                return []
        except Exception as e:
            print(f"Discovery failed: {e}")
            return []

    def run_batch_analysis(self, gas_type: str, **kwargs):
        """Run batch analysis on existing data."""
        print(f"Running Batch Analysis for {gas_type}")
        print("-" * 50)

        try:
            # Use existing batch pipeline
            cmd = [sys.executable, "-m", "gas_analysis.core.run_each_gas", "--gas", gas_type]

            # Add any additional parameters
            for key, value in kwargs.items():
                if key == "avg_top_n":
                    cmd.extend(["--avg-top-n", str(value)])
                elif key == "top_k":
                    cmd.extend(["--top-k", str(value)])

            subprocess.run(cmd, check=True, capture_output=False)

            # Generate summary table
            output_dir = Path("output") / f"{gas_type.lower()}_topavg"
            if output_dir.exists():
                summary_path = generate_results_summary(str(output_dir))
                print(f"Summary table generated: {summary_path}")

            return True

        except subprocess.CalledProcessError as e:
            print(f"Batch analysis failed: {e}")
            return False

    def run_realtime_analysis(self, resource_string: str, duration_seconds: int | None = None):
        """Run real-time analysis with live acquisition."""
        print("Starting Real-Time Analysis")
        print("-" * 40)

        if RealtimeAcquisitionService is None:
            print("RealtimeAcquisitionService not available — install pyvisa and CCS200 drivers.")
            return False

        try:
            # Initialize acquisition service
            self.acquisition_service = RealtimeAcquisitionService(
                integration_time_ms=30.0, target_wavelength=532.0, resource_string=resource_string
            )

            # Connect to spectrometer
            print("Connecting to spectrometer...")
            self.acquisition_service.connect()

            # Capture reference spectrum
            print("Capturing reference spectrum...")
            self.acquisition_service.capture_reference_spectrum()

            # Start acquisition
            print("Starting data acquisition...")
            self.acquisition_service.start()

            self.is_running = True
            start_time = time.time()
            sample_count = 0

            # Analysis loop
            while self.is_running:
                # Check duration limit
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    break

                # Get latest spectrum
                spectrum_data = self.acquisition_service.get_latest_spectrum()
                if spectrum_data:
                    sample_count += 1

                    # Process spectrum (simplified for demo)
                    intensities = spectrum_data["intensities"]

                    # Calculate basic metrics
                    target_intensity = spectrum_data["target_intensity"]
                    snr = target_intensity / max(np.std(intensities), 1e-9)

                    # Store result
                    result = {
                        "timestamp": datetime.now(),
                        "sample_id": f"RT_{sample_count:06d}",
                        "target_intensity": target_intensity,
                        "snr": snr,
                        "wavelength": self.acquisition_service.target_wavelength,
                    }

                    self.results_buffer.append(result)

                    # Display progress
                    if sample_count % 10 == 0:
                        print(f"  Processed {sample_count} samples | SNR: {snr:.1f}")

                time.sleep(0.01)  # Small delay

            # Stop acquisition
            self.acquisition_service.stop()
            self.is_running = False

            print("Real-time analysis completed")
            print(f"   Total samples: {sample_count}")
            print(f"   Duration: {time.time() - start_time:.1f} seconds")

            return True

        except Exception as e:
            print(f"Real-time analysis failed: {e}")
            return False

    def stop_analysis(self):
        """Stop running analysis."""
        self.is_running = False
        if self.acquisition_service:
            self.acquisition_service.stop()

    def launch_dashboard(self):
        """Launch web dashboard."""
        print("Launching Web Dashboard...")

        try:
            # Use existing dashboard
            dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
            return True
        except Exception as e:
            print(f"Dashboard launch failed: {e}")
            return False

    def save_results(self, filename: str = "realtime_results.csv"):
        """Save real-time results to file."""
        if not self.results_buffer:
            print("No results to save")
            return

        import pandas as pd

        df = pd.DataFrame(self.results_buffer)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")


def main():
    """Main entry point for unified pipeline."""
    print("=" * 80)
    print("UNIFIED GAS SENSING PIPELINE")
    print("=" * 80)
    print("Professional integration of real-time acquisition and analysis")
    print()

    pipeline = UnifiedGasSensingPipeline()

    while True:
        print("MAIN MENU")
        print("1. Discover Spectrometers")
        print("2. Run Batch Analysis")
        print("3. Run Real-Time Analysis")
        print("4. Launch Dashboard")
        print("5. Save Results")
        print("6. Exit")
        print()

        try:
            choice = input("Select option (1-6): ").strip()

            if choice == "1":
                resources = pipeline.discover_spectrometers()
                if resources:
                    input("Press Enter to continue...")

            elif choice == "2":
                print("Available gases: Ethanol, Isopropanol, Methanol, MixVOC")
                gas_type = input("Enter gas type: ").strip().title()

                if gas_type in ["Ethanol", "Isopropanol", "Methanol", "MixVOC"]:
                    avg_top_n = input("Average top N frames (default 6): ").strip() or "6"
                    top_k = input("Top K candidates (default 6): ").strip() or "6"

                    pipeline.run_batch_analysis(
                        gas_type, avg_top_n=int(avg_top_n), top_k=int(top_k)
                    )
                else:
                    print("Invalid gas type")

            elif choice == "3":
                resource = input("Enter spectrometer resource string: ").strip()
                if resource:
                    duration = input("Duration in seconds (optional): ").strip()
                    duration = int(duration) if duration else None

                    pipeline.run_realtime_analysis(resource, duration)

            elif choice == "4":
                pipeline.launch_dashboard()

            elif choice == "5":
                filename = input("Output filename (default realtime_results.csv): ").strip()
                filename = filename or "realtime_results.csv"
                pipeline.save_results(filename)

            elif choice == "6":
                print("Goodbye!")
                break

            else:
                print("Invalid option. Please select 1-6.")

            if choice != "6":
                input("\nPress Enter to continue...")
                print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
