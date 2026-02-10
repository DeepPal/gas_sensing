"""Developer entrypoint for running the automation toolkit programmatically."""

from __future__ import annotations

import logging
from pathlib import Path

from slides_automation.config import load_config
from slides_automation.orchestrator import PresentationProject

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    config_path = Path("config/presentation_scientific.yaml")
    config = load_config(config_path)
    project = PresentationProject(config)

    result = project.run()

    if result.google_slides_link:
        logging.info("Google Slides deck: %s", result.google_slides_link)
    if result.pptx_path:
        logging.info("Local PPTX saved to: %s", result.pptx_path)

if __name__ == "__main__":
    main()
