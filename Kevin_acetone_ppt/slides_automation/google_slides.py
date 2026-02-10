from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from .config import GoogleSlidesConfig, ImagePlacement

LOGGER = logging.getLogger(__name__)


class GoogleSlidesAutomator:
    """High-level helper around the Google Slides/Drive APIs."""

    SLIDES_SCOPE = "https://www.googleapis.com/auth/presentations"
    DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"

    def __init__(self, credentials_file: str):
        cred_path = Path(credentials_file).expanduser()
        scopes = [self.SLIDES_SCOPE, self.DRIVE_SCOPE]

        LOGGER.debug("Loading credentials from %s", cred_path)
        creds = service_account.Credentials.from_service_account_file(
            str(cred_path), scopes=scopes
        )
        self._slides: Resource = build("slides", "v1", credentials=creds)
        self._drive: Resource = build("drive", "v3", credentials=creds)

    # ------------------------------------------------------------------
    # Presentation lifecycle helpers
    # ------------------------------------------------------------------
    def clone_template(
        self, template_id: str, *, title: str, folder_id: Optional[str] = None
    ) -> Dict[str, str]:
        """Copy an existing presentation template using the Drive API."""

        metadata: Dict[str, str | List[str]] = {"name": title}
        if folder_id:
            metadata["parents"] = [folder_id]

        LOGGER.info("Cloning template %s into %s", template_id, title)
        result = (
            self._drive.files()
            .copy(fileId=template_id, body=metadata, fields="id, webViewLink")
            .execute()
        )
        return result

    def replace_placeholders(
        self, presentation_id: str, replacements: Mapping[str, str]
    ) -> None:
        """Replace placeholders throughout the presentation."""

        if not replacements:
            LOGGER.debug("No replacements provided; skipping")
            return

        requests = [
            {
                "replaceAllText": {
                    "containsText": {"text": placeholder, "matchCase": False},
                    "replaceText": value,
                }
            }
            for placeholder, value in replacements.items()
        ]

        LOGGER.info(
            "Applying %d text replacements to presentation %s",
            len(requests),
            presentation_id,
        )
        self._slides.presentations().batchUpdate(
            presentationId=presentation_id, body={"requests": requests}
        ).execute()

    def insert_images(
        self,
        presentation_id: str,
        placements: Iterable[ImagePlacement],
        *,
        slide_index_base: int = 0,
    ) -> None:
        """Upload local images to Drive, then embed them on specific slides."""

        placements = list(placements)
        if not placements:
            LOGGER.debug("No image placements configured")
            return

        presentation = self._slides.presentations().get(
            presentationId=presentation_id
        ).execute()
        slides = presentation.get("slides", [])

        requests = []
        for placement in placements:
            slide_index = placement.slide_index - slide_index_base
            if slide_index < 0:
                LOGGER.warning(
                    "Slide index %d is invalid (base=%d); skipping image %s",
                    placement.slide_index,
                    slide_index_base,
                    placement.path,
                )
                continue
            if slide_index >= len(slides):
                LOGGER.warning(
                    "Slide index %d is out of range; skipping image %s",
                    placement.slide_index,
                    placement.path,
                )
                continue

            uploaded = self._upload_image(placement.path)
            if not uploaded:
                continue

            slide_id = slides[slide_index]["objectId"]
            object_id = f"image_{uuid.uuid4().hex}"
            requests.append(
                {
                    "createImage": {
                        "objectId": object_id,
                        "url": f"https://drive.google.com/uc?id={uploaded}",
                        "elementProperties": {
                            "pageObjectId": slide_id,
                            "size": {
                                "width": {
                                    "magnitude": placement.width,
                                    "unit": "INCHES",
                                },
                                "height": {
                                    "magnitude": placement.height,
                                    "unit": "INCHES",
                                },
                            },
                            "transform": {
                                "scaleX": 1,
                                "scaleY": 1,
                                "translateX": placement.left,
                                "translateY": placement.top,
                                "unit": "INCHES",
                            },
                        },
                    }
                }
            )

        if requests:
            LOGGER.info(
                "Embedding %d images into presentation %s",
                len(requests),
                presentation_id,
            )
            self._slides.presentations().batchUpdate(
                presentationId=presentation_id, body={"requests": requests}
            ).execute()

    def share_with_users(
        self, presentation_id: str, emails: Iterable[str], role: str = "reader"
    ) -> None:
        emails = [email for email in emails if email]
        if not emails:
            return

        for email in emails:
            try:
                LOGGER.info("Granting %s access to %s", email, presentation_id)
                self._drive.permissions().create(
                    fileId=presentation_id,
                    body={"type": "user", "role": role, "emailAddress": email},
                    sendNotificationEmail=False,
                ).execute()
            except HttpError as exc:
                LOGGER.error("Failed to share with %s: %s", email, exc)

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------
    def generate_from_config(
        self,
        config: GoogleSlidesConfig,
        *,
        title: str,
        replacements: Mapping[str, str],
    ) -> Dict[str, str]:
        """Clone template, apply replacements, embed images, then share."""

        copy_result = self.clone_template(
            config.template_id, title=title, folder_id=config.folder_id
        )
        presentation_id = copy_result["id"]

        self.replace_placeholders(presentation_id, replacements)
        self.insert_images(
            presentation_id,
            config.image_uploads,
            slide_index_base=config.slide_index_base,
        )
        self.share_with_users(presentation_id, config.share_emails)
        copy_result.setdefault(
            "webViewLink", f"https://docs.google.com/presentation/d/{presentation_id}/edit"
        )
        return copy_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _upload_image(self, path: str) -> Optional[str]:
        file_path = Path(path)
        if not file_path.exists():
            LOGGER.error("Image %s does not exist; skipping", file_path)
            return None

        LOGGER.debug("Uploading image %s to Drive", file_path)
        media = MediaFileUpload(str(file_path))
        uploaded = (
            self._drive.files()
            .create(body={"name": file_path.name}, media_body=media, fields="id")
            .execute()
        )
        return uploaded.get("id")


# typing helpers
