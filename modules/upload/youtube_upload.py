"""
YouTube upload module — OAuth2 authentication and resumable video upload.

Uses Google API v3 with resumable uploads for large files.

Classes:
    YouTubeUploader — Handles OAuth2 auth and video upload to YouTube.
"""

import os
import time
import json
import logging
import http.client
import httplib2
from pathlib import Path
from typing import Optional, Callable, List

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

logger = logging.getLogger(__name__)

# OAuth2 scope for uploading videos
YOUTUBE_UPLOAD_SCOPE = ["https://www.googleapis.com/auth/youtube.upload"]

# Credential storage path
CREDENTIALS_DIR = Path.home() / ".longplay_studio"
CREDENTIALS_FILE = CREDENTIALS_DIR / "youtube_credentials.json"
CLIENT_SECRETS_FILE = CREDENTIALS_DIR / "client_secrets.json"

# Retry configuration
MAX_RETRIES = 10
RETRIABLE_STATUS_CODES = [500, 502, 503, 504]
RETRIABLE_EXCEPTIONS = (httplib2.HttpLib2Error, IOError, http.client.NotConnected,
                        http.client.IncompleteRead, http.client.ImproperConnectionState,
                        http.client.CannotSendRequest, http.client.CannotSendHeader,
                        http.client.ResponseNotReady, http.client.BadStatusLine)


class YouTubeUploader:
    """Handles YouTube OAuth2 authentication and resumable video uploads.

    Usage:
        uploader = YouTubeUploader()
        if uploader.authenticate():
            url = uploader.upload_video(
                video_path="mix.mp4",
                title="My Mix",
                description="Track list...",
                tags=["chill", "music"],
                progress_callback=lambda pct: print(f"{pct:.1f}%"),
            )
            print(f"Uploaded: {url}")
    """

    def __init__(self, client_secrets_path: Optional[str] = None):
        """Initialize the uploader.

        Args:
            client_secrets_path: Path to Google OAuth2 client_secrets.json.
                Defaults to ~/.longplay_studio/client_secrets.json.
        """
        self._client_secrets = Path(client_secrets_path) if client_secrets_path else CLIENT_SECRETS_FILE
        self._credentials: Optional[Credentials] = None
        self._youtube = None
        self._upload_progress: float = 0.0
        self._cancelled = False

        # Ensure credential directory exists
        CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def is_authenticated(self) -> bool:
        """Check if valid (or refreshable) credentials exist.

        Returns:
            True if credentials are loaded and not expired (or can be refreshed).
        """
        if self._credentials and self._credentials.valid:
            return True

        # Try loading from disk
        if CREDENTIALS_FILE.exists():
            try:
                self._credentials = Credentials.from_authorized_user_file(
                    str(CREDENTIALS_FILE), YOUTUBE_UPLOAD_SCOPE
                )
                if self._credentials and self._credentials.valid:
                    return True
                if self._credentials and self._credentials.expired and self._credentials.refresh_token:
                    self._credentials.refresh(Request())
                    self._save_credentials()
                    return self._credentials.valid
            except Exception as exc:
                logger.warning("Failed to load/refresh credentials: %s", exc)
                self._credentials = None

        return False

    def authenticate(self) -> bool:
        """Run the full OAuth2 flow (opens browser).

        If valid credentials already exist on disk they are reused.  Otherwise
        the browser-based consent flow is launched.

        Returns:
            True on success.
        """
        # Quick path — already authed
        if self.is_authenticated():
            self._build_service()
            return True

        # Verify client secrets exist
        if not self._client_secrets.exists():
            logger.error(
                "client_secrets.json not found at %s. "
                "Download it from the Google Cloud Console and place it there.",
                self._client_secrets,
            )
            return False

        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self._client_secrets), YOUTUBE_UPLOAD_SCOPE
            )
            self._credentials = flow.run_local_server(
                port=0,  # Pick a random free port
                prompt="consent",
                success_message="Authentication successful! You can close this tab.",
            )
            self._save_credentials()
            self._build_service()
            return True
        except Exception as exc:
            logger.error("OAuth2 flow failed: %s", exc)
            return False

    def revoke(self) -> bool:
        """Revoke stored credentials and delete the local file.

        Returns:
            True if credentials were removed.
        """
        try:
            if self._credentials:
                self._credentials = None
            if CREDENTIALS_FILE.exists():
                CREDENTIALS_FILE.unlink()
            self._youtube = None
            return True
        except Exception as exc:
            logger.error("Failed to revoke credentials: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_video(
        self,
        video_path: str,
        title: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        category_id: str = "10",
        privacy: str = "private",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Optional[str]:
        """Upload a video to YouTube using resumable upload.

        Args:
            video_path: Path to the video file.
            title: Video title (max 100 characters).
            description: Video description (max 5000 characters).
            tags: List of tags.
            category_id: YouTube category ID. Default "10" (Music).
            privacy: "public", "unlisted", or "private".
            progress_callback: Called with upload percentage (0.0-100.0).

        Returns:
            The YouTube video URL on success, None on failure.
        """
        if not os.path.isfile(video_path):
            logger.error("Video file not found: %s", video_path)
            return None

        if not self.is_authenticated():
            logger.error("Not authenticated. Call authenticate() first.")
            return None

        self._build_service()
        if not self._youtube:
            logger.error("Failed to build YouTube service.")
            return None

        self._upload_progress = 0.0
        self._cancelled = False

        # Clamp values to YouTube limits
        title = title[:100]
        description = description[:5000]
        tags = tags or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": False,
            },
        }

        file_size = os.path.getsize(video_path)
        # Use resumable upload; chunk size 10 MB for large files
        chunk_size = 10 * 1024 * 1024  # 10 MB
        media = MediaFileUpload(
            video_path,
            mimetype="video/*",
            chunksize=chunk_size,
            resumable=True,
        )

        try:
            request = self._youtube.videos().insert(
                part=",".join(body.keys()),
                body=body,
                media_body=media,
            )
            video_id = self._resumable_upload(request, file_size, progress_callback)
            if video_id:
                url = f"https://youtu.be/{video_id}"
                logger.info("Upload complete: %s", url)
                return url
            return None
        except HttpError as exc:
            logger.error("YouTube API error during upload: %s", exc)
            return None
        except Exception as exc:
            logger.error("Unexpected error during upload: %s", exc)
            return None

    def get_upload_progress(self) -> float:
        """Return the current upload progress percentage (0.0 - 100.0)."""
        return self._upload_progress

    def cancel_upload(self):
        """Signal that the current upload should be cancelled."""
        self._cancelled = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_service(self):
        """Build the YouTube Data API v3 service client."""
        if self._youtube is None and self._credentials:
            self._youtube = build("youtube", "v3", credentials=self._credentials)

    def _save_credentials(self):
        """Persist credentials to disk (permissions 0600)."""
        if self._credentials:
            CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
            with open(CREDENTIALS_FILE, "w") as f:
                f.write(self._credentials.to_json())
            # Restrict file permissions to owner-only
            os.chmod(CREDENTIALS_FILE, 0o600)
            logger.info("Credentials saved to %s", CREDENTIALS_FILE)

    def _resumable_upload(
        self,
        request,
        file_size: int,
        progress_callback: Optional[Callable[[float], None]],
    ) -> Optional[str]:
        """Execute a resumable upload with exponential-backoff retry.

        Returns:
            The YouTube video ID on success, None on failure.
        """
        response = None
        error = None
        retry = 0

        while response is None:
            if self._cancelled:
                logger.info("Upload cancelled by user.")
                return None

            try:
                status, response = request.next_chunk()
                if status:
                    pct = status.progress() * 100.0
                    self._upload_progress = pct
                    if progress_callback:
                        progress_callback(pct)
                    logger.debug("Upload progress: %.1f%%", pct)

                if response is not None:
                    video_id = response.get("id")
                    if video_id:
                        self._upload_progress = 100.0
                        if progress_callback:
                            progress_callback(100.0)
                        return video_id
                    else:
                        logger.error("Upload response missing video id: %s", response)
                        return None

            except HttpError as exc:
                if exc.resp.status in RETRIABLE_STATUS_CODES:
                    error = exc
                else:
                    raise

            except RETRIABLE_EXCEPTIONS as exc:
                error = exc

            if error is not None:
                retry += 1
                if retry > MAX_RETRIES:
                    logger.error("Maximum retries exceeded. Last error: %s", error)
                    return None

                wait = min(2 ** retry, 64) + (time.time() % 1)  # jitter
                logger.warning(
                    "Retriable error (attempt %d/%d), retrying in %.1fs: %s",
                    retry, MAX_RETRIES, wait, error,
                )
                time.sleep(wait)
                error = None

        return None
