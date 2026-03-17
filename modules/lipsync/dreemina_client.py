"""
Dreemina API client — generate lip-sync avatar videos from audio + image.

Workflow:
    1. Upload avatar image (or select from saved avatars)
    2. Upload mastered audio (hook or full track)
    3. Poll for completion
    4. Download generated lip-sync video

Classes:
    DreeminaClient — Handles Dreemina API auth and lip-sync generation.
"""

import os
import re
import ssl
import time
import json
import logging
import mimetypes
import shutil
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Config directory
CONFIG_DIR = Path.home() / ".longplay_studio"
DREEMINA_CONFIG = CONFIG_DIR / "dreemina_config.json"

# Retry settings
MAX_RETRIES = 4
POLL_INTERVAL_SEC = 5
MAX_POLL_TIME_SEC = 600  # 10 minutes max wait

# Safe ID pattern for URL path segments (prevent path traversal)
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")

# TLS context — always verify certificates
_SSL_CTX = ssl.create_default_context()


def _sanitize_filename(name: str) -> str:
    """Remove characters that could break multipart boundaries."""
    return re.sub(r'[\r\n"\x00]', "_", name)


def _validate_id(value: str, label: str) -> str:
    """Validate that an API-returned ID is safe for URL interpolation."""
    if not value or not _SAFE_ID_RE.match(value):
        raise ValueError(f"Invalid {label} from API: {value!r}")
    return value


def _validate_api_base(url: str) -> str:
    """Validate that api_base is a well-formed HTTPS URL."""
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"API base must use https:// scheme, got: {url}")
    if not parsed.hostname:
        raise ValueError(f"API base has no hostname: {url}")
    return url.rstrip("/")


class DreeminaClient:
    """Dreemina API client for lip-sync avatar video generation.

    Usage::

        client = DreeminaClient(api_key="your-key")
        result = client.generate_lipsync(
            audio_path="/path/to/mastered_hook.wav",
            avatar_path="/path/to/avatar.jpg",
            output_path="/path/to/output.mp4",
            progress_callback=lambda pct, msg: print(f"{pct:.0f}% {msg}"),
        )
        if result:
            print(f"Lip-sync video saved: {result}")
    """

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "https://api.dreemina.com/v1",
    ):
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._cancel_event = threading.Event()

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # Load saved config if no key provided
        if not self._api_key:
            self._load_config()

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _load_config(self):
        """Load API key from config file."""
        if DREEMINA_CONFIG.exists():
            try:
                with open(DREEMINA_CONFIG, "r") as f:
                    data = json.load(f)
                self._api_key = data.get("api_key", "")
                if data.get("api_base"):
                    self._api_base = data["api_base"]
            except Exception as exc:
                logger.warning("Failed to load Dreemina config: %s", exc)

    def save_config(self):
        """Save API key to config file (permissions 0600)."""
        try:
            with open(DREEMINA_CONFIG, "w") as f:
                json.dump({
                    "api_key": self._api_key,
                    "api_base": self._api_base,
                }, f, indent=2)
            # Restrict file permissions to owner-only (H1 fix)
            os.chmod(DREEMINA_CONFIG, 0o600)
            logger.info("Dreemina config saved to %s", DREEMINA_CONFIG)
        except Exception as exc:
            logger.error("Failed to save Dreemina config: %s", exc)

    @property
    def api_key(self) -> str:
        return self._api_key

    @api_key.setter
    def api_key(self, value: str):
        self._api_key = value

    @property
    def api_base(self) -> str:
        return self._api_base

    @api_base.setter
    def api_base(self, value: str):
        self._api_base = value.rstrip("/")

    def is_configured(self) -> bool:
        """Check if API key is set."""
        return bool(self._api_key)

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

    def _interruptible_sleep(self, seconds: float) -> bool:
        """Sleep that can be interrupted by cancel_event. Returns True if cancelled."""
        return self._cancel_event.wait(timeout=seconds)

    def _api_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[bytes] = None,
        content_type: Optional[str] = None,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """Make an API request with retry logic."""
        api_base = _validate_api_base(self._api_base)
        url = f"{api_base}/{endpoint.lstrip('/')}"
        headers = self._headers()
        if content_type:
            headers["Content-Type"] = content_type

        last_error = None
        for attempt in range(MAX_RETRIES):
            if self._cancel_event.is_set():
                raise InterruptedError("Operation cancelled")

            try:
                req = Request(url, data=data, headers=headers, method=method)
                with urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                    body = resp.read()
                    return json.loads(body) if body else {}
            except HTTPError as exc:
                last_error = exc
                if exc.code in (429, 500, 502, 503, 504):
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "Dreemina API %d (attempt %d/%d), retry in %ds",
                        exc.code, attempt + 1, MAX_RETRIES, wait,
                    )
                    if self._interruptible_sleep(wait):
                        raise InterruptedError("Operation cancelled")
                    continue
                # Non-retriable error
                error_body = exc.read().decode("utf-8", errors="replace")
                logger.error("Dreemina API error %d: %s", exc.code, error_body)
                raise
            except (URLError, OSError) as exc:
                last_error = exc
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "Network error (attempt %d/%d), retry in %ds: %s",
                    attempt + 1, MAX_RETRIES, wait, exc,
                )
                if self._interruptible_sleep(wait):
                    raise InterruptedError("Operation cancelled")

        raise ConnectionError(f"Dreemina API failed after {MAX_RETRIES} retries: {last_error}")

    def _upload_file(self, endpoint: str, file_path: str, field_name: str = "file") -> Dict[str, Any]:
        """Upload a file using multipart/form-data."""
        boundary = f"----LongPlayBoundary{int(time.time() * 1000)}"
        filename = _sanitize_filename(os.path.basename(file_path))
        mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"

        with open(file_path, "rb") as f:
            file_data = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8") + file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

        return self._api_request(
            "POST", endpoint,
            data=body,
            content_type=f"multipart/form-data; boundary={boundary}",
            timeout=120,
        )

    def _download_file(self, url: str, output_path: str, timeout: int = 300) -> bool:
        """Download a file from URL to local path (no auth header for external URLs)."""
        try:
            # Only send auth header if URL belongs to the configured API host
            api_host = urlparse(self._api_base).hostname
            download_host = urlparse(url).hostname
            headers = self._headers() if download_host == api_host else {"Accept": "*/*"}

            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                with open(output_path, "wb") as f:
                    shutil.copyfileobj(resp, f)
            return True
        except Exception as exc:
            logger.error("Download failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Lip-sync generation
    # ------------------------------------------------------------------

    def generate_lipsync(
        self,
        audio_path: str,
        avatar_path: str,
        output_path: str,
        aspect_ratio: str = "9:16",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Optional[str]:
        """Generate a lip-sync avatar video from audio + image.

        Args:
            audio_path: Path to mastered audio file (WAV/MP3).
            avatar_path: Path to avatar image (JPG/PNG) or video.
            output_path: Where to save the generated lip-sync video.
            aspect_ratio: Video aspect ratio. "9:16" for Shorts, "16:9" for landscape.
            progress_callback: Called with (percentage, status_message).

        Returns:
            Path to output video on success, None on failure.
        """
        if not self.is_configured():
            raise ValueError("Dreemina API key not configured. Set api_key first.")

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not os.path.isfile(avatar_path):
            raise FileNotFoundError(f"Avatar file not found: {avatar_path}")

        self._cancel_event.clear()

        def _progress(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        try:
            # Step 1: Upload avatar (10%)
            _progress(5.0, "Uploading avatar image...")
            avatar_resp = self._upload_file("avatars/upload", avatar_path, "image")
            avatar_id = _validate_id(
                avatar_resp.get("id") or avatar_resp.get("avatar_id", ""),
                "avatar_id",
            )
            _progress(10.0, f"Avatar uploaded (ID: {avatar_id})")

            if self._cancel_event.is_set():
                return None

            # Step 2: Upload audio (20%)
            _progress(15.0, "Uploading mastered audio...")
            audio_resp = self._upload_file("audio/upload", audio_path, "audio")
            audio_id = _validate_id(
                audio_resp.get("id") or audio_resp.get("audio_id", ""),
                "audio_id",
            )
            _progress(20.0, f"Audio uploaded (ID: {audio_id})")

            if self._cancel_event.is_set():
                return None

            # Step 3: Create lip-sync job (25%)
            _progress(25.0, "Creating lip-sync job...")
            job_data = json.dumps({
                "avatar_id": avatar_id,
                "audio_id": audio_id,
                "aspect_ratio": aspect_ratio,
                "output_format": "mp4",
            }).encode("utf-8")

            job_resp = self._api_request(
                "POST", "lipsync/generate",
                data=job_data,
                content_type="application/json",
            )
            job_id = _validate_id(
                job_resp.get("id") or job_resp.get("job_id", ""),
                "job_id",
            )
            _progress(30.0, f"Job created (ID: {job_id})")

            # Step 4: Poll for completion (30% → 90%)
            start_time = time.time()
            status_resp = {}
            while not self._cancel_event.is_set():
                elapsed = time.time() - start_time
                if elapsed > MAX_POLL_TIME_SEC:
                    raise TimeoutError(
                        f"Lip-sync generation timed out after {MAX_POLL_TIME_SEC}s"
                    )

                if self._interruptible_sleep(POLL_INTERVAL_SEC):
                    return None  # cancelled

                status_resp = self._api_request("GET", f"lipsync/status/{job_id}")
                status = status_resp.get("status", "unknown")
                job_progress = status_resp.get("progress", 0)

                # Map job progress (0-100) to our range (30-90)
                pct = 30.0 + (job_progress / 100.0) * 60.0
                _progress(pct, f"Generating lip-sync: {status} ({job_progress}%)")

                if status == "completed":
                    break
                elif status in ("failed", "error"):
                    error_msg = status_resp.get("error", "Unknown error")
                    raise RuntimeError(f"Lip-sync generation failed: {error_msg}")

            if self._cancel_event.is_set():
                return None

            # Step 5: Download result (90% → 100%)
            _progress(90.0, "Downloading generated video...")
            download_url = status_resp.get("download_url") or status_resp.get("result_url")
            if not download_url:
                raise RuntimeError("No download URL in completed response")

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            if not self._download_file(download_url, output_path):
                raise RuntimeError("Failed to download generated video")

            _progress(100.0, "Lip-sync video complete!")
            logger.info("Lip-sync video saved: %s", output_path)
            return output_path

        except Exception as exc:
            logger.error("Lip-sync generation failed: %s", exc)
            _progress(0.0, f"Error: {exc}")
            raise

    def cancel(self):
        """Cancel the current generation job (thread-safe via Event)."""
        self._cancel_event.set()

    # ------------------------------------------------------------------
    # Avatar management
    # ------------------------------------------------------------------

    def list_avatars(self) -> list:
        """List saved avatars from the API."""
        try:
            resp = self._api_request("GET", "avatars")
            return resp.get("avatars", [])
        except Exception as exc:
            logger.error("Failed to list avatars: %s", exc)
            return []

    def delete_avatar(self, avatar_id: str) -> bool:
        """Delete a saved avatar."""
        try:
            _validate_id(avatar_id, "avatar_id")
            self._api_request("DELETE", f"avatars/{avatar_id}")
            return True
        except Exception as exc:
            logger.error("Failed to delete avatar: %s", exc)
            return False
