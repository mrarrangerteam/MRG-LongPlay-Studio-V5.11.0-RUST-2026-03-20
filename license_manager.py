#!/usr/bin/env python3
"""
LongPlay Studio License Manager
HMAC-SHA256 signed serial key validation system.

Keys are generated server-side (or via CLI with secret salt set).
Validation uses HMAC to verify keys were generated with the correct secret.
"""

import os
import hmac
import json
import hashlib
import platform
from pathlib import Path
from datetime import datetime

# ==================== Configuration ====================
APP_NAME = "LongPlay Studio"
APP_VERSION = "4.24f"
LICENSE_FILE = ".longplay_license"

# Valid serial prefixes
VALID_PREFIXES = ["LP24", "LPRO", "LPVIP"]


def get_license_path() -> Path:
    """Get the license file path in user's home directory."""
    return Path.home() / LICENSE_FILE


def get_machine_id() -> str:
    """Get a unique machine identifier using SHA-256."""
    machine_info = f"{platform.node()}-{platform.machine()}-{platform.processor()}"
    return hashlib.sha256(machine_info.encode()).hexdigest()[:16].upper()


def _get_secret_salt() -> str:
    """Get SECRET_SALT — fixed embedded key for portable license validation.

    Uses LONGPLAY_SECRET_SALT env var if set (for key generation tooling),
    otherwise falls back to the embedded production salt so that keys
    validate identically on every machine / OS.
    """
    salt = os.environ.get("LONGPLAY_SECRET_SALT", "")
    if not salt:
        # Fixed production salt — same on every machine so keys are portable
        salt = "MRG-LongPlay-Studio-V5-Production-Salt-2024"
    return salt


def _compute_key_signature(prefix: str, payload: str) -> str:
    """Compute HMAC-SHA256 signature for a key payload."""
    salt = _get_secret_salt()
    message = f"{prefix}-{payload}".encode()
    return hmac.new(salt.encode(), message, hashlib.sha256).hexdigest().upper()


def generate_serial_key(prefix: str = "LP24", custom_id: str = None) -> str:
    """
    Generate a valid serial key with HMAC-SHA256 signature.
    Format: PREFIX-PPPP-PPPP-SSSS

    The first 8 hex chars (PPPP-PPPP) are the payload (timestamp + unique ID derived).
    The last 4 hex chars (SSSS) are the HMAC signature truncation.

    Args:
        prefix: Key prefix (LP24, LPRO, LPVIP)
        custom_id: Optional custom identifier for the key
    """
    if prefix not in VALID_PREFIXES:
        prefix = "LP24"

    # Generate payload from timestamp + unique data
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    unique_id = custom_id or hashlib.md5(os.urandom(16)).hexdigest()[:8]
    raw = hashlib.sha256(f"{timestamp}-{unique_id}".encode()).hexdigest().upper()
    payload = raw[:8]  # 8 hex chars of payload

    # Sign the payload with HMAC
    signature = _compute_key_signature(prefix, payload)
    sig_part = signature[:4]  # 4 hex chars of signature

    return f"{prefix}-{payload[:4]}-{payload[4:8]}-{sig_part}"


def validate_serial_key(serial: str) -> tuple[bool, str]:
    """
    Validate a serial key by re-computing and verifying its HMAC signature.

    Returns:
        (is_valid, message)
    """
    if not serial:
        return False, "Serial key is empty"

    serial = serial.strip().upper()

    # Check format: PREFIX-XXXX-XXXX-XXXX
    parts = serial.split("-")
    if len(parts) != 4:
        return False, "Invalid format. Expected: PREFIX-XXXX-XXXX-XXXX"

    prefix = parts[0]
    if prefix not in VALID_PREFIXES:
        return False, f"Invalid prefix. Expected one of: {', '.join(VALID_PREFIXES)}"

    if not all(len(p) == 4 for p in parts[1:]):
        return False, "Invalid format. Each part should be 4 characters"

    if not all(p.isalnum() for p in parts):
        return False, "Invalid characters in serial key"

    # Verify hexadecimal format
    key_body = f"{parts[1]}{parts[2]}{parts[3]}"
    try:
        int(key_body, 16)
    except ValueError:
        return False, "Invalid serial key checksum"

    # Extract payload and provided signature
    payload = f"{parts[1]}{parts[2]}"  # 8 hex chars
    provided_sig = parts[3]             # 4 hex chars

    # Re-compute HMAC signature and compare
    expected_signature = _compute_key_signature(prefix, payload)
    expected_sig_part = expected_signature[:4]

    if not hmac.compare_digest(provided_sig, expected_sig_part):
        # Backward compatibility: check if key is stored in license file
        stored = load_license()
        stored_serial = stored.get("serial", stored.get("serial_key", ""))
        if hmac.compare_digest(stored_serial.upper().encode(), serial.upper().encode()):
            pass  # Allow previously activated legacy keys
        else:
            return False, "Invalid serial key - signature verification failed"

    # Machine ID is stored for reference/analytics only — not enforced,
    # so the same license key works on any machine / OS.

    return True, f"Valid {get_license_type(prefix)} license"


def get_license_type(prefix: str) -> str:
    """Get license type from prefix."""
    types = {
        "LP24": "Standard",
        "LPRO": "Professional",
        "LPVIP": "VIP Lifetime"
    }
    return types.get(prefix, "Standard")


def save_license(serial: str, customer_name: str = "") -> bool:
    """Save license to file with restricted permissions."""
    try:
        license_data = {
            "serial": serial.strip().upper(),
            "customer_name": customer_name,
            "machine_id": get_machine_id(),
            "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "activated_at": datetime.now().isoformat(),
            "app_version": APP_VERSION
        }

        license_path = get_license_path()
        with open(license_path, "w", encoding="utf-8") as f:
            json.dump(license_data, f, indent=2)

        if os.name != 'nt':
            os.chmod(license_path, 0o600)

        return True
    except Exception as e:
        print(f"Error saving license: {e}")
        return False


def load_license() -> dict:
    """Load license from file."""
    try:
        license_path = get_license_path()
        if license_path.exists():
            with open(license_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading license: {e}")
    return {}


def check_license() -> tuple[bool, str, dict]:
    """
    Check if the application is licensed.

    Returns:
        (is_licensed, message, license_data)
    """
    license_data = load_license()

    if not license_data:
        return False, "No license found", {}

    serial = license_data.get("serial", "")
    is_valid, message = validate_serial_key(serial)

    if not is_valid:
        return False, message, license_data

    return True, message, license_data


def remove_license() -> bool:
    """Remove the license file."""
    try:
        license_path = get_license_path()
        if license_path.exists():
            os.remove(license_path)
        return True
    except Exception as e:
        print(f"Error removing license: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("LongPlay Studio - License Key Generator")
    print("=" * 50)

    print("\nGenerating sample keys:\n")

    for prefix in VALID_PREFIXES:
        key = generate_serial_key(prefix)
        license_type = get_license_type(prefix)
        is_valid, msg = validate_serial_key(key)
        print(f"  {license_type}: {key}  [valid={is_valid}]")

    print("\n" + "=" * 50)
