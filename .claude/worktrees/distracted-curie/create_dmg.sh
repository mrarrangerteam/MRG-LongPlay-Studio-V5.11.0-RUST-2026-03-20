#!/bin/bash
# ═══════════════════════════════════════════════════════════
# MRG LongPlay Studio V5.10.0 — DMG Installer Builder
# ═══════════════════════════════════════════════════════════
#
# Usage:
#   ./create_dmg.sh              # build .app + create .dmg
#   ./create_dmg.sh --skip-build # create .dmg from existing .app
#   ./create_dmg.sh --clean      # remove build artifacts
#

set -euo pipefail

# ── Config ──────────────────────────────────────────────────
APP_NAME="LongPlay Studio"
APP_VERSION="5.10.0"
DMG_NAME="LongPlay-Studio-V${APP_VERSION}"
DMG_VOLUME_NAME="LongPlay Studio V${APP_VERSION}"
BUNDLE_ID="com.mrg.longplay-studio"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="${SCRIPT_DIR}/dist"
BUILD_DIR="${SCRIPT_DIR}/build"
DMG_DIR="${SCRIPT_DIR}/dmg_staging"
DMG_OUTPUT="${DIST_DIR}/${DMG_NAME}.dmg"
APP_PATH="${DIST_DIR}/${APP_NAME}.app"

# ── Helpers ─────────────────────────────────────────────────
log()  { echo "[DMG] $*"; }
err()  { echo "[DMG] ERROR: $*" >&2; }
bold() { echo -e "\033[1m$*\033[0m"; }

cleanup_staging() {
    if [ -d "$DMG_DIR" ]; then
        rm -rf "$DMG_DIR"
    fi
    # Detach any leftover mounts
    if hdiutil info 2>/dev/null | grep -q "$DMG_VOLUME_NAME"; then
        hdiutil detach "/Volumes/${DMG_VOLUME_NAME}" -force 2>/dev/null || true
    fi
}

# ── Clean ───────────────────────────────────────────────────
if [[ "${1:-}" == "--clean" ]]; then
    log "Cleaning build artifacts..."
    rm -rf "$DIST_DIR" "$BUILD_DIR" "$DMG_DIR"
    rm -f "${SCRIPT_DIR}/longplay_v55.spec"
    log "Clean complete."
    exit 0
fi

bold "═══════════════════════════════════════════════════"
bold "  MRG LongPlay Studio V${APP_VERSION} — DMG Builder"
bold "═══════════════════════════════════════════════════"
echo

# ── Step 1: Build .app (unless --skip-build) ────────────────
if [[ "${1:-}" != "--skip-build" ]]; then
    if [ -d "$APP_PATH" ]; then
        log ".app already exists at ${APP_PATH}"
        read -p "[DMG] Rebuild? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log "Rebuilding .app..."
            python3 "${SCRIPT_DIR}/build_app.py"
        else
            log "Using existing .app"
        fi
    else
        log "Building .app via build_app.py..."
        python3 "${SCRIPT_DIR}/build_app.py"
    fi
fi

# ── Verify .app exists ─────────────────────────────────────
if [ ! -d "$APP_PATH" ]; then
    err ".app not found at: ${APP_PATH}"
    err "Run without --skip-build, or build manually first:"
    err "  python3 build_app.py"
    exit 1
fi

APP_SIZE=$(du -sh "$APP_PATH" | cut -f1)
log "App bundle: ${APP_PATH} (${APP_SIZE})"

# ── Step 2: Create DMG staging area ─────────────────────────
log "Preparing DMG staging area..."
cleanup_staging
mkdir -p "$DMG_DIR"

# Copy .app
cp -R "$APP_PATH" "$DMG_DIR/"

# Create Applications symlink (drag-to-install)
ln -s /Applications "$DMG_DIR/Applications"

# Add a background hint file (hidden)
mkdir -p "$DMG_DIR/.background"
cat > "$DMG_DIR/.background/README.txt" << 'BGEOF'
Drag LongPlay Studio to Applications to install.
BGEOF

# ── Step 3: Create temporary DMG ────────────────────────────
log "Creating DMG image..."
TEMP_DMG="${DIST_DIR}/${DMG_NAME}-temp.dmg"
mkdir -p "$DIST_DIR"

# Remove old DMG if exists
rm -f "$DMG_OUTPUT" "$TEMP_DMG"

# Calculate size (app size + 20MB headroom)
APP_SIZE_KB=$(du -sk "$DMG_DIR" | cut -f1)
DMG_SIZE_KB=$(( APP_SIZE_KB + 20480 ))

# Create writable DMG
hdiutil create \
    -srcfolder "$DMG_DIR" \
    -volname "$DMG_VOLUME_NAME" \
    -fs HFS+ \
    -fsargs "-c c=64,a=16,e=16" \
    -format UDRW \
    -size "${DMG_SIZE_KB}k" \
    "$TEMP_DMG"

# ── Step 4: Customize DMG appearance ────────────────────────
log "Customizing DMG window..."
MOUNT_DIR="/Volumes/${DMG_VOLUME_NAME}"

# Mount the writable DMG
hdiutil attach "$TEMP_DMG" -readwrite -noverify -noautoopen

# Wait for mount
sleep 2

# Apply Finder view settings via AppleScript
osascript << APPLESCRIPT
tell application "Finder"
    tell disk "${DMG_VOLUME_NAME}"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {200, 120, 800, 500}
        set viewOptions to the icon view options of container window
        set arrangement of viewOptions to not arranged
        set icon size of viewOptions to 100
        -- Position the app icon and Applications alias
        set position of item "${APP_NAME}.app" of container window to {150, 200}
        set position of item "Applications" of container window to {450, 200}
        close
        open
        update without registering applications
        delay 2
        close
    end tell
end tell
APPLESCRIPT

# Hide background folder
SetFile -a V "$MOUNT_DIR/.background" 2>/dev/null || true

# Sync and unmount
sync
hdiutil detach "$MOUNT_DIR"

# ── Step 5: Convert to compressed read-only DMG ─────────────
log "Compressing DMG (UDZO)..."
hdiutil convert "$TEMP_DMG" \
    -format UDZO \
    -imagekey zlib-level=9 \
    -o "$DMG_OUTPUT"

# Clean up temp
rm -f "$TEMP_DMG"
cleanup_staging

# ── Done ────────────────────────────────────────────────────
DMG_SIZE=$(du -sh "$DMG_OUTPUT" | cut -f1)

echo
bold "═══════════════════════════════════════════════════"
bold "  DMG Created Successfully!"
bold "═══════════════════════════════════════════════════"
echo
log "Output:  ${DMG_OUTPUT}"
log "Size:    ${DMG_SIZE}"
log "Volume:  ${DMG_VOLUME_NAME}"
echo
log "To test: open \"${DMG_OUTPUT}\""
echo
