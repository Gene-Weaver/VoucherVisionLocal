#!/bin/bash
# Deploy VoucherVisionLocal — builds all targets on a Mac and publishes a GitHub release.
# Usage: ./deploy.sh [--skip-builds] [--skip-release] [--allow-unsigned]
#
# Modeled on VoucherVisionGO-Editor/deploy.sh. Runs on macOS only (Apple code
# signing and notarization need it). Builds:
#   macOS  Apple Silicon + Intel: .dmg (people download) and .zip (the updater uses it)
#   Linux  x86_64 AppImage
# then notarizes and staples each DMG and creates the GitHub release that the
# installed apps update from. Windows comes later (the "win" block in
# desktop/package.json is ready but not built here).
#
# Before running: set the version in VERSION, commit, and push to main.
# Signing secrets come from .env.signing (gitignored; see .env.signing.example).

set -euo pipefail
cd "$(dirname "$0")"

# Auto-load code-signing secrets if present. Copy .env.signing.example to
# .env.signing and fill in the real values — it is gitignored.
if [ -f .env.signing ]; then
    # shellcheck disable=SC1091
    source .env.signing
fi

REPO="Gene-Weaver/VoucherVisionLocal"
SKIP_BUILDS=0
SKIP_RELEASE=0
ALLOW_UNSIGNED=0

for arg in "$@"; do
    case "$arg" in
        --skip-builds)
            SKIP_BUILDS=1
            ;;
        --skip-release)
            SKIP_RELEASE=1
            ;;
        --allow-unsigned)
            ALLOW_UNSIGNED=1
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: ./deploy.sh [--skip-builds] [--skip-release] [--allow-unsigned]"
            exit 1
            ;;
    esac
done

if [ "$ALLOW_UNSIGNED" -eq 1 ] && [ "$SKIP_RELEASE" -ne 1 ]; then
    echo "ERROR: --allow-unsigned is for local test builds only; add --skip-release."
    exit 1
fi

echo "=== VoucherVisionLocal Deploy ==="
echo ""

# The one version number lives in VERSION; desktop/package.json gets a copy.
VERSION=$(tr -d '[:space:]' < VERSION)
TAG="v${VERSION}"
echo "Version: $VERSION  Tag: $TAG"
echo ""

# 0. Preflight: a release must be exactly a pushed commit on main.
if [ "$(uname)" != "Darwin" ]; then
    echo "ERROR: deploy.sh runs on macOS (Apple signing and notarization need it)."
    exit 1
fi
for tool in git node npm gh xcrun; do
    command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: $tool is not installed."; exit 1; }
done
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: the working tree has uncommitted changes. Commit and push first."
    git status --short | head -20
    exit 1
fi
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    echo "ERROR: releases are built from main (this is $BRANCH)."
    exit 1
fi
git fetch -q origin main
if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
    echo "ERROR: main is not the same as origin/main. Push (or pull) first."
    exit 1
fi
COMMIT=$(git rev-parse HEAD)
echo "Commit: $COMMIT"
echo ""

ART="desktop/build/VoucherVisionLocal-${VERSION}"

# 1-3. Build app (unless --skip-builds)
if [ "$SKIP_BUILDS" -ne 1 ]; then
    unset ELECTRON_RUN_AS_NODE

    echo "Installing desktop dependencies..."
    (cd desktop && npm ci --no-audit --no-fund >/dev/null)
    echo ""

    # Stage once for every target: the parallel builds below share it.
    echo "Staging backend, Python project and bundled uv..."
    (cd desktop && npm run stage -- --targets mac-arm64,mac-x64,linux-x64)
    echo ""

    echo "Building all platforms in parallel..."
    echo ""

    LOGDIR=$(mktemp -d)
    trap 'rm -rf "$LOGDIR"' EXIT

    # Both Mac architectures in ONE electron-builder run, so latest-mac.yml lists
    # both zips. (Separate parallel runs each write latest-mac.yml and the last
    # one wins: Apple Silicon Macs would then update to the Intel build.)
    (cd desktop && npm run build:only -- --mac --arm64 --x64 > "$LOGDIR/mac.raw.log" 2>&1) &
    PID_MAC=$!

    (cd desktop && npm run build:only -- --linux --x64 > "$LOGDIR/linux-x64.raw.log" 2>&1) &
    PID_LINUX=$!

    # Wait for all and track failures
    FAILED=0
    for PID_NAME in "mac:$PID_MAC" "linux-x64:$PID_LINUX"; do
        NAME="${PID_NAME%%:*}"
        PID="${PID_NAME##*:}"
        echo "  [$NAME] Building..."
        if wait "$PID"; then
            echo "  ✓ $NAME completed"
        else
            STATUS=$?
            echo "  ✗ $NAME FAILED (exit $STATUS)"
            FAILED=1
        fi
        # Surface the meaningful per-build lines, including notarization status
        # so a silent notarize-skip can never hide again.
        FILTER='building|packaging|signing|artifact|executing|downloading|notariz|stapl|error|fail|⨯|✗'
        if grep -E -i "$FILTER" "$LOGDIR/$NAME.raw.log" >/dev/null 2>&1; then
            grep -E -i "$FILTER" "$LOGDIR/$NAME.raw.log" | sed "s/^/    [$NAME] /"
        else
            tail -n 20 "$LOGDIR/$NAME.raw.log" | sed "s/^/    [$NAME] /"
        fi
        echo ""
    done

    if [ "$FAILED" = "1" ]; then
        echo "ERROR: One or more builds failed. Aborting."
        exit 1
    fi

    # latest-mac.yml must offer both architectures to the updater.
    for arch in arm64 x64; do
        if ! grep -q "VoucherVisionLocal-${VERSION}-${arch}.zip" desktop/build/latest-mac.yml; then
            echo "ERROR: desktop/build/latest-mac.yml does not list the ${arch} zip."
            exit 1
        fi
    done

    # 3a. The apps must be signed and notarized (unless --allow-unsigned for a test build).
    SIGNED=1
    for APP in desktop/build/mac-arm64/VoucherVisionLocal.app desktop/build/mac/VoucherVisionLocal.app; do
        if [ ! -d "$APP" ]; then
            echo "  ⚠ $APP not found"
            SIGNED=0
            continue
        fi
        if codesign --verify --deep --strict "$APP" >/dev/null 2>&1 && spctl --assess --type execute "$APP" >/dev/null 2>&1; then
            echo "  ✓ signed and accepted by Gatekeeper: $APP"
        else
            echo "  ✗ not signed and notarized: $APP"
            SIGNED=0
        fi
    done
    echo ""
    if [ "$SIGNED" -ne 1 ] && [ "$ALLOW_UNSIGNED" -ne 1 ]; then
        echo "ERROR: unsigned Mac builds cannot update themselves and macOS blocks them."
        echo "  Check .env.signing and the Developer ID certificate, or pass"
        echo "  --allow-unsigned --skip-release for a local test build."
        exit 1
    fi

    # 3b. Notarize and staple macOS DMG containers.
    #
    # electron-builder's `notarize: true` option only notarizes the .app
    # inside the .dmg — it does NOT submit the DMG container itself, so the
    # DMG ends up with a notarized .app but no DMG-level ticket. This means
    # offline users (or users on older macOS) can hit Gatekeeper warnings
    # when mounting the DMG. We fix it here by submitting each finished DMG
    # to Apple separately, then stapling the returned ticket to the DMG.
    #
    # The .app inside is already notarized+stapled by electron-builder, so
    # this is a fast second pass — Apple usually re-uses the existing
    # ticket and responds in a minute or two.
    if [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ] && [ -n "${APPLE_APP_SPECIFIC_PASSWORD:-}" ]; then
        echo "Notarizing and stapling macOS DMG containers..."
        echo ""
        for arch in arm64 x64; do
            DMG="${ART}-${arch}.dmg"
            if [ ! -f "$DMG" ]; then
                echo "  ⚠ $DMG not found, skipping"
                continue
            fi
            echo "  [$arch] Submitting $(basename "$DMG") to Apple..."
            SUBMIT_OUT=$(xcrun notarytool submit "$DMG" \
                --apple-id "$APPLE_ID" \
                --team-id "$APPLE_TEAM_ID" \
                --password "$APPLE_APP_SPECIFIC_PASSWORD" \
                --wait \
                --output-format json 2>&1)
            STATUS=$(echo "$SUBMIT_OUT" | grep -o '"status":[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
            if [ "$STATUS" != "Accepted" ]; then
                echo "  ✗ [$arch] Notarization failed (status: $STATUS)"
                echo "$SUBMIT_OUT" | sed "s/^/    /"
                exit 1
            fi
            echo "  ✓ [$arch] Apple Accepted — stapling..."
            if xcrun stapler staple "$DMG" 2>&1 | sed "s/^/    /"; then
                # Verify the staple was actually applied
                if xcrun stapler validate "$DMG" >/dev/null 2>&1; then
                    echo "  ✓ [$arch] Stapled and validated $(basename "$DMG")"
                else
                    echo "  ✗ [$arch] Staple validation failed for $(basename "$DMG")"
                    exit 1
                fi
            else
                echo "  ✗ [$arch] Stapling failed for $(basename "$DMG")"
                exit 1
            fi
            echo ""
        done
    else
        echo "⚠ Skipping DMG notarization — Apple credentials not set."
        echo "  Set APPLE_ID, APPLE_TEAM_ID, APPLE_APP_SPECIFIC_PASSWORD in"
        echo "  .env.signing (see .env.signing.example for the template)."
        echo ""
    fi
else
    echo "Skipping app builds (--skip-builds)"
    echo ""
fi

# 4. Summary
echo "=== Build outputs ==="
ls -lhS desktop/build/*.dmg desktop/build/*.zip desktop/build/*.AppImage desktop/build/latest*.yml 2>/dev/null
echo ""

# 5. Create GitHub release (unless --skip-release)
if [ "$SKIP_RELEASE" -ne 1 ] && [ "$SKIP_BUILDS" -ne 1 ]; then
    echo "Creating GitHub release $TAG..."

    # Delete existing release/tag if re-deploying same version
    gh release delete "$TAG" --repo "$REPO" --yes 2>/dev/null || true
    git tag -d "$TAG" 2>/dev/null || true
    git push origin ":refs/tags/$TAG" 2>/dev/null || true

    gh release create "$TAG" \
      "${ART}-arm64.dmg#macOS (Apple Silicon)" \
      "${ART}-x64.dmg#macOS (Intel)" \
      "${ART}-arm64.zip" \
      "${ART}-x64.zip" \
      "${ART}-x86_64.AppImage#Linux (64-bit)" \
      "desktop/build/latest-mac.yml" \
      "desktop/build/latest-linux.yml" \
      --repo "$REPO" \
      --target "$COMMIT" \
      --title "VoucherVisionLocal $TAG" \
      --notes "See [README](https://github.com/${REPO}#readme) for details and installation instructions."

    echo ""
    echo "Release created: https://github.com/${REPO}/releases/tag/$TAG"
else
    echo "Skipping GitHub release"
    echo ""
    echo "To release manually:"
    echo "  gh release create $TAG \\"
    echo "    \"${ART}-arm64.dmg\" \\"
    echo "    \"${ART}-x64.dmg\" \\"
    echo "    \"${ART}-arm64.zip\" \\"
    echo "    \"${ART}-x64.zip\" \\"
    echo "    \"${ART}-x86_64.AppImage\" \\"
    echo "    \"desktop/build/latest-mac.yml\" \"desktop/build/latest-linux.yml\" \\"
    echo "    --repo $REPO --target $COMMIT \\"
    echo "    --title \"VoucherVisionLocal $TAG\""
fi
