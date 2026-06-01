#!/usr/bin/env bash
#
# install_supervisor.sh — install / uninstall the DJ-R3X wake-word LaunchAgent.
#
# The supervisor (rex_supervisor.py) runs for your whole login session, listens
# only for "wake up rex", and launches the full robot (main.py) on demand. This
# script renders the plist template with this repo's absolute path and loads it.
#
# Usage:
#   scripts/install_supervisor.sh            # install + start
#   scripts/install_supervisor.sh uninstall  # stop + remove
#   scripts/install_supervisor.sh status     # show launchd status
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.djr3x.supervisor"
TEMPLATE="$PROJECT_ROOT/launchd/$LABEL.plist.template"
AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST="$AGENTS_DIR/$LABEL.plist"
ACTION="${1:-install}"

uninstall() {
    echo "Stopping and removing $LABEL ..."
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Removed $PLIST"
}

case "$ACTION" in
  install)
    if [[ ! -x "$PROJECT_ROOT/venv/bin/python" ]]; then
        echo "ERROR: venv not found at $PROJECT_ROOT/venv. Create it first." >&2
        exit 1
    fi
    if [[ ! -f "$TEMPLATE" ]]; then
        echo "ERROR: template missing: $TEMPLATE" >&2
        exit 1
    fi
    mkdir -p "$AGENTS_DIR" "$PROJECT_ROOT/logs"

    # If already loaded, unload first so we cleanly replace it.
    launchctl unload "$PLIST" 2>/dev/null || true

    sed "s#__PROJECT_ROOT__#$PROJECT_ROOT#g" "$TEMPLATE" > "$PLIST"
    echo "Wrote $PLIST"

    launchctl load "$PLIST"
    echo "Loaded $LABEL — listening for 'wake up rex'."
    echo "Logs: $PROJECT_ROOT/logs/supervisor.out.log (and .err.log)"
    echo
    echo "NOTE: the first run will prompt for Microphone permission for the venv"
    echo "python. Grant it (System Settings > Privacy & Security > Microphone)."
    ;;

  uninstall|remove)
    uninstall
    ;;

  status)
    echo "Plist: $PLIST"
    if [[ -f "$PLIST" ]]; then
        launchctl list | grep "$LABEL" || echo "(installed but not currently listed/running)"
    else
        echo "(not installed)"
    fi
    ;;

  *)
    echo "Usage: $0 [install|uninstall|status]" >&2
    exit 2
    ;;
esac
