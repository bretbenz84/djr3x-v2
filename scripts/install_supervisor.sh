#!/usr/bin/env bash
#
# install_supervisor.sh — install / uninstall the DJ-R3X login LaunchAgents.
#
# Two agents, installed together:
#   com.djr3x.supervisor — the wake-word listener (rex_supervisor.py): runs for
#     your whole login session, listens only for "wake up rex", and launches
#     the full robot (main.py) on demand.
#   com.djr3x.battery — the menu bar battery meter (tools/rex_battery_menubar.py):
#     shows the ESP32 drive base's charge/voltage/current in the macOS menu bar
#     even while the robot is off. Only installed when MOTION_ESP32_PORT is set
#     in .env (without a motion base there is nothing to meter).
#
# This script renders each plist template with this repo's absolute path and
# loads it.
#
# Usage:
#   scripts/install_supervisor.sh            # install + start both
#   scripts/install_supervisor.sh uninstall  # stop + remove both
#   scripts/install_supervisor.sh status     # show launchd status
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPERVISOR_LABEL="com.djr3x.supervisor"
BATTERY_LABEL="com.djr3x.battery"
AGENTS_DIR="$HOME/Library/LaunchAgents"
ACTION="${1:-install}"

install_agent() {  # $1 = label
    local label="$1"
    local template="$PROJECT_ROOT/launchd/$label.plist.template"
    local plist="$AGENTS_DIR/$label.plist"
    if [[ ! -f "$template" ]]; then
        echo "ERROR: template missing: $template" >&2
        return 1
    fi
    # If already loaded, unload first so we cleanly replace it.
    launchctl unload "$plist" 2>/dev/null || true
    sed "s#__PROJECT_ROOT__#$PROJECT_ROOT#g" "$template" > "$plist"
    echo "Wrote $plist"
    launchctl load "$plist"
}

uninstall_agent() {  # $1 = label
    local plist="$AGENTS_DIR/$1.plist"
    launchctl unload "$plist" 2>/dev/null || true
    rm -f "$plist"
    echo "Removed $plist"
}

status_agent() {  # $1 = label
    local plist="$AGENTS_DIR/$1.plist"
    echo "Plist: $plist"
    if [[ -f "$plist" ]]; then
        launchctl list | grep "$1" || echo "(installed but not currently listed/running)"
    else
        echo "(not installed)"
    fi
}

# The battery meter is pointless without a motion base: only install it when
# MOTION_ESP32_PORT has a value in .env (same "unset port = feature off"
# convention as the robot itself).
motion_port_configured() {
    [[ -f "$PROJECT_ROOT/.env" ]] || return 1
    local port
    port="$(sed -n 's/^[[:space:]]*MOTION_ESP32_PORT[[:space:]]*=[[:space:]]*//p' \
            "$PROJECT_ROOT/.env" | tail -1 | tr -d '"'"'" | xargs)"
    [[ -n "$port" ]]
}

case "$ACTION" in
  install)
    if [[ ! -x "$PROJECT_ROOT/venv/bin/python" ]]; then
        echo "ERROR: venv not found at $PROJECT_ROOT/venv. Create it first." >&2
        exit 1
    fi
    mkdir -p "$AGENTS_DIR" "$PROJECT_ROOT/logs"

    install_agent "$SUPERVISOR_LABEL"
    echo "Loaded $SUPERVISOR_LABEL — listening for 'wake up rex'."
    echo "Logs: $PROJECT_ROOT/logs/supervisor.out.log (and .err.log)"
    echo
    echo "NOTE: the first run will prompt for Microphone permission for the venv"
    echo "python. Grant it (System Settings > Privacy & Security > Microphone)."
    echo

    if ! motion_port_configured; then
        echo "MOTION_ESP32_PORT is not set in .env — skipping the menu bar battery"
        echo "meter (re-run this script after configuring the motion base)."
    elif ! "$PROJECT_ROOT/venv/bin/python" -c "import rumps" 2>/dev/null; then
        echo "WARNING: 'rumps' is not installed in the venv — skipping the menu bar"
        echo "battery meter. Fix with: venv/bin/pip install rumps  (then re-run this)"
    else
        install_agent "$BATTERY_LABEL"
        echo "Loaded $BATTERY_LABEL — battery meter is in the menu bar."
        echo "Logs: $PROJECT_ROOT/logs/battery_menubar.out.log (and .err.log)"
    fi
    ;;

  uninstall|remove)
    echo "Stopping and removing $SUPERVISOR_LABEL + $BATTERY_LABEL ..."
    uninstall_agent "$SUPERVISOR_LABEL"
    uninstall_agent "$BATTERY_LABEL"
    ;;

  status)
    status_agent "$SUPERVISOR_LABEL"
    echo
    status_agent "$BATTERY_LABEL"
    ;;

  *)
    echo "Usage: $0 [install|uninstall|status]" >&2
    exit 2
    ;;
esac
