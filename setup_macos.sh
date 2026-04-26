#!/usr/bin/env bash
# setup_macos.sh — Complete macOS environment setup for DJ-R3X v2.
# Target: Apple Silicon macOS. Idempotent — safe to run multiple times.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
PYTHON_VERSION="3.11.9"
VENV_DIR="$PROJECT_DIR/venv"
ZSHRC="$HOME/.zshrc"

# ── Terminal colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${BLUE}▸${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
die()  { echo -e "${RED}✗ ERROR:${NC} $*"; exit 1; }

INSTALLED_ITEMS=()
MANUAL_ATTENTION=()

_is_interactive() {
    [[ -t 0 ]]
}

_prompt_input() {
    local prompt="$1"
    local reply=""
    printf "%b" "$prompt" >&2
    IFS= read -r reply || reply=""
    printf "%s" "$reply"
}

_prompt_secret() {
    local prompt="$1"
    local reply=""
    printf "%b" "$prompt" >&2
    if [[ -t 0 ]]; then
        local old_stty=""
        old_stty="$(stty -g)"
        stty -echo
        IFS= read -r reply || reply=""
        stty "$old_stty"
        printf "\n" >&2
    else
        IFS= read -r reply || reply=""
    fi
    printf "%s" "$reply"
}

_prompt_yes_no() {
    local prompt="$1"
    local default="${2:-n}"
    local reply=""
    reply="$(_prompt_input "$prompt")"
    reply="$(printf "%s" "$reply" | tr '[:upper:]' '[:lower:]')"
    if [[ -z "$reply" ]]; then
        reply="$default"
    fi
    [[ "$reply" == "y" || "$reply" == "yes" ]]
}

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  DJ-R3X v2 — macOS Setup${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ── Platform check ────────────────────────────────────────────────────────────
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    warn "Expected Apple Silicon (arm64), detected: $ARCH. Proceeding anyway."
fi

# ── Helper: ensure Homebrew is in PATH for the current shell ─────────────────
_brew_shellenv() {
    if [[ -x "/opt/homebrew/bin/brew" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -x "/usr/local/bin/brew" ]]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
}
_brew_shellenv

# ── 1. Homebrew ───────────────────────────────────────────────────────────────
log "Checking Homebrew..."
if ! command -v brew &>/dev/null; then
    log "Homebrew not found — installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    _brew_shellenv
    INSTALLED_ITEMS+=("Homebrew")
    ok "Homebrew installed."
else
    log "Homebrew present — updating..."
    brew update --quiet 2>/dev/null || warn "brew update had warnings (network issue?). Continuing."
    ok "Homebrew up to date."
fi

# ── 2. Homebrew packages ──────────────────────────────────────────────────────
BREW_PACKAGES=(cmake portaudio ffmpeg libsndfile boost openblas arduino-cli)

log "Checking Homebrew packages: ${BREW_PACKAGES[*]}"
for pkg in "${BREW_PACKAGES[@]}"; do
    if brew list --formula "$pkg" &>/dev/null; then
        ok "$pkg already installed."
    else
        log "Installing $pkg..."
        brew install "$pkg"
        INSTALLED_ITEMS+=("brew: $pkg")
        ok "$pkg installed."
    fi
done

# ── 2b. Arduino CLI core and libraries ───────────────────────────────────────
log "Checking Arduino CLI support..."
if command -v arduino-cli &>/dev/null; then
    arduino-cli config init >/dev/null 2>&1 || true
    if arduino-cli core update-index >/dev/null 2>&1; then
        if arduino-cli core list 2>/dev/null | grep -q '^arduino:avr[[:space:]]'; then
            ok "Arduino AVR core already installed."
        elif arduino-cli core install arduino:avr >/dev/null 2>&1; then
            INSTALLED_ITEMS+=("Arduino AVR core")
            ok "Arduino AVR core installed."
        else
            warn "Could not install Arduino AVR core."
            MANUAL_ATTENTION+=("Install Arduino AVR support manually: arduino-cli core update-index && arduino-cli core install arduino:avr")
        fi
    else
        warn "Could not update Arduino core index."
        MANUAL_ATTENTION+=("Install Arduino AVR support manually: arduino-cli core update-index && arduino-cli core install arduino:avr")
    fi

    if arduino-cli lib list 2>/dev/null | grep -qi '^FastLED[[:space:]]'; then
        ok "FastLED library already installed."
    elif arduino-cli lib install FastLED >/dev/null 2>&1; then
        INSTALLED_ITEMS+=("Arduino library: FastLED")
        ok "FastLED library installed for Arduino sketches."
    else
        warn "Could not install FastLED with arduino-cli."
        MANUAL_ATTENTION+=("Install FastLED manually: arduino-cli lib install FastLED")
    fi
else
    warn "arduino-cli was not found after Homebrew package installation."
    MANUAL_ATTENTION+=("Install arduino-cli and FastLED manually before uploading Arduino firmware")
fi

# ── 3. pyenv ─────────────────────────────────────────────────────────────────
log "Checking pyenv..."
if brew list --formula pyenv &>/dev/null || command -v pyenv &>/dev/null; then
    ok "pyenv already installed."
else
    log "Installing pyenv via Homebrew..."
    brew install pyenv
    INSTALLED_ITEMS+=("pyenv")
    ok "pyenv installed."
fi

# Add pyenv init to .zshrc if missing
PYENV_INIT_MARKER='eval "$(pyenv init -)"'
if ! grep -qF "$PYENV_INIT_MARKER" "$ZSHRC" 2>/dev/null; then
    log "Adding pyenv init to $ZSHRC..."
    {
        printf '\n# pyenv\n'
        printf 'export PYENV_ROOT="$HOME/.pyenv"\n'
        printf 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"\n'
        printf 'eval "$(pyenv init -)"\n'
    } >> "$ZSHRC"
    INSTALLED_ITEMS+=("pyenv init block in ~/.zshrc")
    ok "pyenv init added to .zshrc."
else
    ok "pyenv init already in .zshrc."
fi

# Initialize pyenv in the current script session
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# ── 4. Python 3.11.9 ─────────────────────────────────────────────────────────
log "Checking Python $PYTHON_VERSION..."
if pyenv versions --bare 2>/dev/null | grep -qx "$PYTHON_VERSION"; then
    ok "Python $PYTHON_VERSION already installed via pyenv."
else
    log "Installing Python $PYTHON_VERSION via pyenv (this may take several minutes)..."
    # Apple Silicon: prefer pre-built if available, fall back to compile
    PYTHON_CONFIGURE_OPTS="--enable-optimizations" pyenv install "$PYTHON_VERSION"
    INSTALLED_ITEMS+=("Python $PYTHON_VERSION (via pyenv)")
    ok "Python $PYTHON_VERSION installed."
fi

log "Setting .python-version to $PYTHON_VERSION..."
pyenv local "$PYTHON_VERSION"
ok "Project local Python pinned to $PYTHON_VERSION."

PYTHON_BIN="$(pyenv which python)"
PYTHON_ACTUAL="$("$PYTHON_BIN" --version 2>&1)"
log "Active interpreter: $PYTHON_BIN ($PYTHON_ACTUAL)"

# ── 5. Virtual environment ────────────────────────────────────────────────────
log "Checking virtual environment..."
if [[ -d "$VENV_DIR" ]] && [[ -x "$VENV_DIR/bin/python" ]]; then
    ok "Virtual environment already exists at $VENV_DIR."
else
    log "Creating virtual environment at $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    INSTALLED_ITEMS+=("venv at $VENV_DIR")
    ok "Virtual environment created."
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# ── 6. pip packages ───────────────────────────────────────────────────────────
REQUIREMENTS="$PROJECT_DIR/requirements.txt"
log "Installing pip packages from requirements.txt..."
if [[ -f "$REQUIREMENTS" ]]; then
    log "Upgrading pip..."
    "$VENV_PIP" install --upgrade pip --quiet
    "$VENV_PIP" install -r "$REQUIREMENTS"
    INSTALLED_ITEMS+=("pip packages from requirements.txt")
    ok "pip packages installed."
else
    warn "requirements.txt not found — skipping pip install."
    MANUAL_ATTENTION+=("requirements.txt missing — run: $VENV_PIP install -r requirements.txt")
fi

# ── 7. Config bootstrap ───────────────────────────────────────────────────────
log "Checking config files..."

# apikeys.py
APIKEYS_SRC="$PROJECT_DIR/apikeys.example.py"
APIKEYS_DST="$PROJECT_DIR/apikeys.py"
if [[ -f "$APIKEYS_DST" ]]; then
    ok "apikeys.py already exists — not overwritten."
else
    if [[ -f "$APIKEYS_SRC" ]]; then
        cp "$APIKEYS_SRC" "$APIKEYS_DST"
        INSTALLED_ITEMS+=("apikeys.py (copied from template)")
        ok "apikeys.py created from apikeys.example.py."
    else
        warn "apikeys.example.py not found — cannot create apikeys.py."
        MANUAL_ATTENTION+=("Create apikeys.py manually (template apikeys.example.py is missing)")
    fi
fi

# .env
ENV_SRC="$PROJECT_DIR/.env.example"
ENV_DST="$PROJECT_DIR/.env"
if [[ -f "$ENV_DST" ]]; then
    ok ".env already exists — not overwritten."
else
    if [[ -f "$ENV_SRC" ]]; then
        cp "$ENV_SRC" "$ENV_DST"
        INSTALLED_ITEMS+=(".env (copied from template)")
        ok ".env created from .env.example."
        MANUAL_ATTENTION+=("If using physical hardware, update MAESTRO_PORT / ARDUINO_*_PORT in: $ENV_DST")
    else
        warn ".env.example not found — cannot create .env."
        MANUAL_ATTENTION+=("Create .env manually (template .env.example is missing)")
    fi
fi

# ── 8. Interactive local configuration ────────────────────────────────────────
_set_env_value() {
    local key="$1"
    local value="$2"
    SETUP_ENV_KEY="$key" SETUP_ENV_VALUE="$value" "$VENV_PYTHON" - "$ENV_DST" <<'PY'
import os
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = os.environ["SETUP_ENV_KEY"]
value = os.environ["SETUP_ENV_VALUE"]

def render_env_value(raw: str) -> str:
    if raw == "":
        return ""
    if re.fullmatch(r"[A-Za-z0-9_./:+@%-]+", raw):
        return raw
    return '"' + raw.replace("\\", "\\\\").replace('"', '\\"') + '"'

lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
replacement = f"{key}={render_env_value(value)}"
for idx, line in enumerate(lines):
    if pattern.match(line):
        lines[idx] = replacement
        break
else:
    lines.append(replacement)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

_write_apikeys() {
    local openai_key="$1"
    local elevenlabs_key="$2"
    SETUP_OPENAI_API_KEY="$openai_key" \
    SETUP_ELEVENLABS_API_KEY="$elevenlabs_key" \
    "$VENV_PYTHON" - "$APIKEYS_DST" "$APIKEYS_SRC" <<'PY'
import ast
import json
import os
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
template_path = Path(sys.argv[2])
if path.exists():
    text = path.read_text(encoding="utf-8")
elif template_path.exists():
    text = template_path.read_text(encoding="utf-8")
else:
    text = (
        "# API credentials — copy this file to apikeys.py and fill in real values.\n"
        "# apikeys.py is excluded from git. Never commit real keys.\n\n"
        'OPENAI_API_KEY = "sk-..."\n'
        'ELEVENLABS_API_KEY = "your-elevenlabs-api-key-here"\n'
    )

def current_value(name: str, fallback: str) -> str:
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*(.+?)\s*$", text, re.MULTILINE)
    if not match:
        return fallback
    try:
        return str(ast.literal_eval(match.group(1)))
    except Exception:
        return fallback

def set_assignment(source: str, name: str, value: str) -> str:
    replacement = f"{name} = {json.dumps(value)}"
    pattern = rf"^\s*{re.escape(name)}\s*=\s*(.+?)\s*$"
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count:
        return updated
    suffix = "" if updated.endswith("\n") else "\n"
    return f"{updated}{suffix}{replacement}\n"

openai = os.environ.get("SETUP_OPENAI_API_KEY") or current_value("OPENAI_API_KEY", "sk-...")
eleven = os.environ.get("SETUP_ELEVENLABS_API_KEY") or current_value(
    "ELEVENLABS_API_KEY",
    "your-elevenlabs-api-key-here",
)

text = set_assignment(text, "OPENAI_API_KEY", openai)
text = set_assignment(text, "ELEVENLABS_API_KEY", eleven)
path.write_text(text if text.endswith("\n") else f"{text}\n", encoding="utf-8")
PY
}

_apikeys_have_placeholders() {
    [[ ! -f "$APIKEYS_DST" ]] && return 0
    grep -qE '"sk-\.\.\."' "$APIKEYS_DST" 2>/dev/null \
    || grep -q 'your-elevenlabs-api-key-here' "$APIKEYS_DST" 2>/dev/null
}

_current_voice_id() {
    "$VENV_PYTHON" - "$PROJECT_DIR/config.py" <<'PY'
import ast
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r'^\s*ELEVENLABS_VOICE_ID\s*=\s*(.+?)\s*$', text, re.MULTILINE)
if not match:
    print("")
    raise SystemExit
try:
    print(ast.literal_eval(match.group(1)))
except Exception:
    print("")
PY
}

_update_voice_id() {
    local voice_id="$1"
    SETUP_ELEVENLABS_VOICE_ID="$voice_id" "$VENV_PYTHON" - "$PROJECT_DIR/config.py" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
voice_id = os.environ["SETUP_ELEVENLABS_VOICE_ID"]
text = path.read_text(encoding="utf-8")
replacement = f"ELEVENLABS_VOICE_ID = {json.dumps(voice_id)}"
new_text, count = re.subn(
    r'^\s*ELEVENLABS_VOICE_ID\s*=\s*["\'][^"\']*["\']',
    replacement,
    text,
    count=1,
    flags=re.MULTILINE,
)
if count != 1:
    raise SystemExit("Could not find ELEVENLABS_VOICE_ID in config.py")
path.write_text(new_text, encoding="utf-8")
PY
}

_list_audio_inputs() {
    "$VENV_PYTHON" - <<'PY'
try:
    import sounddevice as sd
    devices = sd.query_devices()
    default_input = None
    try:
        default_input = sd.default.device[0]
    except Exception:
        pass
    print("Available microphone/input devices:")
    found = False
    for idx, device in enumerate(devices):
        channels = int(device.get("max_input_channels", 0))
        if channels <= 0:
            continue
        found = True
        marker = " (default)" if idx == default_input else ""
        print(f"  [{idx}] {device.get('name', 'Unknown')} — {channels} input channel(s){marker}")
    if not found:
        print("  No input devices reported by PortAudio.")
except Exception as exc:
    print(f"Could not query audio devices: {exc}")
PY
}

_list_cameras() {
    "$VENV_PYTHON" - <<'PY'
import re
import shutil
import subprocess

print("OpenCV camera indices that can be opened:")
try:
    import cv2
    found = False
    for idx in range(8):
        cap = cv2.VideoCapture(idx)
        ok = bool(cap.isOpened())
        cap.release()
        if ok:
            found = True
            print(f"  index {idx}")
    if not found:
        print("  No OpenCV camera indices opened in the 0-7 scan.")
except Exception as exc:
    print(f"  Could not query OpenCV cameras: {exc}")

print("")
print("macOS AVFoundation camera names from ffmpeg:")
if not shutil.which("ffmpeg"):
    print("  ffmpeg not found.")
else:
    proc = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output = proc.stdout + proc.stderr
    in_video = False
    found = False
    for line in output.splitlines():
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            in_video = False
        if not in_video:
            continue
        match = re.search(r"\]\s*\[(\d+)\]\s+(.+)$", line)
        if match:
            found = True
            print(f"  {match.group(2)}")
    if not found:
        print("  No AVFoundation camera names found.")
PY
}

_env_current_value() {
    local key="$1"
    "$VENV_PYTHON" - "$ENV_DST" "$key" <<'PY'
import ast
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
if not path.exists():
    print("")
    raise SystemExit

pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
for line in path.read_text(encoding="utf-8").splitlines():
    if not pattern.match(line) or line.lstrip().startswith("#"):
        continue
    raw = line.split("=", 1)[1].strip()
    if raw == "":
        print("")
        raise SystemExit
    try:
        print(ast.literal_eval(raw))
    except Exception:
        print(raw.strip("\"'"))
    raise SystemExit
print("")
PY
}

_env_value_is_template_placeholder() {
    local key="$1"
    local value="$2"
    case "$key:$value" in
        "MAESTRO_PORT:/dev/tty.usbmodem00000000") return 0 ;;
        "ARDUINO_HEAD_PORT:/dev/tty.usbmodem00000001") return 0 ;;
        "ARDUINO_CHEST_PORT:/dev/tty.usbserial-00000000") return 0 ;;
    esac
    return 1
}

_env_value_is_good_port() {
    local key="$1"
    local value="$2"
    [[ -n "$value" ]] || return 1
    if _env_value_is_template_placeholder "$key" "$value"; then
        return 1
    fi
    return 0
}

_show_current_env_port() {
    local key="$1"
    local label="$2"
    local current=""
    current="$(_env_current_value "$key")"
    if [[ -z "$current" ]]; then
        echo "Current $label ($key): blank / disabled"
    elif _env_value_is_template_placeholder "$key" "$current"; then
        echo "Current $label ($key): $current (template placeholder, probably not usable)"
    else
        echo "Current $label ($key): $current"
    fi
}

_prompt_continue() {
    local prompt="${1:-Press Enter to continue...}"
    _prompt_input "$prompt" >/dev/null
}

_snapshot_dev_names() {
    local output_file="$1"
    ls -1 /dev 2>/dev/null | sort > "$output_file"
}

_print_dev_directory() {
    echo ""
    log "Current /dev directory entries:"
    ls -1 /dev 2>/dev/null | sed 's/^/  /'
    echo ""
}

_print_new_dev_entries() {
    local before_file="$1"
    local after_file="$2"
    local new_entries=""
    new_entries="$(comm -13 "$before_file" "$after_file" || true)"
    if [[ -z "$new_entries" ]]; then
        warn "No new /dev entries appeared."
        return 1
    fi
    log "New /dev entries:"
    printf "%s\n" "$new_entries" | sed 's#^#  /dev/#'
    return 0
}

_serial_candidates_from_snapshot_diff() {
    local before_file="$1"
    local after_file="$2"
    comm -13 "$before_file" "$after_file" | while IFS= read -r name; do
        case "$name" in
            cu.usbmodem*|tty.usbmodem*|cu.usbserial*|tty.usbserial*|\
            cu.wchusbserial*|tty.wchusbserial*|cu.SLAB_USBtoUART*|tty.SLAB_USBtoUART*|\
            cu.usb*|tty.usb*)
                printf "/dev/%s\n" "$name"
                ;;
        esac
    done
}

_all_serial_candidates() {
    ls -1 /dev 2>/dev/null | while IFS= read -r name; do
        case "$name" in
            cu.usbmodem*|tty.usbmodem*|cu.usbserial*|tty.usbserial*|\
            cu.wchusbserial*|tty.wchusbserial*|cu.SLAB_USBtoUART*|tty.SLAB_USBtoUART*|\
            cu.usb*|tty.usb*)
                printf "/dev/%s\n" "$name"
                ;;
        esac
    done
}

_rank_serial_candidates() {
    local kind="$1"
    local candidate_file="$2"
    while IFS= read -r path; do
        [[ -n "$path" ]] || continue
        local name="${path#/dev/}"
        local score=90
        case "$kind:$name" in
            chest:cu.usbserial*|chest:cu.wchusbserial*|chest:cu.SLAB_USBtoUART*) score=10 ;;
            chest:tty.usbserial*|chest:tty.wchusbserial*|chest:tty.SLAB_USBtoUART*) score=11 ;;
            chest:cu.usbmodem*) score=30 ;;
            chest:tty.usbmodem*) score=31 ;;
            head:cu.usbmodem*|maestro:cu.usbmodem*) score=10 ;;
            head:tty.usbmodem*|maestro:tty.usbmodem*) score=11 ;;
            head:cu.usbserial*|head:cu.wchusbserial*|head:cu.SLAB_USBtoUART*) score=30 ;;
            maestro:cu.usbserial*|maestro:cu.wchusbserial*|maestro:cu.SLAB_USBtoUART*) score=30 ;;
            head:tty.usbserial*|head:tty.wchusbserial*|head:tty.SLAB_USBtoUART*) score=31 ;;
            maestro:tty.usbserial*|maestro:tty.wchusbserial*|maestro:tty.SLAB_USBtoUART*) score=31 ;;
            *:cu.usb*) score=50 ;;
            *:tty.usb*) score=51 ;;
        esac
        printf "%03d\t%s\n" "$score" "$path"
    done < "$candidate_file" | sort -n | cut -f2-
}

_print_numbered_serial_choices() {
    local ranked_file="$1"
    local current="$2"
    local idx=1
    local path=""
    while IFS= read -r path; do
        [[ -n "$path" ]] || continue
        local suffix=""
        case "$path" in
            /dev/cu.*) suffix=" (recommended for app use on macOS)" ;;
            /dev/tty.*) suffix=" (alternate tty device)" ;;
        esac
        if [[ -n "$current" && "$path" == "$current" ]]; then
            suffix="$suffix (current)"
        fi
        echo "  [$idx] $path$suffix"
        idx=$((idx + 1))
    done < "$ranked_file"
}

_serial_choice_by_index() {
    local ranked_file="$1"
    local choice="$2"
    sed -n "${choice}p" "$ranked_file"
}

_maybe_open_usb_driver_links() {
    echo ""
    warn "No likely USB serial device appeared."
    echo "Common clone-driver links:"
    echo "  CH340 / CH341: https://www.wch-ic.com/downloads/CH34XSER_MAC_ZIP.html"
    echo "  Silicon Labs CP210x: https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers"
    echo "  FTDI VCP: https://ftdichip.com/drivers/vcp-drivers/"
    echo ""
    if _prompt_yes_no "Open those driver pages in your browser? [y/N] " "n"; then
        open "https://www.wch-ic.com/downloads/CH34XSER_MAC_ZIP.html" >/dev/null 2>&1 || true
        open "https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers" >/dev/null 2>&1 || true
        open "https://ftdichip.com/drivers/vcp-drivers/" >/dev/null 2>&1 || true
    fi
}

_scan_for_serial_device_with_retry() {
    local label="$1"
    local after_file="$2"
    local candidate_file="$3"

    while true; do
        _snapshot_dev_names "$after_file"
        _print_new_dev_entries "$HARDWARE_BASELINE_FILE" "$after_file" || true
        _serial_candidates_from_snapshot_diff "$HARDWARE_BASELINE_FILE" "$after_file" > "$candidate_file"

        if [[ -s "$candidate_file" ]]; then
            return 0
        fi

        _maybe_open_usb_driver_links
        echo ""
        echo "After installing drivers, swapping cables, or reconnecting $label, the device may need to be unplugged and plugged back in."
        if _prompt_yes_no "Scan again for $label? [Y/n] " "y"; then
            _prompt_continue "Connect or reconnect $label by USB, then press Enter..."
            continue
        fi

        warn "Continuing without a newly detected $label device."
        return 1
    done
}

SELECTED_DEVICE_PORT=""
SELECTED_DEVICE_CHANGED=0
_select_serial_port_for_env() {
    local label="$1"
    local env_key="$2"
    local kind="$3"
    local candidates_file="$4"
    SELECTED_DEVICE_PORT=""
    SELECTED_DEVICE_CHANGED=0

    _show_current_env_port "$env_key" "$label"

    local ranked_file=""
    ranked_file="$(mktemp)"
    _rank_serial_candidates "$kind" "$candidates_file" > "$ranked_file"

    if [[ -s "$ranked_file" ]]; then
        log "Likely serial device(s) for $label:"
    else
        warn "No likely serial device was detected for $label."
        echo "Serial-looking devices currently attached:"
        _all_serial_candidates | sed 's/^/  /' || true
    fi

    local detected=""
    detected="$(sed -n '1p' "$ranked_file")"
    local current=""
    current="$(_env_current_value "$env_key")"
    local reply=""

    if [[ -s "$ranked_file" ]]; then
        _print_numbered_serial_choices "$ranked_file" "$current"
        echo "On macOS, the /dev/cu.* entry is usually the right one for Arduino/Maestro serial communication."
    fi

    if _env_value_is_good_port "$env_key" "$current"; then
        echo "Press Enter to keep the current value, enter a number above, or enter a custom /dev path."
        reply="$(_prompt_input "Choice for $env_key [keep current]: ")"
        if [[ -z "$reply" || "$reply" == "keep" || "$reply" == "skip" ]]; then
            SELECTED_DEVICE_PORT="$current"
            rm -f "$ranked_file"
            return 0
        elif [[ "$reply" =~ ^[0-9]+$ ]]; then
            reply="$(_serial_choice_by_index "$ranked_file" "$reply")"
            if [[ -z "$reply" ]]; then
                warn "That number is not in the detected-device list — keeping current value."
                SELECTED_DEVICE_PORT="$current"
                rm -f "$ranked_file"
                return 0
            fi
        elif [[ "$reply" == "use" && -n "$detected" ]]; then
            reply="$detected"
        fi
    else
        if [[ -n "$detected" ]]; then
            echo "Press Enter to use the recommended device, enter a number above, type 'skip', or enter a custom /dev path."
            reply="$(_prompt_input "Choice for $env_key [$detected]: ")"
            if [[ -z "$reply" ]]; then
                reply="$detected"
            elif [[ "$reply" == "skip" ]]; then
                SELECTED_DEVICE_PORT="$current"
                rm -f "$ranked_file"
                return 0
            elif [[ "$reply" =~ ^[0-9]+$ ]]; then
                reply="$(_serial_choice_by_index "$ranked_file" "$reply")"
                if [[ -z "$reply" ]]; then
                    warn "That number is not in the detected-device list — leaving $env_key unchanged."
                    SELECTED_DEVICE_PORT="$current"
                    rm -f "$ranked_file"
                    return 0
                fi
            fi
        else
            echo "Enter a custom /dev path, or press Enter to keep the current value."
            reply="$(_prompt_input "Choice for $env_key: ")"
            if [[ -z "$reply" ]]; then
                SELECTED_DEVICE_PORT="$current"
                rm -f "$ranked_file"
                return 0
            fi
        fi
    fi

    if [[ -n "$reply" ]]; then
        _set_env_value "$env_key" "$reply"
        SELECTED_DEVICE_PORT="$reply"
        SELECTED_DEVICE_CHANGED=1
        INSTALLED_ITEMS+=(".env $env_key=$reply")
        ok "$env_key set to $reply."
    fi
    rm -f "$ranked_file"
}

_choose_arduino_board_fqbn() {
    local label="$1"
    local default_fqbn="$2"
    {
        echo ""
        echo "What kind of Arduino is connected for $label?"
        echo "  [1] Default for this device: $default_fqbn"
        echo "  [2] Arduino Uno / Uno-compatible: arduino:avr:uno"
        echo "  [3] Arduino Nano ATmega328P, current bootloader: arduino:avr:nano"
        echo "  [4] Arduino Nano ATmega328P, old bootloader / many CH340 clones: arduino:avr:nano:cpu=atmega328old"
        echo "  [5] Custom FQBN"
    } >&2
    local choice=""
    choice="$(_prompt_input "Arduino board [1]: ")"
    case "$choice" in
        ""|1) printf "%s" "$default_fqbn" ;;
        2) printf "%s" "arduino:avr:uno" ;;
        3) printf "%s" "arduino:avr:nano" ;;
        4) printf "%s" "arduino:avr:nano:cpu=atmega328old" ;;
        5)
            local custom=""
            custom="$(_prompt_input "Custom FQBN: ")"
            printf "%s" "$custom"
            ;;
        *)
            warn "Unknown board choice — using $default_fqbn." >&2
            printf "%s" "$default_fqbn"
            ;;
    esac
}

_arduino_compile_failure_action() {
    local label="$1"
    while true; do
        {
            echo ""
            echo "$label firmware did not compile."
            echo "This is often the wrong Arduino type/bootloader selection, or a missing Arduino library."
            echo "Choose what to do next:"
            echo "  [1] Change Arduino type/bootloader and try again"
            echo "  [2] Try compiling again with the same selection"
            echo "  [3] Skip firmware upload for now"
        } >&2
        local choice=""
        choice="$(_prompt_input "Next step [1]: ")"
        choice="$(printf "%s" "$choice" | tr '[:upper:]' '[:lower:]')"
        case "$choice" in
            ""|1|board|b|change|c) printf "%s" "change_board"; return ;;
            2|retry|r|same) printf "%s" "retry"; return ;;
            3|skip|s|n|no) printf "%s" "skip"; return ;;
            *) warn "Please choose 1, 2, or 3." >&2 ;;
        esac
    done
}

_arduino_upload_failure_action() {
    local label="$1"
    while true; do
        {
            echo ""
            echo "$label firmware did not upload."
            echo "This is often the wrong bootloader, a busy/wrong serial port, or a charge-only USB cable."
            echo "Choose what to do next:"
            echo "  [1] Change Arduino type/bootloader and try again"
            echo "  [2] Change serial port and try again"
            echo "  [3] Try uploading again with the same settings"
            echo "  [4] Skip firmware upload for now"
        } >&2
        local choice=""
        choice="$(_prompt_input "Next step [1]: ")"
        choice="$(printf "%s" "$choice" | tr '[:upper:]' '[:lower:]')"
        case "$choice" in
            ""|1|board|b|bootloader|change|c) printf "%s" "change_board"; return ;;
            2|port|p) printf "%s" "change_port"; return ;;
            3|retry|r|same) printf "%s" "retry"; return ;;
            4|skip|s|n|no) printf "%s" "skip"; return ;;
            *) warn "Please choose 1, 2, 3, or 4." >&2 ;;
        esac
    done
}

_configure_servo_limits_interactive() {
    echo ""
    warn "Servo limits must be measured safely in the Pololu Maestro Control Center before powering DJ-R3X servos."
    echo "Use the values shown in the Maestro app, like: 496 - 2496"
    echo "The values are stored in .env as microseconds. config.py converts them to quarter-microseconds at runtime."
    echo "Startup fails closed for non-numeric values, values outside 300-3000, or incomplete min/max pairs."
    echo "Press Enter on a servo only if the current .env/default values are safe for this build."
    echo ""

    if ! _prompt_yes_no "Save servo min/max limits to .env now? [y/N] " "n"; then
        warn "Skipping servo limit programming."
        MANUAL_ATTENTION+=("Update .env servo limit overrides after measuring Maestro min/max values before powering servos")
        return
    fi

    "$VENV_PYTHON" - "$PROJECT_DIR/config.py" "$ENV_DST" <<'PY'
import ast
import re
import sys
from pathlib import Path

try:
    _tty = open("/dev/tty", "r+", encoding="utf-8", buffering=1)
except OSError:
    _tty = None


def prompt_input(prompt: str) -> str:
    if _tty is None:
        return input(prompt)
    _tty.write(prompt)
    _tty.flush()
    line = _tty.readline()
    if line == "":
        raise EOFError("No input available on /dev/tty")
    return line.rstrip("\n")

config_path = Path(sys.argv[1])
env_path = Path(sys.argv[2])
text = config_path.read_text(encoding="utf-8")
match = re.search(r"SERVO_CHANNELS\s*=\s*(\{.*?\n\})", text, re.DOTALL)
if not match:
    raise SystemExit("Could not find SERVO_CHANNELS in config.py")

channels = ast.literal_eval(match.group(1))
channel_rows = sorted(channels.items(), key=lambda item: int(item[1]["ch"]))
updates: dict[str, tuple[float, float]] = {}
friendly_names = {
    "neck": "neck pan / left-right head turn",
    "headlift": "head lift / up-down head height",
    "headtilt": "head tilt / nod angle",
    "visor": "visor",
    "elbow": "left elbow",
    "hand": "left hand/wrist",
    "pokerarm": "right poker arm",
    "heroarm": "right hero arm",
}

def env_key(name: str, suffix: str) -> str:
    return f"SERVO_{name.upper()}_{suffix}_US"

def parse_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        key = key.strip()
        value = raw.strip().strip("'\"")
        env[key] = value
    return env

def write_env_values(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    used: set[str] = set()
    rendered: list[str] = []
    for line in lines:
        if "=" not in line or line.lstrip().startswith("#"):
            rendered.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in values:
            rendered.append(f"{key}={values[key]}")
            used.add(key)
        else:
            rendered.append(line)
    for key, value in values.items():
        if key not in used:
            rendered.append(f"{key}={value}")
    path.write_text("\n".join(rendered) + "\n", encoding="utf-8")

env_values = parse_env(env_path)

def fmt_us(q_us: int) -> str:
    value = q_us / 4
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")

def fmt_raw_us(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")

def current_us(name: str, key: str, fallback_q_us: int) -> float:
    raw = env_values.get(env_key(name, key))
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return fallback_q_us / 4

def parse_limits(raw: str) -> tuple[float, float] | None:
    nums = re.findall(r"\d+(?:\.\d+)?", raw)
    if len(nums) < 2:
        return None
    lo_us = float(nums[0])
    hi_us = float(nums[1])
    if lo_us <= 0 or hi_us <= 0:
        return None
    if not (300 <= lo_us <= 3000 and 300 <= hi_us <= 3000):
        return None
    if lo_us > hi_us:
        lo_us, hi_us = hi_us, lo_us
    return lo_us, hi_us

print("Servo channels:")
for name, cfg in channel_rows:
    ch = int(cfg["ch"])
    friendly = friendly_names.get(name, name)
    current_min = current_us(name, "MIN", int(cfg["min"]))
    current_max = current_us(name, "MAX", int(cfg["max"]))
    current = f"{fmt_raw_us(current_min)} - {fmt_raw_us(current_max)}"
    neutral = fmt_us(int(cfg["neutral"]))
    prompt = (
        f"  ch {ch} {name} ({friendly}) current {current} us, "
        f"neutral {neutral} us. New min-max: "
    )
    while True:
        raw = prompt_input(prompt).strip()
        if not raw:
            break
        parsed = parse_limits(raw)
        if parsed is None:
            print("    Enter two Maestro microsecond values like 496 - 2496, within 300 - 3000, or press Enter to keep current.")
            continue
        lo_us, hi_us = parsed
        neutral_us = int(cfg["neutral"]) / 4
        if not (lo_us <= neutral_us <= hi_us):
            print(
                f"    Warning: current neutral {fmt_raw_us(neutral_us)} us is outside "
                f"{fmt_raw_us(lo_us)} - {fmt_raw_us(hi_us)} us. Update neutral later if needed."
            )
        updates[name] = (lo_us, hi_us)
        break

if not updates:
    print("No servo limit changes requested.")
    raise SystemExit(0)

new_values: dict[str, str] = {}
for name, (lo_us, hi_us) in updates.items():
    new_values[env_key(name, "MIN")] = fmt_raw_us(lo_us)
    new_values[env_key(name, "MAX")] = fmt_raw_us(hi_us)

write_env_values(env_path, new_values)
print("Updated .env servo limit overrides:")
for name, (lo_us, hi_us) in sorted(updates.items(), key=lambda item: int(channels[item[0]]["ch"])):
    ch = int(channels[name]["ch"])
    friendly = friendly_names.get(name, name)
    print(
        f"  ch {ch} {name} ({friendly}): "
        f"{fmt_raw_us(lo_us)} - {fmt_raw_us(hi_us)} us "
        f"({round(lo_us * 4)} - {round(hi_us * 4)} q-us runtime)"
    )
PY
    INSTALLED_ITEMS+=(".env servo min/max limits reviewed")
}

_compile_and_upload_arduino() {
    local label="$1"
    local sketch_dir="$2"
    local port="$3"
    local default_fqbn="${4:-arduino:avr:nano}"

    if ! command -v arduino-cli &>/dev/null; then
        warn "arduino-cli is not available — skipping $label firmware upload."
        MANUAL_ATTENTION+=("Upload $label firmware manually after installing arduino-cli")
        return
    fi
    if [[ -z "$port" ]]; then
        warn "No serial port selected for $label — skipping firmware upload."
        return
    fi
    if [[ ! -d "$sketch_dir" ]]; then
        warn "Sketch directory missing for $label: $sketch_dir"
        MANUAL_ATTENTION+=("Sketch directory missing for $label: $sketch_dir")
        return
    fi

    local fqbn=""
    fqbn="$(_choose_arduino_board_fqbn "$label" "$default_fqbn")"
    if [[ -z "$fqbn" ]]; then
        warn "No FQBN entered — skipping $label firmware upload."
        return
    fi

    while true; do
        log "Compiling $label firmware ($fqbn)..."
        if arduino-cli compile --fqbn "$fqbn" "$sketch_dir"; then
            ok "$label firmware compiled."
        else
            warn "$label firmware compile failed."
            local compile_action=""
            compile_action="$(_arduino_compile_failure_action "$label")"
            case "$compile_action" in
                change_board)
                    fqbn="$(_choose_arduino_board_fqbn "$label" "$default_fqbn")"
                    [[ -n "$fqbn" ]] || return
                    continue
                    ;;
                retry)
                    continue
                    ;;
                skip)
                    MANUAL_ATTENTION+=("Skipped $label firmware upload after compile failure")
                    return
                    ;;
                *)
                    continue
                    ;;
            esac
        fi

        while true; do
            log "Uploading $label firmware to $port..."
            if arduino-cli upload -p "$port" --fqbn "$fqbn" "$sketch_dir"; then
                INSTALLED_ITEMS+=("$label firmware uploaded")
                ok "$label firmware uploaded."
                return
            fi

            warn "$label firmware upload failed."
            local upload_action=""
            upload_action="$(_arduino_upload_failure_action "$label")"
            case "$upload_action" in
                change_board)
                    fqbn="$(_choose_arduino_board_fqbn "$label" "$default_fqbn")"
                    [[ -n "$fqbn" ]] || return
                    break
                    ;;
                change_port)
                    local new_port=""
                    new_port="$(_prompt_input "Upload port [$port]: ")"
                    if [[ -n "$new_port" ]]; then
                        port="$new_port"
                    fi
                    ;;
                retry)
                    ;;
                skip)
                    MANUAL_ATTENTION+=("Skipped $label firmware upload after upload failure")
                    return
                    ;;
                *)
                    ;;
            esac
        done
    done
}

_guided_arduino_device_setup() {
    local label="$1"
    local env_key="$2"
    local kind="$3"
    local sketch_dir="$4"
    local connect_prompt="$5"
    local wiring_prompt="$6"
    local default_fqbn="${7:-arduino:avr:nano}"

    echo ""
    echo -e "${BOLD}$label${NC}"
    _prompt_continue "$connect_prompt"

    local after_file=""
    local candidate_file=""
    after_file="$(mktemp)"
    candidate_file="$(mktemp)"
    _scan_for_serial_device_with_retry "$label" "$after_file" "$candidate_file" || true

    _select_serial_port_for_env "$label" "$env_key" "$kind" "$candidate_file"

    if [[ -n "$SELECTED_DEVICE_PORT" ]] && _prompt_yes_no "Compile and upload $label firmware now? [y/N] " "n"; then
        _compile_and_upload_arduino "$label" "$sketch_dir" "$SELECTED_DEVICE_PORT" "$default_fqbn"
    fi

    echo ""
    echo "$wiring_prompt"
    _prompt_continue "Press Enter when you are ready to continue..."

    cp "$after_file" "$HARDWARE_BASELINE_FILE"
    rm -f "$after_file" "$candidate_file"
}

_guided_maestro_setup() {
    echo ""
    echo -e "${BOLD}Pololu Maestro servo controller${NC}"

    warn "Servo safety: determine every servo limit in the Pololu Maestro Control Center first."
    echo "Write down min and max values, then save them to .env before connecting powered servos."
    echo "Do not connect the Maestro to live servo power until those limits are programmed."
    _configure_servo_limits_interactive

    echo ""
    echo "Disconnect the Pololu Maestro USB now, if it is connected. Keep servo power disconnected."
    _prompt_continue "Press Enter once the Maestro USB is disconnected..."
    _print_dev_directory

    if [[ -z "$HARDWARE_BASELINE_FILE" ]]; then
        HARDWARE_BASELINE_FILE="$(mktemp)"
    fi
    _snapshot_dev_names "$HARDWARE_BASELINE_FILE"

    echo ""
    echo "Connect the Pololu Maestro by USB only. Keep servo power disconnected and do not power live servos yet."
    _prompt_continue "Press Enter after connecting the unpowered Maestro..."

    local after_file=""
    local candidate_file=""
    after_file="$(mktemp)"
    candidate_file="$(mktemp)"
    _scan_for_serial_device_with_retry "Pololu Maestro" "$after_file" "$candidate_file" || true

    _select_serial_port_for_env "Pololu Maestro" "MAESTRO_PORT" "maestro" "$candidate_file"

    cp "$after_file" "$HARDWARE_BASELINE_FILE"
    rm -f "$after_file" "$candidate_file"
}

HARDWARE_BASELINE_FILE=""
_configure_droid_hardware_interactive() {
    if [[ ! -f "$ENV_DST" ]]; then
        warn ".env is missing — skipping droid hardware setup."
        return
    fi

    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  Droid Hardware Setup${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    local setup_leds=0
    local setup_servos=0
    if _prompt_yes_no "Set up Arduino LED controllers now? [Y/n] " "y"; then
        setup_leds=1
    fi
    if _prompt_yes_no "Set up a Pololu Maestro servo controller now? [y/N] " "n"; then
        setup_servos=1
    fi

    if [[ "$setup_leds" -eq 0 && "$setup_servos" -eq 0 ]]; then
        warn "No physical hardware component selected — skipping droid hardware setup."
        return
    fi

    if [[ "$setup_leds" -eq 1 ]]; then
        echo ""
        echo "Disconnect the chest and head LED Arduinos now."
        _prompt_continue "Press Enter once the LED Arduino USB device(s) are disconnected..."

        _print_dev_directory

        HARDWARE_BASELINE_FILE="$(mktemp)"
        _snapshot_dev_names "$HARDWARE_BASELINE_FILE"
    fi

    if [[ "$setup_leds" -eq 1 ]]; then
        _guided_arduino_device_setup \
            "Chest LED Arduino" \
            "ARDUINO_CHEST_PORT" \
            "chest" \
            "$PROJECT_DIR/arduino/chest_nano" \
            "Connect the chest LED Arduino by USB, then press Enter..." \
            "After upload, connect the chest LEDs: data to Arduino pin 6, plus 5V and ground to the LED power rails." \
            "arduino:avr:nano"

        _guided_arduino_device_setup \
            "Head LED Arduino" \
            "ARDUINO_HEAD_PORT" \
            "head" \
            "$PROJECT_DIR/arduino/head_nano" \
            "Connect the head LED Arduino by USB, then press Enter..." \
            "After upload, connect the eyes data line to Arduino pin 6, then daisy-chain the data output from the eyes into the mouth PCB. Connect shared 5V and ground." \
            "arduino:avr:uno"
    else
        local current_led_port=""
        for key in ARDUINO_HEAD_PORT ARDUINO_CHEST_PORT; do
            current_led_port="$(_env_current_value "$key")"
            if [[ -z "$current_led_port" ]] || _env_value_is_template_placeholder "$key" "$current_led_port"; then
                _set_env_value "$key" ""
                INSTALLED_ITEMS+=(".env $key disabled")
            fi
        done
        ok "LED Arduino setup skipped."
    fi

    if [[ "$setup_servos" -eq 1 ]]; then
        _guided_maestro_setup
    else
        local current_maestro_port=""
        current_maestro_port="$(_env_current_value "MAESTRO_PORT")"
        if [[ -z "$current_maestro_port" ]] || _env_value_is_template_placeholder "MAESTRO_PORT" "$current_maestro_port"; then
            _set_env_value "MAESTRO_PORT" ""
            INSTALLED_ITEMS+=(".env MAESTRO_PORT disabled")
        fi
        ok "Pololu Maestro setup skipped."
    fi

    rm -f "$HARDWARE_BASELINE_FILE"
    HARDWARE_BASELINE_FILE=""

    ok "Droid hardware setup prompts complete."
    MANUAL_ATTENTION+=("Activate the venv and run DJ-R3X: source venv/bin/activate && python main.py")
}

_configure_software_only_hardware_ports() {
    if [[ ! -f "$ENV_DST" ]]; then
        return
    fi
    echo ""
    log "Software-only hardware settings"
    echo "For a Mac-only test run, blank hardware ports disable servos and LED controllers cleanly."
    local key=""
    for key in MAESTRO_PORT ARDUINO_HEAD_PORT ARDUINO_CHEST_PORT; do
        _show_current_env_port "$key" "$key"
    done
    if _prompt_yes_no "Blank any template placeholder hardware ports in .env? [Y/n] " "y"; then
        local current=""
        for key in MAESTRO_PORT ARDUINO_HEAD_PORT ARDUINO_CHEST_PORT; do
            current="$(_env_current_value "$key")"
            if [[ -z "$current" ]] || _env_value_is_template_placeholder "$key" "$current"; then
                _set_env_value "$key" ""
                INSTALLED_ITEMS+=(".env $key disabled for software-only mode")
            fi
        done
        ok "Template hardware ports blanked; real existing port values were preserved."
    fi
}

_configure_local_interactive() {
    if ! _is_interactive; then
        warn "Non-interactive shell detected — skipping mic/camera/API prompts."
        MANUAL_ATTENTION+=("Fill in apikeys.py with OpenAI and ElevenLabs API keys")
        MANUAL_ATTENTION+=("Update AUDIO_DEVICE_INDEX and camera settings in .env")
        return
    fi

    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  Local Interactive Configuration${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    if [[ -f "$APIKEYS_DST" ]]; then
        log "API keys"
        echo "Leave a field blank to keep the current value."
        local openai_key=""
        local elevenlabs_key=""
        openai_key="$(_prompt_secret "OpenAI API key: ")"
        elevenlabs_key="$(_prompt_secret "ElevenLabs API key: ")"
        if [[ -n "$openai_key" || -n "$elevenlabs_key" ]]; then
            _write_apikeys "$openai_key" "$elevenlabs_key"
            INSTALLED_ITEMS+=("apikeys.py updated")
            ok "apikeys.py updated."
        fi
        if _apikeys_have_placeholders; then
            MANUAL_ATTENTION+=("apikeys.py still has placeholder values — fill in real OpenAI and ElevenLabs keys")
        fi
    fi

    local current_voice_id=""
    current_voice_id="$(_current_voice_id)"
    if [[ -n "$current_voice_id" ]]; then
        echo ""
        log "ElevenLabs voice"
        echo "Current ELEVENLABS_VOICE_ID in config.py: $current_voice_id"
        if _prompt_yes_no "Use a different ElevenLabs voice ID? [y/N] " "n"; then
            local new_voice_id=""
            new_voice_id="$(_prompt_input "New ElevenLabs voice ID: ")"
            if [[ -n "$new_voice_id" ]]; then
                _update_voice_id "$new_voice_id"
                INSTALLED_ITEMS+=("config.py ELEVENLABS_VOICE_ID updated")
                ok "config.py updated with the new ElevenLabs voice ID."
            else
                warn "No voice ID entered — leaving config.py unchanged."
            fi
        fi
    else
        MANUAL_ATTENTION+=("Could not read ELEVENLABS_VOICE_ID from config.py")
    fi

    echo ""
    log "Physical droid hardware"
    if _prompt_yes_no "Are you setting this up for a physical DJ-R3X droid? [y/N] " "n"; then
        _configure_droid_hardware_interactive
    else
        _configure_software_only_hardware_ports
    fi

    if [[ -f "$ENV_DST" ]]; then
        echo ""
        log "Microphone selection"
        _list_audio_inputs
        echo ""
        local audio_choice=""
        audio_choice="$(_prompt_input "AUDIO_DEVICE_INDEX to use (blank = keep current, 'none' = disable): ")"
        if [[ -n "$audio_choice" ]]; then
            if [[ "$audio_choice" == "none" ]]; then
                _set_env_value "AUDIO_DEVICE_INDEX" ""
                INSTALLED_ITEMS+=(".env AUDIO_DEVICE_INDEX disabled")
                ok "AUDIO_DEVICE_INDEX disabled in .env."
            elif [[ "$audio_choice" =~ ^[0-9]+$ ]]; then
                _set_env_value "AUDIO_DEVICE_INDEX" "$audio_choice"
                INSTALLED_ITEMS+=(".env AUDIO_DEVICE_INDEX=$audio_choice")
                ok "AUDIO_DEVICE_INDEX set to $audio_choice."
            else
                warn "Audio choice must be a numeric device index or 'none' — leaving .env unchanged."
            fi
        fi

        echo ""
        log "Camera selection"
        _list_cameras
        echo ""
        echo "Enter an OpenCV index like 0, a device-name hint like FaceTime, or 'none' to disable."
        local camera_choice=""
        camera_choice="$(_prompt_input "Camera selection (blank = keep current): ")"
        if [[ -n "$camera_choice" ]]; then
            if [[ "$camera_choice" == "none" ]]; then
                _set_env_value "CAMERA_INDEX" ""
                _set_env_value "CAMERA_DEVICE_NAME" ""
                INSTALLED_ITEMS+=(".env camera disabled")
                ok "Camera disabled in .env."
            elif [[ "$camera_choice" =~ ^[0-9]+$ ]]; then
                _set_env_value "CAMERA_INDEX" "$camera_choice"
                _set_env_value "CAMERA_DEVICE_NAME" ""
                INSTALLED_ITEMS+=(".env CAMERA_INDEX=$camera_choice")
                ok "CAMERA_INDEX set to $camera_choice and CAMERA_DEVICE_NAME cleared."
            else
                _set_env_value "CAMERA_INDEX" ""
                _set_env_value "CAMERA_DEVICE_NAME" "$camera_choice"
                INSTALLED_ITEMS+=(".env CAMERA_DEVICE_NAME=$camera_choice")
                ok "CAMERA_DEVICE_NAME set to '$camera_choice' and CAMERA_INDEX cleared."
            fi
        fi
    fi
}

_configure_local_interactive

# ── 9. Asset and model downloads ──────────────────────────────────────────────
SETUP_ASSETS="$PROJECT_DIR/setup_assets.py"
if [[ -f "$SETUP_ASSETS" ]]; then
    log "Running setup_assets.py (model downloads + database init)..."
    "$VENV_PYTHON" "$SETUP_ASSETS"
    ok "setup_assets.py completed."
else
    warn "setup_assets.py not found — skipping model downloads and DB init."
    MANUAL_ATTENTION+=("setup_assets.py missing — run it once available: $VENV_PYTHON setup_assets.py")
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  Setup Summary${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ── Venv activation check ─────────────────────────────────────────────────────
# $VIRTUAL_ENV is set by the shell's `activate` script — this script cannot set
# it for the user's interactive session, so we detect and warn prominently.
if [[ "${VIRTUAL_ENV:-}" == "$VENV_DIR" ]]; then
    echo -e "${GREEN}${BOLD}✓  Virtual environment is active.${NC}"
else
    echo -e "${RED}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}${BOLD}║  ⚠  VIRTUAL ENVIRONMENT IS NOT ACTIVE           ║${NC}"
    echo -e "${RED}${BOLD}╠══════════════════════════════════════════════════╣${NC}"
    echo -e "${RED}${BOLD}║                                                  ║${NC}"
    echo -e "${RED}${BOLD}║  The venv was created but THIS shell session is  ║${NC}"
    echo -e "${RED}${BOLD}║  not inside it. Shell scripts cannot activate a  ║${NC}"
    echo -e "${RED}${BOLD}║  venv for you — you must run this yourself:      ║${NC}"
    echo -e "${RED}${BOLD}║                                                  ║${NC}"
    echo -e "${RED}${BOLD}║    source venv/bin/activate                      ║${NC}"
    echo -e "${RED}${BOLD}║                                                  ║${NC}"
    echo -e "${RED}${BOLD}║  Do this BEFORE running python, pip, or any      ║${NC}"
    echo -e "${RED}${BOLD}║  project commands.                               ║${NC}"
    echo -e "${RED}${BOLD}║                                                  ║${NC}"
    echo -e "${RED}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
fi

echo ""

if [[ ${#INSTALLED_ITEMS[@]} -gt 0 ]]; then
    echo -e "${GREEN}${BOLD}Installed / configured this run:${NC}"
    for item in "${INSTALLED_ITEMS[@]}"; do
        echo -e "  ${GREEN}✓${NC} $item"
    done
else
    echo -e "${GREEN}Nothing to install — environment already fully set up.${NC}"
fi

echo ""

if [[ ${#MANUAL_ATTENTION[@]} -gt 0 ]]; then
    echo -e "${YELLOW}${BOLD}⚠  Manual attention required:${NC}"
    for item in "${MANUAL_ATTENTION[@]}"; do
        echo -e "  ${YELLOW}→${NC} $item"
    done
else
    echo -e "${GREEN}No manual steps required — you're ready to go.${NC}"
fi

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  Next steps${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}${BOLD}  Step 1 — Activate the virtual environment (REQUIRED):${NC}"
echo ""
echo -e "    ${BOLD}source venv/bin/activate${NC}"
echo ""
echo -e "  Your prompt will change to show ${BOLD}(venv)${NC} when it is active."
echo -e "  You must do this in every new terminal session before"
echo -e "  running any project commands."
echo ""
echo -e "${BOLD}  Step 2 — Start DJ-R3X:${NC}"
echo ""
echo -e "    ${BOLD}python main.py${NC}"
echo ""
