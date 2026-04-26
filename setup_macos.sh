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
BREW_PACKAGES=(cmake portaudio ffmpeg libsndfile boost openblas)

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
