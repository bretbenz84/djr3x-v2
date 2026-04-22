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
    # Warn if placeholder values are still in place
    if grep -qE '"sk-\.\.\."' "$APIKEYS_DST" 2>/dev/null \
    || grep -q 'your-elevenlabs-api-key-here' "$APIKEYS_DST" 2>/dev/null; then
        MANUAL_ATTENTION+=("apikeys.py still has placeholder values — fill in real OpenAI and ElevenLabs keys")
    fi
else
    if [[ -f "$APIKEYS_SRC" ]]; then
        cp "$APIKEYS_SRC" "$APIKEYS_DST"
        INSTALLED_ITEMS+=("apikeys.py (copied from template)")
        ok "apikeys.py created from apikeys.example.py."
        MANUAL_ATTENTION+=("Fill in real API keys in: $APIKEYS_DST")
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
        MANUAL_ATTENTION+=("Update hardware device paths in: $ENV_DST")
        MANUAL_ATTENTION+=("  - CAMERA_INDEX: run: python3 -c \"import cv2; [print(i) for i in range(5) if cv2.VideoCapture(i).isOpened()]\"")
        MANUAL_ATTENTION+=("  - MAESTRO_PORT / ARDUINO_*_PORT: check 'ls /dev/tty.usb*' with hardware connected")
    else
        warn ".env.example not found — cannot create .env."
        MANUAL_ATTENTION+=("Create .env manually (template .env.example is missing)")
    fi
fi

# ── 8. Asset and model downloads ──────────────────────────────────────────────
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
