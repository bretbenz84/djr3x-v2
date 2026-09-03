#!/usr/bin/env python3
"""
tools/flex_ctl.py — read (and only when explicitly told to, write) XMOS XVF3800
DSP parameters on the reSpeaker Flex over USB.

    ./venv/bin/python tools/flex_ctl.py                    # dump the audio-relevant set
    ./venv/bin/python tools/flex_ctl.py VERSION AEC_AECCONVERGED PP_AGCGAIN
    ./venv/bin/python tools/flex_ctl.py --list             # every known parameter
    ./venv/bin/python tools/flex_ctl.py --write PP_AGCONOFF 0

Reads are harmless and can run while Rex is up (they are vendor control
transfers on the device endpoint; the audio streams are untouched). WRITES
change the live DSP — volatile until SAVE_CONFIGURATION, and REBOOT /
CLEAR_CONFIGURATION reset everything. Owner rule (2026-09-02): never write or
reflash without Bret's explicit go. `--write` therefore prints what it is
about to do and requires typing "yes".

Why this exists instead of vendoring Seeed's xvf_host.py: their script has no
license file, prints protocol chatter on every read, and needs a system libusb
on macOS. This uses the same wire protocol (bRequest 0, wIndex = resource id,
wValue = command id with bit 7 set for reads, one status byte ahead of the
payload; status 64 = retry) through pyusb + the wheel-bundled libusb, so it
works out of the venv with no Homebrew step.

Verified against firmware 1.0.0 "ua-io16-6ch-sqr" (VID 0x2886, PID 0x001e).
"""

from __future__ import annotations

import argparse
import struct
import sys
import time

VID = 0x2886
_STATUS_OK = 0
_STATUS_RETRY = 64
_TIMEOUT_MS = 5000

# name: (resid, cmdid, count, access, type, description)
# Resource ids: 48 app, 33 AEC, 35 audio manager, 17 post-processing, 20 LED/DoA.
PARAMETERS: dict[str, tuple[int, int, int, str, str, str]] = {
    "VERSION": (48, 0, 3, "ro", "uint8", "firmware version major.minor.patch"),
    "BLD_MSG": (48, 1, 50, "ro", "char", "build configuration name (e.g. ua-io16-6ch-sqr)"),
    "USB_BIT_DEPTH": (48, 8, 2, "rw", "uint8", "USB bit depth IN, OUT (16/24/32) — SETTING REBOOTS the chip"),
    "SAVE_CONFIGURATION": (48, 9, 1, "wo", "uint8", "persist the current parameters to flash"),
    "CLEAR_CONFIGURATION": (48, 10, 1, "wo", "uint8", "wipe saved parameters, revert to defaults"),
    "REBOOT": (48, 7, 1, "wo", "uint8", "reboot; resets every unsaved parameter"),
    # AEC
    "SHF_BYPASS": (33, 70, 1, "rw", "uint8", "bypass the whole AEC/beamformer (SHF) block"),
    "AEC_NUM_MICS": (33, 71, 1, "ro", "int32", "mic inputs into the AEC"),
    "AEC_NUM_FARENDS": (33, 72, 1, "ro", "int32", "far-end (reference) inputs into the AEC"),
    "AEC_MIC_ARRAY_TYPE": (33, 73, 1, "ro", "int32", "1 linear, 2 square/circular"),
    "AEC_AZIMUTH_VALUES": (33, 75, 4, "ro", "float", "beam azimuths (rad): beam1, beam2, free-running, auto-select"),
    "AEC_SPENERGY_VALUES": (33, 80, 4, "ro", "float", "speech energy per beam (>0 = speech)"),
    "AEC_AECPATHCHANGE": (33, 0, 1, "ro", "int32", "echo path change detected (0/1)"),
    "AEC_HPFONOFF": (33, 1, 1, "rw", "int32", "mic high-pass: 0 off, 1 70Hz, 2 125Hz, 3 150Hz, 4 180Hz"),
    "AEC_AECCONVERGED": (33, 3, 1, "ro", "int32", "adaptive filter converged (0/1) — read DURING/after playback"),
    "AEC_AECEMPHASISONOFF": (33, 4, 1, "rw", "int32", "pre/de-emphasis: 0 off, 1 on, 2 on_eq"),
    "AEC_FAR_EXTGAIN": (33, 5, 1, "rw", "float", "external gain (dB) applied to the reference"),
    "AEC_RT60": (33, 9, 1, "ro", "float", "RT60 estimate (s); negative/denormal = not estimated yet"),
    "AEC_ASROUTONOFF": (33, 35, 1, "rw", "int32", "1 = ASR beam outputs, 0 = raw AEC residual per mic"),
    "AEC_ASROUTGAIN": (33, 36, 1, "rw", "float", "fixed linear gain on the ASR output"),
    "AEC_FIXEDBEAMSONOFF": (33, 37, 1, "rw", "int32", "fixed focused beams instead of free-running (0/1)"),
    "AEC_FIXEDBEAMSAZIMUTH_VALUES": (33, 81, 2, "rw", "float", "fixed beam 1, 2 azimuths (rad)"),
    "AEC_FIXEDBEAMSGATING": (33, 83, 1, "rw", "uint8", "silence inactive fixed beams (0/1)"),
    # audio manager (routing, gains, delay)
    "AUDIO_MGR_MIC_GAIN": (35, 0, 1, "rw", "float", "pre-SHF microphone gain (linear)"),
    "AUDIO_MGR_REF_GAIN": (35, 1, 1, "rw", "float", "pre-SHF reference gain (linear)"),
    "AUDIO_MGR_SELECTED_AZIMUTHS": (35, 11, 2, "ro", "float", "processed DoA, auto-select beam DoA (rad)"),
    "AUDIO_MGR_SELECTED_CHANNELS": (35, 12, 2, "rw", "uint8", "beams routed to the 'user chosen' outputs"),
    "AUDIO_MGR_OP_L": (35, 15, 2, "rw", "uint8", "USB ch0 source: <category>,<source>"),
    "AUDIO_MGR_OP_R": (35, 19, 2, "rw", "uint8", "USB ch1 source: <category>,<source>"),
    "AUDIO_MGR_OP_CH3": (35, 28, 2, "rw", "uint8", "USB ch2 source (unsupported on the 6ch build: status 65)"),
    "AUDIO_MGR_OP_CH4": (35, 29, 2, "rw", "uint8", "USB ch3 source"),
    "AUDIO_MGR_OP_CH5": (35, 30, 2, "rw", "uint8", "USB ch4 source"),
    "AUDIO_MGR_OP_CH6": (35, 31, 2, "rw", "uint8", "USB ch5 source"),
    "AUDIO_MGR_FAR_END_DSP_ENABLE": (35, 25, 1, "rw", "uint8", "far-end DSP on the playback path (0/1)"),
    "AUDIO_MGR_SYS_DELAY": (35, 26, 1, "rw", "int32", "samples of delay applied to the reference before the AEC"),
    # post-processing (applies to the Conference output, USB ch0)
    "PP_AGCONOFF": (17, 10, 1, "rw", "int32", "AGC on the Conference output (0/1)"),
    "PP_AGCMAXGAIN": (17, 11, 1, "rw", "float", "AGC max linear gain [1..1000]"),
    "PP_AGCDESIREDLEVEL": (17, 12, 1, "rw", "float", "AGC target power [1e-8..1]"),
    "PP_AGCGAIN": (17, 13, 1, "rw", "float", "AGC current linear gain"),
    "PP_AGCTIME": (17, 14, 1, "rw", "float", "AGC ramp time constant (s)"),
    "PP_LIMITONOFF": (17, 19, 1, "rw", "int32", "limiter on the Conference output (0/1)"),
    "PP_MIN_NS": (17, 21, 1, "rw", "float", "stationary noise-suppression gain floor [0..1]"),
    "PP_MIN_NN": (17, 22, 1, "rw", "float", "non-stationary noise-suppression gain floor [0..1]"),
    "PP_ECHOONOFF": (17, 23, 1, "rw", "int32", "residual echo suppression (0/1)"),
    "PP_GAMMA_E": (17, 24, 1, "rw", "float", "echo over-subtraction, direct/early [0..2]"),
    "PP_GAMMA_ETAIL": (17, 25, 1, "rw", "float", "echo over-subtraction, tail [0..2]"),
    "PP_GAMMA_ENL": (17, 26, 1, "rw", "float", "non-linear echo over-subtraction [0..5]"),
    "PP_NLAEC_MODE": (17, 28, 1, "rw", "int32", "non-linear AEC training: 0 normal, 1 train, 2 train2"),
    "PP_DTSENSITIVE": (17, 31, 1, "rw", "int32", "echo suppression vs double-talk trade-off"),
    "PP_ATTNS_MODE": (17, 32, 1, "rw", "int32", "extra AGC gain reduction during non-speech (0/1)"),
    # LED / DoA
    "LED_EFFECT": (20, 12, 1, "rw", "uint8", "0 off, 1 breath, 2 rainbow, 3 single, 4 doa, 5 ring"),
    "DOA_VALUE": (20, 18, 2, "ro", "uint16", "direction of arrival 0-359, speech-detected flag"),
}

# What a bare `flex_ctl.py` prints — the parameters that decide what our pipeline hears.
DEFAULT_DUMP = [
    "VERSION", "BLD_MSG", "USB_BIT_DEPTH", "AEC_MIC_ARRAY_TYPE", "AEC_NUM_MICS",
    "AEC_NUM_FARENDS", "AUDIO_MGR_OP_L", "AUDIO_MGR_OP_R", "AUDIO_MGR_SELECTED_CHANNELS",
    "AEC_ASROUTONOFF", "AEC_ASROUTGAIN", "AEC_AECCONVERGED", "AEC_AECPATHCHANGE",
    "AEC_HPFONOFF", "AEC_AECEMPHASISONOFF", "AEC_FAR_EXTGAIN", "AEC_RT60",
    "AEC_FIXEDBEAMSONOFF", "AUDIO_MGR_MIC_GAIN", "AUDIO_MGR_REF_GAIN",
    "AUDIO_MGR_SYS_DELAY", "AUDIO_MGR_FAR_END_DSP_ENABLE", "PP_AGCONOFF",
    "PP_AGCMAXGAIN", "PP_AGCGAIN", "PP_AGCDESIREDLEVEL", "PP_ECHOONOFF",
    "PP_NLAEC_MODE", "PP_DTSENSITIVE", "PP_MIN_NS", "PP_MIN_NN", "PP_LIMITONOFF",
    "AEC_AZIMUTH_VALUES", "DOA_VALUE",
]

_FMT = {"uint8": ("B", 1), "uint16": ("H", 2), "int32": ("i", 4), "uint32": ("I", 4), "float": ("f", 4)}


class FlexDevice:
    """One open XVF3800 control channel. Use as a context manager."""

    def __init__(self, dev, backend):
        self.dev = dev
        self._backend = backend
        self.vid = int(getattr(dev, "idVendor", 0))
        self.pid = int(getattr(dev, "idProduct", 0))

    # -- transport -----------------------------------------------------------
    def _ctrl_in(self, resid: int, cmdid: int, length: int):
        import usb.util
        return self.dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, 0x80 | cmdid, resid, length, _TIMEOUT_MS,
        )

    def _ctrl_out(self, resid: int, cmdid: int, payload: bytes) -> None:
        import usb.util
        self.dev.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, cmdid, resid, payload, _TIMEOUT_MS,
        )

    # -- API -----------------------------------------------------------------
    def read(self, name: str):
        resid, cmdid, count, access, typ, _ = PARAMETERS[name]
        if access == "wo":
            raise ValueError(f"{name} is write-only")
        if typ == "char":
            length = count + 1
        else:
            length = count * _FMT[typ][1] + 1
        for attempt in range(100):
            resp = bytes(self._ctrl_in(resid, cmdid, length))
            if not resp:
                raise IOError(f"{name}: empty response")
            if resp[0] == _STATUS_OK:
                break
            if resp[0] != _STATUS_RETRY:
                raise IOError(f"{name}: device status {resp[0]}")
            time.sleep(0.01)
        else:
            raise IOError(f"{name}: device kept asking to retry")
        body = resp[1:]
        if typ == "char":
            return body.rstrip(b"\x00").decode("utf-8", errors="replace")
        code, size = _FMT[typ]
        return struct.unpack("<" + code * count, body[: count * size])

    def write(self, name: str, values) -> None:
        resid, cmdid, count, access, typ, _ = PARAMETERS[name]
        if access == "ro":
            raise ValueError(f"{name} is read-only")
        values = list(values)
        if typ == "char":
            payload = str(values[0]).encode("utf-8")
        else:
            if len(values) != count:
                raise ValueError(f"{name} takes {count} value(s), got {len(values)}")
            code, _ = _FMT[typ]
            cast = float if typ == "float" else int
            payload = struct.pack("<" + code * count, *[cast(v) for v in values])
        self._ctrl_out(resid, cmdid, payload)

    def close(self) -> None:
        try:
            import usb.util
            usb.util.dispose_resources(self.dev)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open_device(vid: int = VID) -> FlexDevice:
    """Find the Flex on USB. Raises RuntimeError with a plain reason when absent."""
    try:
        import libusb_package
        import usb.backend.libusb1
        import usb.core
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(f"pyusb/libusb-package not installed ({exc}); pip install pyusb libusb-package")
    backend = usb.backend.libusb1.get_backend(find_library=libusb_package.find_library)
    if backend is None:
        raise RuntimeError("no libusb backend available (libusb-package wheel missing?)")
    devices = list(usb.core.find(find_all=True, idVendor=vid, backend=backend) or [])
    if not devices:
        raise RuntimeError(f"no USB device with VID 0x{vid:04x} (is the reSpeaker plugged in?)")
    devices.sort(key=lambda d: int(getattr(d, "idProduct", 0)))
    return FlexDevice(devices[0], backend)


def read_param(name: str):
    """One-shot convenience: read a single parameter (opens and closes the device)."""
    with open_device() as dev:
        return dev.read(name.upper())


def snapshot(names=None) -> dict:
    """Read several parameters; a failed read becomes the string 'ERR ...'.

    Used by tools/mic_check.py's aec test to record the DSP state alongside the
    measurement, so a later regression can be tied to a changed parameter.
    """
    out: dict = {}
    try:
        dev = open_device()
    except Exception as exc:
        return {"_error": str(exc)}
    with dev:
        out["_usb"] = f"VID=0x{dev.vid:04x} PID=0x{dev.pid:04x}"
        for name in names or DEFAULT_DUMP:
            try:
                val = dev.read(name)
                out[name] = list(val) if isinstance(val, tuple) else val
            except Exception as exc:
                out[name] = f"ERR {exc}"
    return out


def _fmt(val) -> str:
    if isinstance(val, tuple):
        return ", ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in val)
    return str(val)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="parameter names to read (default: the audio-relevant set)")
    ap.add_argument("--list", action="store_true", help="list every known parameter and exit")
    ap.add_argument("--write", nargs="+", metavar=("NAME", "VALUE"),
                    help="WRITE a parameter (asks for confirmation; volatile until SAVE_CONFIGURATION)")
    args = ap.parse_args(argv)

    if args.list:
        for name, (resid, cmdid, count, access, typ, desc) in PARAMETERS.items():
            print(f"{name:<32} {access:<2} {typ:<6} x{count:<2} {desc}")
        return 0

    try:
        dev = open_device()
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1

    with dev:
        print(f"reSpeaker Flex VID=0x{dev.vid:04x} PID=0x{dev.pid:04x}")
        if args.write:
            name = args.write[0].upper()
            if name not in PARAMETERS:
                print(f"error: unknown parameter {name}")
                return 2
            values = args.write[1:]
            if not values:
                print("error: --write needs NAME and at least one VALUE")
                return 2
            before = None
            try:
                before = _fmt(dev.read(name))
            except Exception:
                pass
            print(f"ABOUT TO WRITE {name} = {' '.join(values)}   (currently: {before})")
            print("This changes the live DSP on the mic array. Owner rule: only with Bret's go.")
            if input("type yes to proceed: ").strip().lower() != "yes":
                print("aborted")
                return 3
            dev.write(name, values)
            try:
                print(f"{name:<32} now: {_fmt(dev.read(name))}")
            except Exception as exc:
                print(f"{name:<32} written (read-back failed: {exc})")
            return 0

        names = [n.upper() for n in args.names] or DEFAULT_DUMP
        for name in names:
            if name not in PARAMETERS:
                print(f"{name:<32} (unknown parameter — see --list)")
                continue
            try:
                print(f"{name:<32} {_fmt(dev.read(name))}")
            except Exception as exc:
                print(f"{name:<32} ERR {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
