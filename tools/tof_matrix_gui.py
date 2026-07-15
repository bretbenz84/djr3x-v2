#!/usr/bin/env python3
"""Live 8x8 depth-grid GUI for the DFRobot Matrix ToF bench firmware.

A proper desktop window (Qt / PySide6): a fixed, ordered 8x8 heat-map that
repaints in place — no scrolling. Pairs with firmware/tof_matrix_test running on
a spare ESP32, which streams one frame per line over USB serial:

    D,v0,v1,...,v63          64 distances in mm, row-major (index = y*8 + x)
    # ...                    human-readable status from the firmware

Features:
  - 8x8 colored grid, near = red -> far = blue, dim = no return, mm shown per cell
  - live orientation controls (Rotate / Flip-H / Flip-V) to match how the board
    is physically mounted — the raw sensor order is arbitrary until you calibrate it
  - mm/cm unit toggle, Pause, a colorbar legend, and near/far/center + FPS stats
  - read-only: never writes to the serial port, safe against the sensor firmware

Usage:
    ./venv/bin/python tools/tof_matrix_gui.py                 # auto-detect port
    ./venv/bin/python tools/tof_matrix_gui.py --port /dev/cu.usbserial-0001
    ./venv/bin/python tools/tof_matrix_gui.py --list          # list serial ports

Keys: R rotate · H flip-H · V flip-V · U toggle mm/cm · Space pause · Q quit.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time

try:
    import serial  # pyserial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial not installed — run: ./venv/bin/pip install pyserial")

try:
    from PySide6.QtCore import Qt, QThread, Signal, QRectF
    from PySide6.QtGui import QColor, QPainter, QFont, QKeySequence, QShortcut
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QHBoxLayout, QVBoxLayout,
    )
except ImportError:
    sys.exit("PySide6 not installed — run: ./venv/bin/pip install PySide6")

BAUD = 115200
GRID = 8
NCELLS = GRID * GRID
RAMP_NEAR_MM = 50       # color-ramp bounds; distances clamped to this before hue map
RAMP_FAR_MM = 4000      # 4000 == sensor ceiling / no target in range
CENTER_IDX = (27, 28, 35, 36)   # the 4 middle zones of the 8x8 grid


# ---- port discovery ------------------------------------------------------
def find_port() -> str | None:
    for p in list_ports.comports():
        blob = f"{p.device} {p.description} {p.manufacturer}".lower()
        if any(k in blob for k in ("usbserial", "slab", "wch", "cp210", "ch340", "silicon labs")):
            return p.device
    hits = (glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.SLAB*")
            + glob.glob("/dev/cu.wchusbserial*") + glob.glob("/dev/ttyUSB*"))
    return hits[0] if hits else None


def list_serial_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print("Serial ports:")
    for p in ports:
        print(f"  {p.device:24s} {p.description}")


# ---- color mapping -------------------------------------------------------
def dist_to_qcolor(mm: int) -> QColor:
    """Near -> red, far -> blue (HSV hue 0..240). Invalid (<=0) -> dark gray."""
    if mm <= 0:
        return QColor(38, 38, 42)
    d = max(RAMP_NEAR_MM, min(RAMP_FAR_MM, mm))
    t = (d - RAMP_NEAR_MM) / (RAMP_FAR_MM - RAMP_NEAR_MM)   # 0 near .. 1 far
    hue = t * 240.0                                          # 0=red .. 240=blue
    c = QColor()
    c.setHsvF(hue / 360.0, 0.85, 0.95)
    return c


# ---- serial reader thread ------------------------------------------------
class SerialReader(QThread):
    """Reads the firmware stream on a worker thread; emits into the GUI thread.

    Cross-thread Qt signals are delivered on the receiver's (GUI) thread via the
    event loop, so the widgets are only ever touched from the main thread.
    """
    frame = Signal(list)     # 64 ints (mm), row-major
    status = Signal(str)     # a "# ..." firmware line
    note = Signal(str)       # connection / error notice for the status bar

    def __init__(self, port: str, baud: int, auto_detected: bool = False) -> None:
        super().__init__()
        self._port = port
        self._baud = baud
        self._auto = auto_detected   # was the port auto-detected (vs an explicit --port)?
        self._running = True

    def stop(self) -> None:
        self._running = False

    def _sleep(self, ms_total: int) -> bool:
        """Sleep ~ms_total in 100 ms slices; return False the moment stop() is seen
        so no wait path can make quit unresponsive."""
        for _ in range(max(1, ms_total // 100)):
            if not self._running:
                return False
            self.msleep(100)
        return self._running

    def run(self) -> None:
        open_fails = 0
        while self._running:
            try:
                # ValueError (bad baud) and OSError are NOT subclasses of
                # SerialException — catch them too so an invalid rate surfaces as a
                # retry note instead of silently killing this thread (GUI would then
                # hang forever on "waiting for first frame").
                ser = serial.Serial(self._port, self._baud, timeout=1)
            except (serial.SerialException, ValueError, OSError) as e:
                open_fails += 1
                hint = ""
                # Auto-detected boards can re-enumerate under a new /dev node on
                # unplug/replug; re-scan and adopt the new path so a replug
                # reconnects on its own. Never override an explicit --port.
                if self._auto and open_fails >= 2:
                    newp = find_port()
                    if newp and newp != self._port:
                        self.note.emit(f"port moved {self._port} → {newp}")
                        self._port = newp
                        open_fails = 0
                        continue
                    hint = " (device may have re-enumerated; pass --port if it won't reconnect)"
                self.note.emit(f"can't open {self._port}: {e}{hint} — retrying in 2s")
                if not self._sleep(2000):
                    return
                continue

            open_fails = 0
            self.note.emit(f"connected {self._port} @ {self._baud}")
            stop_requested = False
            try:
                while self._running:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    if line.startswith("D,"):
                        parts = line[2:].split(",")
                        if len(parts) != NCELLS:
                            continue
                        try:
                            self.frame.emit([int(p) for p in parts])
                        except ValueError:
                            continue
                    elif line.startswith("#"):
                        self.status.emit(line)
            except (serial.SerialException, OSError) as e:
                # Back off before reconnecting so a flapping link (open OK but
                # readline faults instantly) can't busy-spin open/close/emit.
                self.note.emit(f"serial dropped: {e} — reconnecting")
                stop_requested = not self._sleep(1000)
            finally:
                try:
                    ser.close()
                except Exception:
                    pass
            if stop_requested:
                return


# ---- the grid widget -----------------------------------------------------
class GridWidget(QWidget):
    """Paints the 8x8 grid + a colorbar. Holds the current frame and the
    display orientation (rotation in 90 deg CW steps, plus H/V flips)."""

    def __init__(self) -> None:
        super().__init__()
        self._frame = [0] * NCELLS
        self._rot = 0            # 0..3, each = 90 deg clockwise
        self._flip_h = False
        self._flip_v = False
        self._unit = "mm"        # "mm" or "cm"
        self.setMinimumSize(560, 480)

    # -- state setters (called from the GUI thread) --
    def set_frame(self, frame: list[int]) -> None:
        self._frame = frame
        self.update()

    def rotate(self) -> None:
        self._rot = (self._rot + 1) % 4
        self.update()

    def flip_h(self) -> None:
        self._flip_h = not self._flip_h
        self.update()

    def flip_v(self) -> None:
        self._flip_v = not self._flip_v
        self.update()

    def toggle_unit(self) -> None:
        self._unit = "cm" if self._unit == "mm" else "mm"
        self.update()

    def unit(self) -> str:
        return self._unit

    def orientation_text(self) -> str:
        bits = [f"rot {self._rot * 90}°"]
        if self._flip_h:
            bits.append("flip-H")
        if self._flip_v:
            bits.append("flip-V")
        return " · ".join(bits)

    def oriented(self) -> list[list[int]]:
        """Return the frame as an 8x8 matrix with the current orientation applied.

        Pure Python (no numpy dependency): start from row-major M[y][x], apply
        `rot` 90-deg clockwise turns, then optional horizontal/vertical flips.
        """
        m = [self._frame[y * GRID:(y + 1) * GRID] for y in range(GRID)]
        for _ in range(self._rot):                       # 90 deg CW: M'[r][c] = M[N-1-c][r]
            m = [[m[GRID - 1 - c][r] for c in range(GRID)] for r in range(GRID)]
        if self._flip_h:
            m = [list(reversed(row)) for row in m]
        if self._flip_v:
            m = list(reversed(m))
        return m

    # -- painting --
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, False)
        qp.fillRect(self.rect(), QColor(18, 18, 20))

        pad = 8
        label_l = 24          # room for Y labels on the left
        label_t = 20          # room for X labels on top
        bar_w = 54            # colorbar strip on the right
        avail_w = self.width() - pad * 2 - label_l - bar_w
        avail_h = self.height() - pad * 2 - label_t
        grid_px = max(0, min(avail_w, avail_h))
        cell = grid_px // GRID
        if cell < 4:
            return
        grid_px = cell * GRID
        gx = pad + label_l
        gy = pad + label_t

        mat = self.oriented()

        axis_font = QFont("Menlo", 9)
        qp.setFont(axis_font)
        qp.setPen(QColor(150, 150, 155))
        for i in range(GRID):
            qp.drawText(QRectF(gx + i * cell, pad - 2, cell, label_t),
                        Qt.AlignCenter, str(i))
            qp.drawText(QRectF(pad - 2, gy + i * cell, label_l, cell),
                        Qt.AlignCenter, str(i))

        val_font = QFont("Menlo", max(8, min(15, cell // 4)))
        val_font.setBold(True)
        for r in range(GRID):
            for c in range(GRID):
                mm = mat[r][c]
                col = dist_to_qcolor(mm)
                x = gx + c * cell
                y = gy + r * cell
                qp.fillRect(x, y, cell - 1, cell - 1, col)
                # contrasting text
                lum = 0.299 * col.red() + 0.587 * col.green() + 0.114 * col.blue()
                qp.setPen(QColor(0, 0, 0) if lum > 140 else QColor(255, 255, 255))
                qp.setFont(val_font)
                if mm <= 0:
                    txt = "—"
                elif self._unit == "cm":
                    txt = f"{mm / 10:.0f}"
                else:
                    txt = str(mm)
                qp.drawText(QRectF(x, y, cell - 1, cell - 1), Qt.AlignCenter, txt)

        self._paint_colorbar(qp, self.width() - pad - bar_w + 8, gy, grid_px)

    def _paint_colorbar(self, qp: QPainter, x: int, y: int, h: int) -> None:
        w = 16
        steps = max(1, h)
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            mm = RAMP_FAR_MM - t * (RAMP_FAR_MM - RAMP_NEAR_MM)   # near at bottom
            qp.fillRect(x, y + i, w, 1, dist_to_qcolor(int(mm)))
        qp.setFont(QFont("Menlo", 8))
        qp.setPen(QColor(150, 150, 155))
        unit = self._unit
        far = "4000" if unit == "mm" else "400"
        near = "50" if unit == "mm" else "5"
        qp.drawText(QRectF(x + w + 3, y - 6, 40, 14), Qt.AlignLeft, f"{far}")
        qp.drawText(QRectF(x + w + 3, y + h - 8, 40, 14), Qt.AlignLeft, f"{near}")
        qp.drawText(QRectF(x + w + 3, y + h // 2 - 7, 40, 14), Qt.AlignLeft, unit)


# ---- main window ---------------------------------------------------------
class MatrixWindow(QMainWindow):
    def __init__(self, port: str, baud: int, auto_detected: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("DFRobot 8x8 Matrix ToF — VL53L7CX @ 0x33")
        self._paused = False
        self._count = 0
        self._fps = 0.0
        self._last_t: float | None = None

        self.grid = GridWidget()

        # top control/stat bar
        self.stat = QLabel("waiting for first frame (firmware settles ~5s after boot)…")
        self.stat.setStyleSheet("color:#ddd; font-family:Menlo; font-size:12px;")
        self.orient = QLabel("")
        self.orient.setStyleSheet("color:#9ad; font-family:Menlo; font-size:12px;")

        def button(text, slot):
            b = QPushButton(text)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(slot)
            return b

        bar = QHBoxLayout()
        bar.addWidget(button("Rotate", self._rotate))
        bar.addWidget(button("Flip H", self._flip_h))
        bar.addWidget(button("Flip V", self._flip_v))
        bar.addWidget(button("mm/cm", self._toggle_unit))
        self.pause_btn = button("Pause", self._toggle_pause)
        bar.addWidget(self.pause_btn)
        bar.addStretch(1)
        bar.addWidget(self.orient)

        root = QVBoxLayout()
        root.addWidget(self.stat)
        root.addLayout(bar)
        root.addWidget(self.grid, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self._sync_orient()

        for key, slot in (("R", self._rotate), ("H", self._flip_h), ("V", self._flip_v),
                          ("U", self._toggle_unit), ("Space", self._toggle_pause),
                          ("Q", self.close)):
            QShortcut(QKeySequence(key), self, activated=slot)

        # serial thread
        self.reader = SerialReader(port, baud, auto_detected)
        self.reader.frame.connect(self._on_frame)
        self.reader.status.connect(self._on_status)
        self.reader.note.connect(self._on_note)
        self.reader.start()

    # -- control slots --
    def _rotate(self):
        self.grid.rotate(); self._sync_orient()

    def _flip_h(self):
        self.grid.flip_h(); self._sync_orient()

    def _flip_v(self):
        self.grid.flip_v(); self._sync_orient()

    def _toggle_unit(self):
        self.grid.toggle_unit()

    def _toggle_pause(self):
        self._paused = not self._paused
        self.pause_btn.setText("Resume" if self._paused else "Pause")

    def _sync_orient(self):
        self.orient.setText(self.grid.orientation_text())

    # -- data slots (GUI thread) --
    def _on_frame(self, frame: list[int]) -> None:
        now = time.time()
        if self._last_t is not None:
            dt = now - self._last_t
            if dt > 0:
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0 else 0.8 * self._fps + 0.2 * inst
        self._last_t = now
        self._count += 1
        if self._paused:
            return
        self.grid.set_frame(frame)

        valid = [d for d in frame if d > 0]
        cvals = [frame[i] for i in CENTER_IDX if frame[i] > 0]
        near = min(valid) if valid else 0
        far = max(valid) if valid else 0
        center = round(sum(cvals) / len(cvals)) if cvals else 0
        # Report in the same unit the grid/colorbar are showing, so the whole
        # window agrees when the user toggles mm/cm.
        scale = 10.0 if self.grid.unit() == "cm" else 1.0
        u = self.grid.unit()
        self.stat.setText(
            f"frame {self._count:>6}   {self._fps:4.1f} fps   valid {len(valid):2d}/64   "
            f"near {near / scale:>4.0f} {u}   far {far / scale:>4.0f} {u}   "
            f"center {center / scale:>4.0f} {u}"
        )

    def _on_status(self, line: str) -> None:
        self.stat.setText(line)

    def _on_note(self, line: str) -> None:
        self.stat.setText(line)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.reader.stop()
        # If the worker is wedged in a blocking serial call and doesn't unwind in
        # time, force it down — otherwise the C++ QThread gets destroyed while
        # still running at interpreter teardown, which SIGABRTs.
        if not self.reader.wait(2500):
            self.reader.terminate()
            self.reader.wait(1000)
        super().closeEvent(event)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live 8x8 ToF depth-grid GUI (Qt).")
    ap.add_argument("--port", help="serial port (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=BAUD, help=f"baud (default {BAUD})")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    args = ap.parse_args()

    if args.list:
        list_serial_ports()
        return

    if args.baud <= 0:
        sys.exit(f"Invalid --baud {args.baud}: must be a positive integer.")

    port = args.port or find_port()
    if not port:
        sys.exit("No serial port found. Plug in the ESP32 or pass --port (see --list).")

    app = QApplication(sys.argv)
    win = MatrixWindow(port, args.baud, auto_detected=args.port is None)
    win.resize(720, 620)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
