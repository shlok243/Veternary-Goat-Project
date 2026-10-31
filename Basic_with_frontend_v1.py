"""
Goat AI Training Model - 4-Panel Dashboard
-------------------------------------------
 ┌──────────────────────┬──────────────────────┐
 │ TOP LEFT             │ TOP RIGHT            │
 │ Demo / training video│ Main camera (TPU     │
 │                      │ part, rod goes in)   │
 ├──────────────────────┼──────────────────────┤
 │ BOTTOM LEFT          │ BOTTOM RIGHT         │
 │ Second camera        │ OpenCV output: rod   │
 │                      │ shadow outlined      │
 └──────────────────────┴──────────────────────┘

Install:  pip install PyQt5 opencv-python numpy
Run:      python goat_ai_dashboard_fixed.py        (no arguments needed)

Fixed setup (change the constants under "Fixed settings" below if needed):
  training video = demo.mp4 next to this file, main camera = 2, second camera = 4
  college logo   = ycce_logo.png next to this file (shown top right; skipped if missing)

Keys:
  C          capture reference (rod OUT, LEDs ON)  <- do this first
  Mouse drag on the top-right camera = select the TPU part (ROI)
  R          reset ROI to full frame
  E          cycle the side the rod enters from (left/right/top/bottom)
  Up / Down  shadow sensitivity (Darkness %)
  Space      pause / play the demo video
  S          save snapshot of the OpenCV output
  H          show / hide the bottom hint bar
  F11        fullscreen on/off      Esc  leave fullscreen      Q  quit
"""

import os
import sys
import threading
import time

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

# ----------------------------- Theme (matches the PPT) -----------------------------
BG = "#0a1a2f"          # slide background
PANEL = "#12263f"       # card background
BORDER = "#1f3b5e"
TEAL = "#2dd4bf"        # accent / headings
TEXT = "#e8eef7"
MUTED = "#7f93ad"
AMBER = "#fbbf24"

ENTRY_SIDES = ["left", "right", "top", "bottom"]
PROJECT_TITLE = "GOAT AI TRAINING MODEL  ·  ROD SHADOW TRACKER"

# ----------------------------- Fixed settings (edit here) -----------------------------
# Folder of this script (or of the .exe if you package it later)
BASE = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
    else os.path.dirname(os.path.abspath(__file__))
VIDEO_FILE = os.path.join(BASE, "demo.mp4")   # training video, keep it next to this file
CAM_MAIN = 2              # camera number: TPU part (top right)
CAM_SECOND = 4            # camera number: second camera (bottom left)
HEADING = "GOAT ARTIFICIAL INSEMINATION"      # title shown at the top of the screen
LOGO_FILES = ["ycce_logo.png", "ycce_logo.jpg", "ycce_logo.jpeg", "ycce_logo.webp"]  # first one found is used
LOGO_HEIGHT = 64          # logo height in pixels (top right)
START_FULLSCREEN = False  # set True for the final monitor setup (Esc / F11 leaves it)
LOCK_CAMERA = False       # True = freeze auto exposure/focus (only if brightness pumps)


# ----------------------------- Video sources (background threads) -----------------------------
class Grabber(threading.Thread):
    """Reads a camera or video file in the background and keeps the newest frame."""

    def __init__(self, source, is_file=False, lock_exposure=False):
        super().__init__(daemon=True)
        self.source = source
        self.is_file = is_file
        self.lock_exposure = lock_exposure
        self.frame = None
        self.ok = False
        self.paused = False
        self.running = True
        self._lock = threading.Lock()

    def run(self):
        if not self.is_file and os.name == "nt":
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)  # more reliable on Windows
        else:
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.ok = False
            return
        self.ok = True

        if not self.is_file:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # Webcams on Windows REMEMBER the last settings they were given, so we always set
            # white balance back to AUTO (the old manual setting caused the colour cast).
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            if self.lock_exposure:  # optional: keep brightness stable for shadow detection
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # manual
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            else:                   # normal: put exposure and focus back to automatic
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # auto
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps and fps > 1 else 1.0 / 30

        while self.running:
            t0 = time.time()
            if self.paused:
                time.sleep(0.05)
                continue
            ok, f = cap.read()
            if not ok:
                if self.is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
                else:
                    time.sleep(0.05)
                continue
            with self._lock:
                self.frame = f
            if self.is_file:
                time.sleep(max(0.0, delay - (time.time() - t0)))
        cap.release()

    def get(self):
        with self._lock:
            return self.frame

    def stop(self):
        self.running = False


# ----------------------------- Shadow detection (same idea as the tracker script) -----------------------------
def preprocess(frame, roi, blur_k):
    x, y, w, h = roi
    crop = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    k = max(3, blur_k) | 1
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    return crop, gray.astype(np.float32)


def find_shadow(gray, ref, drop_pct, min_area):
    ratio = gray / ref                       # 1.0 = same as the rod-out reference
    mask = (ratio < 1.0 - drop_pct / 100.0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    return max(cnts, key=cv2.contourArea) if cnts else None


def tip_point(contour, side):
    pts = contour.reshape(-1, 2)
    if side == "left":
        return tuple(int(v) for v in pts[pts[:, 0].argmax()])
    if side == "right":
        return tuple(int(v) for v in pts[pts[:, 0].argmin()])
    if side == "top":
        return tuple(int(v) for v in pts[pts[:, 1].argmax()])
    return tuple(int(v) for v in pts[pts[:, 1].argmin()])


# ----------------------------- UI pieces -----------------------------
class ViewLabel(QtWidgets.QLabel):
    """Shows a frame scaled to fit (keeps aspect ratio). Optionally lets you drag a box."""

    selected = QtCore.pyqtSignal(int, int, int, int)  # x, y, w, h in FRAME coordinates

    def __init__(self, selectable=False):
        super().__init__()
        self.selectable = selectable
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setMinimumSize(160, 90)
        self.setStyleSheet(f"color:{MUTED}; background:#0d2036; border-radius:6px;")
        self._frame_wh = None
        self._origin = None
        self._band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)

    def show_text(self, text):
        self._frame_wh = None
        self.setPixmap(QtGui.QPixmap())
        self.setText(text)

    def show_frame(self, bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]
        img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._frame_wh = (w, h)
        self.setPixmap(pix)

    # --- mouse -> frame coordinates ---
    def _to_frame(self, pos):
        pm = self.pixmap()
        if pm is None or pm.isNull() or self._frame_wh is None:
            return None
        ox = (self.width() - pm.width()) / 2
        oy = (self.height() - pm.height()) / 2
        fx = (pos.x() - ox) / pm.width() * self._frame_wh[0]
        fy = (pos.y() - oy) / pm.height() * self._frame_wh[1]
        return fx, fy

    def mousePressEvent(self, e):
        if self.selectable and e.button() == QtCore.Qt.LeftButton:
            self._origin = e.pos()
            self._band.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
            self._band.show()

    def mouseMoveEvent(self, e):
        if self._origin is not None:
            self._band.setGeometry(QtCore.QRect(self._origin, e.pos()).normalized())

    def mouseReleaseEvent(self, e):
        if self._origin is None:
            return
        a = self._to_frame(self._origin)
        b = self._to_frame(e.pos())
        self._origin = None
        self._band.hide()
        if a is None or b is None:
            return
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        w, h = self._frame_wh
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 > 20 and y1 - y0 > 20:
            self.selected.emit(int(x0), int(y0), int(x1 - x0), int(y1 - y0))


class VideoPanel(QtWidgets.QFrame):
    def __init__(self, title, selectable=False):
        super().__init__()
        self.setObjectName("panel")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 14)
        lay.setSpacing(8)

        head = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel(title.upper())
        f = QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold)
        f.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 2)
        self.title.setFont(f)
        self.title.setStyleSheet(f"color:{TEAL};")
        self.status = QtWidgets.QLabel("")
        self.status.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
        self.status.setStyleSheet(f"color:{MUTED};")
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(self.status)
        lay.addLayout(head)

        self.view = ViewLabel(selectable)
        lay.addWidget(self.view, 1)

    def set_status(self, text, color=MUTED):
        self.status.setText(text)
        self.status.setStyleSheet(f"color:{color};")


# ----------------------------- Main window -----------------------------
class Dashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Goat Artificial Insemination")
        self.resize(1400, 860)

        # video sources
        self.g_video = Grabber(VIDEO_FILE, is_file=True)
        self.g_main = Grabber(CAM_MAIN, lock_exposure=LOCK_CAMERA)
        self.g_second = Grabber(CAM_SECOND)
        for g in (self.g_video, self.g_main, self.g_second):
            g.start()

        # detection state
        self.roi = None            # (x, y, w, h) in main-camera frame coords; None = full frame
        self.ref = None
        self.calib_left = 0
        self.calib_acc = None
        self.calib_n = 30
        self.side_idx = 0
        self.drop = 35             # % darker than reference counts as shadow
        self.min_area = 400
        self.blur = 7
        self.tip_hist = []
        self.last_output = None

        # layout
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(14, 10, 14, 8)
        outer.setSpacing(8)

        # header: centred heading, college logo on the right
        header = QtWidgets.QFrame()
        header.setObjectName("header")
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(6, 0, 6, 8)
        logo_pix = self._load_logo()
        title = QtWidgets.QLabel(HEADING)
        title.setAlignment(QtCore.Qt.AlignCenter)
        tf = QtGui.QFont("Segoe UI", 24, QtGui.QFont.Bold)
        tf.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 3)
        title.setFont(tf)
        title.setStyleSheet(f"color:{TEXT};")
        left_pad = QtWidgets.QWidget()          # balances the logo so the heading stays centred
        hl.addWidget(left_pad)
        hl.addWidget(title, 1)
        if logo_pix is not None:
            logo = QtWidgets.QLabel()
            logo.setPixmap(logo_pix)
            # white badge so the logo looks right on the dark background (works for any logo)
            logo.setStyleSheet("background:#ffffff; border-radius:10px; padding:6px;")
            hl.addWidget(logo)
            left_pad.setFixedWidth(logo_pix.width() + 12)
        else:
            left_pad.setFixedWidth(0)
        outer.addWidget(header)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(12)
        self.p_video = VideoPanel("Training video")
        self.p_main = VideoPanel("Main camera  ·  drag to select part", selectable=True)
        self.p_second = VideoPanel("Second camera")
        self.p_output = VideoPanel("OpenCV output  ·  rod shadow")
        grid.addWidget(self.p_video, 0, 0)    # top left
        grid.addWidget(self.p_main, 0, 1)     # top right
        grid.addWidget(self.p_second, 1, 0)   # bottom left
        grid.addWidget(self.p_output, 1, 1)   # bottom right
        for i in (0, 1):
            grid.setRowStretch(i, 1)
            grid.setColumnStretch(i, 1)
        outer.addLayout(grid, 1)

        self.footer = QtWidgets.QLabel(
            f"{PROJECT_TITLE}      C calibrate · drag = select part · R reset · E entry side · "
            "↑/↓ sensitivity · Space video · S snapshot · H hide · F11 fullscreen · Q quit")
        self.footer.setFont(QtGui.QFont("Segoe UI", 8))
        self.footer.setStyleSheet(f"color:{MUTED};")
        outer.addWidget(self.footer)

        self.setStyleSheet(f"""
            QMainWindow, QWidget#root {{ background:{BG}; }}
            QFrame#panel {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; }}
            QFrame#panel QLabel {{ border:none; background:transparent; }}
            QFrame#header {{ background:transparent; border:none; border-bottom:2px solid {BORDER}; }}
        """)
        # the video labels keep their own darker background
        for p in (self.p_video, self.p_main, self.p_second, self.p_output):
            p.view.setStyleSheet(f"color:{MUTED}; background:#0d2036; border-radius:6px;")

        self.p_main.view.selected.connect(self.set_roi)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    # ---------- logo ----------
    def _load_logo(self):
        """Returns the scaled logo pixmap, or None if no logo file is next to this script."""
        for name in LOGO_FILES:
            path = os.path.join(BASE, name)
            if os.path.exists(path):
                pix = QtGui.QPixmap(path)
                if not pix.isNull():
                    return pix.scaledToHeight(LOGO_HEIGHT, QtCore.Qt.SmoothTransformation)
        return None

    # ---------- actions ----------
    def set_roi(self, x, y, w, h):
        self.roi = (x, y, w, h)
        self.ref = None
        self.tip_hist.clear()

    def start_calibration(self):
        self.calib_left = self.calib_n
        self.calib_acc = None
        self.tip_hist.clear()

    # ---------- main loop ----------
    def show_source(self, panel, grabber, label):
        f = grabber.get()
        if f is None:
            panel.view.show_text("NO SIGNAL" if not grabber.ok else "LOADING…")
            panel.set_status("NO SIGNAL" if not grabber.ok else "", MUTED)
            return None
        panel.view.show_frame(f)
        return f

    def tick(self):
        self.show_source(self.p_video, self.g_video, "video")
        self.show_source(self.p_second, self.g_second, "cam2")

        frame = self.g_main.get()
        if frame is None:
            self.p_main.view.show_text("NO SIGNAL" if not self.g_main.ok else "LOADING…")
            self.p_output.view.show_text("WAITING FOR CAMERA")
            return

        H, W = frame.shape[:2]
        x, y, w, h = self.roi if self.roi else (0, 0, W, H)
        x, y = min(x, W - 2), min(y, H - 2)
        w, h = min(w, W - x), min(h, H - y)
        roi = (x, y, w, h)

        # main camera panel: draw ROI box so you can see what's being analysed
        shown = frame.copy()
        if self.roi:
            cv2.rectangle(shown, (x, y), (x + w, y + h), (191, 212, 45), 2)  # teal (BGR)
        self.p_main.view.show_frame(shown)
        self.p_main.set_status("", MUTED)

        crop, gray = preprocess(frame, roi, self.blur)
        view = crop.copy()

        # --- calibration in progress ---
        if self.calib_left > 0:
            self.calib_acc = gray.copy() if self.calib_acc is None else self.calib_acc + gray
            self.calib_left -= 1
            if self.calib_left == 0:
                self.ref = np.maximum(self.calib_acc / self.calib_n, 10.0)
            self.p_output.set_status(f"CALIBRATING… {self.calib_n - self.calib_left}/{self.calib_n}", AMBER)
        elif self.ref is None or self.ref.shape != gray.shape:
            self.ref = None
            self.p_output.set_status("PRESS C TO CALIBRATE (ROD OUT)", AMBER)
        else:
            cnt = find_shadow(gray, self.ref, self.drop, self.min_area)
            if cnt is not None:
                overlay = view.copy()
                cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), -1)
                view = cv2.addWeighted(overlay, 0.25, view, 0.75, 0)
                cv2.drawContours(view, [cnt], -1, (191, 212, 45), 2)  # teal outline

                self.tip_hist.append(tip_point(cnt, ENTRY_SIDES[self.side_idx]))
                self.tip_hist = self.tip_hist[-5:]
                sx = int(np.mean([p[0] for p in self.tip_hist]))
                sy = int(np.mean([p[1] for p in self.tip_hist]))
                cv2.circle(view, (sx, sy), 10, (36, 191, 251), 2)
                cv2.drawMarker(view, (sx, sy), (36, 191, 251), cv2.MARKER_CROSS, 24, 2)
                cv2.putText(view, "ROD TIP", (sx + 14, sy - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 191, 251), 2)
                self.p_output.set_status(
                    f"ROD DETECTED  ·  enters {ENTRY_SIDES[self.side_idx]}  ·  sens {self.drop}%", TEAL)
            else:
                self.tip_hist.clear()
                self.p_output.set_status(
                    f"NO ROD  ·  enters {ENTRY_SIDES[self.side_idx]}  ·  sens {self.drop}%", MUTED)

        self.last_output = view
        self.p_output.view.show_frame(view)

    # ---------- keys ----------
    def keyPressEvent(self, e):
        k = e.key()
        if k == QtCore.Qt.Key_Q:
            self.close()
        elif k == QtCore.Qt.Key_C:
            self.start_calibration()
        elif k == QtCore.Qt.Key_R:
            self.roi = None
            self.ref = None
            self.tip_hist.clear()
        elif k == QtCore.Qt.Key_E:
            self.side_idx = (self.side_idx + 1) % len(ENTRY_SIDES)
            self.tip_hist.clear()
        elif k == QtCore.Qt.Key_Up:
            self.drop = min(90, self.drop + 5)
        elif k == QtCore.Qt.Key_Down:
            self.drop = max(5, self.drop - 5)
        elif k == QtCore.Qt.Key_Space:
            self.g_video.paused = not self.g_video.paused
        elif k == QtCore.Qt.Key_S and self.last_output is not None:
            name = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(name, self.last_output)
        elif k == QtCore.Qt.Key_H:
            self.footer.setVisible(not self.footer.isVisible())
        elif k == QtCore.Qt.Key_F11:
            self.setWindowState(self.windowState() ^ QtCore.Qt.WindowFullScreen)
        elif k == QtCore.Qt.Key_Escape and self.isFullScreen():
            self.showNormal()

    def closeEvent(self, e):
        for g in (self.g_video, self.g_main, self.g_second):
            g.stop()
        e.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = Dashboard()
    if START_FULLSCREEN:
        win.showFullScreen()
    else:
        win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()