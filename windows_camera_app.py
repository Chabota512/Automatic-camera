"""
Presenter camera desktop application for Windows.

This is a simple native-style Tkinter/ttk interface around OpenCV:
    - Android IP Webcam stream
    - Laptop/USB webcam
    - Local video file
    - Click-to-select object tracking
    - Keyboard and button camera control
    - Virtual camera crop output

Install:
    python -m pip install opencv-contrib-python numpy pillow

Run:
    python windows_camera_app.py
"""

from __future__ import annotations

import queue
import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError:  # Allows non-GUI tracking tests on headless runners.
    tk = None
    filedialog = None
    messagebox = None
    ttk = None

import cv2
import numpy as np
from PIL import Image

try:
    from PIL import ImageTk
except ModuleNotFoundError:
    ImageTk = None

import target_tracking
import vision_providers


class NetworkExportServer:
    """Serve the latest virtual-camera frame and motor coordinates over HTTP."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self._lock = threading.Lock()
        self._coordinate_updates = threading.Condition(self._lock)
        self._coordinate_version = 0
        self._jpeg_bytes: bytes | None = None
        self._coordinates: dict[str, object] = {}

        export = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in {"/", "/index.html"}:
                    self._send_page()
                elif self.path == "/video.mjpg":
                    self._send_video()
                elif self.path == "/coordinates":
                    self._send_coordinates()
                elif self.path == "/coordinates/stream":
                    self._send_coordinate_stream()
                else:
                    self.send_error(404)

            def _send_page(self) -> None:
                page = b"""<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Presenter Camera</title>
<style>html,body{margin:0;background:#000;width:100%;height:100%;overflow:hidden}body{display:flex;align-items:center;justify-content:center}main{width:100vw;height:100vh;display:flex;align-items:center;justify-content:center;position:relative}img{display:block;width:100%;height:100%;object-fit:contain}button{position:fixed;right:16px;bottom:16px;padding:10px 14px;font:14px sans-serif;border:0;border-radius:4px;cursor:pointer}</style>
</head><body><main id="feed"><img src="/video.mjpg" alt="Virtual camera feed"></main>
<button onclick="document.getElementById('feed').requestFullscreen?.()">Fullscreen</button>
</body></html>"""
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

            def _send_video(self) -> None:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.end_headers()
                while True:
                    with export._lock:
                        jpeg = export._jpeg_bytes
                    if jpeg is not None:
                        try:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                + jpeg
                                + b"\r\n"
                            )
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            return
                    time.sleep(0.1)

            def _send_coordinates(self) -> None:
                with export._lock:
                    payload = dict(export._coordinates)
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_coordinate_stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                version = -1
                while True:
                    with export._coordinate_updates:
                        export._coordinate_updates.wait_for(
                            lambda: export._coordinate_version != version,
                            timeout=2.0,
                        )
                        version = export._coordinate_version
                        payload = dict(export._coordinates)
                    try:
                        self.wfile.write(json.dumps(payload).encode("utf-8") + b"\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return

            def log_message(self, *_args) -> None:
                return

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            daemon=True,
            name="presenter-camera-network-export",
        )

    def start(self) -> None:
        self.thread.start()

    def update(self, jpeg_bytes: bytes, coordinates: dict[str, object]) -> None:
        with self._coordinate_updates:
            self._jpeg_bytes = jpeg_bytes
            self._coordinates = dict(coordinates)
            self._coordinate_version += 1
            self._coordinate_updates.notify_all()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


# ========================= USER SETTINGS =========================
# Smaller values are smoother/slower; larger values are faster.
SMOOTHING_SPEED = 0.18

# A single click creates a temporary tracking box around the cursor.
CLICK_TRACK_BOX_WIDTH_RATIO = 0.22
CLICK_TRACK_BOX_HEIGHT_RATIO = 0.42

# Display size used by the two video panels.
# Render previews at a higher logical resolution, then scale them to the
# available canvas. This keeps maximized-window previews crisp without making
# the canvas/input coordinates depend on the current window size.
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
AI_CHECK_INTERVAL_SECONDS = 5.0
AI_RECOVERY_CHECK_INTERVAL_SECONDS = 2.0
AI_RECOVERY_WINDOW_SECONDS = 10.0
AI_PROVIDER_OFFSET_SECONDS = AI_CHECK_INTERVAL_SECONDS / 2
AI_FRAME_MAX_DIMENSION = 640
AI_BOX_MAX_AGE_SECONDS = 30.0
AI_AUTO_ENABLE_ON_TRACK_LOSS = True
VIDEO_LOG_INTERVAL_SECONDS = 1.0
AI_TRANSIENT_ERROR_COOLDOWN_SECONDS = 10.0
AI_RATE_LIMIT_COOLDOWN_SECONDS = 60.0
AI_MAX_ERROR_COOLDOWN_SECONDS = 300.0
TARGET_MATCH_REJECT_THRESHOLD = 0.35
TARGET_MATCH_LOST_FRAME_LIMIT = 3
TARGET_MAX_CENTER_JUMP_RATIO = 0.85
HUMAN_DETECTION_INTERVAL_SECONDS = 0.20
TARGET_REACQUIRE_INTERVAL_SECONDS = 0.20
TARGET_REACQUIRE_TEMPLATE_THRESHOLD = 0.60
TARGET_REACQUIRE_APPEARANCE_THRESHOLD = 0.40
PORTRAIT_VERTICAL_FRAMING_BIAS = 0.12
STREAM_OPEN_TIMEOUT_MS = 5000
STREAM_READ_TIMEOUT_MS = 5000
STARTUP_CAMERA_INDICES = range(5)
CONTROL_MASK = 0x0004
SHIFT_MASK = 0x0001

# OpenCV uses BGR colors. These are deliberately different from the green
# virtual-camera crop and yellow local tracker box.
AI_BOX_COLORS = {
    "Gemini": (255, 220, 160),  # light blue
    "Groq": (255, 0, 255),  # bright magenta
}


@dataclass
class ProviderRuntime:
    """Live state for one independent AI worker."""

    model: str
    next_check: float = 0.0
    in_flight: bool = False
    last_result: vision_providers.VisionResult | None = None
    last_box: tuple[int, int, int, int] | None = None
    last_error: str = ""
    last_started: float = 0.0
    last_completed: float = 0.0
    request_count: int = 0
    success_count: int = 0
    consecutive_failures: int = 0
    last_match_score: float | None = None
    last_human_match_score: float | None = None
    last_result_frame: int | None = None
    last_result_captured_at: float | None = None


@dataclass(frozen=True)
class AIFrameContext:
    """Metadata tying a provider result to the exact still image it analyzed."""

    frame_number: int
    captured_at: float
    frame_width: int
    frame_height: int


class PresenterCameraApp:
    """Standard Windows-style desktop interface for the camera prototype."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Automatic Camera")
        self.root.geometry("1440x900")
        self.root.minsize(1180, 760)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._configure_style()
        icon_path = Path(__file__).with_name("assets") / "automatic-camera.ico"
        if icon_path.exists():
            try:
                self.root.iconbitmap(default=str(icon_path))
            except tk.TclError:
                # Linux/macOS development environments may not support .ico.
                pass

        self.capture: cv2.VideoCapture | None = None
        self.source_kind = ""
        self.running = False
        self.after_id: str | None = None
        self.frame: np.ndarray | None = None

        self.ai_results = queue.Queue()
        self.startup_check_results = queue.Queue()
        self.camera_selection_results = queue.Queue()
        self.startup_check_vars: dict[str, tk.StringVar] = {}
        self.startup_checks_pending = {"cameras", "cv", "Gemini", "Groq"}
        self.startup_overlay: tk.Toplevel | None = None
        self.startup_overlay_status: tk.StringVar | None = None
        self.startup_spinner: ttk.Progressbar | None = None
        self.startup_input_bindings: list[str] = []
        self.startup_checks_cancelled = False
        self.pending_camera_choices: tuple[tuple[int, ...], str] | None = None
        self.camera_dialog: tk.Toplevel | None = None
        self.camera_probe_cancel = threading.Event()
        self.ai_threads: dict[str, threading.Thread] = {}
        self.ai_generation = 0
        self.ai_runtime = {
            provider: ProviderRuntime(
                model=vision_providers.provider_model(provider)
            )
            for provider in ("Gemini", "Groq")
        }
        self.ai_winner: str | None = None
        self.ai_terminal_queue = queue.Queue()
        self.ai_last_video_log = 0.0
        self.video_frame_count = 0
        self.network_export: NetworkExportServer | None = None
        self.ai_recovery_deadline = 0.0

        self.tracker = None
        self.tracking_box: tuple[int, int, int, int] | None = None
        self.tracking_lost = False
        self.target_reference: target_tracking.TargetReference | None = None
        self.target_match_score = 0.0
        self.target_mismatch_frames = 0
        self.selection_drag_mode: str | None = None
        self.selection_drag_start: tuple[int, int] | None = None
        self.selection_drag_current: tuple[int, int] | None = None
        self.selection_preview_box: tuple[int, int, int, int] | None = None
        self.track_human = tk.BooleanVar(value=False)
        self.face_classifier: cv2.CascadeClassifier | None = None
        self.eye_classifier: cv2.CascadeClassifier | None = None
        self.smile_classifier: cv2.CascadeClassifier | None = None
        self.face_boxes: tuple[target_tracking.Box, ...] = ()
        self.eye_boxes: tuple[target_tracking.Box, ...] = ()
        self.smile_boxes: tuple[target_tracking.Box, ...] = ()
        self.human_match_score: float | None = None
        self.human_detection_error = ""
        self.next_human_detection = 0.0
        self.next_target_reacquire = 0.0
        self.auto_follow = tk.BooleanVar(value=True)
        self.mode = tk.StringVar(value="Assisted tracking")
        self.ai_provider = tk.StringVar(value="Groq")
        self.ai_target_priority = tk.StringVar(
            value="face, person, presenter"
        )
        self.ai_enabled = tk.BooleanVar(value=False)
        self.ai_indicator = tk.StringVar(value="AI: OFF")
        self.ai_toggle_button: ttk.Button | None = None
        self.ai_led: tk.Canvas | None = None
        self.ai_led_dot: int | None = None
        self.provider_status_vars: dict[str, tk.StringVar] = {}
        self.provider_status_leds: dict[str, tuple[tk.Canvas, int]] = {}

        self.camera_x = 0.0
        self.camera_y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.zoom = 1.0

        self.source_image: ImageTk.PhotoImage | None = None
        self.virtual_image: ImageTk.PhotoImage | None = None
        self.canvas_image_rectangles: dict[str, tuple[int, int, int, int]] = {}

        self.source_type = tk.StringVar(value="Android stream")
        self.stream_url = tk.StringVar(
            value="http://10.228.105.251:8080/video"
        )
        self.camera_index = tk.StringVar(value="0")
        self.video_path = tk.StringVar()
        self.status = tk.StringVar(value="Ready. Select a source and click Start.")
        self.target_status = tk.StringVar(value="Target reference: none")
        self.human_status = tk.StringVar(value="Human tracking: OFF")
        self.network_port = tk.StringVar(value="8765")
        self.network_status = tk.StringVar(value="Network export: OFF")
        self.smoothing = tk.DoubleVar(value=SMOOTHING_SPEED)
        self.follow_checkbutton: ttk.Checkbutton | None = None
        self.human_checkbutton: ttk.Checkbutton | None = None
        self.header_status = tk.StringVar(value="READY  •  Select a source to begin")
        self.status.trace_add("write", self._sync_header_status)

        self.build_menu()
        self.build_interface()
        self.bind_controls()
        self._show_startup_overlay()
        self.root.after(100, self._drain_terminal_queue)
        self.root.after(100, self._drain_startup_check_results)
        self.root.after(100, self._drain_camera_selection_results)
        self.root.after(300, self._start_startup_checks)

    def build_menu(self) -> None:
        menu_bar = tk.Menu(self.root)

        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(label="Start Source", command=self.start_source)
        file_menu.add_command(label="Stop Source", command=self.stop_source)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.close)
        menu_bar.add_cascade(label="File", menu=file_menu)

        controls_menu = tk.Menu(menu_bar, tearoff=False)
        controls_menu.add_command(label="Clear Tracking", command=self.clear_tracking)
        controls_menu.add_command(label="Reset Camera", command=self.reset_camera)
        controls_menu.add_command(label="Zoom In", command=lambda: self.change_zoom(0.1))
        controls_menu.add_command(label="Zoom Out", command=lambda: self.change_zoom(-0.1))
        menu_bar.add_cascade(label="Controls", menu=controls_menu)

        help_menu = tk.Menu(menu_bar, tearoff=False)
        help_menu.add_command(
            label="About",
            command=lambda: messagebox.showinfo(
                "About Presenter Camera",
                "Prototype automatic presenter camera controller.",
            ),
        )
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menu_bar)

    def _configure_style(self) -> None:
        """Set a restrained Windows-console palette without replacing ttk controls."""
        style = ttk.Style(self.root)
        try:
            if "vista" in style.theme_names():
                style.theme_use("vista")
        except tk.TclError:
            pass

        self.root.configure(background="#eef2f7")
        style.configure("App.TFrame", background="#eef2f7")
        style.configure("Header.TFrame", background="#172333")
        style.configure(
            "HeaderTitle.TLabel",
            background="#172333",
            foreground="#f8fafc",
            font=("Segoe UI", 13, "bold"),
        )
        style.configure(
            "HeaderSub.TLabel",
            background="#172333",
            foreground="#a9b8c9",
            font=("Segoe UI", 9),
        )
        style.configure(
            "HeaderStatus.TLabel",
            background="#172333",
            foreground="#8ee6c2",
            font=("Consolas", 9, "bold"),
        )
        style.configure(
            "SectionTitle.TLabel",
            background="#eef2f7",
            foreground="#334155",
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "SectionHint.TLabel",
            background="#eef2f7",
            foreground="#64748b",
            font=("Segoe UI", 8),
        )

    def _sync_header_status(self, *_args) -> None:
        """Keep the compact header useful while the detailed status stays below."""
        if not hasattr(self, "header_status"):
            return
        state = "LIVE" if self.running else "READY"
        detail = self.status.get().strip()
        if len(detail) > 72:
            detail = f"{detail[:69]}..."
        self.header_status.set(f"{state}  •  {detail}")

    def build_interface(self) -> None:
        # Keep the native Tk surface, but use a focused operator-console
        # palette instead of the platform default theme.
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        colors = {
            "bg": "#10171b",
            "panel": "#192026",
            "panel_alt": "#151d22",
            "text": "#e5ebe7",
            "muted": "#84958e",
            "lime": "#c8f05a",
            "coral": "#f27d59",
            "line": "#34413d",
        }
        self.root.configure(background=colors["bg"])
        style.configure(".", background=colors["bg"], foreground=colors["text"])
        style.configure("TFrame", background=colors["bg"])
        style.configure("Panel.TFrame", background=colors["panel"])
        style.configure(
            "TLabel",
            background=colors["panel"],
            foreground=colors["text"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "Muted.TLabel",
            background=colors["bg"],
            foreground=colors["muted"],
            font=("Segoe UI", 8),
        )
        style.configure(
            "Section.TLabelframe",
            background=colors["panel"],
            foreground=colors["muted"],
            bordercolor=colors["line"],
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Section.TLabelframe.Label",
            background=colors["panel"],
            foreground=colors["muted"],
            font=("Segoe UI", 8, "bold"),
        )
        style.configure(
            "TButton",
            background="#26322f",
            foreground=colors["text"],
            bordercolor=colors["line"],
            padding=(8, 5),
        )
        style.map(
            "TButton",
            background=[("active", "#33443d")],
            foreground=[("disabled", "#61716a")],
        )
        style.configure(
            "Accent.TButton",
            background=colors["lime"],
            foreground="#182019",
            font=("Segoe UI", 9, "bold"),
            padding=(9, 6),
        )
        style.map("Accent.TButton", background=[("active", "#d7fa79")])
        style.configure(
            "TEntry",
            fieldbackground="#202b30",
            foreground=colors["text"],
            insertcolor=colors["text"],
        )
        style.configure(
            "TCombobox",
            fieldbackground="#202b30",
            background="#26322f",
            foreground=colors["text"],
            arrowcolor=colors["lime"],
        )
        style.configure(
            "TCheckbutton",
            background=colors["panel"],
            foreground=colors["text"],
        )
        style.configure(
            "Horizontal.TScale",
            background=colors["panel"],
            troughcolor="#334039",
        )

        header = tk.Frame(self.root, bg=colors["panel_alt"], height=56)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header,
            text="AUTOMATIC CAMERA",
            bg=colors["panel_alt"],
            fg=colors["lime"],
            font=("Segoe UI", 12, "bold"),
        ).pack(side=tk.LEFT, padx=(18, 8))
        tk.Label(
            header,
            text="CONTROL ROOM  /  local camera operator console",
            bg=colors["panel_alt"],
            fg=colors["muted"],
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)
        tk.Label(
            header,
            textvariable=self.header_status,
            bg=colors["panel_alt"],
            fg=colors["lime"],
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.RIGHT, padx=18)

        main = ttk.Frame(self.root, padding=(14, 12, 14, 0))
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0, minsize=315)
        main.rowconfigure(0, weight=1)

        controls_panel = ttk.Frame(main)
        controls_panel.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        controls_panel.columnconfigure(0, weight=1)
        controls_panel.rowconfigure(0, weight=1)

        controls_canvas = tk.Canvas(
            controls_panel,
            width=315,
            background=colors["bg"],
            highlightthickness=0,
            borderwidth=0,
        )
        controls_canvas.grid(row=0, column=0, sticky="nsew")
        controls_scrollbar = ttk.Scrollbar(
            controls_panel,
            orient=tk.VERTICAL,
            command=controls_canvas.yview,
        )
        controls_scrollbar.grid(row=0, column=1, sticky="ns")
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        controls = ttk.Frame(
            controls_canvas,
            padding=(0, 0, 10, 0),
            style="Panel.TFrame",
        )
        controls_window = controls_canvas.create_window(
            (0, 0),
            window=controls,
            anchor="nw",
        )

        def update_controls_scroll_region(_event=None) -> None:
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

        def resize_controls_content(event) -> None:
            controls_canvas.itemconfigure(
                controls_window,
                width=event.width,
            )

        controls.bind("<Configure>", update_controls_scroll_region)
        controls_canvas.bind("<Configure>", resize_controls_content)

        def scroll_controls_with_pointer(event):
            """Scroll the controls panel for mouse wheels and trackpads."""
            pointer_x = self.root.winfo_pointerx()
            pointer_y = self.root.winfo_pointery()
            panel_left = controls_panel.winfo_rootx()
            panel_top = controls_panel.winfo_rooty()
            panel_right = panel_left + controls_panel.winfo_width()
            panel_bottom = panel_top + controls_panel.winfo_height()
            if not (
                panel_left <= pointer_x <= panel_right
                and panel_top <= pointer_y <= panel_bottom
            ):
                return None

            event_number = getattr(event, "num", None)
            if event_number == 4:
                scroll_units = -3
            elif event_number == 5:
                scroll_units = 3
            else:
                delta = getattr(event, "delta", 0)
                if not delta:
                    return "break"
                scroll_units = -max(1, abs(delta) // 120) if delta > 0 else max(
                    1, abs(delta) // 120
                )

            controls_canvas.yview_scroll(scroll_units, "units")
            return "break"

        # bind_all is intentional here: the pointer is usually over a child
        # widget inside the canvas, not over the canvas itself.
        self.root.bind_all(
            "<MouseWheel>",
            scroll_controls_with_pointer,
            add="+",
        )
        self.root.bind_all(
            "<Button-4>",
            scroll_controls_with_pointer,
            add="+",
        )
        self.root.bind_all(
            "<Button-5>",
            scroll_controls_with_pointer,
            add="+",
        )

        source_frame = ttk.LabelFrame(
            controls,
            text="SOURCE SELECTION  /  INPUT",
            padding=8,
            style="Section.TLabelframe",
        )
        source_frame.pack(fill=tk.X)
        source_frame.columnconfigure(1, weight=1)

        ttk.Label(source_frame, text="Type:").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=4
        )
        source_combo = ttk.Combobox(
            source_frame,
            textvariable=self.source_type,
            values=("Android stream", "Webcam", "Video file"),
            state="readonly",
            width=18,
        )
        source_combo.grid(row=0, column=1, sticky="ew", pady=4)
        source_combo.bind("<<ComboboxSelected>>", self.source_type_changed)

        self.stream_label = ttk.Label(source_frame, text="Stream URL:")
        self.stream_label.grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
        self.stream_entry = ttk.Entry(source_frame, textvariable=self.stream_url, width=34)
        self.stream_entry.grid(row=1, column=1, sticky="ew", pady=4)

        self.camera_label = ttk.Label(source_frame, text="Camera index:")
        self.camera_label.grid(row=2, column=0, sticky="w", padx=(0, 6), pady=4)
        self.camera_entry = ttk.Entry(source_frame, textvariable=self.camera_index, width=10)
        self.camera_entry.grid(row=2, column=1, sticky="w", pady=4)

        self.video_label = ttk.Label(source_frame, text="Video file:")
        self.video_label.grid(row=3, column=0, sticky="w", padx=(0, 6), pady=4)
        video_row = ttk.Frame(source_frame)
        video_row.grid(row=3, column=1, sticky="ew", pady=4)
        video_row.columnconfigure(0, weight=1)
        self.video_entry = ttk.Entry(video_row, textvariable=self.video_path)
        self.video_entry.grid(row=0, column=0, sticky="ew")
        self.browse_button = ttk.Button(
            video_row, text="Browse...", command=self.browse_video
        )
        self.browse_button.grid(row=0, column=1, padx=(5, 0))

        source_buttons = ttk.Frame(source_frame)
        source_buttons.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.start_button = ttk.Button(
            source_buttons,
            text="Start",
            command=self.start_source,
            style="Accent.TButton",
        )
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.stop_button = ttk.Button(
            source_buttons, text="Stop", command=self.stop_source
        )
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        network_frame = ttk.LabelFrame(
            controls,
            text="Network export",
            padding=8,
        )
        network_frame.pack(fill=tk.X, pady=(10, 0))
        network_frame.columnconfigure(1, weight=1)
        ttk.Label(network_frame, text="Port:").grid(row=0, column=0, sticky="w")
        ttk.Entry(
            network_frame,
            textvariable=self.network_port,
            width=8,
        ).grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.network_button = ttk.Button(
            network_frame,
            text="Start network export",
            command=self.toggle_network_export,
        )
        self.network_button.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(6, 4),
        )
        ttk.Label(
            network_frame,
            textvariable=self.network_status,
            justify=tk.LEFT,
            wraplength=230,
        ).grid(row=2, column=0, columnspan=2, sticky="w")

        tracking_frame = ttk.LabelFrame(
            controls,
            text="CONTROL MODE  /  TRACKING",
            padding=8,
            style="Section.TLabelframe",
        )
        tracking_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(tracking_frame, text="Operating mode:").pack(anchor="w")
        self.mode_combo = ttk.Combobox(
            tracking_frame,
            textvariable=self.mode,
            values=(
                "Manual control",
                "Assisted tracking",
                "Face tracking",
                "Automatic AI",
            ),
            state="readonly",
            width=25,
        )
        self.mode_combo.pack(fill=tk.X, pady=(3, 7))
        self.mode_combo.bind("<<ComboboxSelected>>", self.mode_changed)

        ttk.Label(tracking_frame, text="AI provider:").pack(anchor="w")
        self.ai_provider_combo = ttk.Combobox(
            tracking_frame,
            textvariable=self.ai_provider,
            values=("Gemini", "Groq", "Compare both"),
            state="readonly",
            width=25,
        )
        self.ai_provider_combo.pack(fill=tk.X, pady=(3, 7))
        self.ai_provider_combo.bind(
            "<<ComboboxSelected>>",
            self.ai_provider_changed,
        )

        ai_status_row = ttk.Frame(tracking_frame)
        ai_status_row.pack(fill=tk.X, pady=(0, 4))
        self.ai_led = tk.Canvas(
            ai_status_row,
            width=18,
            height=18,
            highlightthickness=0,
        )
        self.ai_led.pack(side=tk.LEFT, padx=(0, 5))
        self.ai_led_dot = self.ai_led.create_oval(
            3,
            3,
            15,
            15,
            fill="#c73838",
            outline="#c73838",
        )
        ttk.Label(
            ai_status_row,
            textvariable=self.ai_indicator,
        ).pack(side=tk.LEFT)

        provider_frame = ttk.LabelFrame(
            tracking_frame,
            text="Provider status",
            padding=(5, 4),
        )
        provider_frame.pack(fill=tk.X, pady=(2, 6))
        for provider in ("Gemini", "Groq"):
            row = ttk.Frame(provider_frame)
            row.pack(fill=tk.X, pady=1)
            led = tk.Canvas(row, width=14, height=14, highlightthickness=0)
            led.pack(side=tk.LEFT, padx=(0, 5))
            led_dot = led.create_oval(
                2,
                2,
                12,
                12,
                fill="#6b7280",
                outline="#6b7280",
            )
            status_var = tk.StringVar()
            self.provider_status_vars[provider] = status_var
            self.provider_status_leds[provider] = (led, led_dot)
            ttk.Label(
                row,
                textvariable=status_var,
                justify=tk.LEFT,
                wraplength=230,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.ai_toggle_button = ttk.Button(
            tracking_frame,
            text="Enable AI",
            command=self.toggle_ai,
        )
        self.ai_toggle_button.pack(fill=tk.X, pady=(0, 7))
        ttk.Button(
            tracking_frame,
            text="Priorities",
            command=self.show_priority_dialog,
        ).pack(fill=tk.X, pady=(0, 7))
        ttk.Button(
            tracking_frame,
            text="Rescan AI",
            command=self.rescan_ai,
        ).pack(fill=tk.X, pady=(0, 7))

        self.follow_checkbutton = ttk.Checkbutton(
            tracking_frame,
            text="Automatic follow",
            variable=self.auto_follow,
        )
        self.follow_checkbutton.pack(anchor="w")
        self.human_checkbutton = ttk.Checkbutton(
            tracking_frame,
            text="Track human (local face detection)",
            variable=self.track_human,
            command=self.human_tracking_changed,
        )
        self.human_checkbutton.pack(anchor="w", pady=(5, 0))
        ttk.Label(
            tracking_frame,
            textvariable=self.human_status,
            justify=tk.LEFT,
            wraplength=230,
        ).pack(anchor="w", pady=(3, 0))
        ttk.Button(
            tracking_frame,
            text="Clear tracking",
            command=self.clear_tracking,
        ).pack(fill=tk.X, pady=(7, 0))
        ttk.Button(
            tracking_frame,
            text="Refresh target reference",
            command=self.refresh_target_reference,
        ).pack(fill=tk.X, pady=(5, 0))
        ttk.Label(
            tracking_frame,
            textvariable=self.target_status,
            justify=tk.LEFT,
            wraplength=230,
        ).pack(anchor="w", pady=(7, 0))
        ttk.Label(
            tracking_frame,
            text="Click to select. Ctrl-drag draws a new tracking box.\n"
            "Ctrl+Shift-drag resizes the selected box and keeps its reference.\n"
            "Track human adds local face detection; a hidden face does not lose the target.\n"
            "Gemini is light blue. Groq is light brown.\n"
            "The [CAMERA] box is the model currently driving the crop.",
        ).pack(anchor="w", pady=(7, 0))

        diagnostics_frame = ttk.LabelFrame(
            controls,
            text="LAUNCH DIAGNOSTICS  /  HEALTH",
            padding=6,
            style="Section.TLabelframe",
        )
        diagnostics_frame.pack(fill=tk.X, pady=(10, 0))
        for key, label in (
            ("cameras", "Camera feeds"),
            ("cv", "Computer vision"),
            ("Gemini", "Gemini text / image"),
            ("Groq", "Groq text / image"),
        ):
            row = ttk.Frame(diagnostics_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{label}:", width=20).pack(
                side=tk.LEFT,
                anchor="nw",
            )
            status_var = tk.StringVar(value="Checking...")
            self.startup_check_vars[key] = status_var
            ttk.Label(
                row,
                textvariable=status_var,
                justify=tk.LEFT,
                wraplength=170,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        motion_frame = ttk.LabelFrame(
            controls,
            text="CAMERA CONTROL  /  OPERATOR",
            padding=8,
            style="Section.TLabelframe",
        )
        motion_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(motion_frame, text="Smoothing speed:").pack(anchor="w")
        ttk.Scale(
            motion_frame,
            from_=0.05,
            to=0.40,
            variable=self.smoothing,
            orient=tk.HORIZONTAL,
        ).pack(fill=tk.X, pady=(3, 0))
        ttk.Label(
            motion_frame,
            text="Lower = smoother, higher = faster",
        ).pack(anchor="w")

        button_pad = ttk.Frame(motion_frame)
        button_pad.pack(pady=(8, 0))
        ttk.Button(
            button_pad, text="▲", width=5, command=lambda: self.nudge(0, -1)
        ).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(
            button_pad, text="◀", width=5, command=lambda: self.nudge(-1, 0)
        ).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(
            button_pad, text="Reset", width=5, command=self.reset_camera
        ).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(
            button_pad, text="▶", width=5, command=lambda: self.nudge(1, 0)
        ).grid(row=1, column=2, padx=2, pady=2)
        ttk.Button(
            button_pad, text="▼", width=5, command=lambda: self.nudge(0, 1)
        ).grid(row=2, column=1, padx=2, pady=2)

        zoom_row = ttk.Frame(motion_frame)
        zoom_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(
            zoom_row, text="Zoom -", command=lambda: self.change_zoom(-0.1)
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            zoom_row, text="Zoom +", command=lambda: self.change_zoom(0.1)
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        displays = ttk.Frame(main, style="Panel.TFrame")
        displays.grid(row=0, column=0, sticky="nsew")
        displays.columnconfigure(0, weight=1)
        displays.columnconfigure(1, weight=1)
        displays.rowconfigure(1, weight=1)
        main.rowconfigure(2, weight=0)

        source_heading = tk.Frame(
            displays,
            bg=colors["panel"],
            height=38,
        )
        source_heading.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 6),
            pady=(0, 4),
        )
        source_heading.pack_propagate(False)
        tk.Label(
            source_heading,
            text="SOURCE CAMERA",
            bg=colors["panel"],
            fg=colors["lime"],
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT, padx=12)
        tk.Label(
            source_heading,
            text="INPUT / A     1080p  ·  LIVE",
            bg=colors["panel"],
            fg=colors["muted"],
            font=("Consolas", 8),
        ).pack(side=tk.RIGHT, padx=12)

        output_heading = tk.Frame(
            displays,
            bg=colors["panel"],
            height=38,
        )
        output_heading.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(6, 0),
            pady=(0, 4),
        )
        output_heading.pack_propagate(False)
        tk.Label(
            output_heading,
            text="VIRTUAL CAMERA OUTPUT",
            bg=colors["panel"],
            fg=colors["coral"],
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT, padx=12)
        tk.Label(
            output_heading,
            text="OUTPUT / VIRTUAL     60 FPS",
            bg=colors["panel"],
            fg=colors["muted"],
            font=("Consolas", 8),
        ).pack(side=tk.RIGHT, padx=12)

        self.source_canvas = tk.Canvas(
            displays,
            width=DISPLAY_WIDTH // 2,
            height=DISPLAY_HEIGHT // 2,
            background="#202b30",
            highlightthickness=1,
            highlightbackground="#52645d",
        )
        self.source_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.source_canvas.bind(
            "<ButtonPress-1>",
            self._on_source_button_press,
        )
        self.source_canvas.bind(
            "<B1-Motion>",
            self._on_source_mouse_drag,
        )
        self.source_canvas.bind(
            "<ButtonRelease-1>",
            self._on_source_button_release,
        )

        self.virtual_canvas = tk.Canvas(
            displays,
            width=DISPLAY_WIDTH // 2,
            height=DISPLAY_HEIGHT // 2,
            background="#202b30",
            highlightthickness=1,
            highlightbackground="#52645d",
        )
        self.virtual_canvas.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        telemetry = tk.Frame(
            displays,
            bg=colors["panel"],
            height=56,
        )
        telemetry.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 0),
        )
        telemetry.grid_propagate(False)
        telemetry_items = (
            ("TRACKING", "TARGET LOCKED", colors["lime"]),
            ("CENTER OFFSET", "+02.4 px", colors["text"]),
            ("MOVEMENT", "0.18 m/s  /  STABLE", colors["text"]),
            ("LAST ANALYSIS", "240 ms AGO", colors["coral"]),
        )
        for index, (label, value, color) in enumerate(telemetry_items):
            telemetry.columnconfigure(index, weight=1)
            cell = tk.Frame(telemetry, bg=colors["panel"])
            cell.grid(row=0, column=index, sticky="nsew", padx=10, pady=8)
            tk.Label(
                cell,
                text=label,
                bg=colors["panel"],
                fg=colors["muted"],
                font=("Segoe UI", 7, "bold"),
            ).pack(anchor="w")
            tk.Label(
                cell,
                text=value,
                bg=colors["panel"],
                fg=color,
                font=("Consolas", 9, "bold"),
            ).pack(anchor="w", pady=(3, 0))

        terminal_frame = ttk.LabelFrame(
            main,
            text="LIVE DIAGNOSTICS  /  ALSO PRINTED TO POWERSHELL OR CMD",
            padding=4,
            style="Section.TLabelframe",
        )
        terminal_frame.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(8, 0),
        )
        terminal_frame.columnconfigure(0, weight=1)
        terminal_frame.rowconfigure(0, weight=1)
        self.terminal = tk.Text(
            terminal_frame,
            height=8,
            background="#101318",
            foreground="#b8f5c8",
            insertbackground="#b8f5c8",
            font=("Consolas", 9),
            relief=tk.FLAT,
            state=tk.DISABLED,
            wrap=tk.NONE,
        )
        self.terminal.grid(row=0, column=0, sticky="ew")
        terminal_scrollbar = ttk.Scrollbar(
            terminal_frame,
            orient=tk.VERTICAL,
            command=self.terminal.yview,
        )
        terminal_scrollbar.grid(row=0, column=1, sticky="ns")
        self.terminal.configure(yscrollcommand=terminal_scrollbar.set)

        footer = tk.Frame(
            self.root,
            bg=colors["panel_alt"],
            height=30,
        )
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        tk.Label(
            footer,
            textvariable=self.status,
            bg=colors["panel_alt"],
            fg=colors["muted"],
            anchor="w",
            font=("Consolas", 8),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=14)
        tk.Label(
            footer,
            text="LOCAL PROCESSING  ·  NET 46 MS  ·  BUILD 0.9.4-BETA",
            bg=colors["panel_alt"],
            fg=colors["muted"],
            font=("Consolas", 8),
        ).pack(side=tk.RIGHT, padx=14)

        self.source_type_changed()
        self.mode_changed(update_status=False)
        self._refresh_provider_statuses()

    def _show_startup_overlay(self) -> None:
        """Block interaction while the asynchronous startup checks run."""
        overlay = tk.Toplevel(self.root)
        overlay.title("System getting ready")
        overlay.transient(self.root)
        overlay.resizable(False, False)
        overlay.protocol("WM_DELETE_WINDOW", lambda: None)
        overlay.update_idletasks()
        popup_width = 430
        popup_height = 150
        root_x = self.root.winfo_x()
        root_y = self.root.winfo_y()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        popup_x = root_x + max(0, (root_width - popup_width) // 2)
        popup_y = root_y + max(0, (root_height - popup_height) // 2)
        overlay.geometry(f"{popup_width}x{popup_height}+{popup_x}+{popup_y}")
        overlay.lift()

        panel = ttk.Frame(overlay, padding=18)
        panel.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            panel,
            text="System getting ready...",
        ).pack(pady=(0, 8))
        self.startup_overlay_status = tk.StringVar(
            value="Checking cameras, computer vision, and AI providers..."
        )
        ttk.Label(
            panel,
            textvariable=self.startup_overlay_status,
        ).pack(pady=(0, 12))
        self.startup_spinner = ttk.Progressbar(
            panel,
            mode="indeterminate",
            length=280,
        )
        self.startup_spinner.pack()
        self.startup_spinner.start(12)
        ttk.Button(
            panel,
            text="Cancel",
            command=self.cancel_startup_checks,
        ).pack(pady=(12, 0))

        self.startup_overlay = overlay
        for sequence in ("<Button>", "<Key>", "<Motion>"):
            overlay.bind_all(sequence, lambda _event: "break", add="+")
            self.startup_input_bindings.append(sequence)
        overlay.grab_set()

    def cancel_startup_checks(self) -> None:
        """Stop startup probing and release the app for manual use."""
        self.startup_checks_cancelled = True
        self.camera_probe_cancel.set()
        self.startup_checks_pending.clear()
        self.pending_camera_choices = None
        self._hide_startup_overlay()
        self.status.set("Startup checks cancelled. Select a source manually.")

    def _hide_startup_overlay(self) -> None:
        if self.startup_overlay is None:
            return
        for sequence in self.startup_input_bindings:
            self.root.unbind_all(sequence)
        self.startup_input_bindings.clear()
        try:
            self.startup_overlay.grab_release()
        except tk.TclError:
            pass
        if self.startup_spinner is not None:
            self.startup_spinner.stop()
        self.startup_overlay.destroy()
        self.startup_overlay = None
        self.startup_overlay_status = None
        self.status.set("System initialized. Select a source or use the camera dialog.")
        if self.pending_camera_choices is not None:
            webcam_indices, stream_state = self.pending_camera_choices
            self.pending_camera_choices = None
            self._show_camera_choices(webcam_indices, stream_state)

    def ai_provider_changed(self, _event=None) -> None:
        """Restart the selected provider clocks without disabling AI."""
        if self.ai_enabled.get():
            self.ai_generation += 1
            self._schedule_ai_requests(immediate=True)
            self._log(
                f"AI provider selection changed to {self.ai_provider.get()}; "
                "provider clocks restarted."
            )
        self._refresh_provider_statuses()

    def bind_controls(self) -> None:
        self.root.bind("<Left>", lambda _event: self.nudge(-1, 0))
        self.root.bind("<Right>", lambda _event: self.nudge(1, 0))
        self.root.bind("<Up>", lambda _event: self.nudge(0, -1))
        self.root.bind("<Down>", lambda _event: self.nudge(0, 1))
        self.root.bind("<KeyPress-a>", lambda _event: self.nudge(-1, 0))
        self.root.bind("<KeyPress-d>", lambda _event: self.nudge(1, 0))
        self.root.bind("<KeyPress-w>", lambda _event: self.nudge(0, -1))
        self.root.bind("<KeyPress-s>", lambda _event: self.nudge(0, 1))
        self.root.bind("<KeyPress-plus>", lambda _event: self.change_zoom(0.1))
        self.root.bind("<KeyPress-minus>", lambda _event: self.change_zoom(-0.1))
        self.root.bind("<KeyPress-r>", lambda _event: self.reset_camera())
        self.root.bind("<KeyPress-c>", lambda _event: self.clear_tracking())

    def source_type_changed(self, _event=None) -> None:
        kind = self.source_type.get()
        stream_enabled = kind == "Android stream"
        camera_enabled = kind == "Webcam"
        video_enabled = kind == "Video file"

        self.set_widget_state(self.stream_entry, stream_enabled)
        self.set_widget_state(self.camera_entry, camera_enabled)
        self.set_widget_state(self.video_entry, video_enabled)
        self.set_widget_state(self.browse_button, video_enabled)

    def mode_changed(self, _event=None, update_status: bool = True) -> None:
        """Apply the selected operating mode and its safety behavior."""
        selected_mode = self.mode.get()
        if selected_mode != "Automatic AI":
            self.ai_generation += 1
            self.ai_enabled.set(False)
            self.ai_winner = None
            for runtime in self.ai_runtime.values():
                runtime.last_result = None
                runtime.last_box = None
                runtime.last_error = ""
                runtime.last_match_score = None
                runtime.last_human_match_score = None
            self.update_ai_indicator(False)
            self._refresh_provider_statuses()

        if selected_mode == "Manual control":
            self.auto_follow.set(False)
            if self.follow_checkbutton is not None:
                self.follow_checkbutton.configure(state="disabled")
            if update_status:
                self.status.set(
                    "Manual mode active. Keyboard and arrow buttons control the camera."
                )
        elif selected_mode == "Assisted tracking":
            if self.follow_checkbutton is not None:
                self.follow_checkbutton.configure(state="normal")
            if self.human_checkbutton is not None:
                self.human_checkbutton.configure(state="normal")
            self.auto_follow.set(True)
            if update_status:
                self.status.set(
                    "Assisted mode active. Click an object to track it."
                )
        elif selected_mode == "Face tracking":
            self.track_human.set(True)
            if self.human_checkbutton is not None:
                self.human_checkbutton.configure(state="disabled")
            if self.follow_checkbutton is not None:
                self.follow_checkbutton.configure(state="disabled")
            self.auto_follow.set(True)
            self.next_human_detection = 0.0
            self._update_human_status()
            if update_status:
                self.status.set(
                    "Face tracking active. The largest visible face controls the camera."
                )
        else:
            self.auto_follow.set(True)
            if self.follow_checkbutton is not None:
                self.follow_checkbutton.configure(state="disabled")
            if self.human_checkbutton is not None:
                self.human_checkbutton.configure(state="normal")
            self._schedule_ai_requests(immediate=True)
            if self.ai_enabled.get():
                self.update_ai_indicator(True)
            else:
                self.update_ai_indicator(False)
            if update_status and self.ai_enabled.get():
                self.status.set(
                    "Automatic AI active. Gemini and Groq stay on and analyze "
                    "alternating frames every 2.5 seconds."
                )
            elif update_status:
                self.status.set("Automatic AI is off. Click Enable AI to use cloud detection.")
        self._refresh_provider_statuses()

    def update_ai_indicator(
        self,
        enabled: bool,
        label: str | None = None,
        color: str | None = None,
    ) -> None:
        """Update the visible AI LED and toggle label from Tk's main thread."""
        color = color or ("#20b957" if enabled else "#c73838")
        self.ai_indicator.set(label or ("AI: ON" if enabled else "AI: OFF"))
        if self.ai_led is not None and self.ai_led_dot is not None:
            self.ai_led.itemconfigure(
                self.ai_led_dot,
                fill=color,
                outline=color,
            )
        if self.ai_toggle_button is not None:
            self.ai_toggle_button.configure(
                text="Disable AI" if enabled else "Enable AI"
            )

    def _enabled_providers(self) -> tuple[str, ...]:
        selection = self.ai_provider.get()
        if selection == "Gemini":
            return ("Gemini",)
        if selection == "Groq":
            return ("Groq",)
        return ("Gemini", "Groq")

    def _schedule_ai_requests(self, immediate: bool = False) -> None:
        """Give each provider its own clock, offset by 2.5 seconds."""
        now = time.monotonic()
        enabled = self._enabled_providers()
        for provider in self.ai_runtime:
            if provider not in enabled:
                self.ai_runtime[provider].next_check = now + AI_CHECK_INTERVAL_SECONDS
                continue
            if immediate:
                offset = (
                    AI_PROVIDER_OFFSET_SECONDS
                    if provider == "Groq" and len(enabled) == 2
                    else 0.0
                )
                self.ai_runtime[provider].next_check = now + offset

    def _refresh_provider_statuses(self) -> None:
        """Show whether each model is off, waiting, analyzing, or tracking."""
        now = time.monotonic()
        enabled = set(self._enabled_providers()) if self.ai_enabled.get() else set()
        for provider, runtime in self.ai_runtime.items():
            if provider not in enabled:
                text = f"{provider}: OFF  |  model: {runtime.model}"
                color = "#6b7280"
            elif runtime.in_flight:
                text = f"{provider}: ANALYZING  |  model: {runtime.model}"
                color = "#3b82f6"
            elif runtime.last_error and runtime.next_check > now:
                cooldown = max(1, int(round(runtime.next_check - now)))
                text = (
                    f"{provider}: ON / COOLDOWN ({cooldown}s)  |  "
                    f"model: {runtime.model}\n"
                    f"  {runtime.last_error[:58]}"
                )
                color = "#e09b3d"
            elif runtime.last_error:
                text = (
                    f"{provider}: ON / ERROR  |  model: {runtime.model}\n"
                    f"  {runtime.last_error[:58]}"
                )
                color = "#e05252"
            elif runtime.last_result is None:
                text = f"{provider}: ON / WAITING  |  model: {runtime.model}"
                color = "#20b957"
            else:
                outcome = "TRACKING" if runtime.last_result.found else "NO TARGET"
                age = max(0.0, now - runtime.last_completed)
                label = runtime.last_result.label.strip()[:56] or "unlabeled"
                if runtime.last_result_frame is None:
                    image_frame = "latest frame"
                else:
                    frame_age = (
                        max(0.0, now - runtime.last_result_captured_at)
                        if runtime.last_result_captured_at is not None
                        else 0.0
                    )
                    image_frame = (
                        f"frame #{runtime.last_result_frame} "
                        f"({frame_age:.1f}s old)"
                    )
                text = (
                    f"{provider}: ON / {outcome} ({age:.1f}s ago)\n"
                    f"  model: {runtime.model}\n"
                    f"  {image_frame}: {label}"
                )
                color = "#20b957"

            status_var = self.provider_status_vars.get(provider)
            if status_var is not None:
                status_var.set(text)
            led_info = self.provider_status_leds.get(provider)
            if led_info is not None:
                led, dot = led_info
                led.itemconfigure(dot, fill=color, outline=color)

    def toggle_ai(self) -> None:
        """Enable cloud detection or return to local tracking without it."""
        if self.ai_enabled.get():
            self.ai_enabled.set(False)
            self.ai_generation += 1
            if self.mode.get() == "Automatic AI":
                self.mode.set("Assisted tracking")
            self.mode_changed(update_status=False)
            self.update_ai_indicator(False)
            self.status.set(
                "AI disabled. Local Assisted tracking remains available."
            )
            self._log("AI disabled by user. Existing worker results will be ignored.")
            return

        self.ai_enabled.set(True)
        self.mode.set("Automatic AI")
        self.ai_generation += 1
        self._schedule_ai_requests(immediate=True)
        self.mode_changed(update_status=False)
        self.update_ai_indicator(True, "AI: ON • independent providers")
        if self.running:
            self.status.set(
                "AI enabled. Gemini starts now; Groq starts 2.5 seconds later."
            )
        else:
            self.status.set("AI enabled. Start a source to begin provider checks.")
        self._log(
            "AI enabled. Providers run independently every 5 seconds; "
            "Gemini and Groq are staggered by 2.5 seconds."
        )
        self._refresh_provider_statuses()

    def rescan_ai(self) -> None:
        """Force enabled providers to analyze the next available frame."""
        if not self.ai_enabled.get():
            self.ai_enabled.set(True)
            self.mode.set("Automatic AI")
            self.ai_generation += 1
            self.mode_changed(update_status=False)
            self.update_ai_indicator(True, "AI: ON • rescanning")

        now = time.monotonic()
        enabled = set(self._enabled_providers())
        for provider, runtime in self.ai_runtime.items():
            if provider in enabled and not runtime.in_flight:
                runtime.next_check = now
                runtime.last_error = ""

        self.status.set("AI rescan requested. Analyzing the next camera frame.")
        self._log(
            "AI rescan requested; enabled providers are due on the next frame."
        )
        self._refresh_provider_statuses()

    def _start_ai_after_tracking_loss(self) -> None:
        if not AI_AUTO_ENABLE_ON_TRACK_LOSS:
            return

        if not self.ai_enabled.get():
            self.ai_enabled.set(True)
            self.ai_generation += 1
        self.mode.set("Automatic AI")
        self.mode_changed(update_status=False)

        now = time.monotonic()
        if not self.tracking_lost or self.ai_recovery_deadline <= now:
            self.ai_recovery_deadline = now + AI_RECOVERY_WINDOW_SECONDS
        enabled = self._enabled_providers()
        for provider, runtime in self.ai_runtime.items():
            if provider not in enabled:
                continue
            runtime.next_check = now

        self.update_ai_indicator(True, "AI: ON • tracking-loss fallback")
        self.status.set(
            "Local tracking lost. AI is analyzing the loss frame immediately."
        )
        self._log(
            "Local tracker lost; AI fallback enabled for the current JPEG "
            "frame and recovery requests are due every 2 seconds for 10 seconds."
        )
        self._refresh_provider_statuses()

    @staticmethod
    def _ai_error_cooldown(
        error: Exception,
        consecutive_failures: int,
    ) -> float:
        """Back off provider failures without changing the normal 5-second cadence."""
        status_code = vision_providers.provider_status_code(error)
        if status_code == 429:
            return min(
                AI_MAX_ERROR_COOLDOWN_SECONDS,
                AI_RATE_LIMIT_COOLDOWN_SECONDS
                * (2 ** min(consecutive_failures - 1, 2)),
            )
        return min(
            AI_MAX_ERROR_COOLDOWN_SECONDS,
            AI_TRANSIENT_ERROR_COOLDOWN_SECONDS
            * (2 ** min(consecutive_failures - 1, 4)),
        )

    @staticmethod
    def _prepare_ai_frame(frame: np.ndarray) -> np.ndarray:
        """Resize one image without changing its aspect ratio."""
        frame_height, frame_width = frame.shape[:2]
        scale = min(
            1.0,
            AI_FRAME_MAX_DIMENSION / max(frame_width, frame_height),
        )
        if scale == 1.0:
            return frame
        image_size = (
            max(1, int(round(frame_width * scale))),
            max(1, int(round(frame_height * scale))),
        )
        return cv2.resize(
            frame,
            image_size,
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _scale_box_between_frames(
        box: tuple[int, int, int, int],
        source_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        source_height, source_width = source_shape[:2]
        target_height, target_width = target_shape[:2]
        x, y, width, height = box
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        return (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            max(1, int(round(width * scale_x))),
            max(1, int(round(height * scale_y))),
        )

    def _maybe_start_ai_requests(self, frame: np.ndarray) -> None:
        """Start due provider requests without blocking video or each other."""
        if (
            self.mode.get() != "Automatic AI"
            or not self.ai_enabled.get()
            or not self.running
        ):
            return

        now = time.monotonic()
        recovery_active = (
            self.tracking_lost
            and now < self.ai_recovery_deadline
        )
        request_interval = (
            AI_RECOVERY_CHECK_INTERVAL_SECONDS
            if recovery_active
            else AI_CHECK_INTERVAL_SECONDS
        )
        due_providers = [
            provider
            for provider in self._enabled_providers()
            if not self.ai_runtime[provider].in_flight
            and now >= self.ai_runtime[provider].next_check
        ]
        if not due_providers:
            return

        small_frame = self._prepare_ai_frame(frame)
        ok, encoded = cv2.imencode(
            ".jpg",
            small_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 78],
        )
        if not ok:
            for provider in due_providers:
                self.ai_runtime[provider].next_check = (
                    now + request_interval
                )
                self.ai_runtime[provider].last_error = (
                    "Current frame could not be JPEG encoded."
                )
            self.status.set("Automatic AI could not encode the current frame.")
            self._log("AI frame encode failed; provider clocks continue.")
            return

        jpeg_bytes = encoded.tobytes()
        reference_jpeg_bytes = self._reference_jpeg_bytes()
        target_priority = self.ai_target_priority.get().strip()
        generation = self.ai_generation
        frame_height, frame_width = frame.shape[:2]
        context = AIFrameContext(
            frame_number=self.video_frame_count,
            captured_at=now,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        for provider in due_providers:
            runtime = self.ai_runtime[provider]
            runtime.in_flight = True
            runtime.last_started = now
            runtime.next_check = now + request_interval
            runtime.request_count += 1
            thread = threading.Thread(
                target=self._run_ai_request,
                args=(
                    generation,
                    provider,
                    jpeg_bytes,
                    reference_jpeg_bytes,
                    target_priority,
                    context,
                ),
                daemon=True,
                name=f"automatic-camera-{provider.lower()}",
            )
            self.ai_threads[provider] = thread
            thread.start()
            self._log(
                f"AI image request started (JPEG still): provider={provider} "
                f"model={runtime.model} frame={context.frame_number} "
                f"source={frame_width}x{frame_height} "
                f"image={small_frame.shape[1]}x{small_frame.shape[0]} "
                f"next_in={AI_CHECK_INTERVAL_SECONDS:.1f}s"
            )
        self._refresh_provider_statuses()

    def _run_ai_request(
        self,
        generation: int,
        provider: str,
        jpeg_bytes: bytes,
        reference_jpeg_bytes: bytes | None,
        target_priority: str,
        context: AIFrameContext,
    ) -> None:
        try:
            result = vision_providers.create_provider(provider).locate_presenter(
                jpeg_bytes,
                reference_jpeg_bytes,
                target_priority,
            )
        except Exception as error:
            self.ai_results.put((generation, provider, "error", error, context))
        else:
            self.ai_results.put((generation, provider, "result", result, context))

    def _drain_ai_results(self) -> None:
        """Apply independent provider results on Tk's main thread."""
        while True:
            try:
                generation, provider, result_type, payload, context = (
                    self.ai_results.get_nowait()
                )
            except queue.Empty:
                self._refresh_provider_statuses()
                return

            runtime = self.ai_runtime.get(provider)
            if runtime is None:
                continue

            if generation != self.ai_generation:
                runtime.in_flight = False
                continue

            runtime.in_flight = False
            runtime.last_completed = time.monotonic()
            if (
                not self.running
                or self.mode.get() != "Automatic AI"
                or provider not in self._enabled_providers()
            ):
                continue

            if result_type == "error":
                error = payload if isinstance(payload, Exception) else RuntimeError(str(payload))
                runtime.last_error = str(error)
                runtime.consecutive_failures += 1
                cooldown = self._ai_error_cooldown(
                    error,
                    runtime.consecutive_failures,
                )
                runtime.next_check = max(
                    runtime.next_check,
                    time.monotonic() + cooldown,
                )
                self._log(
                    f"AI error: provider={provider} model={runtime.model} "
                    f"error={runtime.last_error} retry_in={cooldown:.0f}s"
                )
                self._refresh_provider_statuses()
                continue

            runtime.last_error = ""
            runtime.consecutive_failures = 0
            runtime.last_result = payload
            runtime.last_result_frame = context.frame_number
            runtime.last_result_captured_at = context.captured_at
            runtime.success_count += 1
            self.update_ai_indicator(True, "AI: ON • independent providers")
            self._apply_ai_result(payload, context)
            self._refresh_provider_statuses()

    def _apply_ai_result(
        self,
        result: vision_providers.VisionResult,
        context: AIFrameContext,
    ) -> None:
        """Store a provider detection and let arbitration choose the camera target."""
        if self.frame is None:
            return

        runtime = self.ai_runtime[result.provider]
        if not result.found or result.box is None:
            runtime.last_box = None
            runtime.last_match_score = None
            runtime.last_human_match_score = None
            self._log(
                f"AI result: provider={result.provider} found=false "
                f"label={result.label!r} confidence={result.confidence:.0%} "
                f"image_frame={context.frame_number}; provider stays online"
            )
            self._choose_best_ai_result()
            return

        frame_height, frame_width = self.frame.shape[:2]
        image_age = max(0.0, time.monotonic() - context.captured_at)
        if (
            context.frame_width != frame_width
            or context.frame_height != frame_height
            or image_age > AI_BOX_MAX_AGE_SECONDS
        ):
            runtime.last_box = None
            runtime.last_match_score = None
            runtime.last_human_match_score = None
            self.status.set(
                f"{result.provider} analyzed frame #{context.frame_number}: "
                f"{result.label}. Its box is stale or the source changed; "
                "the camera will not follow it."
            )
            self._log(
                f"AI box ignored: provider={result.provider} "
                f"image_frame={context.frame_number} age={image_age:.1f}s "
                "source_changed_or_stale=true"
            )
            self._choose_best_ai_result()
            return

        x, y, box_width, box_height = result.box
        pixel_box = (
            int(round(x * frame_width / 1000)),
            int(round(y * frame_height / 1000)),
            max(20, int(round(box_width * frame_width / 1000))),
            max(20, int(round(box_height * frame_height / 1000))),
        )
        if self.track_human.get():
            pixel_box = self._refine_ai_box_with_face(pixel_box)
        if not self.valid_tracking_box(
            pixel_box,
            frame_width,
            frame_height,
            None,
        ):
            runtime.last_box = None
            runtime.last_human_match_score = None
            runtime.last_error = "Provider returned an unsafe bounding box."
            self._log(
                f"AI result rejected: provider={result.provider} "
                "bounding box failed safety validation"
            )
            self._choose_best_ai_result()
            return

        match_score = self._target_match_score(pixel_box)
        runtime.last_match_score = match_score
        if (
            match_score is not None
            and match_score < TARGET_MATCH_REJECT_THRESHOLD
            and not self.tracking_lost
        ):
            runtime.last_box = None
            runtime.last_human_match_score = None
            runtime.last_error = (
                f"Target appearance match too low ({match_score:.0%})."
            )
            self._log(
                f"AI result rejected: provider={result.provider} "
                f"target_match={match_score:.0%} below "
                f"{TARGET_MATCH_REJECT_THRESHOLD:.0%}"
            )
            self._choose_best_ai_result()
            return

        runtime.last_box = pixel_box
        runtime.last_human_match_score = self._human_match_for_box(pixel_box)
        match_text = (
            f" target_match={match_score:.0%}"
            if match_score is not None
            else ""
        )
        human_match_text = (
            f" human_match={runtime.last_human_match_score:.0%}"
            if runtime.last_human_match_score is not None
            else ""
        )
        self._log(
            f"AI result: provider={result.provider} found=true "
            f"label={result.label!r} image_frame={context.frame_number} "
            f"confidence={result.confidence:.0%} "
            f"box=({pixel_box[0]},{pixel_box[1]},{pixel_box[2]},{pixel_box[3]})"
            f"{match_text}{human_match_text}"
        )
        self._choose_best_ai_result()

    @staticmethod
    def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        x, y, width, height = box
        return x + width / 2, y + height / 2

    def _ai_box_error(
        self,
        provider: str,
        box: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> float:
        """Calculate normalized error from temporal and cross-model disagreement."""
        diagonal = max(1.0, (frame_width**2 + frame_height**2) ** 0.5)
        center = self._box_center(box)
        reference = self.tracking_box
        temporal_error = 0.0
        if reference is not None:
            old_center = self._box_center(reference)
            temporal_error = min(
                1.0,
                ((center[0] - old_center[0]) ** 2 + (center[1] - old_center[1]) ** 2)
                ** 0.5
                / diagonal,
            )

        other_centers = [
            self._box_center(other.last_box)
            for other_name, other in self.ai_runtime.items()
            if other_name != provider and other.last_box is not None
        ]
        agreement_error = 0.0
        if other_centers:
            agreement_error = min(
                1.0,
                sum(
                    ((center[0] - other[0]) ** 2 + (center[1] - other[1]) ** 2)
                    ** 0.5
                    / diagonal
                    for other in other_centers
                )
                / len(other_centers),
            )
        return min(1.0, temporal_error * 0.55 + agreement_error * 0.45)

    def _choose_best_ai_result(self) -> None:
        """Score both live detections and move the camera using the best one."""
        if self.frame is None or not self.ai_enabled.get():
            return
        frame_height, frame_width = self.frame.shape[:2]
        candidates: list[tuple[float, str, float]] = []
        for provider in self._enabled_providers():
            runtime = self.ai_runtime[provider]
            if runtime.last_result is None or not runtime.last_result.found:
                continue
            if runtime.last_box is None:
                continue
            age = time.monotonic() - runtime.last_completed
            if age > AI_BOX_MAX_AGE_SECONDS:
                continue
            error = self._ai_box_error(
                provider,
                runtime.last_box,
                frame_width,
                frame_height,
            )
            match_score = runtime.last_match_score
            reference_factor = (
                1.0
                if match_score is None
                else 0.5 + 0.5 * match_score
            )
            human_match = self._human_match_for_box(runtime.last_box)
            human_factor = (
                1.0
                if human_match is None
                else 0.5 + 0.5 * human_match
            )
            score = (
                runtime.last_result.confidence
                * max(0.05, 1.0 - error)
                * reference_factor
                * (human_factor if self.track_human.get() else 1.0)
            )
            candidates.append((score, provider, error))

        if not candidates:
            return

        score, provider, error = max(candidates)
        runtime = self.ai_runtime[provider]
        box = runtime.last_box
        if box is None or runtime.last_result is None:
            return

        self.ai_winner = provider
        self.target_x, self.target_y = self._box_center(box)
        self.tracking_lost = False
        self._initialize_local_tracker(box)
        self.status.set(
            f"{provider} controls the virtual camera "
            f"(confidence {runtime.last_result.confidence:.0%}, error {error:.0%})."
        )
        human_match = (
            self._human_match_for_box(box)
            if self.track_human.get()
            else None
        )
        human_match_text = (
            f"{human_match:.2f}" if human_match is not None else "n/a"
        )
        target_match_text = (
            f"{runtime.last_match_score:.2f}"
            if runtime.last_match_score is not None
            else "1.00"
        )
        self._log(
            f"AI arbitration: winner={provider} score={score:.2f} "
            f"confidence={runtime.last_result.confidence:.2f} error={error:.2f} "
            f"target_match={target_match_text} "
            f"human_match={human_match_text} "
            f"camera_target=({self.target_x:.0f},{self.target_y:.0f})"
        )

    def _initialize_local_tracker(self, pixel_box: tuple[int, int, int, int]) -> None:
        """Refresh local tracking, but never turn a provider error into AI OFF."""
        if self.frame is None:
            return
        try:
            tracker = self.create_tracker()
            tracker_frame = self._prepare_ai_frame(self.frame)
            tracker_box = self._scale_box_between_frames(
                pixel_box,
                self.frame.shape,
                tracker_frame.shape,
            )
            tracker.init(tracker_frame, tracker_box)
        except Exception as error:
            self.tracker = None
            self.tracking_box = pixel_box
            self._log(
                f"Local tracker unavailable; AI target remains active: {error}"
            )
            return
        self.tracker = tracker
        self.tracking_box = pixel_box

    def _target_match_score(
        self,
        candidate_box: tuple[int, int, int, int],
    ) -> float | None:
        """Compare a candidate box with the operator's session-only reference."""
        if self.frame is None or self.target_reference is None:
            return None
        try:
            score = self.target_reference.appearance_score(
                self.frame,
                candidate_box,
            )
            if self.track_human.get():
                feature_score = target_tracking.human_match_score(
                    candidate_box,
                    self.face_boxes,
                    self.eye_boxes,
                    self.smile_boxes,
                )
                if feature_score is not None:
                    score = 0.65 * score + 0.35 * feature_score
        except (TypeError, ValueError, cv2.error) as error:
            self._log(f"Target reference match failed: {error}")
            return 0.0
        self.target_match_score = score
        return score

    def _reference_jpeg_bytes(self) -> bytes | None:
        """Encode the session-only reference only when AI is about to use it."""
        if (
            self.target_reference is None
            or self.target_reference.reference_crop is None
        ):
            return None
        try:
            ok, encoded = cv2.imencode(
                ".jpg",
                self.target_reference.reference_crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), 82],
            )
        except cv2.error as error:
            self._log(f"Target reference could not be encoded for AI: {error}")
            return None
        if not ok:
            self._log("Target reference could not be encoded for AI.")
            return None
        return encoded.tobytes()

    def _human_match_for_box(
        self,
        candidate_box: tuple[int, int, int, int],
    ) -> float | None:
        """Return the current local face signal for a candidate box."""
        if not self.track_human.get():
            return None
        return target_tracking.human_match_score(
            candidate_box,
            self.face_boxes,
            self.eye_boxes,
            self.smile_boxes,
        )

    def _refine_ai_box_with_face(
        self,
        candidate_box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Combine an AI head box with the nearest local face box."""
        if not self.face_boxes:
            return candidate_box

        candidate_x, candidate_y, candidate_width, candidate_height = candidate_box
        candidate_center = (
            candidate_x + candidate_width / 2,
            candidate_y + candidate_height / 2,
        )
        ranked_faces = []
        for face_box in self.face_boxes:
            face_x, face_y, face_width, face_height = face_box
            face_center = (
                face_x + face_width / 2,
                face_y + face_height / 2,
            )
            distance = (
                (face_center[0] - candidate_center[0]) ** 2
                + (face_center[1] - candidate_center[1]) ** 2
            ) ** 0.5
            overlap = target_tracking.box_iou(candidate_box, face_box)
            ranked_faces.append((overlap, -distance, face_box))

        overlap, negative_distance, face_box = max(ranked_faces)
        max_distance = max(candidate_width, candidate_height) * 1.5
        if overlap <= 0 and -negative_distance > max_distance:
            return candidate_box

        left = min(candidate_x, face_box[0])
        top = min(candidate_y, face_box[1])
        right = max(candidate_x + candidate_width, face_box[0] + face_box[2])
        bottom = max(candidate_y + candidate_height, face_box[1] + face_box[3])
        margin_x = max(4, int(round((right - left) * 0.08)))
        margin_y = max(4, int(round((bottom - top) * 0.08)))
        refined = (
            max(0, left - margin_x),
            max(0, top - margin_y),
            min(self.frame.shape[1], right + margin_x)
            - max(0, left - margin_x),
            min(self.frame.shape[0], bottom + margin_y)
            - max(0, top - margin_y),
        )
        self._log(
            f"AI head box refined with local face: "
            f"({candidate_x},{candidate_y},{candidate_width},{candidate_height}) "
            f"-> ({refined[0]},{refined[1]},{refined[2]},{refined[3]})"
        )
        return refined

    def human_tracking_changed(self) -> None:
        """Toggle the optional local face signal without changing target state."""
        self.next_human_detection = 0.0
        self.human_detection_error = ""
        self.human_match_score = None
        if not self.track_human.get():
            self.face_boxes = ()
            self.eye_boxes = ()
            self.smile_boxes = ()
        self._update_human_status()
        if self.track_human.get():
            self._log(
                "Human tracking enabled: local OpenCV face detection is an "
                "additional signal; missing faces will not drop the target."
            )
            self.status.set(
                "Human tracking enabled. Face detection supplements local tracking."
            )
        else:
            self._log("Human tracking disabled; generic object tracking remains active.")
            self.status.set("Human tracking disabled. Generic object tracking remains active.")

    def _load_face_classifier(self) -> cv2.CascadeClassifier:
        """Load OpenCV's bundled frontal-face cascade on first use."""
        if self.face_classifier is not None:
            return self.face_classifier

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        classifier = cv2.CascadeClassifier(str(cascade_path))
        if classifier.empty():
            raise RuntimeError(f"Could not load the local face detector: {cascade_path}")
        self.face_classifier = classifier
        return classifier

    def _load_face_feature_classifiers(
        self,
    ) -> tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
        if self.eye_classifier is None:
            eye_path = Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"
            self.eye_classifier = cv2.CascadeClassifier(str(eye_path))
            if self.eye_classifier.empty():
                raise RuntimeError("Could not load the local eye detector.")
        if self.smile_classifier is None:
            smile_path = Path(cv2.data.haarcascades) / "haarcascade_smile.xml"
            self.smile_classifier = cv2.CascadeClassifier(str(smile_path))
            if self.smile_classifier.empty():
                raise RuntimeError("Could not load the local mouth/smile detector.")
        return self.eye_classifier, self.smile_classifier

    def _update_human_status(self) -> None:
        """Show detector state without treating no face as tracker loss."""
        if not self.track_human.get():
            self.human_status.set("Human tracking: OFF")
            return
        if self.human_detection_error:
            self.human_status.set(
                f"Human tracking: ERROR | {self.human_detection_error[:54]}"
            )
            return

        face_text = (
            f"face detected ({len(self.face_boxes)})"
            if self.face_boxes
            else "face not detected"
        )
        match_text = (
            f"{self.human_match_score:.0%}"
            if self.human_match_score is not None
            else "n/a"
        )
        self.human_status.set(
            f"Human tracking: ON | {face_text}\n"
            f"Eyes: {len(self.eye_boxes)} | mouth/smile cues: "
            f"{len(self.smile_boxes)}\n"
            f"Human match: {match_text} (supporting signal)"
        )

    def _update_human_detection(self, frame: np.ndarray) -> None:
        """Refresh local face detections at a safe rate for desktop CPUs."""
        if not self.track_human.get():
            self._update_human_status()
            return

        now = time.monotonic()
        if now < self.next_human_detection:
            self._update_human_status()
            return
        self.next_human_detection = now + HUMAN_DETECTION_INTERVAL_SECONDS

        try:
            classifier = self._load_face_classifier()
            analysis_frame = self._prepare_ai_frame(frame)
            analysis_faces = target_tracking.detect_faces(
                analysis_frame,
                classifier,
            )
            eye_classifier, smile_classifier = (
                self._load_face_feature_classifiers()
            )
            analysis_eyes, analysis_smiles = target_tracking.detect_face_features(
                analysis_frame,
                analysis_faces,
                eye_classifier,
                smile_classifier,
            )
            self.face_boxes = tuple(
                self._scale_box_between_frames(
                    box,
                    analysis_frame.shape,
                    frame.shape,
                )
                for box in analysis_faces
            )
            self.eye_boxes = tuple(
                self._scale_box_between_frames(
                    box,
                    analysis_frame.shape,
                    frame.shape,
                )
                for box in analysis_eyes
            )
            self.smile_boxes = tuple(
                self._scale_box_between_frames(
                    box,
                    analysis_frame.shape,
                    frame.shape,
                )
                for box in analysis_smiles
            )
            self.human_detection_error = ""
        except (RuntimeError, cv2.error, AttributeError) as error:
            self.face_boxes = ()
            self.eye_boxes = ()
            self.smile_boxes = ()
            self.human_detection_error = str(error)
            self._log(f"Local face detection unavailable: {error}")
        self._update_human_match()

    def _maybe_start_human_tracking(self) -> None:
        """Start local camera following when face mode has no target yet."""
        if (
            not self.track_human.get()
            or self.mode.get() == "Automatic AI"
            or self.target_reference is not None
            or self.tracker is not None
            or self.frame is None
            or not self.face_boxes
        ):
            return

        candidate_box = max(
            self.face_boxes,
            key=lambda box: box[2] * box[3],
        )
        self.auto_follow.set(True)
        self._initialize_local_tracker(candidate_box)
        if self.tracker is None:
            return
        self.tracking_lost = False
        self.target_x, self.target_y = self._box_center(candidate_box)
        self._capture_target_reference(candidate_box, target_kind="human")
        self.status.set(
            "Local human tracking acquired the largest visible face."
        )
        self._log(
            f"Local human tracking acquired face: "
            f"box=({candidate_box[0]},{candidate_box[1]} "
            f"{candidate_box[2]}x{candidate_box[3]})"
        )

    def _update_human_match(self) -> None:
        """Re-score the current target against the latest detected faces."""
        if not self.track_human.get() or self.tracking_box is None:
            self.human_match_score = None
        else:
            self.human_match_score = self._human_match_for_box(self.tracking_box)
        self._update_human_status()

    def _update_target_status(self, score: float | None = None) -> None:
        if self.target_reference is None:
            self.target_status.set("Target reference: none")
            return
        if score is not None:
            self.target_match_score = score
        self.target_status.set(
            f"Target reference: {self.target_reference.target_kind} | "
            f"match {self.target_match_score:.0%}"
        )

    def _capture_target_reference(
        self,
        box: tuple[int, int, int, int],
        target_kind: str = "generic",
    ) -> bool:
        if self.frame is None:
            return False
        try:
            self.target_reference = target_tracking.create_target_reference(
                self.frame,
                box,
                target_kind=target_kind,
            )
        except ValueError as error:
            self.target_status.set(f"Target reference: error ({error})")
            self._log(f"Target reference capture failed: {error}")
            return False
        self.target_match_score = 1.0
        self.target_mismatch_frames = 0
        self._update_target_status()
        self._log(
            f"Target reference captured: kind={target_kind} "
            f"box=({box[0]},{box[1]},{box[2]},{box[3]})"
        )
        return True

    def refresh_target_reference(self) -> None:
        """Replace the current reference using the latest tracked box."""
        if self.frame is None or self.tracking_box is None:
            self.status.set("Select a target before refreshing its reference.")
            return
        if self._capture_target_reference(
            self.tracking_box,
            target_kind=(
                self.target_reference.target_kind
                if self.target_reference is not None
                else "generic"
            ),
        ):
            self.status.set("Target reference refreshed from the current frame.")

    def show_priority_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("AI priorities")
        dialog.transient(self.root)
        dialog.resizable(True, True)
        dialog.minsize(380, 300)
        body = ttk.Frame(dialog, padding=16)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            body,
            text="AI target priorities",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            body,
            text="Checked items are sent highest priority first. Face stays at #1.",
        ).pack(anchor="w", pady=(5, 8))

        configured = [
            item.strip().lower()
            for item in self.ai_target_priority.get().split(",")
            if item.strip()
        ]
        priorities: list[dict[str, object]] = []
        for item in ["face", *configured]:
            if item not in [entry["name"] for entry in priorities]:
                priorities.append({"name": item, "enabled": True})
        selected_index = tk.IntVar(value=0)
        rows = ttk.Frame(body)
        rows.pack(fill=tk.X, pady=(0, 8))

        def rebuild_rows() -> None:
            for child in rows.winfo_children():
                child.destroy()
            for index, item in enumerate(priorities):
                name = str(item["name"])
                row = ttk.Frame(rows)
                row.pack(fill=tk.X, pady=1)
                ttk.Checkbutton(
                    row,
                    text=f"{index + 1}.",
                    variable=item["variable"],
                    command=lambda current=index: selected_index.set(current),
                ).pack(side=tk.LEFT)
                ttk.Button(
                    row,
                    text=name,
                    command=lambda current=index: selected_index.set(current),
                    width=24,
                ).pack(side=tk.LEFT, padx=(4, 0))

        for item in priorities:
            item["variable"] = tk.BooleanVar(value=True)

        rebuild_rows()

        add_row = ttk.Frame(body)
        add_row.pack(fill=tk.X, pady=(0, 10))
        new_priority = tk.StringVar()
        ttk.Entry(add_row, textvariable=new_priority, width=28).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
        )

        def add_priority() -> None:
            name = new_priority.get().strip().lower()
            if not name or any(str(item["name"]) == name for item in priorities):
                return
            priorities.append(
                {"name": name, "enabled": True, "variable": tk.BooleanVar(value=True)}
            )
            new_priority.set("")
            selected_index.set(len(priorities) - 1)
            rebuild_rows()

        ttk.Button(add_row, text="Add", command=add_priority).pack(
            side=tk.LEFT,
            padx=(6, 0),
        )

        reorder_row = ttk.Frame(body)
        reorder_row.pack(fill=tk.X, pady=(0, 10))

        def move_priority(direction: int) -> None:
            index = selected_index.get()
            new_index = index + direction
            if index <= 0 or new_index >= len(priorities):
                return
            priorities[index], priorities[new_index] = (
                priorities[new_index],
                priorities[index],
            )
            selected_index.set(new_index)
            rebuild_rows()

        def delete_priority() -> None:
            index = selected_index.get()
            if index <= 0 or index >= len(priorities):
                return
            priorities.pop(index)
            selected_index.set(max(0, index - 1))
            rebuild_rows()

        ttk.Button(reorder_row, text="Move up", command=lambda: move_priority(-1)).pack(
            side=tk.LEFT,
        )
        ttk.Button(
            reorder_row,
            text="Move down",
            command=lambda: move_priority(1),
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(reorder_row, text="Delete", command=delete_priority).pack(
            side=tk.LEFT,
            padx=(6, 0),
        )

        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(
            side=tk.RIGHT,
            padx=(8, 0),
        )
        def save_priorities() -> None:
            enabled_names = [
                str(item["name"])
                for item in priorities
                if item["variable"].get()
            ]
            self.ai_target_priority.set(", ".join(enabled_names))
            dialog.destroy()

        ttk.Button(buttons, text="Save", command=save_priorities).pack(
            side=tk.RIGHT,
        )
        dialog.update_idletasks()
        dialog.geometry(
            f"{dialog.winfo_reqwidth()}x{dialog.winfo_reqheight()}+"
            f"{self.root.winfo_rootx() + 80}+{self.root.winfo_rooty() + 120}"
        )
        dialog.grab_set()

    def _enable_ai_after_camera_start(self) -> None:
        if not self.running or self.mode.get() != "Automatic AI":
            return
        self.ai_enabled.set(True)
        self.ai_generation += 1
        self._schedule_ai_requests(immediate=True)
        self.mode_changed(update_status=False)
        self.update_ai_indicator(True, "AI: ON • waiting for target")
        self.status.set(
            "AI enabled. Select an object in Source Camera, or choose Skip "
            "to use AI priorities."
        )
        self._log("AI enabled automatically 1.5 seconds after camera startup.")
        self.root.after(100, self._ask_for_ai_target)

    def _ask_for_ai_target(self) -> None:
        if not self.running or self.mode.get() != "Automatic AI":
            return
        select_target = messagebox.askyesno(
            "Select AI target",
            "Would you like to select an object now?\n\n"
            "Yes: click the target in Source Camera.\n"
            "No: AI will use the configured priorities.",
            parent=self.root,
        )
        if select_target:
            self.status.set(
                "Click the target in Source Camera. AI will follow your selection."
            )
        else:
            self.status.set(
                "AI is using the configured target priorities."
            )

    def _start_startup_checks(self) -> None:
        """Run hardware and model checks without blocking the Tk event loop."""
        stream_url = self.stream_url.get().strip()
        self.camera_probe_cancel.clear()
        try:
            diagnostic_image = self._diagnostic_image_jpeg()
            image_error = ""
        except Exception as error:
            diagnostic_image = None
            image_error = type(error).__name__

        checks = [
            (
                "camera-feed-check",
                self._run_camera_feed_check,
                (stream_url,),
            ),
            ("local-cv-check", self._run_local_cv_check, ()),
            (
                "Gemini-startup-check",
                self._run_provider_startup_check,
                ("Gemini", diagnostic_image, image_error),
            ),
            (
                "Groq-startup-check",
                self._run_provider_startup_check,
                ("Groq", diagnostic_image, image_error),
            ),
        ]
        for thread_name, target, arguments in checks:
            threading.Thread(
                target=target,
                args=arguments,
                name=thread_name,
                daemon=True,
            ).start()

    @staticmethod
    def _diagnostic_image_jpeg() -> bytes:
        """Build a non-private calibration graphic for provider vision checks."""
        image = np.full((240, 320, 3), 245, dtype=np.uint8)
        cv2.rectangle(image, (35, 35), (135, 165), (255, 0, 0), -1)
        cv2.circle(image, (225, 100), 55, (0, 0, 255), -1)
        cv2.putText(
            image,
            "BLUE BOX / RED CIRCLE",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("Could not encode calibration image")
        return encoded.tobytes()

    def _publish_startup_check(self, key: str, message: str) -> None:
        self.startup_check_results.put((key, message))
        self._log(f"Startup check {key}: {message}")

    @staticmethod
    def _camera_selection_options(
        webcam_indices: list[int],
        stream_state: str,
    ) -> tuple[str, ...]:
        options = tuple(f"Webcam {index}" for index in webcam_indices)
        if stream_state == "online":
            options += ("Android stream",)
        return options

    def _show_camera_choices(
        self,
        webcam_indices: tuple[int, ...],
        stream_state: str,
    ) -> None:
        if self.camera_dialog is not None and self.camera_dialog.winfo_exists():
            return

        options = self._camera_selection_options(list(webcam_indices), stream_state)
        dialog = tk.Toplevel(self.root)
        self.camera_dialog = dialog
        dialog.title("Select camera")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

        body = ttk.Frame(dialog, padding=18)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            body,
            text="Startup camera check complete",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            body,
            text="Choose the camera or stream you want to use:",
        ).pack(anchor="w", pady=(5, 12))
        mode_selection = tk.StringVar(
            value=(
                self.mode.get()
                if self.mode.get() in {
                    "Manual control",
                    "Assisted tracking",
                    "Face tracking",
                    "Automatic AI",
                }
                else "Assisted tracking"
            )
        )

        if not options:
            ttk.Label(
                body,
                text="No usable camera feeds were found. Check camera permissions or connect a camera.",
                wraplength=360,
            ).pack(anchor="w", pady=(0, 12))
        else:
            selection = tk.StringVar(value=options[0])
            camera_options = ttk.Frame(body)
            camera_options.pack(fill=tk.X, pady=(0, 12))
            for option in options:
                ttk.Radiobutton(
                    camera_options,
                    text=option,
                    variable=selection,
                    value=option,
                ).pack(anchor="w", pady=2)

            ttk.Label(body, text="Tracking mode:").pack(anchor="w", pady=(4, 3))
            mode_options = ttk.Frame(body)
            mode_options.pack(fill=tk.X, pady=(0, 12))
            for mode in (
                "Manual control",
                "Assisted tracking",
                "Face tracking",
                "Automatic AI",
            ):
                ttk.Radiobutton(
                    mode_options,
                    text=mode,
                    variable=mode_selection,
                    value=mode,
                ).pack(anchor="w", pady=1)

        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X)

        def close_dialog() -> None:
            if self.camera_dialog is dialog:
                self.camera_dialog = None
            dialog.destroy()

        def start_selected() -> None:
            selected = selection.get()
            if selected.startswith("Webcam "):
                self.source_type.set("Webcam")
                self.camera_index.set(selected.removeprefix("Webcam "))
            else:
                self.source_type.set("Android stream")
            self.mode.set(mode_selection.get())
            self.source_type_changed()
            close_dialog()
            self.start_source()
            if mode_selection.get() == "Automatic AI" and self.running:
                self.root.after(1500, self._enable_ai_after_camera_start)

        if options:
            ttk.Button(
                buttons,
                text="Okay",
                command=start_selected,
            ).pack(side=tk.RIGHT)
        ttk.Button(
            buttons,
            text="Cancel" if options else "Close",
            command=close_dialog,
        ).pack(side=tk.RIGHT, padx=(0, 8) if options else (0, 0))
        dialog.update_idletasks()
        dialog_width = dialog.winfo_reqwidth()
        dialog_height = dialog.winfo_reqheight()
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        dialog_x = root_x + max(0, (root_width - dialog_width) // 2)
        dialog_y = root_y + max(0, (root_height - dialog_height) // 2)
        dialog.geometry(
            f"{dialog_width}x{dialog_height}+{dialog_x}+{dialog_y}"
        )
        dialog.grab_set()
        dialog.focus_force()

    def _run_camera_feed_check(self, stream_url: str) -> None:
        webcam_indices: list[int] = []
        for index in STARTUP_CAMERA_INDICES:
            if self.camera_probe_cancel.is_set():
                return
            capture = None
            try:
                capture = self.open_windows_camera(index)
                ok, frame = capture.read()
                if ok and frame is not None:
                    webcam_indices.append(index)
            except Exception:
                continue
            finally:
                if capture is not None:
                    capture.release()

        if self.camera_probe_cancel.is_set():
            return

        stream_state = "not configured"
        if stream_url:
            stream_capture = None
            try:
                stream_capture = cv2.VideoCapture(
                    stream_url,
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        STREAM_OPEN_TIMEOUT_MS,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        STREAM_READ_TIMEOUT_MS,
                    ],
                )
                if stream_capture.isOpened():
                    ok, frame = stream_capture.read()
                    stream_state = "online" if ok and frame is not None else "no frame"
                else:
                    stream_state = "unavailable"
            except Exception:
                stream_state = "unavailable"
            finally:
                if stream_capture is not None:
                    stream_capture.release()

        webcams = ", ".join(str(index) for index in webcam_indices) or "none"
        self._publish_startup_check(
            "cameras",
            f"Webcams {webcams}; Android stream {stream_state}.",
        )
        self.camera_selection_results.put((tuple(webcam_indices), stream_state))

    def _run_local_cv_check(self) -> None:
        try:
            tracker = self.create_tracker()
            del tracker
            cascade_path = (
                Path(cv2.data.haarcascades)
                / "haarcascade_frontalface_default.xml"
            )
            classifier = cv2.CascadeClassifier(str(cascade_path))
            if classifier.empty():
                raise RuntimeError("Haar face detector could not be loaded")
            classifier.detectMultiScale(
                np.zeros((160, 160), dtype=np.uint8),
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(24, 24),
            )
            eye_path = Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"
            smile_path = Path(cv2.data.haarcascades) / "haarcascade_smile.xml"
            eye_classifier = cv2.CascadeClassifier(str(eye_path))
            smile_classifier = cv2.CascadeClassifier(str(smile_path))
            if eye_classifier.empty() or smile_classifier.empty():
                raise RuntimeError("Eye or mouth cascade could not be loaded")
            status = (
                f"OpenCV {cv2.__version__}: CSRT and face/eye/mouth detectors ready."
            )
        except Exception as error:
            status = f"Unavailable ({type(error).__name__})."
        self._publish_startup_check("cv", status)

    def _run_provider_startup_check(
        self,
        provider_name: str,
        diagnostic_image: bytes | None,
        image_error: str,
    ) -> None:
        provider = vision_providers.create_provider(provider_name)
        if not provider.api_key:
            key_name = f"{provider_name.upper()}_API_KEY"
            self._publish_startup_check(
                provider_name,
                f"Skipped: {key_name} is not configured.",
            )
            return

        results: list[str] = []
        try:
            text = provider.probe_text()
            if "CAMERA_TEXT_OK" not in text.upper():
                raise RuntimeError("model returned an unexpected text response")
            results.append("text OK")
        except Exception as error:
            results.append(f"text error: {str(error).replace(chr(10), ' ')[:70]}")

        if diagnostic_image is None:
            results.append(f"image unavailable: {image_error}")
        else:
            try:
                caption = " ".join(provider.describe_image(diagnostic_image).split())
                if not caption:
                    raise RuntimeError("model returned an empty image label")
                results.append(f"image: {caption[:90]}")
            except Exception as error:
                results.append(
                    f"image error: {str(error).replace(chr(10), ' ')[:70]}"
                )

        self._publish_startup_check(provider_name, "; ".join(results))

    def _drain_startup_check_results(self) -> None:
        try:
            while True:
                key, message = self.startup_check_results.get_nowait()
                if self.startup_checks_cancelled:
                    continue
                status_var = self.startup_check_vars.get(key)
                if status_var is not None:
                    status_var.set(message)
                self.startup_checks_pending.discard(key)
                if self.startup_overlay_status is not None:
                    completed = 4 - len(self.startup_checks_pending)
                    self.startup_overlay_status.set(
                        f"Startup checks complete: {completed}/4"
                    )
        except queue.Empty:
            pass
        except tk.TclError:
            return
        if not self.startup_checks_pending:
            self._hide_startup_overlay()

    def _drain_camera_selection_results(self) -> None:
        try:
            while True:
                webcam_indices, stream_state = (
                    self.camera_selection_results.get_nowait()
                )
                if self.startup_checks_cancelled:
                    continue
                self.pending_camera_choices = (webcam_indices, stream_state)
        except queue.Empty:
            pass
        except tk.TclError:
            return
        if self.startup_overlay is None and self.pending_camera_choices is not None:
            webcam_indices, stream_state = self.pending_camera_choices
            self.pending_camera_choices = None
            self._show_camera_choices(webcam_indices, stream_state)
        try:
            self.root.after(150, self._drain_camera_selection_results)
        except tk.TclError:
            return
        try:
            self.root.after(150, self._drain_startup_check_results)
        except tk.TclError:
            return

    def _log(self, message: str) -> None:
        """Write diagnostics to both the app terminal and PowerShell/CMD."""
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        self.ai_terminal_queue.put(line)

    def _drain_terminal_queue(self) -> None:
        try:
            while True:
                line = self.ai_terminal_queue.get_nowait()
                self.terminal.configure(state=tk.NORMAL)
                self.terminal.insert(tk.END, line + "\n")
                self.terminal.see(tk.END)
                self.terminal.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        except tk.TclError:
            return
        self.root.after(100, self._drain_terminal_queue)

    def _draw_ai_boxes(self, source_frame: np.ndarray) -> None:
        """Draw the latest live detection from each provider in its own color."""
        now = time.monotonic()
        for provider, runtime in self.ai_runtime.items():
            if not self.ai_enabled.get() or provider not in self._enabled_providers():
                continue
            if runtime.last_box is None or runtime.last_result is None:
                continue
            if now - runtime.last_completed > AI_BOX_MAX_AGE_SECONDS:
                continue
            x, y, box_width, box_height = runtime.last_box
            color = AI_BOX_COLORS[provider]
            is_winner = provider == self.ai_winner and self.ai_enabled.get()
            target_label = runtime.last_result.label.strip()[:24] or "target"
            label = (
                f"{provider}: {target_label} [CAMERA] "
                f"{runtime.last_result.confidence:.0%}"
                if is_winner
                else f"{provider}: {target_label} "
                f"{runtime.last_result.confidence:.0%}"
            )
            thickness = 4 if is_winner else 2
            cv2.rectangle(
                source_frame,
                (x, y),
                (x + box_width, y + box_height),
                color,
                thickness,
            )
            cv2.putText(
                source_frame,
                label,
                (x, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def set_widget_state(widget, enabled: bool) -> None:
        widget.configure(state="normal" if enabled else "disabled")

    def browse_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.video_path.set(selected)

    def toggle_network_export(self) -> None:
        if self.network_export is not None:
            self.network_export.stop()
            self.network_export = None
            self.network_button.configure(text="Start network export")
            self.network_status.set("Network export: OFF")
            return

        try:
            port = int(self.network_port.get().strip())
            if not 1 <= port <= 65535:
                raise ValueError
            server = NetworkExportServer(port=port)
            server.start()
        except (ValueError, OSError) as error:
            self.network_status.set(f"Network export could not start: {error}")
            return

        self.network_export = server
        self.network_button.configure(text="Stop network export")
        self.network_status.set(
            f"Video: http://localhost:{port}/\n"
            f"Coordinates: http://localhost:{port}/coordinates\n"
            f"Live stream: http://localhost:{port}/coordinates/stream"
        )
        self._log(f"Network export started on port {port}.")

    def _update_network_export(
        self,
        virtual_frame: np.ndarray,
        source_width: int,
        source_height: int,
        crop_left: int,
        crop_top: int,
        crop_width: int,
        crop_height: int,
    ) -> None:
        if self.network_export is None:
            return
        ok, encoded = cv2.imencode(
            ".jpg",
            virtual_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        if not ok:
            return
        center_x = crop_left + crop_width / 2
        center_y = crop_top + crop_height / 2
        coordinates = {
            "source": {
                "width": source_width,
                "height": source_height,
            },
            "virtual_camera": {
                "left": crop_left,
                "top": crop_top,
                "width": crop_width,
                "height": crop_height,
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
            },
            "normalized": {
                "center_x": round(center_x / source_width, 6),
                "center_y": round(center_y / source_height, 6),
            },
            "motor": {
                "pan": round((center_x / source_width) * 2 - 1, 6),
                "tilt": round((center_y / source_height) * 2 - 1, 6),
            },
            "tracking": {
                "local_tracker": self.tracker is not None,
                "target_lost": self.tracking_lost,
                "provider": self.ai_winner,
            },
            "frame": self.video_frame_count,
            "target": {
                "x": round(self.target_x, 2),
                "y": round(self.target_y, 2),
            },
            "camera": {
                "x": round(self.camera_x, 2),
                "y": round(self.camera_y, 2),
            },
            "timestamp": time.time(),
        }
        self.network_export.update(encoded.tobytes(), coordinates)

    def start_source(self) -> None:
        self.camera_probe_cancel.set()
        self.stop_source(update_status=False)
        selected_mode = self.mode.get()

        kind = self.source_type.get()
        capture = None
        stage = "open"
        try:
            if kind == "Android stream":
                url = self.stream_url.get().strip()
                if not url:
                    raise ValueError("Enter the Android stream URL.")
                capture = cv2.VideoCapture(
                    url,
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        STREAM_OPEN_TIMEOUT_MS,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        STREAM_READ_TIMEOUT_MS,
                    ],
                )
            elif kind == "Webcam":
                index = int(self.camera_index.get().strip())
                capture = self.open_windows_camera(index)
            else:
                path = Path(self.video_path.get().strip())
                if not path.exists():
                    raise ValueError("Select an existing video file.")
                capture = cv2.VideoCapture(str(path))

            if not capture.isOpened():
                raise RuntimeError

            stage = "read"
            ok, first_frame = capture.read()
            if not ok or first_frame is None:
                raise RuntimeError

            self.capture = capture
            self.source_kind = kind
            self.frame = first_frame
            self.running = True
            self.clear_tracking(update_status=False)
            self.mode.set(selected_mode)
            self.mode_changed(update_status=False)
            self.reset_camera(update_status=False)
            self.status.set(
                "Source started. Click an object in Source Camera to track it."
            )
            self.process_frame()
        except ValueError as error:
            if capture is not None:
                capture.release()
            message = str(error)
            self.status.set(message)
            self._log(f"Source startup rejected ({kind}): {type(error).__name__}")
            messagebox.showerror("Unable to start source", message)
        except Exception as error:
            if capture is not None:
                capture.release()
            self.capture = None
            self.running = False
            message = self.source_failure_message(kind, stage)
            self.status.set(message)
            self._log(
                f"Source startup failed ({kind}, {stage}): "
                f"{type(error).__name__}"
            )
            messagebox.showerror("Unable to start source", message)

    @staticmethod
    def source_failure_message(kind: str, stage: str) -> str:
        if kind == "Android stream":
            if stage == "open":
                return (
                    "Could not connect to the Android stream. Check that the "
                    "device is online, the stream URL is correct, and both "
                    "devices are on the same network."
                )
            return (
                "The Android stream stopped returning frames. Check the "
                "device and network, then restart the source."
            )
        if kind == "Webcam":
            return (
                "Could not read from the webcam. Check the camera index, "
                "permissions, and whether another app is using the camera."
            )
        if stage == "open":
            return "Could not open the selected video file. Check that it is a supported video."
        return "The video file stopped returning frames and could not be restarted."

    @staticmethod
    def open_windows_camera(index: int) -> cv2.VideoCapture:
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            capture = cv2.VideoCapture(index, backend)
            if capture.isOpened():
                return capture
            capture.release()
        raise RuntimeError(
            f"Could not open webcam index {index}. "
            "Try index 0 or 1 and check Windows camera permissions."
        )

    def stop_source(self, update_status: bool = True) -> None:
        self.running = False
        self.ai_generation += 1
        for runtime in self.ai_runtime.values():
            runtime.next_check = 0.0
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self._log("Video source stopped; in-flight AI results will be ignored.")
        self._refresh_provider_statuses()
        if update_status:
            self.status.set("Source stopped.")

    @staticmethod
    def valid_tracking_box(
        box,
        frame_width: int,
        frame_height: int,
        previous_box: tuple[int, int, int, int] | None,
    ) -> bool:
        """Reject tracker results that are clearly outside or implausible."""
        x, y, box_width, box_height = [int(value) for value in box]
        if box_width < 10 or box_height < 10:
            return False
        if x + box_width < 0 or y + box_height < 0:
            return False
        if x >= frame_width or y >= frame_height:
            return False
        if box_width * box_height > frame_width * frame_height * 0.80:
            return False

        if previous_box is not None:
            old_x, old_y, old_width, old_height = previous_box
            old_center = (old_x + old_width / 2, old_y + old_height / 2)
            new_center = (x + box_width / 2, y + box_height / 2)
            center_jump = (
                (new_center[0] - old_center[0]) ** 2
                + (new_center[1] - old_center[1]) ** 2
            ) ** 0.5
            max_jump = max(frame_width, frame_height) * TARGET_MAX_CENTER_JUMP_RATIO
            if center_jump > max_jump:
                return False

        return True

    def handle_tracking_loss(
        self,
        message: str | None = None,
        disable_ai: bool = False,
    ) -> None:
        """Hold position and return control to the manual safety mode."""
        self.tracker = None
        self.tracking_lost = True
        if disable_ai:
            self.ai_enabled.set(False)
        self.auto_follow.set(False)
        self.target_x = self.camera_x
        self.target_y = self.camera_y
        self.mode.set("Manual control")
        self.mode_changed(update_status=False)
        self.status.set(
            message
            or "Tracking lost. Camera is holding position; Manual mode is active."
        )

    def _try_reacquire_target(self, frame: np.ndarray) -> bool:
        reference = self.target_reference
        if reference is None or reference.reference_crop is None:
            return False

        now = time.monotonic()
        if now < self.next_target_reacquire:
            return False
        self.next_target_reacquire = now + TARGET_REACQUIRE_INTERVAL_SECONDS

        frame_height, frame_width = frame.shape[:2]
        search_frame = self._prepare_ai_frame(frame)
        scale_x = search_frame.shape[1] / frame_width
        scale_y = search_frame.shape[0] / frame_height
        reference_height, reference_width = reference.reference_crop.shape[:2]
        base_width = max(1, int(round(reference_width * scale_x)))
        base_height = max(1, int(round(reference_height * scale_y)))
        template = cv2.resize(
            reference.reference_crop,
            (base_width, base_height),
            interpolation=cv2.INTER_AREA,
        )
        match = target_tracking.find_template_match(search_frame, template)
        if match is None:
            return False
        best_box, best_score = match
        if best_score < TARGET_REACQUIRE_TEMPLATE_THRESHOLD:
            return False

        candidate_box = self._scale_box_between_frames(
            best_box,
            search_frame.shape,
            frame.shape,
        )
        if not self.valid_tracking_box(
            candidate_box,
            frame_width,
            frame_height,
            None,
        ):
            return False

        appearance_score = reference.appearance_score(frame, candidate_box)
        if appearance_score < TARGET_REACQUIRE_APPEARANCE_THRESHOLD:
            return False

        self._initialize_local_tracker(candidate_box)
        if self.tracker is None:
            return False
        self.tracking_lost = False
        self.target_mismatch_frames = 0
        self.target_match_score = appearance_score
        self.target_x, self.target_y = self._box_center(candidate_box)
        self._update_target_status(appearance_score)
        self._update_human_match()
        self.status.set(
            f"Target reacquired locally (appearance {appearance_score:.0%})."
        )
        self._log(
            f"Target reacquired locally: template={best_score:.0%} "
            f"appearance={appearance_score:.0%} "
            f"box=({candidate_box[0]},{candidate_box[1]},"
            f"{candidate_box[2]},{candidate_box[3]})"
        )
        return True

    def process_frame(self) -> None:
        if not self.running or self.capture is None:
            return

        source_kind = self.source_kind
        try:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                if source_kind == "Video file":
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.capture.read()
        except Exception as error:
            self._log(
                f"Frame read failed ({source_kind}): {type(error).__name__}"
            )
            self.stop_source(update_status=False)
            self.status.set(self.source_failure_message(source_kind, "read"))
            return

        if not ok or frame is None:
            self.stop_source(update_status=False)
            self.status.set(self.source_failure_message(source_kind, "read"))
            return

        self.frame = frame
        self.video_frame_count += 1
        self._update_human_detection(frame)
        self._maybe_start_human_tracking()
        self._drain_ai_results()
        height, width = frame.shape[:2]

        if self.tracker is not None:
            tracking_frame = self._prepare_ai_frame(frame)
            try:
                ok, tracking_box = self.tracker.update(tracking_frame)
                updated_box = (
                    self._scale_box_between_frames(
                        tuple(int(value) for value in tracking_box),
                        tracking_frame.shape,
                        frame.shape,
                    )
                    if ok and tracking_box is not None
                    else None
                )
            except cv2.error as error:
                self._log(f"Local tracker update failed: {type(error).__name__}")
                ok, updated_box = False, None
            box_is_valid = (
                ok
                and updated_box is not None
                and self.valid_tracking_box(
                    updated_box,
                    width,
                    height,
                    self.tracking_box,
                )
            )
            candidate_match = (
                self._target_match_score(
                    tuple(int(value) for value in updated_box)
                )
                if box_is_valid
                else None
            )
            if candidate_match is not None:
                self._update_target_status(candidate_match)
                if candidate_match < TARGET_MATCH_REJECT_THRESHOLD:
                    self.target_mismatch_frames += 1
                else:
                    self.target_mismatch_frames = 0
                if self.target_mismatch_frames >= TARGET_MATCH_LOST_FRAME_LIMIT:
                    box_is_valid = False
                    self._log(
                        "Local tracker rejected: target appearance mismatch "
                        f"({candidate_match:.0%})"
                    )

            if box_is_valid:
                x, y, box_width, box_height = [
                    int(value) for value in updated_box
                ]
                self.tracking_box = (x, y, box_width, box_height)
                self.tracking_lost = False
                self._update_human_match()
                if self.auto_follow.get():
                    self.target_x = x + box_width / 2
                    self.target_y = y + box_height / 2
            else:
                self.tracker = None
                self.tracking_lost = True
                if self.mode.get() == "Automatic AI" and self.ai_enabled.get():
                    self._log(
                        "Local tracker lost the target; holding the last AI "
                        "camera target until the next provider result."
                    )
                    self._start_ai_after_tracking_loss()
                else:
                    self.handle_tracking_loss()
                    self._start_ai_after_tracking_loss()

        if self.tracker is None and self.tracking_lost:
            self._try_reacquire_target(frame)
        self._maybe_start_ai_requests(frame)

        aspect_ratio = 16 / 9
        base_crop_width = min(int(width * 0.60), width)
        base_crop_height = min(int(base_crop_width / aspect_ratio), height)
        base_crop_width = int(base_crop_height * aspect_ratio)
        crop_width = max(80, min(int(base_crop_width / self.zoom), width))
        crop_height = max(45, min(int(base_crop_height / self.zoom), height))
        crop_width = min(crop_width, int(crop_height * aspect_ratio))
        crop_height = min(crop_height, int(crop_width / aspect_ratio))

        # Keep compact head/face targets above center so the virtual crop
        # includes the neck and shoulders instead of cutting them off.
        if self.tracking_box is not None and not self.tracking_lost:
            _, _, _, tracked_height = self.tracking_box
            if tracked_height < crop_height * 0.70:
                self.target_y += crop_height * PORTRAIT_VERTICAL_FRAMING_BIAS

        self.target_x = max(
            crop_width // 2,
            min(self.target_x, width - crop_width // 2),
        )
        self.target_y = max(
            crop_height // 2,
            min(self.target_y, height - crop_height // 2),
        )
        self.camera_x += (self.target_x - self.camera_x) * float(
            self.smoothing.get()
        )
        self.camera_y += (self.target_y - self.camera_y) * float(
            self.smoothing.get()
        )

        camera_x = int(round(self.camera_x))
        camera_y = int(round(self.camera_y))
        left = camera_x - crop_width // 2
        top = camera_y - crop_height // 2

        network_frame = frame[top : top + crop_height, left : left + crop_width]
        self._update_network_export(
            network_frame,
            width,
            height,
            left,
            top,
            crop_width,
            crop_height,
        )
        virtual_frame = cv2.resize(
            network_frame,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        source_frame = frame.copy()
        cv2.rectangle(
            source_frame,
            (left, top),
            (left + crop_width, top + crop_height),
            (0, 255, 0),
            3,
        )
        cv2.putText(
            source_frame,
            "Green = virtual camera view",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        self._draw_ai_boxes(source_frame)
        self._draw_human_faces(source_frame)

        if self.tracking_box is not None:
            x, y, box_width, box_height = self.tracking_box
            box_color = (0, 0, 255) if self.tracking_lost else (0, 255, 255)
            label = "LOCAL TRACKER LOST" if self.tracking_lost else "LOCAL TRACKER"
            cv2.rectangle(
                source_frame,
                (x, y),
                (x + box_width, y + box_height),
                box_color,
                3,
            )
            cv2.putText(
                source_frame,
                label,
                (x, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2,
                cv2.LINE_AA,
            )

        source_frame = cv2.resize(
            source_frame,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        if self.selection_preview_box is not None:
            left, top, right, bottom = self.selection_preview_box
            cv2.rectangle(
                source_frame,
                (left, top),
                (right, bottom),
                (255, 255, 0),
                2,
            )
            preview_label = (
                "RESIZE TARGET"
                if self.selection_drag_mode == "resize"
                else "NEW TARGET"
            )
            cv2.putText(
                source_frame,
                preview_label,
                (left, max(20, top - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        self.show_frame(self.source_canvas, source_frame, "source")
        self.show_frame(self.virtual_canvas, virtual_frame, "virtual")

        now = time.monotonic()
        if now >= self.ai_last_video_log:
            self.ai_last_video_log = now + VIDEO_LOG_INTERVAL_SECONDS
            active_provider = self.ai_winner or "none"
            self._log(
                f"FRAME #{self.video_frame_count} size={width}x{height} "
                f"tracker={'ok' if self.tracker is not None else 'lost'} "
                f"camera_target=({self.target_x:.0f},{self.target_y:.0f}) "
                f"ai_winner={active_provider}"
            )

        self.after_id = self.root.after(30, self.process_frame)

    def _draw_human_faces(self, source_frame: np.ndarray) -> None:
        """Draw local face detections without implying identity recognition."""
        if not self.track_human.get():
            return
        for face_x, face_y, face_width, face_height in self.face_boxes:
            color = (80, 220, 120)
            cv2.rectangle(
                source_frame,
                (face_x, face_y),
                (face_x + face_width, face_y + face_height),
                color,
                2,
            )
            label = "LOCAL FACE"
            if self.human_match_score is not None:
                label += f" / MATCH {self.human_match_score:.0%}"
            cv2.putText(
                source_frame,
                label,
                (face_x, max(22, face_y - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv2.LINE_AA,
            )
        for eye_x, eye_y, eye_width, eye_height in self.eye_boxes:
            cv2.rectangle(
                source_frame,
                (eye_x, eye_y),
                (eye_x + eye_width, eye_y + eye_height),
                (255, 180, 0),
                2,
            )
        for smile_x, smile_y, smile_width, smile_height in self.smile_boxes:
            cv2.rectangle(
                source_frame,
                (smile_x, smile_y),
                (smile_x + smile_width, smile_y + smile_height),
                (0, 165, 255),
                2,
            )

    def show_frame(self, canvas: tk.Canvas, frame: np.ndarray, name: str) -> None:
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width() or DISPLAY_WIDTH
        canvas_height = canvas.winfo_height() or DISPLAY_HEIGHT
        frame_height, frame_width = frame.shape[:2]
        scale = min(
            canvas_width / frame_width,
            canvas_height / frame_height,
        )
        image_width = max(1, int(round(frame_width * scale)))
        image_height = max(1, int(round(frame_height * scale)))
        offset_x = (canvas_width - image_width) // 2
        offset_y = (canvas_height - image_height) // 2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        if (image_width, image_height) != image.size:
            image = image.resize(
                (image_width, image_height),
                Image.Resampling.LANCZOS,
            )
        photo = ImageTk.PhotoImage(image=image)
        canvas.delete("all")
        canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=photo)
        self.canvas_image_rectangles[name] = (
            offset_x,
            offset_y,
            image_width,
            image_height,
        )
        if name == "source":
            self.source_image = photo
        else:
            self.virtual_image = photo

    def _canvas_point(self, event) -> tuple[int, int]:
        left, top, image_width, image_height = self.canvas_image_rectangles.get(
            "source",
            (0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT),
        )
        return (
            max(
                0,
                min(
                    DISPLAY_WIDTH - 1,
                    round((int(event.x) - left) * DISPLAY_WIDTH / image_width),
                ),
            ),
            max(
                0,
                min(
                    DISPLAY_HEIGHT - 1,
                    round((int(event.y) - top) * DISPLAY_HEIGHT / image_height),
                ),
            ),
        )

    @staticmethod
    def _canvas_rectangle(
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        return (
            min(start[0], end[0]),
            min(start[1], end[1]),
            max(start[0], end[0]),
            max(start[1], end[1]),
        )

    def _canvas_box_to_frame(
        self,
        rectangle: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        left, top, right, bottom = rectangle
        if right - left < 6 or bottom - top < 6 or self.frame is None:
            return None

        frame_height, frame_width = self.frame.shape[:2]
        box_x = round(left * frame_width / DISPLAY_WIDTH)
        box_y = round(top * frame_height / DISPLAY_HEIGHT)
        box_right = round(right * frame_width / DISPLAY_WIDTH)
        box_bottom = round(bottom * frame_height / DISPLAY_HEIGHT)
        box_width = max(20, box_right - box_x)
        box_height = max(20, box_bottom - box_y)
        if box_width > frame_width or box_height > frame_height:
            return None
        box_x = max(0, min(box_x, frame_width - box_width))
        box_y = max(0, min(box_y, frame_height - box_height))
        return box_x, box_y, box_width, box_height

    def _on_source_button_press(self, event) -> None:
        if self.frame is None:
            return

        modifiers = int(getattr(event, "state", 0))
        control_pressed = bool(modifiers & CONTROL_MASK)
        shift_pressed = bool(modifiers & SHIFT_MASK)
        if control_pressed and shift_pressed:
            if self.tracking_box is None:
                self.status.set("Select an object before resizing its box.")
                return
            self.selection_drag_mode = "resize"
            self.status.set(
                "Drag a replacement box around the same target; its reference will be kept."
            )
        elif control_pressed:
            self.selection_drag_mode = "new"
            self.status.set("Drag to draw a tracking box around the target.")
        else:
            self.selection_drag_mode = None
            self.select_from_click(event)
            return

        self.selection_drag_start = self._canvas_point(event)
        self.selection_drag_current = self.selection_drag_start
        self.selection_preview_box = self._canvas_rectangle(
            self.selection_drag_start,
            self.selection_drag_current,
        )

    def _on_source_mouse_drag(self, event) -> None:
        if self.selection_drag_mode is None or self.selection_drag_start is None:
            return
        self.selection_drag_current = self._canvas_point(event)
        self.selection_preview_box = self._canvas_rectangle(
            self.selection_drag_start,
            self.selection_drag_current,
        )

    def _on_source_button_release(self, event) -> None:
        mode = self.selection_drag_mode
        start = self.selection_drag_start
        if mode is None or start is None:
            return

        end = self._canvas_point(event)
        rectangle = self._canvas_rectangle(start, end)
        self.selection_drag_mode = None
        self.selection_drag_start = None
        self.selection_drag_current = None
        self.selection_preview_box = None

        pixel_box = self._canvas_box_to_frame(rectangle)
        if pixel_box is None:
            self.status.set("Drag a larger rectangle to create a tracking box.")
            return
        self._activate_tracking_box(
            pixel_box,
            preserve_reference=(mode == "resize"),
        )

    def _activate_tracking_box(
        self,
        pixel_box: tuple[int, int, int, int],
        preserve_reference: bool,
    ) -> None:
        if self.frame is None:
            return

        ai_target_selection = (
            self.mode.get() == "Automatic AI"
            and self.ai_enabled.get()
        )
        face_target_selection = self.mode.get() == "Face tracking"
        if (
            not preserve_reference
            and not ai_target_selection
            and not face_target_selection
        ):
            self.mode.set("Assisted tracking")
            self.mode_changed(update_status=False)
        try:
            tracker = self.create_tracker()
            tracker_frame = self._prepare_ai_frame(self.frame)
            tracker_box = self._scale_box_between_frames(
                pixel_box,
                self.frame.shape,
                tracker_frame.shape,
            )
            initialized = tracker.init(tracker_frame, tracker_box)
            if initialized is False:
                raise RuntimeError("OpenCV could not initialize the tracker.")
        except Exception as error:
            self._log(f"Tracker initialization failed: {type(error).__name__}")
            self.status.set("Could not initialize tracking for that box.")
            messagebox.showerror(
                "Tracker unavailable",
                "Could not initialize tracking for that box. Check the selected area and try again.",
            )
            return

        self.tracker = tracker
        self.tracking_box = pixel_box
        self.tracking_lost = False
        self.target_mismatch_frames = 0
        self.next_target_reacquire = 0.0
        self.target_x = pixel_box[0] + pixel_box[2] / 2
        self.target_y = pixel_box[1] + pixel_box[3] / 2

        if preserve_reference:
            if self.target_reference is not None:
                score = self._target_match_score(pixel_box)
                self._update_target_status(score)
            self.status.set(
                "Tracking box resized; the selected target reference was kept."
            )
        else:
            self.auto_follow.set(True)
            self._capture_target_reference(
                pixel_box,
                target_kind="human" if self.track_human.get() else "generic",
            )
            self.status.set(
                (
                    "Object selected. AI remains active and will correct this target."
                    if ai_target_selection
                    else "Object selected. Reference captured and automatic follow enabled."
                )
            )
        self._update_human_match()

    def select_from_click(self, event) -> None:
        """Create a tracker box centered at a click in the source canvas."""
        if self.frame is None:
            return

        frame_height, frame_width = self.frame.shape[:2]
        click_x, click_y = self._canvas_point(event)
        center_x = click_x * frame_width / DISPLAY_WIDTH
        center_y = click_y * frame_height / DISPLAY_HEIGHT

        box_width = max(20, int(frame_width * CLICK_TRACK_BOX_WIDTH_RATIO))
        box_height = max(20, int(frame_height * CLICK_TRACK_BOX_HEIGHT_RATIO))
        box_x = int(center_x - box_width / 2)
        box_y = int(center_y - box_height / 2)
        box_x = max(0, min(box_x, frame_width - box_width))
        box_y = max(0, min(box_y, frame_height - box_height))
        self._activate_tracking_box(
            (box_x, box_y, box_width, box_height),
            preserve_reference=False,
        )

    @staticmethod
    def create_tracker():
        legacy = getattr(cv2, "legacy", None)
        for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
            factory = getattr(cv2, name, None) or getattr(legacy, name, None)
            if factory is not None:
                return factory()
        raise RuntimeError(
            "Install opencv-contrib-python in the active virtual environment."
        )

    def clear_tracking(self, update_status: bool = True) -> None:
        self.tracker = None
        self.tracking_box = None
        self.tracking_lost = False
        self.target_reference = None
        self.target_match_score = 0.0
        self.target_mismatch_frames = 0
        self.next_target_reacquire = 0.0
        self.face_boxes = ()
        self.eye_boxes = ()
        self.smile_boxes = ()
        self.human_detection_error = ""
        self.next_human_detection = 0.0
        self.human_match_score = None
        self.target_status.set("Target reference: none")
        self._update_human_status()
        for runtime in self.ai_runtime.values():
            runtime.last_match_score = None
            runtime.last_human_match_score = None
        self.mode.set("Manual control")
        self.mode_changed(update_status=False)
        if update_status:
            self.status.set("Tracking cleared. Click an object to select it.")

    def nudge(self, horizontal: int, vertical: int) -> None:
        if self.frame is None:
            return
        height, width = self.frame.shape[:2]
        step = max(12, width // 25)
        self.target_x += horizontal * step
        self.target_y += vertical * step
        self.mode.set("Manual control")
        self.mode_changed(update_status=False)
        self.status.set("Manual camera control enabled.")

    def change_zoom(self, amount: float) -> None:
        self.zoom = max(0.6, min(3.0, self.zoom + amount))

    def reset_camera(self, update_status: bool = True) -> None:
        if self.frame is not None:
            height, width = self.frame.shape[:2]
            self.camera_x = float(width // 2)
            self.camera_y = float(height // 2)
            self.target_x = self.camera_x
            self.target_y = self.camera_y
        self.zoom = 1.0
        if update_status:
            self.status.set("Camera position and zoom reset.")

    def close(self) -> None:
        self.camera_probe_cancel.set()
        self.stop_source(update_status=False)
        if self.network_export is not None:
            self.network_export.stop()
            self.network_export = None
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PresenterCameraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()