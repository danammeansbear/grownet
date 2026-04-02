import cv2
import tkinter as tk
from tkinter import ttk
from plantcv import plantcv as pcv
from PIL import Image, ImageTk
import pandas as pd
import logging
import json
import pickle
import copy
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from tkinter import filedialog, messagebox
import os
import threading
import time
import queue
import serial
import serial.tools.list_ports
from datetime import datetime, timedelta


MAX_CAMERA_SLOTS = 20
DEFAULT_VISIBLE_CAMERA_SLOTS = 4
CAMERA_PANEL_COLUMNS = 4
DEFAULT_AUTO_ANALYSIS_INTERVAL_SECONDS = 5
# On Windows try MSMF (built-in cameras) then DSHOW; both avoid the obsensor errors from CAP_ANY
_CV2_CAM_BACKENDS = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if os.name == "nt" else [cv2.CAP_ANY]
PLANTCV_PIPELINES = [
    "Get ROI Info",
    "Photosynthetic Analysis",
    "Health Status Analysis",
    "Growth Rate Analysis",
    "Nutrient Deficiency Detection",
    "Machine Learning Detection",
    "Plant Morphology Analysis",
]

# ── Greenhouse SaaS colour palette ────────────────────────────────────────────
THEME = {
    "bg":             "#0f1f0f",   # root / page background
    "panel":          "#152615",   # toolbar / notebook panel strip
    "card":           "#1e3a1e",   # camera / sensor card background
    "card_header":    "#2a4d2a",   # card header strip
    "border":         "#3a5c3a",   # card / widget borders
    "primary":        "#4caf50",   # primary green accent
    "primary_dk":     "#2e7d32",   # dark primary (button bg)
    "primary_lt":     "#81c784",   # light primary (headings, labels)
    "accent":         "#00e676",   # bright accent (live indicators)
    "text":           "#e8f5e9",   # main text (near-white)
    "text_muted":     "#a5d6a7",   # secondary / label text
    "text_dim":       "#567a56",   # dim / placeholder text
    "warning":        "#ffca28",   # warning state
    "error":          "#ef5350",   # error / disconnected
    "success":        "#66bb6a",   # connected / streaming
    "treeview_row":   "#1e3a1e",   # treeview row background
    "treeview_sel":   "#2e7d32",   # treeview selected row
    "scrollbar":      "#243c24",   # scrollbar trough
    "entry":          "#152615",   # combobox / spinbox / entry bg
    "graph_bg":       "#111f11",   # matplotlib figure background
    "graph_ax":       "#172a17",   # matplotlib axes background
    "gauge_arc":      "#4caf50",   # gauge active arc
    "gauge_track":    "#2a4d2a",   # gauge background ring
    "gauge_bg":       "#1e3a1e",   # gauge canvas background
}

# ── Leafy AI Copilot constants ────────────────────────────────────────────────
_LEAFY_SYSTEM = (
    "You are Leafy \U0001f331, a friendly and expert greenhouse crop-steering AI assistant "
    "embedded in the GreenSight Plant Intelligence Platform. "
    "You help growers optimise their crops using environmental, hydroponic, and plant "
    "health sensor data. You specialise in:\n"
    "\u2022 Crop steering (vegetative \u2194 generative balance via EC, temperature, VPD, light)\n"
    "\u2022 VPD management (leaf-surface vapour pressure deficit, optimal 0.8\u20131.2\u00a0kPa)\n"
    "\u2022 DLI optimisation (Daily Light Integral, target 12\u201330\u00a0mol/m\u00b2/day)\n"
    "\u2022 Hydroponic EC/pH strategy (pH 5.5\u20136.5; EC by crop stage)\n"
    "\u2022 Nutrient deficiency diagnosis from colour/spectral metrics\n"
    "\u2022 IPM and environmental threats\n"
    "\nGuidelines:\n"
    "\u2022 Be concise, practical, and priority-ranked. Use bullet points and emojis.\n"
    "\u2022 When you see out-of-range readings, flag them first with severity (\U0001f7e1 warning / \U0001f534 critical).\n"
    "\u2022 Always provide an actionable next step for each issue.\n"
    "\u2022 Speak like a knowledgeable but friendly grower, not a textbook."
)

_LEAFY_GREETING = (
    "Hi there! \U0001f44b  I\u2019m Leafy, your greenhouse crop-steering Copilot.\n\n"
    "I can help you with:\n"
    "\u2022 \U0001f4ca Analyze All Sensors \u2014 instant crop-health snapshot\n"
    "\u2022 \U0001f4a7 Hydroponic EC / pH / nutrient advice\n"
    "\u2022 \U0001f321 VPD and climate steering\n"
    "\u2022 \U0001f33f Plant health and deficiency diagnosis\n\n"
    "Paste your OpenAI API key above for AI-powered advice, or click "
    "\"Analyze Now\" for instant rule-based guidance!"
)


class ScrollableFrame(tk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        _bg = self.cget('bg')
        self.canvas = tk.Canvas(self, bg=_bg, highlightthickness=0)
        self.scrollbar_y = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar_x = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=_bg)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar_y.pack(side="right", fill="y")
        self.scrollbar_x.pack(side="bottom", fill="x")

        self.canvas.bind("<ButtonPress-1>", self._start_pan)
        self.canvas.bind("<B1-Motion>", self._pan_canvas)

        self.zoom_scale = 1.0

    def zoom(self, scale_factor):
        self.zoom_scale *= scale_factor
        self.canvas.scale("all", 0, 0, scale_factor, scale_factor)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _pan_canvas(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)


class CircularGauge(tk.Canvas):
    """Custom circular gauge widget for displaying environmental data"""
    def __init__(self, parent, title, min_value, max_value, **kwargs):
        kwargs.setdefault('bg', THEME["gauge_bg"])
        kwargs.setdefault('highlightthickness', 0)
        super().__init__(parent, **kwargs)
        self.title = title
        self.min_value = min_value
        self.max_value = max_value
        self.value = min_value
        self.width = kwargs.get('width', 160)
        self.height = kwargs.get('height', 160)
        self.configure(width=self.width, height=self.height)
        self.draw()

    def set_value(self, value):
        self.value = max(self.min_value, min(self.max_value, value))
        self.draw()

    def draw(self):
        T = THEME
        self.delete("all")
        self.configure(bg=T["gauge_bg"])

        cx, cy = self.width / 2, self.height / 2
        r = (self.width - 44) / 2
        start = 135
        span = 270

        ratio = (self.value - self.min_value) / max(self.max_value - self.min_value, 1)
        ratio = max(0.0, min(1.0, ratio))

        # Background track arc
        self.create_arc(cx - r, cy - r, cx + r, cy + r,
                        start=start, extent=span,
                        outline=T["gauge_track"], width=9, style="arc")

        # Value arc
        if ratio > 0:
            self.create_arc(cx - r, cy - r, cx + r, cy + r,
                            start=start, extent=span * ratio,
                            outline=T["gauge_arc"], width=9, style="arc")

        # Inner circle rim
        self.create_oval(cx - r + 14, cy - r + 14, cx + r - 14, cy + r - 14,
                         outline=T["border"], width=1)

        # Value text
        self.create_text(cx, cy - 4,
                         text=f"{self.value:.1f}",
                         font=("Segoe UI", 14, "bold"), fill=T["accent"])

        # Unit / title text
        self.create_text(cx, cy + 16,
                         text=self.title,
                         font=("Segoe UI", 7), fill=T["text_muted"])

        # Min / max labels
        self.create_text(cx - r * 0.82, cy + r * 0.58,
                         text=str(self.min_value), font=("Segoe UI", 7), fill=T["text_dim"])
        self.create_text(cx + r * 0.82, cy + r * 0.58,
                         text=str(self.max_value), font=("Segoe UI", 7), fill=T["text_dim"])


class PlantDetectionDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("GreenSight — Plant Intelligence Platform")
        self.is_closing = False
        self.max_camera_slots = MAX_CAMERA_SLOTS
        self.visible_camera_slots = DEFAULT_VISIBLE_CAMERA_SLOTS
        self.camera_panel_columns = CAMERA_PANEL_COLUMNS
        self.camera_probe_limit = MAX_CAMERA_SLOTS
        self.analysis_options = PLANTCV_PIPELINES

        # Apply greenhouse SaaS theme before building any widgets
        self._apply_theme()

        # Branded header bar (above the notebook)
        self._create_header()

        # Create tabbed interface
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.plant_tab   = ttk.Frame(self.notebook)
        self.env_tab     = ttk.Frame(self.notebook)
        self.sensor_tab  = ttk.Frame(self.notebook)
        self.comm_tab    = ttk.Frame(self.notebook)
        self.hist_tab    = ttk.Frame(self.notebook)
        self.hydro_tab   = ttk.Frame(self.notebook)
        self.auto_tab    = ttk.Frame(self.notebook)
        self.morph_tab   = ttk.Frame(self.notebook)

        self.notebook.add(self.plant_tab,  text='  🌿  Plant Detection  ')
        self.notebook.add(self.env_tab,    text='  📊  Environmental Data  ')
        self.notebook.add(self.sensor_tab, text='  🔧  Sensor Config  ')
        self.notebook.add(self.comm_tab,   text='  🗣  Plant Communication  ')
        self.notebook.add(self.hist_tab,   text='  📋  Plant History  ')
        self.notebook.add(self.hydro_tab,  text='  💧  Hydroponics  ')
        self.notebook.add(self.auto_tab,   text='  ⚡  Automations  ')
        self.notebook.add(self.morph_tab,  text='  🫘  3D Morphology  ')
        
        # Plant Detection Tab Variables — cameras populated asynchronously after UI is shown
        self.cameras = []
        self.visible_camera_slots = DEFAULT_VISIBLE_CAMERA_SLOTS
        self.camera_streams = [None] * self.max_camera_slots
        self.is_running = [False] * self.max_camera_slots
        self.grids = []
        self.last_update_time = [datetime.min] * self.max_camera_slots
        self.analysis_completed = [False] * self.max_camera_slots
        self.update_threads = [None] * self.max_camera_slots
        self.display_threads = [None] * self.max_camera_slots
        self.analysis_threads = [None] * self.max_camera_slots
        self.analysis_in_progress = [False] * self.max_camera_slots
        self.last_analysis_time = [0.0] * self.max_camera_slots
        self.growth_history = [[] for _ in range(self.max_camera_slots)]
        self.latest_analysis_results = [pd.DataFrame() for _ in range(self.max_camera_slots)]
        self.ml_model = None
        self.ml_model_path = None
        self.ml_feature_columns = [
            "area_px",
            "perimeter_px",
            "aspect_ratio",
            "solidity",
            "extent",
            "greenness",
            "exg_index",
            "yellow_ratio",
            "brown_ratio",
            "purple_ratio",
        ]
        
        # Stream optimization variables
        self.frame_queues = [queue.Queue(maxsize=2) for _ in range(self.max_camera_slots)]
        self.stream_quality = [50] * self.max_camera_slots
        self.stream_fps = [5] * self.max_camera_slots
        self.last_frame_time = [0] * self.max_camera_slots
        self.current_frames = [None] * self.max_camera_slots
        self.stream_active = [False] * self.max_camera_slots
        
        # Environmental Tab Variables
        self.serial_port = None
        self.serial_thread = None
        self.is_serial_running = False
        self.env_data = {
            'temperature': [],
            'humidity': [],
            'co2': [],
            'light': [],
            'soil_moisture': [],
            'time': []
        }
        self.env_data_lock = threading.Lock()
        self.plantcv_lock = threading.RLock()
        self.max_history_points = 100  # Store last 100 readings
        self.env_graph_update_pending = False
        self.history_window_var = tk.StringVar(value=str(self.max_history_points))

        # ── Sensor Config Tab State ──────────────────────────────────────────
        # Each sensor entry: {name, port, baud, topic, camera_panel, enabled}
        self.sensor_entries = []        # list of dicts holding tk.StringVar / tk.BooleanVar per row
        self.sensor_rows_frame = None   # populated by create_sensor_config_widgets

        # RabbitMQ connection vars
        self.rmq_host_var     = tk.StringVar(value="localhost")
        self.rmq_port_var     = tk.StringVar(value="5672")
        self.rmq_vhost_var    = tk.StringVar(value="/")
        self.rmq_user_var     = tk.StringVar(value="guest")
        self.rmq_pass_var     = tk.StringVar(value="guest")
        self.rmq_exchange_var = tk.StringVar(value="greensight")
        self.rmq_status_var   = tk.StringVar(value="Not connected")
        self._rmq_connection  = None
        self._rmq_channel     = None

        # ── Plant Communication Tab State ────────────────────────────────────
        # Per-camera health snapshot: list of dicts, one per max_camera_slots
        # Each dict: {ndvi, greenness, exg, yellow_ratio, brown_ratio, purple_ratio,
        #             nutrient_flag, health_score, health_label, advice, rois, timestamp}
        self.comm_panels     = []        # list of UI widget dicts (built by create_plant_comm_widgets)
        self.comm_data       = [None] * MAX_CAMERA_SLOTS  # latest health snapshot per slot
        self.comm_auto_var   = tk.BooleanVar(value=True)  # auto-update toggle

        # ── Plant History Tab State ──────────────────────────────────────────
        self.plant_history         = []            # list of history record dicts
        self.hist_tree             = None          # ttk.Treeview for the audit log
        self.hist_detail_tree      = None          # ttk.Treeview for per-ROI detail
        self.hist_filter_panel_var = tk.StringVar(value="All Panels")
        self.hist_filter_type_var  = tk.StringVar(value="All Types")
        self.hist_search_var       = tk.StringVar(value="")
        self.hist_row_count_var    = tk.StringVar(value="0 records")
        self._hist_stat_total      = tk.StringVar(value="—")
        self._hist_stat_today      = tk.StringVar(value="—")
        self._hist_stat_avg        = tk.StringVar(value="—")
        self._hist_stat_last       = tk.StringVar(value="—")
        self._hist_sort_rev        = {}

        # Forecast sub-panel state
        self.hist_forecast_panel_var   = tk.StringVar(value="All Panels")
        self.hist_forecast_metric_var  = tk.StringVar(value="Health Score")
        self.hist_forecast_horizon_var = tk.IntVar(value=6)
        self.hist_forecast_fig         = None
        self.hist_forecast_canvas      = None
        self.hist_forecast_fcst_tree   = None   # predictions table widget

        # ── Camera Section / Role assignment ─────────────────────────────────
        # Section i  =>  overhead = slot (2*i),  side/canopy = slot (2*i + 1)
        # Role is auto-derived from slot index; no manual config needed.
        self._section_pair_label = None   # tk.Label written by set_visible_sections

        # Latest morphology data per section for 3D reconstruction
        # key = section number (1-based int)
        # value = {"overhead": pd.DataFrame, "side": pd.DataFrame, "timestamp": str}
        self.morph_section_data = {}

        # ── 3D Morphology viewer state ────────────────────────────────────────
        self._morph_fig          = None   # matplotlib Figure
        self._morph_canvas       = None   # FigureCanvasTkAgg
        self._morph_ax           = None   # Axes3D
        self._morph_azim         = tk.DoubleVar(value=-60)
        self._morph_elev         = tk.DoubleVar(value=25)
        self._morph_section_var  = tk.StringVar(value="All Sections")
        self._morph_loaded_file  = tk.StringVar(value="")
        self._morph_show_axes    = tk.BooleanVar(value=True)
        self._morph_show_grid    = tk.BooleanVar(value=True)
        self._morph_dragging     = False
        self._morph_drag_start   = None

        # ── DJI / RTMP drone stream registry ─────────────────────────────────
        # Each entry: {"label": str, "url": str, "type": "rtmp"|"youtube"|"custom"}
        # These are added to self.cameras alongside physical USB/MSMF cameras.
        self.drone_streams = []     # authoritative list; cameras list is rebuilt from this + local cams

        # ── Hydroponics Tab State ────────────────────────────────────
        # Nutrient channels tracked
        HYDRO_CHANNELS = ['ph', 'ec', 'tds', 'water_temp', 'do', 'orp', 'nitrogen', 'phosphorus', 'potassium']
        self.hydro_channels    = HYDRO_CHANNELS
        self.hydro_history     = {ch: [] for ch in HYDRO_CHANNELS}
        self.hydro_time        = []
        self.hydro_lock        = threading.Lock()
        self.hydro_serial_port = None
        self.hydro_serial_thread = None
        self.hydro_serial_running = False
        self.hydro_gauges      = {}      # channel -> CircularGauge widget
        self.hydro_val_labels  = {}      # channel -> tk.StringVar for current value
        self.hydro_alert_labels = {}     # channel -> tk.Label for alert badge
        self.hydro_log_tree    = None    # ttk.Treeview audit log
        self.hydro_fig         = None
        self.hydro_canvas      = None
        self.hydro_last_ts     = tk.StringVar(value='Never')
        self.hydro_status_var  = tk.StringVar(value='● Disconnected')
        # Serial / MQTT connection vars
        self.hydro_port_var    = tk.StringVar(value='')
        self.hydro_baud_var    = tk.StringVar(value='115200')
        self.hydro_mqtt_host_var    = tk.StringVar(value='localhost')
        self.hydro_mqtt_port_var    = tk.StringVar(value='1883')
        self.hydro_mqtt_topic_var   = tk.StringVar(value='hydroponics/nutrients')
        self.hydro_mqtt_user_var    = tk.StringVar(value='')
        self.hydro_mqtt_pass_var    = tk.StringVar(value='')
        self._hydro_mqtt_client     = None
        self._hydro_input_mode      = tk.StringVar(value='Serial')  # 'Serial' | 'MQTT' | 'Manual'
        # Setpoints and thresholds (target range low, high)
        self.hydro_targets = {
            'ph':          (5.5,  6.5),
            'ec':          (1.2,  2.4),
            'tds':         (800, 1600),
            'water_temp':  (18.0, 24.0),
            'do':          (6.0,  9.0),
            'orp':         (200,  400),
            'nitrogen':    (150,  250),
            'phosphorus':  (30,    60),
            'potassium':   (150,  300),
        }
        self.hydro_target_vars = {
            ch: (tk.StringVar(value=str(lo)), tk.StringVar(value=str(hi)))
            for ch, (lo, hi) in self.hydro_targets.items()
        }
        # Dosing pump state (8 pumps)
        self.hydro_pump_vars   = [tk.BooleanVar(value=False) for _ in range(8)]
        self.hydro_pump_names  = ['pH Up', 'pH Down', 'Nutrient A', 'Nutrient B',
                                   'Cal-Mag', 'Flush', 'Pump 7', 'Pump 8']
        # Graph channel selection
        self.hydro_graph_ch_var = tk.StringVar(value='ph')
        self.hydro_max_history  = 200

        # ── Automation Engine State ──────────────────────────────────────
        # Rule dict keys: id, name, enabled, trigger_type, trigger_channel,
        #   trigger_op, trigger_value, trigger_time, action_type, action_target,
        #   action_duration, cooldown_s, last_triggered, trigger_count, notes
        self.automations           = []          # list of rule dicts
        self._auto_id_counter      = 0           # monotonic rule ID
        self._auto_engine_running  = False
        self._auto_engine_thread   = None
        self._auto_sensor_cache    = {}          # channel -> latest float value
        self._auto_sensor_lock     = threading.Lock()
        self._auto_log_tree        = None        # ttk.Treeview execution log
        self._auto_rule_tree       = None        # ttk.Treeview rule list
        self._auto_engine_interval = 5           # seconds between evaluations
        # Trigger types and action types exposed in the UI
        self.AUTO_TRIGGER_TYPES  = [
            'hydro_threshold', 'env_threshold', 'health_threshold', 'schedule'
        ]
        self.AUTO_ACTION_TYPES   = [
            'pump_on', 'pump_off', 'pump_pulse',
            'alert_popup', 'log_event', 'analyze_panel',
        ]
        # Operator choices for threshold triggers
        self.AUTO_OPS = ['<', '<=', '>', '>=', '==', '!=']
        # All channel names exposed for threshold triggers
        self._hydro_ch_keys  = ['ph', 'ec', 'tds', 'water_temp', 'do', 'orp',
                                 'nitrogen', 'phosphorus', 'potassium']
        self._env_ch_keys    = ['temperature', 'humidity', 'co2', 'light', 'soil_moisture']
        self._health_ch_keys = ['health_score', 'ndvi']
        # Editor StringVars (populated when a row is selected / new rule opened)
        self._ae = {}   # editor field vars, built in create_automations_widgets

        # ── Leafy AI Copilot state ─────────────────────────────────────────────
        self._leafy_window   = None
        self._leafy_chat     = []        # conversation history for multi-turn context
        self._leafy_api_key  = tk.StringVar(value="")
        self._leafy_model    = tk.StringVar(value="gpt-4o")
        self._leafy_thinking = False
        self._leafy_chat_txt = None      # Text widget reference (built lazily)
        self._leafy_status_lbl = None
        self._leafy_input_var  = None

        # Initialize PlantCV debug parameters
        pcv.params.debug = None
        pcv.params.dpi = 100

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Build Plant Detection UI
        self.create_plant_detection_widgets()
        
        # Build Environmental Data UI
        self.create_environmental_widgets()

        # Build Sensor Config UI
        self.create_sensor_config_widgets()

        # Build Plant Communication UI
        self.create_plant_comm_widgets()

        # Build Plant History UI
        self.create_plant_history_widgets()

        # Build Hydroponics UI
        self.create_hydro_widgets()

        # Build Automations UI
        self.create_automations_widgets()

        # Build 3D Morphology viewer UI
        self.create_morphology_widgets()

        # Load Leafy config (API key + model) saved from a previous session
        self._leafy_load_config()

        # Bind mouse wheel for zoom in Plant Detection tab
        self.root.bind("<Control-MouseWheel>", self.mouse_wheel_zoom)
        
        # Handle application close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Probe for cameras in a background thread after the UI is shown
        self.root.after(200, self._start_camera_probe)

    # ── Sensor Config tab ──────────────────────────────────────────────────────

    def create_sensor_config_widgets(self):
        T = THEME
        _btn  = dict(bg=T["primary_dk"], fg=T["text"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary"], activeforeground=T["text"],
                     cursor="hand2", bd=0)
        _sbtn = dict(bg=T["card_header"], fg=T["primary_lt"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary_dk"], activeforeground=T["text"],
                     cursor="hand2", bd=0)
        _lf   = dict(bg=T["card"], fg=T["primary"], font=("Segoe UI", 10, "bold"),
                     relief="groove", bd=1)
        _ent  = dict(bg=T["entry"], fg=T["text"], relief="flat",
                     highlightthickness=1, highlightcolor=T["border"],
                     highlightbackground=T["border"],
                     insertbackground=T["primary"], bd=0, font=("Segoe UI", 9))
        _spn  = dict(bg=T["entry"], fg=T["text"], relief="flat",
                     buttonbackground=T["card_header"],
                     highlightthickness=1, highlightcolor=T["border"],
                     highlightbackground=T["border"],
                     insertbackground=T["primary"], bd=0, font=("Segoe UI", 9))
        _lbl  = lambda parent, text, bold=False: tk.Label(
            parent, text=text, bg=T["card"], fg=T["text_muted"],
            font=("Segoe UI", 9, "bold" if bold else "normal"))

        outer = tk.Frame(self.sensor_tab, bg=T["panel"])
        outer.pack(fill="both", expand=True, padx=12, pady=10)

        # ── RabbitMQ Connection ────────────────────────────────────────────────
        rmq_lf = tk.LabelFrame(outer, text=" 🐇  RabbitMQ Connection ", padx=12, pady=10, **_lf)
        rmq_lf.pack(fill="x", padx=4, pady=(0, 10))

        # Row 0 — host / port / vhost
        r0 = tk.Frame(rmq_lf, bg=T["card"])
        r0.pack(fill="x", pady=(0, 6))
        _lbl(r0, "Host:").pack(side="left", padx=(0, 4))
        tk.Entry(r0, textvariable=self.rmq_host_var, width=22, **_ent).pack(side="left", padx=(0, 12))
        _lbl(r0, "Port:").pack(side="left", padx=(0, 4))
        tk.Spinbox(r0, from_=1, to=65535, width=7, textvariable=self.rmq_port_var, **_spn).pack(side="left", padx=(0, 12))
        _lbl(r0, "Virtual Host:").pack(side="left", padx=(0, 4))
        tk.Entry(r0, textvariable=self.rmq_vhost_var, width=10, **_ent).pack(side="left", padx=(0, 12))
        _lbl(r0, "Exchange:").pack(side="left", padx=(0, 4))
        tk.Entry(r0, textvariable=self.rmq_exchange_var, width=16, **_ent).pack(side="left")

        # Row 1 — user / pass / buttons
        r1 = tk.Frame(rmq_lf, bg=T["card"])
        r1.pack(fill="x")
        _lbl(r1, "Username:").pack(side="left", padx=(0, 4))
        tk.Entry(r1, textvariable=self.rmq_user_var, width=16, **_ent).pack(side="left", padx=(0, 12))
        _lbl(r1, "Password:").pack(side="left", padx=(0, 4))
        tk.Entry(r1, textvariable=self.rmq_pass_var, width=16, show="•", **_ent).pack(side="left", padx=(0, 16))

        tk.Button(r1, text="⚡  Connect",    command=self._rmq_connect,    **_btn).pack(side="left", padx=(0, 6))
        tk.Button(r1, text="⏹  Disconnect", command=self._rmq_disconnect, **_btn).pack(side="left", padx=(0, 16))
        tk.Button(r1, text="🧪  Test",       command=self._rmq_test,       **_sbtn).pack(side="left")

        # Status pill
        rmq_status_lf = tk.Frame(rmq_lf, bg=T["card"])
        rmq_status_lf.pack(fill="x", pady=(8, 0))
        _lbl(rmq_status_lf, "Status:").pack(side="left", padx=(0, 6))
        self._rmq_status_lbl = tk.Label(rmq_status_lf, textvariable=self.rmq_status_var,
                                        bg=T["card"], fg=T["error"],
                                        font=("Segoe UI", 9, "bold"))
        self._rmq_status_lbl.pack(side="left")

        # ── Arduino Sensor List ────────────────────────────────────────────────
        sensor_lf = tk.LabelFrame(outer, text=" 🔌  Arduino Sensors ", padx=12, pady=10, **_lf)
        sensor_lf.pack(fill="both", expand=True, padx=4, pady=(0, 10))

        # Column-header row
        hdr = tk.Frame(sensor_lf, bg=T["card_header"])
        hdr.pack(fill="x", pady=(0, 4))
        _COL_WIDTHS = [14, 8, 7, 18, 12, 8, 5]
        _COL_LABELS = ["Sensor Name", "COM Port", "Baud", "RabbitMQ Topic / Key",
                       "Camera Panel", "Enabled", ""]
        for w, lbl_text in zip(_COL_WIDTHS, _COL_LABELS):
            tk.Label(hdr, text=lbl_text, width=w, anchor="w",
                     bg=T["card_header"], fg=T["primary_lt"],
                     font=("Segoe UI", 8, "bold")).pack(side="left", padx=4, pady=3)

        # Scrollable rows container
        scroll_wrap = tk.Frame(sensor_lf, bg=T["card"])
        scroll_wrap.pack(fill="both", expand=True)
        scroll_canvas = tk.Canvas(scroll_wrap, bg=T["card"], highlightthickness=0)
        vsb = tk.Scrollbar(scroll_wrap, orient="vertical", command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)
        self.sensor_rows_frame = tk.Frame(scroll_canvas, bg=T["card"])
        scroll_canvas.create_window((0, 0), window=self.sensor_rows_frame, anchor="nw")
        self.sensor_rows_frame.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
        self._sensor_scroll_canvas = scroll_canvas

        # Action buttons bar
        btn_bar = tk.Frame(sensor_lf, bg=T["card"])
        btn_bar.pack(fill="x", pady=(8, 0))
        tk.Button(btn_bar, text="➕  Add Sensor",       command=self._sensor_add_row,    **_btn).pack(side="left", padx=(0, 6))
        tk.Button(btn_bar, text="💾  Save Config",      command=self._sensor_save,       **_btn).pack(side="left", padx=(0, 6))
        tk.Button(btn_bar, text="📂  Load Config",      command=self._sensor_load,       **_btn).pack(side="left", padx=(0, 6))
        tk.Button(btn_bar, text="▶  Activate All",     command=self._sensor_activate_all, **_sbtn).pack(side="left", padx=(0, 6))
        tk.Button(btn_bar, text="⏹  Deactivate All",   command=self._sensor_deactivate_all, **_sbtn).pack(side="left")

        # Add a default empty row to give a starting point
        self._sensor_add_row()

    def _sensor_add_row(self, data=None):
        """Append one editable sensor row to the sensor list."""
        T = THEME
        data = data or {}
        row_idx = len(self.sensor_entries)

        _ent = dict(bg=T["entry"], fg=T["text"], relief="flat",
                    highlightthickness=1, highlightcolor=T["border"],
                    highlightbackground=T["border"],
                    insertbackground=T["primary"], bd=0, font=("Segoe UI", 9))
        _spn = dict(bg=T["entry"], fg=T["text"], relief="flat",
                    buttonbackground=T["card_header"],
                    highlightthickness=1, highlightcolor=T["border"],
                    highlightbackground=T["border"],
                    insertbackground=T["primary"], bd=0, font=("Segoe UI", 9))

        row_frame = tk.Frame(self.sensor_rows_frame, bg=T["card"],
                             highlightbackground=T["border"], highlightthickness=1)
        row_frame.pack(fill="x", padx=2, pady=2)

        vars_ = {
            "name":    tk.StringVar(value=data.get("name",    f"Sensor {row_idx + 1}")),
            "port":    tk.StringVar(value=data.get("port",    "COM3")),
            "baud":    tk.StringVar(value=data.get("baud",    "9600")),
            "topic":   tk.StringVar(value=data.get("topic",   f"sensor.{row_idx + 1}")),
            "camera":  tk.StringVar(value=data.get("camera",  "Panel 1")),
            "enabled": tk.BooleanVar(value=data.get("enabled", True)),
            "frame":   row_frame,
        }

        baud_choices = ["9600", "19200", "38400", "57600", "115200"]
        panel_choices = [f"Panel {i+1}" for i in range(self.max_camera_slots)] + ["(none)"]

        tk.Entry(row_frame, textvariable=vars_["name"],  width=14, **_ent).pack(side="left", padx=(4, 3), pady=4)
        tk.Entry(row_frame, textvariable=vars_["port"],  width=8,  **_ent).pack(side="left", padx=(0, 3), pady=4)
        ttk.Combobox(row_frame, textvariable=vars_["baud"],
                     values=baud_choices, width=6, state="normal").pack(side="left", padx=(0, 3), pady=4)
        tk.Entry(row_frame, textvariable=vars_["topic"], width=18, **_ent).pack(side="left", padx=(0, 3), pady=4)
        ttk.Combobox(row_frame, textvariable=vars_["camera"],
                     values=panel_choices, width=11, state="readonly").pack(side="left", padx=(0, 3), pady=4)
        tk.Checkbutton(row_frame, variable=vars_["enabled"], text="On",
                       bg=T["card"], fg=T["text"], selectcolor=T["primary_dk"],
                       activebackground=T["card"], activeforeground=T["primary_lt"],
                       font=("Segoe UI", 9)).pack(side="left", padx=(0, 6), pady=4)
        tk.Button(row_frame, text="✕", width=2,
                  bg=T["card_header"], fg=T["error"], relief="flat",
                  activebackground=T["card"], activeforeground=T["error"],
                  cursor="hand2", bd=0, font=("Segoe UI", 9, "bold"),
                  command=lambda f=row_frame, v=vars_: self._sensor_remove_row(f, v)
                  ).pack(side="left", padx=(0, 4), pady=4)

        # Status dot (updated when activated)
        status_dot = tk.Label(row_frame, text="●", bg=T["card"], fg=T["text_dim"],
                              font=("Segoe UI", 10))
        status_dot.pack(side="right", padx=(0, 8))
        vars_["status_dot"] = status_dot

        self.sensor_entries.append(vars_)
        self._sensor_scroll_canvas.update_idletasks()
        self._sensor_scroll_canvas.configure(
            scrollregion=self._sensor_scroll_canvas.bbox("all"))

    def _sensor_remove_row(self, frame, vars_):
        frame.destroy()
        self.sensor_entries = [e for e in self.sensor_entries if e is not vars_]

    def _sensor_get_config(self):
        """Return current sensor configuration as a list of dicts (JSON-serialisable)."""
        result = []
        for e in self.sensor_entries:
            result.append({
                "name":    e["name"].get(),
                "port":    e["port"].get(),
                "baud":    e["baud"].get(),
                "topic":   e["topic"].get(),
                "camera":  e["camera"].get(),
                "enabled": e["enabled"].get(),
            })
        return result

    def _sensor_get_rmq_config(self):
        return {
            "host":     self.rmq_host_var.get().strip(),
            "port":     int(self.rmq_port_var.get().strip() or 5672),
            "vhost":    self.rmq_vhost_var.get().strip(),
            "user":     self.rmq_user_var.get().strip(),
            "password": self.rmq_pass_var.get(),
            "exchange": self.rmq_exchange_var.get().strip(),
        }

    def _sensor_save(self):
        path = filedialog.asksaveasfilename(
            title="Save Sensor Configuration",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not path:
            return
        config = {
            "rabbitmq": self._sensor_get_rmq_config(),
            "sensors":  self._sensor_get_config(),
        }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(config, fh, indent=2)
            messagebox.showinfo("Sensor Config", f"Configuration saved to:\n{path}")
            logging.info(f"Sensor config saved to {path}")
        except Exception as exc:
            messagebox.showerror("Sensor Config", f"Could not save: {exc}")

    def _sensor_load(self):
        path = filedialog.askopenfilename(
            title="Load Sensor Configuration",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        except Exception as exc:
            messagebox.showerror("Sensor Config", f"Could not load: {exc}")
            return

        rmq = config.get("rabbitmq", {})
        self.rmq_host_var.set(rmq.get("host",     "localhost"))
        self.rmq_port_var.set(str(rmq.get("port", 5672)))
        self.rmq_vhost_var.set(rmq.get("vhost",   "/"))
        self.rmq_user_var.set(rmq.get("user",     "guest"))
        self.rmq_pass_var.set(rmq.get("password", "guest"))
        self.rmq_exchange_var.set(rmq.get("exchange", "greensight"))

        # Rebuild sensor rows
        for e in list(self.sensor_entries):
            e["frame"].destroy()
        self.sensor_entries.clear()

        for sensor in config.get("sensors", []):
            self._sensor_add_row(data=sensor)

        messagebox.showinfo("Sensor Config", f"Loaded {len(self.sensor_entries)} sensor(s).")
        logging.info(f"Sensor config loaded from {path}")

    def _sensor_activate_all(self):
        for e in self.sensor_entries:
            e["enabled"].set(True)
            e["status_dot"].config(fg=THEME["success"])
        logging.info("All sensors marked enabled")

    def _sensor_deactivate_all(self):
        for e in self.sensor_entries:
            e["enabled"].set(False)
            e["status_dot"].config(fg=THEME["text_dim"])
        logging.info("All sensors marked disabled")

    # ── RabbitMQ helpers ───────────────────────────────────────────────────────

    def _rmq_set_status(self, text, color):
        self.rmq_status_var.set(text)
        if hasattr(self, "_rmq_status_lbl"):
            self._rmq_status_lbl.config(fg=color)

    def _rmq_connect(self):
        cfg = self._sensor_get_rmq_config()
        try:
            import pika
        except ImportError:
            self._rmq_set_status("pika not installed — run: pip install pika", THEME["warning"])
            messagebox.showerror("RabbitMQ", "pika is not installed.\nRun:  pip install pika")
            return
        try:
            self._rmq_disconnect()
            credentials = pika.PlainCredentials(cfg["user"], cfg["password"])
            parameters  = pika.ConnectionParameters(
                host=cfg["host"], port=cfg["port"],
                virtual_host=cfg["vhost"], credentials=credentials,
                connection_attempts=2, retry_delay=1, socket_timeout=5)
            self._rmq_connection = pika.BlockingConnection(parameters)
            self._rmq_channel    = self._rmq_connection.channel()
            self._rmq_channel.exchange_declare(
                exchange=cfg["exchange"], exchange_type="topic", durable=True)
            self._rmq_set_status(
                f"● Connected  —  {cfg['host']}:{cfg['port']}{cfg['vhost']}  [{cfg['exchange']}]",
                THEME["success"])
            logging.info(f"RabbitMQ connected to {cfg['host']}:{cfg['port']}")
        except Exception as exc:
            self._rmq_set_status(f"● Connection failed: {exc}", THEME["error"])
            logging.error(f"RabbitMQ connection failed: {exc}")

    def _rmq_disconnect(self):
        try:
            if self._rmq_connection and self._rmq_connection.is_open:
                self._rmq_connection.close()
        except Exception:
            pass
        self._rmq_connection = None
        self._rmq_channel    = None
        self._rmq_set_status("● Disconnected", THEME["error"])

    def _rmq_test(self):
        """Publish a test heartbeat message on each configured sensor topic."""
        if self._rmq_channel is None:
            messagebox.showwarning("RabbitMQ", "Not connected. Click Connect first.")
            return
        cfg = self._sensor_get_rmq_config()
        sent = 0
        try:
            import pika
            for e in self.sensor_entries:
                if not e["enabled"].get():
                    continue
                topic = e["topic"].get().strip() or "sensor.test"
                body  = json.dumps({
                    "sensor":    e["name"].get(),
                    "camera":    e["camera"].get(),
                    "heartbeat": True,
                    "timestamp": datetime.now().isoformat(),
                }).encode()
                self._rmq_channel.basic_publish(
                    exchange=cfg["exchange"],
                    routing_key=topic,
                    body=body,
                    properties=pika.BasicProperties(content_type="application/json", delivery_mode=1))
                e["status_dot"].config(fg=THEME["accent"])
                sent += 1
            messagebox.showinfo("RabbitMQ Test",
                                f"Sent {sent} heartbeat message(s) to exchange '{cfg['exchange']}'.")
        except Exception as exc:
            messagebox.showerror("RabbitMQ Test", f"Publish failed:\n{exc}")
            logging.error(f"RabbitMQ test publish failed: {exc}")

    # ── Plant Communication tab ────────────────────────────────────────────────

    # ── Health scoring helpers ─────────────────────────────────────────────────

    @staticmethod
    def _compute_ndvi_proxy(greenness, exg_index):
        """
        Estimate a pseudo-NDVI from RGB metrics.
        True NDVI requires a near-infrared channel; this is an RGB approximation
        using the Excess Green Index and relative greenness.
        Range normalised to [-1, 1].
        """
        # ExG is roughly proportional to (G – R+B); scale to pseudo-NDVI range
        proxy = (greenness - 0.333) * 3.0 * 0.7 + (exg_index - 0.5) * 0.3
        return max(-1.0, min(1.0, proxy))

    @staticmethod
    def _health_label_from_score(score):
        if score >= 0.75:
            return ("Excellent",  "#00e676")
        if score >= 0.55:
            return ("Good",       "#66bb6a")
        if score >= 0.35:
            return ("Fair",       "#ffca28")
        if score >= 0.15:
            return ("Poor",       "#ffa726")
        return ("Critical",   "#ef5350")

    @staticmethod
    def _nutrient_flags(yellow_ratio, brown_ratio, purple_ratio, lab_a_median=None, lab_b_median=None):
        flags = []
        if yellow_ratio > 0.15:
            flags.append("⚠ Nitrogen / Iron deficiency (yellowing)")
        if brown_ratio > 0.12:
            flags.append("⚠ Potassium deficiency or tip-burn (browning)")
        if purple_ratio > 0.08:
            flags.append("⚠ Phosphorus deficiency (purple tint)")
        if lab_b_median is not None and lab_b_median > 148:
            flags.append("⚠ Chlorosis-like hue shift (LAB b↑)")
        if lab_a_median is not None and lab_a_median < 118:
            flags.append("⚠ Reduced red–green contrast (pale tissue)")
        return flags or ["✓ No obvious nutrient deficiency detected"]

    @staticmethod
    def _advice_from_flags(flags):
        advice_map = {
            "Nitrogen":    "Apply nitrogen-rich fertiliser; check pH for iron lock-out.",
            "Iron":        "Apply nitrogen-rich fertiliser; check pH for iron lock-out.",
            "Potassium":   "Supplement with potassium; ensure adequate watering.",
            "Phosphorus":  "Apply phosphorus-rich fertiliser; check root health.",
            "Chlorosis":   "Check iron/manganese availability; test soil pH.",
            "pale":        "Increase light intensity or inspect for root stress.",
        }
        seen = set()
        out = []
        for f in flags:
            for keyword, tip in advice_map.items():
                if keyword.lower() in f.lower() and tip not in seen:
                    seen.add(tip)
                    out.append(tip)
        return out or ["Monitor plant closely. Re-analyse after next watering cycle."]

    # ── UI builder ─────────────────────────────────────────────────────────────

    def create_plant_comm_widgets(self):
        T = THEME
        _btn  = dict(bg=T["primary_dk"], fg=T["text"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary"], activeforeground=T["text"],
                     cursor="hand2", bd=0)
        _sbtn = dict(bg=T["card_header"], fg=T["primary_lt"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary_dk"], activeforeground=T["text"],
                     cursor="hand2", bd=0)

        outer = tk.Frame(self.comm_tab, bg=T["panel"])
        outer.pack(fill="both", expand=True)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = tk.Frame(outer, bg=T["panel"])
        toolbar.pack(fill="x", padx=12, pady=(8, 4))

        tk.Button(toolbar, text="🔄  Analyse All Cameras",
                  command=self._comm_analyse_all, **_btn).pack(side="left", padx=(0, 6))
        tk.Button(toolbar, text="📋  Export Report",
                  command=self._comm_export_report, **_sbtn).pack(side="left", padx=(0, 16))

        tk.Checkbutton(toolbar, text="Auto-update on analyse",
                       variable=self.comm_auto_var,
                       bg=T["panel"], fg=T["text"], selectcolor=T["primary_dk"],
                       activebackground=T["panel"], activeforeground=T["primary_lt"],
                       font=("Segoe UI", 9)).pack(side="left")

        # ── Legend strip ──────────────────────────────────────────────────────
        legend = tk.Frame(outer, bg=T["card_header"])
        legend.pack(fill="x", padx=12, pady=(0, 6))
        for label_text, color in [("Excellent ≥0.75", "#00e676"), ("Good ≥0.55","#66bb6a"),
                                   ("Fair ≥0.35","#ffca28"), ("Poor ≥0.15","#ffa726"),
                                   ("Critical <0.15","#ef5350")]:
            dot = tk.Label(legend, text="●", fg=color, bg=T["card_header"],
                           font=("Segoe UI", 11))
            dot.pack(side="left", padx=(12, 2), pady=4)
            tk.Label(legend, text=label_text, bg=T["card_header"],
                     fg=T["text_muted"], font=("Segoe UI", 8)).pack(side="left", padx=(0, 8), pady=4)

        # ── Scrollable panel grid ─────────────────────────────────────────────
        scroll_outer = tk.Frame(outer, bg=T["panel"])
        scroll_outer.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        comm_canvas = tk.Canvas(scroll_outer, bg=T["panel"], highlightthickness=0)
        vsb = tk.Scrollbar(scroll_outer, orient="vertical", command=comm_canvas.yview)
        comm_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        comm_canvas.pack(side="left", fill="both", expand=True)

        self._comm_panels_frame = tk.Frame(comm_canvas, bg=T["panel"])
        comm_canvas.create_window((0, 0), window=self._comm_panels_frame, anchor="nw")
        self._comm_panels_frame.bind(
            "<Configure>",
            lambda e: comm_canvas.configure(scrollregion=comm_canvas.bbox("all")))
        self._comm_canvas = comm_canvas

        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            comm_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        comm_canvas.bind("<MouseWheel>", _on_mousewheel)

        # Build one card per camera slot
        COLS = 2
        for i in range(self.max_camera_slots):
            row_idx = i // COLS
            col_idx = i % COLS
            self._comm_panels_frame.columnconfigure(col_idx, weight=1)

            card = tk.Frame(self._comm_panels_frame, bg=T["card"],
                            highlightbackground=T["border"], highlightthickness=1)
            card.grid(row=row_idx, column=col_idx, padx=8, pady=8, sticky="nsew")
            card.columnconfigure(0, weight=1)

            # Card header
            hdr = tk.Frame(card, bg=T["card_header"], height=28)
            hdr.grid(row=0, column=0, columnspan=3, sticky="ew")
            hdr.grid_propagate(False)
            hdr_lbl = tk.Label(hdr, text=f"  🌿  Panel {i+1}  —  No data yet",
                               bg=T["card_header"], fg=T["primary_lt"],
                               font=("Segoe UI", 9, "bold"))
            hdr_lbl.pack(side="left", padx=6, pady=4)
            status_dot = tk.Label(hdr, text="●", bg=T["card_header"],
                                  fg=T["text_dim"], font=("Segoe UI", 12))
            status_dot.pack(side="right", padx=10, pady=4)

            # NDVI bar row
            ndvi_row = tk.Frame(card, bg=T["card"])
            ndvi_row.grid(row=1, column=0, sticky="ew", padx=10, pady=(8, 2))
            tk.Label(ndvi_row, text="NDVI proxy:", bg=T["card"],
                     fg=T["text_muted"], font=("Segoe UI", 8)).pack(side="left")
            ndvi_bar_bg = tk.Frame(ndvi_row, bg=T["scrollbar"], height=10, width=180)
            ndvi_bar_bg.pack(side="left", padx=(8, 4), pady=2)
            ndvi_bar_bg.pack_propagate(False)
            ndvi_bar_fill = tk.Frame(ndvi_bar_bg, bg=T["text_dim"], height=10, width=1)
            ndvi_bar_fill.place(x=0, y=0, relheight=1.0, width=1)
            ndvi_val_lbl = tk.Label(ndvi_row, text="—", bg=T["card"],
                                    fg=T["accent"], font=("Segoe UI", 8, "bold"))
            ndvi_val_lbl.pack(side="left")

            # Health score row
            score_row = tk.Frame(card, bg=T["card"])
            score_row.grid(row=2, column=0, sticky="ew", padx=10, pady=2)
            tk.Label(score_row, text="Health:", bg=T["card"],
                     fg=T["text_muted"], font=("Segoe UI", 8)).pack(side="left")
            health_lbl = tk.Label(score_row, text="—",
                                  bg=T["card"], fg=T["text_dim"],
                                  font=("Segoe UI", 10, "bold"))
            health_lbl.pack(side="left", padx=(8, 0))
            score_lbl = tk.Label(score_row, text="",
                                 bg=T["card"], fg=T["text_dim"],
                                 font=("Segoe UI", 8))
            score_lbl.pack(side="left", padx=(6, 0))

            # ROI count row
            roi_row = tk.Frame(card, bg=T["card"])
            roi_row.grid(row=3, column=0, sticky="ew", padx=10, pady=2)
            tk.Label(roi_row, text="Plants detected:", bg=T["card"],
                     fg=T["text_muted"], font=("Segoe UI", 8)).pack(side="left")
            roi_count_lbl = tk.Label(roi_row, text="—",
                                     bg=T["card"], fg=T["text"],
                                     font=("Segoe UI", 8, "bold"))
            roi_count_lbl.pack(side="left", padx=(8, 0))

            # Nutrient flags box
            tk.Label(card, text="Nutrient Screen:", bg=T["card"],
                     fg=T["text_muted"], font=("Segoe UI", 8, "bold")
                     ).grid(row=4, column=0, sticky="w", padx=10, pady=(6, 2))
            nutrient_text = tk.Text(card, height=3, state="disabled",
                                    bg=T["card"], fg=T["text"], relief="flat",
                                    font=("Segoe UI", 8), bd=0,
                                    wrap="word",
                                    highlightthickness=1,
                                    highlightbackground=T["border"],
                                    insertbackground=T["primary"])
            nutrient_text.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 4))

            # Advice box
            tk.Label(card, text="Recommendations:", bg=T["card"],
                     fg=T["text_muted"], font=("Segoe UI", 8, "bold")
                     ).grid(row=6, column=0, sticky="w", padx=10, pady=(4, 2))
            advice_text = tk.Text(card, height=3, state="disabled",
                                  bg=T["card"], fg=T["primary_lt"], relief="flat",
                                  font=("Segoe UI", 8), bd=0,
                                  wrap="word",
                                  highlightthickness=1,
                                  highlightbackground=T["border"],
                                  insertbackground=T["primary"])
            advice_text.grid(row=7, column=0, sticky="ew", padx=10, pady=(0, 4))

            # Per-ROI detail treeview
            tk.Label(card, text="Per-ROI Detail:", bg=T["card"],
                     fg=T["text_muted"], font=("Segoe UI", 8, "bold")
                     ).grid(row=8, column=0, sticky="w", padx=10, pady=(4, 2))
            roi_tree = ttk.Treeview(card, columns=("roi","ndvi","health","yellow","brown","purple"),
                                    show="headings", height=3)
            for col, hd, w in [("roi","ROI",60), ("ndvi","NDVI~",60), ("health","Health",80),
                                ("yellow","Yellow%",65), ("brown","Brown%",65), ("purple","Purple%",65)]:
                roi_tree.heading(col, text=hd)
                roi_tree.column(col, width=w, anchor="center")
            roi_tree.grid(row=9, column=0, sticky="ew", padx=10, pady=(0, 6))

            # Timestamp footer
            ts_lbl = tk.Label(card, text="Last updated: —",
                              bg=T["card"], fg=T["text_dim"],
                              font=("Segoe UI", 7))
            ts_lbl.grid(row=10, column=0, sticky="e", padx=10, pady=(0, 6))

            self.comm_panels.append({
                "card":          card,
                "hdr_lbl":       hdr_lbl,
                "status_dot":    status_dot,
                "ndvi_bar_fill": ndvi_bar_fill,
                "ndvi_bar_bg":   ndvi_bar_bg,
                "ndvi_val_lbl":  ndvi_val_lbl,
                "health_lbl":    health_lbl,
                "score_lbl":     score_lbl,
                "roi_count_lbl": roi_count_lbl,
                "nutrient_text": nutrient_text,
                "advice_text":   advice_text,
                "roi_tree":      roi_tree,
                "ts_lbl":        ts_lbl,
            })

        # Initially hide panels beyond visible_camera_slots
        self._comm_update_visibility()

    def _comm_update_visibility(self):
        for i, panel in enumerate(self.comm_panels):
            if i < self.visible_camera_slots:
                panel["card"].grid()
            else:
                panel["card"].grid_remove()
        if hasattr(self, "_comm_canvas"):
            self._comm_canvas.update_idletasks()
            self._comm_canvas.configure(
                scrollregion=self._comm_canvas.bbox("all"))

    def push_plant_comm_update(self, camera_idx, roi_metrics_list, measurements_list):
        """
        Called after each analysis run.  Receives a list of roi_metrics dicts and
        the corresponding list of measurements dicts (from collect_plantcv_measurements).
        Computes health scores and pushes the result to the comm panel.
        """
        if not self.comm_auto_var.get():
            return
        if camera_idx >= len(self.comm_panels):
            return

        T = THEME
        rois = []
        agg_ndvi      = 0.0
        agg_greenness = 0.0
        agg_exg       = 0.0
        agg_yellow    = 0.0
        agg_brown     = 0.0
        agg_purple    = 0.0
        all_flags     = []
        n = max(len(roi_metrics_list), 1)

        for roi_i, (metrics, meas) in enumerate(zip(roi_metrics_list, measurements_list), start=1):
            greenness   = metrics.get("greenness",    0.0)
            exg         = metrics.get("exg_index",    0.0)
            yellow      = metrics.get("yellow_ratio", 0.0)
            brown       = metrics.get("brown_ratio",  0.0)
            purple      = metrics.get("purple_ratio", 0.0)

            # Pull LAB medians from PlantCV observations if available
            color_obs   = meas.get("color",  {})
            lab_a_obs   = meas.get("lab_a",  {})
            lab_b_obs   = meas.get("lab_b",  {})
            lab_a_med   = lab_a_obs.get("gray_median",  {}).get("value") if lab_a_obs else None
            lab_b_med   = lab_b_obs.get("gray_median",  {}).get("value") if lab_b_obs else None

            ndvi_proxy  = self._compute_ndvi_proxy(greenness, exg)

            # Health score: weighted composite
            ndvi_norm   = (ndvi_proxy + 1.0) / 2.0          # 0..1
            defect_pen  = yellow * 0.4 + brown * 0.3 + purple * 0.3
            score       = max(0.0, min(1.0, ndvi_norm * 0.6 + (1.0 - defect_pen) * 0.4))
            h_label, h_color = self._health_label_from_score(score)

            flags       = self._nutrient_flags(yellow, brown, purple, lab_a_med, lab_b_med)
            all_flags.extend(flags)

            agg_ndvi      += ndvi_proxy
            agg_greenness += greenness
            agg_exg       += exg
            agg_yellow    += yellow
            agg_brown     += brown
            agg_purple    += purple

            rois.append({
                "label":      f"ROI {roi_i}",
                "ndvi":       round(ndvi_proxy, 3),
                "score":      round(score, 3),
                "h_label":    h_label,
                "h_color":    h_color,
                "yellow":     round(yellow * 100, 1),
                "brown":      round(brown  * 100, 1),
                "purple":     round(purple * 100, 1),
                "flags":      flags,
            })

        # Aggregate across all ROIs
        agg_ndvi    /= n
        avg_score    = sum(r["score"] for r in rois) / n
        agg_h_label, agg_h_color = self._health_label_from_score(avg_score)

        # Deduplicate flags
        unique_flags = list(dict.fromkeys(all_flags))
        advice       = self._advice_from_flags(unique_flags)

        snapshot = {
            "rois":        rois,
            "ndvi":        round(agg_ndvi, 3),
            "score":       round(avg_score, 3),
            "h_label":     agg_h_label,
            "h_color":     agg_h_color,
            "flags":       unique_flags,
            "advice":      advice,
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_rois":      len(rois),
        }
        self.comm_data[camera_idx] = snapshot

        # Push to UI in main thread
        if not self.is_closing:
            self.root.after(0, lambda idx=camera_idx, s=snapshot: self._comm_render_panel(idx, s))

    def _comm_render_panel(self, idx, snapshot):
        """Update a single comm panel card with fresh snapshot data."""
        if idx >= len(self.comm_panels):
            return
        T   = THEME
        p   = self.comm_panels[idx]
        n   = snapshot["n_rois"]
        col = snapshot["h_color"]

        # Header
        p["hdr_lbl"].config(
            text=f"  🌿  Panel {idx+1}  —  {n} plant{'s' if n != 1 else ''} detected")
        p["status_dot"].config(fg=col)

        # NDVI bar (bar_bg is 180px wide, map -1..1 → 0..180)
        ndvi = snapshot["ndvi"]
        bar_width = max(2, int((ndvi + 1.0) / 2.0 * 180))
        p["ndvi_bar_fill"].place(x=0, y=0, relheight=1.0, width=bar_width)
        p["ndvi_bar_fill"].config(bg=col)
        p["ndvi_val_lbl"].config(text=f"{ndvi:+.3f}", fg=col)

        # Health
        p["health_lbl"].config(text=snapshot["h_label"], fg=col)
        p["score_lbl"].config(text=f"(score {snapshot['score']:.2f})", fg=T["text_muted"])

        # ROI count
        p["roi_count_lbl"].config(text=str(n))

        # Nutrient flags
        flag_text = "\n".join(snapshot["flags"])
        p["nutrient_text"].config(state="normal")
        p["nutrient_text"].delete("1.0", "end")
        p["nutrient_text"].insert("1.0", flag_text)
        p["nutrient_text"].config(state="disabled")
        # Update text area height dynamically
        p["nutrient_text"].config(height=max(2, min(6, len(snapshot["flags"]) + 1)))

        # Advice
        adv_text = "\n".join(f"• {a}" for a in snapshot["advice"])
        p["advice_text"].config(state="normal")
        p["advice_text"].delete("1.0", "end")
        p["advice_text"].insert("1.0", adv_text)
        p["advice_text"].config(state="disabled")
        p["advice_text"].config(height=max(2, min(6, len(snapshot["advice"]) + 1)))

        # Per-ROI tree
        tree = p["roi_tree"]
        for row in tree.get_children():
            tree.delete(row)
        for r in snapshot["rois"]:
            tree.insert("", "end", values=(
                r["label"], r["ndvi"], r["h_label"],
                f"{r['yellow']}%", f"{r['brown']}%", f"{r['purple']}%"))

        # Timestamp
        p["ts_lbl"].config(text=f"Last updated: {snapshot['timestamp']}")

    def _comm_analyse_all(self):
        """Trigger a fresh analysis on all visible panels that have a current frame."""
        for i in range(self.visible_camera_slots):
            if self.current_frames[i] is not None:
                frame         = self.current_frames[i].copy()
                analysis_type = self.grids[i]["analysis_dropdown"].get()
                self.start_analysis_job(i, frame, analysis_type)

    def _comm_export_report(self):
        """Export the latest plant communication snapshots to a JSON file."""
        report = []
        for i, snap in enumerate(self.comm_data):
            if snap is not None:
                report.append({"panel": i + 1, **snap})
        if not report:
            messagebox.showwarning("Export Report", "No plant health data available yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Plant Health Report",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
            messagebox.showinfo("Export Report", f"Report saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export Report", f"Could not save:\n{exc}")

    # ── Plant History tab ─────────────────────────────────────────────────────

    def create_plant_history_widgets(self):
        T = THEME
        _btn  = dict(bg=T["primary_dk"], fg=T["text"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary"], activeforeground=T["text"],
                     cursor="hand2", bd=0)
        _sbtn = dict(bg=T["card_header"], fg=T["primary_lt"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary_dk"], activeforeground=T["text"],
                     cursor="hand2", bd=0)
        _ent  = dict(bg=T["entry"], fg=T["text"], relief="flat",
                     highlightthickness=1, highlightcolor=T["border"],
                     highlightbackground=T["border"],
                     insertbackground=T["primary"], bd=0, font=("Segoe UI", 9))

        outer = tk.Frame(self.hist_tab, bg=T["panel"])
        outer.pack(fill="both", expand=True)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = tk.Frame(outer, bg=T["panel"], pady=6)
        toolbar.pack(fill="x", padx=8)

        tk.Label(toolbar, text="Filter Panel:", bg=T["panel"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(4, 2))
        panel_choices = ["All Panels"] + [f"Panel {i + 1}" for i in range(MAX_CAMERA_SLOTS)]
        self._hist_panel_cb = ttk.Combobox(
            toolbar, textvariable=self.hist_filter_panel_var,
            values=panel_choices, state="readonly", width=12,
            font=("Segoe UI", 9))
        self._hist_panel_cb.pack(side="left", padx=(0, 8))

        tk.Label(toolbar, text="Analysis Type:", bg=T["panel"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 2))
        type_choices = ["All Types"] + PLANTCV_PIPELINES
        self._hist_type_cb = ttk.Combobox(
            toolbar, textvariable=self.hist_filter_type_var,
            values=type_choices, state="readonly", width=22,
            font=("Segoe UI", 9))
        self._hist_type_cb.pack(side="left", padx=(0, 8))

        tk.Label(toolbar, text="🔍", bg=T["panel"], fg=T["text_muted"],
                 font=("Segoe UI", 10)).pack(side="left")
        self._hist_search_entry = tk.Entry(
            toolbar, textvariable=self.hist_search_var, width=20, **_ent)
        self._hist_search_entry.pack(side="left", padx=(2, 8))

        tk.Button(toolbar, text="Apply Filters", **_btn,
                  command=self._hist_apply_filters).pack(side="left", padx=(0, 4))
        tk.Button(toolbar, text="↺ Reset", **_sbtn,
                  command=self._hist_reset_filters).pack(side="left", padx=(0, 12))

        tk.Button(toolbar, text="⬇ Export CSV", **_btn,
                  command=self._hist_export_csv).pack(side="right", padx=(4, 4))
        tk.Button(toolbar, text="🗑 Clear History", **_sbtn,
                  command=self._hist_clear).pack(side="right", padx=(0, 4))
        tk.Label(toolbar, textvariable=self.hist_row_count_var,
                 bg=T["panel"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="right", padx=(0, 16))

        # ── Summary strip ─────────────────────────────────────────────────────
        summ = tk.Frame(outer, bg=T["card_header"], pady=5)
        summ.pack(fill="x", padx=8, pady=(0, 4))

        for label, var_attr in [
            ("Total Records",    "_hist_stat_total"),
            ("Runs Today",       "_hist_stat_today"),
            ("Avg Health Score", "_hist_stat_avg"),
            ("Last Panel",       "_hist_stat_last"),
        ]:
            col = tk.Frame(summ, bg=T["card_header"])
            col.pack(side="left", padx=20)
            tk.Label(col, textvariable=getattr(self, var_attr),
                     bg=T["card_header"], fg=T["accent"],
                     font=("Segoe UI", 13, "bold")).pack()
            tk.Label(col, text=label,
                     bg=T["card_header"], fg=T["text_muted"],
                     font=("Segoe UI", 8)).pack()

        # ── Main audit treeview ───────────────────────────────────────────────
        tree_frame = tk.Frame(outer, bg=T["panel"])
        tree_frame.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        columns   = ("#", "Timestamp", "Panel", "Type", "Plants",
                     "NDVI~", "Score", "Health", "Yellow%", "Brown%", "Purple%", "Flags")
        col_widths = (44, 148, 72, 190, 55, 62, 55, 85, 68, 68, 68, 260)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        self.hist_tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings",
            yscrollcommand=vsb.set, xscrollcommand=hsb.set,
            selectmode="browse", height=14)
        vsb.configure(command=self.hist_tree.yview)
        hsb.configure(command=self.hist_tree.xview)

        for col, w in zip(columns, col_widths):
            self.hist_tree.heading(col, text=col,
                                   command=lambda c=col: self._hist_sort_by(c))
            self.hist_tree.column(col, width=w, minwidth=w, stretch=(col == "Flags"))

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.hist_tree.pack(fill="both", expand=True)

        # Health-label row colour tags
        _health_tags = {
            "Excellent": ("#001a00", T["accent"]),
            "Good":      ("#001500", T["primary"]),
            "Fair":      ("#1a1a00", T["warning"]),
            "Poor":      ("#1a0a00", "#ff7043"),
            "Critical":  ("#1a0000", T["error"]),
        }
        for tag, (bg_c, fg_c) in _health_tags.items():
            self.hist_tree.tag_configure(tag, background=bg_c, foreground=fg_c)

        self.hist_tree.bind("<<TreeviewSelect>>", self._hist_on_select)

        # ── Per-ROI detail panel ──────────────────────────────────────────────
        detail_outer = tk.Frame(outer, bg=T["card"], bd=1, relief="groove")
        detail_outer.pack(fill="x", padx=8, pady=(0, 6))

        tk.Label(detail_outer, text="  ROI Detail for Selected Record",
                 bg=T["card_header"], fg=T["text_muted"],
                 font=("Segoe UI", 9, "bold")).pack(fill="x", ipady=3)

        det_cols   = ("ROI", "NDVI~", "Score", "Health", "Yellow%", "Brown%", "Purple%", "Flags")
        det_widths = (60, 62, 55, 85, 68, 68, 68, 300)

        det_inner = tk.Frame(detail_outer, bg=T["card"])
        det_inner.pack(fill="x")
        det_vsb = ttk.Scrollbar(det_inner, orient="vertical")
        self.hist_detail_tree = ttk.Treeview(
            det_inner, columns=det_cols, show="headings",
            yscrollcommand=det_vsb.set,
            selectmode="browse", height=4)
        det_vsb.configure(command=self.hist_detail_tree.yview)
        for col, w in zip(det_cols, det_widths):
            self.hist_detail_tree.heading(col, text=col)
            self.hist_detail_tree.column(col, width=w, minwidth=w, stretch=(col == "Flags"))
        for tag, (bg_c, fg_c) in _health_tags.items():
            self.hist_detail_tree.tag_configure(tag, background=bg_c, foreground=fg_c)
        det_vsb.pack(side="right", fill="y")
        self.hist_detail_tree.pack(fill="x")

        # ── Forecast panel ────────────────────────────────────────────────────
        fc_outer = tk.LabelFrame(outer, text=" 📈  Crop Health Forecast  (Holt-Winters Exponential Smoothing) ",
                                 bg=T["card"], fg=T["primary_lt"],
                                 font=("Segoe UI", 10, "bold"),
                                 relief="groove", bd=1)
        fc_outer.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        # Forecast toolbar
        fc_bar = tk.Frame(fc_outer, bg=T["card"], pady=4)
        fc_bar.pack(fill="x")

        tk.Label(fc_bar, text="Panel:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(6, 2))
        fc_panel_cb = ttk.Combobox(
            fc_bar, textvariable=self.hist_forecast_panel_var,
            values=["All Panels"] + [f"Panel {i+1}" for i in range(MAX_CAMERA_SLOTS)],
            state="readonly", width=12, font=("Segoe UI", 9))
        fc_panel_cb.pack(side="left", padx=(0, 10))

        tk.Label(fc_bar, text="Metric:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 2))
        fc_metric_cb = ttk.Combobox(
            fc_bar, textvariable=self.hist_forecast_metric_var,
            values=["Health Score", "NDVI~", "Yellow%", "Brown%", "Purple%"],
            state="readonly", width=14, font=("Segoe UI", 9))
        fc_metric_cb.pack(side="left", padx=(0, 10))

        tk.Label(fc_bar, text="Periods ahead:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 2))
        tk.Spinbox(fc_bar, from_=1, to=30, textvariable=self.hist_forecast_horizon_var,
                   width=4, bg=T["entry"], fg=T["text"], relief="flat",
                   highlightthickness=1, highlightbackground=T["border"],
                   buttonbackground=T["card_header"],
                   insertbackground=T["primary"], font=("Segoe UI", 9)).pack(side="left", padx=(0, 10))

        tk.Button(fc_bar, text="⚡ Run Forecast", **_btn,
                  command=self._hist_run_forecast).pack(side="left", padx=(0, 6))

        self._fc_status_lbl = tk.Label(fc_bar, text="", bg=T["card"],
                                       fg=T["text_muted"], font=("Segoe UI", 8, "italic"))
        self._fc_status_lbl.pack(side="left", padx=(6, 0))

        # Forecast body: chart left, table right
        fc_body = tk.Frame(fc_outer, bg=T["card"])
        fc_body.pack(fill="both", expand=True)

        # Matplotlib chart
        self.hist_forecast_fig = plt.figure(figsize=(7, 3.2), tight_layout=True)
        self.hist_forecast_fig.patch.set_facecolor(T["graph_bg"])
        self.hist_forecast_canvas = FigureCanvasTkAgg(
            self.hist_forecast_fig, master=fc_body)
        self.hist_forecast_canvas.get_tk_widget().pack(
            side="left", fill="both", expand=True, padx=(4, 0), pady=4)

        # Predictions treeview
        fc_tbl_frame = tk.Frame(fc_body, bg=T["card"], width=240)
        fc_tbl_frame.pack(side="right", fill="y", padx=(4, 6), pady=4)
        fc_tbl_frame.pack_propagate(False)

        tk.Label(fc_tbl_frame, text="Predicted Values",
                 bg=T["card_header"], fg=T["primary_lt"],
                 font=("Segoe UI", 9, "bold")).pack(fill="x", ipady=3)

        fc_vsb = ttk.Scrollbar(fc_tbl_frame, orient="vertical")
        self.hist_forecast_fcst_tree = ttk.Treeview(
            fc_tbl_frame, columns=("Step", "Value", "Lo 95", "Hi 95"),
            show="headings", selectmode="none",
            yscrollcommand=fc_vsb.set, height=7)
        fc_vsb.configure(command=self.hist_forecast_fcst_tree.yview)
        for col, w in [("Step", 46), ("Value", 60), ("Lo 95", 60), ("Hi 95", 60)]:
            self.hist_forecast_fcst_tree.heading(col, text=col)
            self.hist_forecast_fcst_tree.column(col, width=w, minwidth=w)
        fc_vsb.pack(side="right", fill="y")
        self.hist_forecast_fcst_tree.pack(fill="both", expand=True)

        # Initial blank chart
        self._fc_draw_blank("Run a forecast to see predictions.")

    def _hist_append_record(self, idx, analysis_type):
        """
        Append a history record for camera *idx* using the latest comm_data snapshot
        (already computed by push_plant_comm_update).  Called from perform_analysis.
        """
        snap = self.comm_data[idx] if idx < len(self.comm_data) else None
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if snap:
            yellow_avg = round(sum(r["yellow"] for r in snap["rois"]) / max(len(snap["rois"]), 1), 1)
            brown_avg  = round(sum(r["brown"]  for r in snap["rois"]) / max(len(snap["rois"]), 1), 1)
            purple_avg = round(sum(r["purple"] for r in snap["rois"]) / max(len(snap["rois"]), 1), 1)
            record = {
                "id":            len(self.plant_history) + 1,
                "timestamp":     ts,
                "camera_idx":    idx,
                "panel":         f"Panel {idx + 1}",
                "analysis_type": analysis_type,
                "n_plants":      snap["n_rois"],
                "ndvi":          snap["ndvi"],
                "score":         snap["score"],
                "h_label":       snap["h_label"],
                "h_color":       snap["h_color"],
                "yellow":        yellow_avg,
                "brown":         brown_avg,
                "purple":        purple_avg,
                "flags":         snap["flags"],
                "rois":          snap["rois"],
            }
        else:
            record = {
                "id":            len(self.plant_history) + 1,
                "timestamp":     ts,
                "camera_idx":    idx,
                "panel":         f"Panel {idx + 1}",
                "analysis_type": analysis_type,
                "n_plants":      0,
                "ndvi":          0.0,
                "score":         0.0,
                "h_label":       "—",
                "h_color":       THEME["text_muted"],
                "yellow":        0.0,
                "brown":         0.0,
                "purple":        0.0,
                "flags":         [],
                "rois":          [],
            }

        self.plant_history.append(record)
        if not self.is_closing:
            self.root.after(0, lambda r=record: self._hist_add_row_to_tree(r))

    def _hist_add_row_to_tree(self, record):
        """Insert one record into the audit treeview and refresh summary stats."""
        if self.hist_tree is None:
            return
        if not self._hist_record_passes_filter(record):
            self._hist_update_stats()
            return

        flags_str = "; ".join(record["flags"]) if record["flags"] else "—"
        tag  = record["h_label"] if record["h_label"] in (
            "Excellent", "Good", "Fair", "Poor", "Critical") else ""
        atype = record["analysis_type"]
        if len(atype) > 28:
            atype = atype[:26] + "…"

        self.hist_tree.insert("", "end", iid=str(record["id"]), tags=(tag,), values=(
            record["id"],
            record["timestamp"],
            record["panel"],
            atype,
            record["n_plants"],
            f"{record['ndvi']:+.3f}",
            f"{record['score']:.2f}",
            record["h_label"],
            f"{record['yellow']:.1f}%",
            f"{record['brown']:.1f}%",
            f"{record['purple']:.1f}%",
            flags_str,
        ))
        self.hist_tree.yview_moveto(1.0)
        self._hist_update_stats()

    def _hist_record_passes_filter(self, record):
        """Return True if *record* satisfies the current filter settings."""
        panel_f = self.hist_filter_panel_var.get()
        type_f  = self.hist_filter_type_var.get()
        search  = self.hist_search_var.get().strip().lower()
        if panel_f != "All Panels" and record["panel"] != panel_f:
            return False
        if type_f != "All Types" and record["analysis_type"] != type_f:
            return False
        if search:
            row_text = " ".join(str(v) for v in (
                record["timestamp"], record["panel"], record["analysis_type"],
                record["h_label"], " ".join(record["flags"]))).lower()
            if search not in row_text:
                return False
        return True

    def _hist_apply_filters(self):
        """Rebuild the treeview showing only records that pass the current filters."""
        if self.hist_tree is None:
            return
        for iid in self.hist_tree.get_children():
            self.hist_tree.delete(iid)
        for record in self.plant_history:
            if self._hist_record_passes_filter(record):
                self._hist_add_row_to_tree(record)
        self._hist_update_stats()

    def _hist_reset_filters(self):
        """Clear all filters and reload the full history."""
        self.hist_filter_panel_var.set("All Panels")
        self.hist_filter_type_var.set("All Types")
        self.hist_search_var.set("")
        self._hist_apply_filters()

    def _hist_update_stats(self):
        """Refresh the summary strip."""
        total = len(self.plant_history)
        self._hist_stat_total.set(str(total))
        today_str   = datetime.now().strftime("%Y-%m-%d")
        today_count = sum(1 for r in self.plant_history if r["timestamp"].startswith(today_str))
        self._hist_stat_today.set(str(today_count))
        if total:
            avg  = sum(r["score"] for r in self.plant_history) / total
            self._hist_stat_avg.set(f"{avg:.2f}")
            self._hist_stat_last.set(self.plant_history[-1]["panel"])
        else:
            self._hist_stat_avg.set("—")
            self._hist_stat_last.set("—")
        visible = len(self.hist_tree.get_children()) if self.hist_tree else 0
        self.hist_row_count_var.set(f"{visible} of {total} records")

    def _hist_on_select(self, event=None):
        """Populate the ROI detail panel for the selected history row."""
        if self.hist_tree is None or self.hist_detail_tree is None:
            return
        sel = self.hist_tree.selection()
        if not sel:
            return
        iid = sel[0]
        record = next((r for r in self.plant_history if str(r["id"]) == iid), None)
        if record is None:
            return
        for row in self.hist_detail_tree.get_children():
            self.hist_detail_tree.delete(row)
        for roi in record["rois"]:
            tag = roi["h_label"] if roi["h_label"] in (
                "Excellent", "Good", "Fair", "Poor", "Critical") else ""
            self.hist_detail_tree.insert("", "end", tags=(tag,), values=(
                roi["label"],
                f"{roi['ndvi']:+.3f}",
                f"{roi['score']:.2f}",
                roi["h_label"],
                f"{roi['yellow']:.1f}%",
                f"{roi['brown']:.1f}%",
                f"{roi['purple']:.1f}%",
                "; ".join(roi["flags"]) if roi["flags"] else "—",
            ))

    def _hist_sort_by(self, col):
        """Sort the audit treeview by *col* (toggles ascending/descending)."""
        if self.hist_tree is None:
            return
        iids = list(self.hist_tree.get_children())
        if not iids:
            return
        id_map = {str(r["id"]): r for r in self.plant_history}
        col_key = {
            "#":        lambda r: r["id"],
            "Timestamp":lambda r: r["timestamp"],
            "Panel":    lambda r: r["panel"],
            "Type":     lambda r: r["analysis_type"],
            "Plants":   lambda r: r["n_plants"],
            "NDVI~":    lambda r: r["ndvi"],
            "Score":    lambda r: r["score"],
            "Health":   lambda r: r["h_label"],
            "Yellow%":  lambda r: r["yellow"],
            "Brown%":   lambda r: r["brown"],
            "Purple%":  lambda r: r["purple"],
            "Flags":    lambda r: "; ".join(r["flags"]),
        }
        key_fn  = col_key.get(col, lambda r: r["id"])
        reverse = not self._hist_sort_rev.get(col, False)
        self._hist_sort_rev[col] = reverse
        records = [id_map[iid] for iid in iids if iid in id_map]
        records.sort(key=key_fn, reverse=reverse)
        for iid in self.hist_tree.get_children():
            self.hist_tree.delete(iid)
        for record in records:
            if self._hist_record_passes_filter(record):
                self._hist_add_row_to_tree(record)

    def _hist_export_csv(self):
        """Export the currently visible (filtered) history rows to a CSV file."""
        import csv as _csv
        path = filedialog.asksaveasfilename(
            title="Save Plant History CSV",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not path:
            return
        try:
            headers = ["#", "Timestamp", "Panel", "Analysis Type", "Plants",
                       "NDVI~", "Score", "Health", "Yellow%", "Brown%", "Purple%", "Flags"]
            visible_iids = set(self.hist_tree.get_children()) if self.hist_tree else set()
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = _csv.writer(fh)
                writer.writerow(headers)
                for record in self.plant_history:
                    if str(record["id"]) not in visible_iids:
                        continue
                    writer.writerow([
                        record["id"],
                        record["timestamp"],
                        record["panel"],
                        record["analysis_type"],
                        record["n_plants"],
                        f"{record['ndvi']:+.3f}",
                        f"{record['score']:.2f}",
                        record["h_label"],
                        f"{record['yellow']:.1f}",
                        f"{record['brown']:.1f}",
                        f"{record['purple']:.1f}",
                        "; ".join(record["flags"]),
                    ])
            messagebox.showinfo("Export CSV", f"History exported to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export CSV", f"Could not save:\n{exc}")

    def _hist_clear(self):
        """Clear all history records after user confirmation."""
        if not messagebox.askyesno(
                "Clear History",
                "This will permanently delete all plant history records.\n\nContinue?"):
            return
        self.plant_history.clear()
        if self.hist_tree:
            for iid in self.hist_tree.get_children():
                self.hist_tree.delete(iid)
        if self.hist_detail_tree:
            for iid in self.hist_detail_tree.get_children():
                self.hist_detail_tree.delete(iid)
        self._hist_update_stats()

    # ── Forecasting engine ─────────────────────────────────────────────────────
    # Primary model: Holt-Winters (Triple) Exponential Smoothing via statsmodels.
    # Holt-Winters (Brown 1959 / Holt 1957 / Winters 1960) decomposes a series
    # into level, trend, and seasonal components each updated with its own
    # smoothing factor (alpha, beta, gamma).
    # Forecast equation: yhat_{t+h} = (L_t + h * B_t) + S_{t+h-m}
    #
    # Fallback when statsmodels is unavailable or the series has < 4 points:
    #   Holt's Double Exponential Smoothing (level + trend, pure numpy)

    @staticmethod
    def _hw_double_exp_smooth(series, h, alpha=0.4, beta=0.2):
        """
        Holt's Double Exponential Smoothing — handles level and additive trend.
        Returns (forecast, lower_80, upper_80) each of length *h*.
        Reference: Holt (1957), 'Forecasting seasonals and trends by
        exponentially weighted moving averages', ONR Research Memo 52.
        """
        y = np.array(series, dtype=float)
        n = len(y)
        # Initialise
        L  = y[0]
        B  = (y[-1] - y[0]) / max(n - 1, 1)
        # Smoothing pass
        residuals = []
        for i in range(1, n):
            L_prev, B_prev = L, B
            L = alpha * y[i]       + (1.0 - alpha) * (L_prev + B_prev)
            B = beta  * (L - L_prev) + (1.0 - beta)  * B_prev
            residuals.append(y[i] - (L_prev + B_prev))
        sigma = float(np.std(residuals) if residuals else 0.05)
        z80   = 1.282
        fcst, lo, hi = [], [], []
        for k in range(1, h + 1):
            f      = L + k * B
            margin = z80 * sigma * np.sqrt(k)
            fcst.append(f)
            lo.append(f - margin)
            hi.append(f + margin)
        return fcst, lo, hi

    def _hist_series_for(self, panel_filter, metric):
        """Extract an ordered list of metric values from plant_history."""
        key_map = {
            "Health Score": "score",
            "NDVI~":  "ndvi",
            "Yellow%": "yellow",
            "Brown%":  "brown",
            "Purple%": "purple",
        }
        key = key_map.get(metric, "score")
        return [
            r[key] for r in self.plant_history
            if panel_filter == "All Panels" or r["panel"] == panel_filter
        ]

    def _hist_run_forecast(self):
        """
        Collect the selected metric time-series, fit Holt-Winters Exponential
        Smoothing (statsmodels, additive trend with damping), fall back to
        Holt's Double ES (pure numpy) when statsmodels is unavailable or
        the series is too short (< 4 points), then render the chart and
        populate the predictions table.
        """
        T       = THEME
        panel   = self.hist_forecast_panel_var.get()
        metric  = self.hist_forecast_metric_var.get()
        horizon = max(1, min(30, self.hist_forecast_horizon_var.get()))
        series  = self._hist_series_for(panel, metric)

        if len(series) < 2:
            self._fc_status_lbl.config(
                text="Need >= 2 data points to forecast.", fg=T["warning"])
            self._fc_draw_blank("Not enough data — run more analyses first.")
            return

        self._fc_status_lbl.config(text="Computing...", fg=T["text_muted"])
        self.root.update_idletasks()

        bounds = {
            "Health Score": (0.0,    1.0),
            "NDVI~":        (-1.0,   1.0),
            "Yellow%":      (0.0, 100.0),
            "Brown%":       (0.0, 100.0),
            "Purple%":      (0.0, 100.0),
        }
        lo_b, hi_b = bounds.get(metric, (-1e9, 1e9))
        model_name = ""

        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            if len(series) < 4:
                raise ValueError("too short for statsmodels HW")
            fit = ExponentialSmoothing(
                series,
                trend="add",
                damped_trend=True,
                initialization_method="estimated",
            ).fit(optimized=True, remove_bias=True)
            fcst_arr = fit.forecast(horizon)
            sigma    = float(np.std(fit.resid)) if hasattr(fit, "resid") and len(fit.resid) > 0 else 0.05
            z80      = 1.282
            steps    = np.arange(1, horizon + 1)
            lo_arr   = (fcst_arr - z80 * sigma * np.sqrt(steps)).tolist()
            hi_arr   = (fcst_arr + z80 * sigma * np.sqrt(steps)).tolist()
            fcst     = fcst_arr.tolist()
            model_name = "Holt-Winters ES (additive trend, damped) — statsmodels"
        except Exception:
            fcst, lo_arr, hi_arr = self._hw_double_exp_smooth(series, horizon)
            model_name = f"Holt Double ES (numpy, n={len(series)})"

        # Clamp to physically meaningful range
        fcst   = [max(lo_b, min(hi_b, v)) for v in fcst]
        lo_arr = [max(lo_b, min(hi_b, v)) for v in lo_arr]
        hi_arr = [max(lo_b, min(hi_b, v)) for v in hi_arr]

        self._fc_plot(series, fcst, lo_arr, hi_arr, metric, panel, model_name)
        self._fc_populate_table(fcst, lo_arr, hi_arr)
        self._fc_status_lbl.config(text=f"OK  {model_name}", fg=T["success"])

    def _fc_draw_blank(self, message=""):
        """Draw a placeholder message on the forecast figure."""
        T   = THEME
        if self.hist_forecast_fig is None:
            return
        fig = self.hist_forecast_fig
        fig.clear()
        ax  = fig.add_subplot(111)
        ax.set_facecolor(T["graph_ax"])
        ax.text(0.5, 0.5, message,
                ha="center", va="center",
                color=T["text_muted"], fontsize=10,
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(T["border"])
        if self.hist_forecast_canvas:
            self.hist_forecast_canvas.draw()

    def _fc_plot(self, history, forecast, lo, hi, metric, panel, model_name):
        """Render historical observations + Holt-Winters forecast with CI band."""
        T   = THEME
        fig = self.hist_forecast_fig
        fig.clear()
        ax  = fig.add_subplot(111)
        ax.set_facecolor(T["graph_ax"])
        fig.patch.set_facecolor(T["graph_bg"])

        n      = len(history)
        x_hist = list(range(n))
        x_fc   = list(range(n - 1, n + len(forecast)))

        # Observed
        ax.plot(x_hist, history,
                color=T["primary"], linewidth=1.8,
                marker="o", markersize=4, label="Observed", zorder=3)

        # Forecast continuation
        ax.plot(x_fc, [history[-1]] + forecast,
                color=T["accent"], linewidth=2, linestyle="--",
                marker="o", markersize=4, label="Forecast", zorder=4)

        # 80% confidence interval band
        ci_x = list(range(n, n + len(forecast)))
        ax.fill_between(ci_x, lo, hi,
                        color=T["accent"], alpha=0.12, label="80% CI")

        # Divider between history and forecast
        ax.axvline(x=n - 1, color=T["text_dim"], linewidth=0.8, linestyle=":")

        ax.set_title(f"{metric}  |  {panel}  |  {model_name}",
                     color=T["text_muted"], fontsize=7.5, pad=4)
        ax.set_xlabel("Analysis run #", color=T["text_muted"], fontsize=8)
        ax.set_ylabel(metric,           color=T["text_muted"], fontsize=8)
        ax.tick_params(colors=T["text_dim"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(T["border"])
        ax.legend(facecolor=T["card"], edgecolor=T["border"],
                  labelcolor=T["text_muted"], fontsize=7, loc="best")
        ax.grid(axis="y", color=T["border"], linewidth=0.4, alpha=0.5)
        self.hist_forecast_canvas.draw()

    def _fc_populate_table(self, forecast, lo, hi):
        """Fill the forecast predictions treeview."""
        tree = self.hist_forecast_fcst_tree
        if tree is None:
            return
        for iid in tree.get_children():
            tree.delete(iid)
        for i, (f, l, h) in enumerate(zip(forecast, lo, hi), start=1):
            tree.insert("", "end", values=(
                f"+{i}", f"{f:.4f}", f"{l:.4f}", f"{h:.4f}"))

    # ── Hydroponics tab ────────────────────────────────────────────────────────

    # Human-readable channel labels + units
    _HYDRO_META = {
        'ph':          ("pH",            "",       0,    14,   6.0),
        'ec':          ("EC",            "mS/cm",  0,     5,   1.8),
        'tds':         ("TDS",           "ppm",    0,  3000, 1200),
        'water_temp':  ("Water Temp",    "°C",     0,    40,  21.0),
        'do':          ("Dissolved O₂",  "mg/L",   0,    15,   7.5),
        'orp':         ("ORP",           "mV",  -200,   600,  300),
        'nitrogen':    ("Nitrogen (N)",  "ppm",    0,   500,  200),
        'phosphorus':  ("Phosphorus (P)","ppm",    0,   200,   45),
        'potassium':   ("Potassium (K)", "ppm",    0,   600,  225),
    }

    # Deficiency / toxicity advisory table
    _HYDRO_ADVICE = {
        'ph':  {
            'low':  "pH too low — add pH Up solution (potassium hydroxide).",
            'high': "pH too high — add pH Down solution (phosphoric acid).",
        },
        'ec':  {
            'low':  "EC too low — nutrients are under-dosed; increase nutrient solution strength.",
            'high': "EC too high — dilute reservoir with fresh water; risk of nutrient burn.",
        },
        'tds': {
            'low':  "TDS too low — nutrient concentration insufficient; top up nutrient mix.",
            'high': "TDS too high — flush reservoir and prepare fresh solution.",
        },
        'water_temp': {
            'low':  "Water temp too low — root growth slows; warm reservoir or use aquarium heater.",
            'high': "Water temp too high — dissolved O₂ drops and root rot risk rises; chill reservoir.",
        },
        'do':  {
            'low':  "DO too low — increase aeration (air pump / diffuser); lower water temperature.",
            'high': "DO slightly high — normal, minor benefit.",
        },
        'orp': {
            'low':  "ORP low — water may be stagnant or contaminated; consider H₂O₂ treatment.",
            'high': "ORP very high — excess oxidiser; flush reservoir.",
        },
        'nitrogen':   {'low': "N deficiency — yellowing older leaves; increase N source (calcium nitrate).",
                       'high': "N excess — dark green, stunted; reduce nitrates."},
        'phosphorus': {'low': "P deficiency — purple/red stems; add monopotassium phosphate.",
                       'high': "P excess — can lock out zinc/iron; reduce P dose."},
        'potassium':  {'low': "K deficiency — brown leaf edges; increase potassium sulfate.",
                       'high': "K excess — can antagonise calcium/magnesium; reduce K dose."},
    }

    def create_hydro_widgets(self):
        T = THEME
        _btn  = dict(bg=T["primary_dk"], fg=T["text"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary"], activeforeground=T["text"],
                     cursor="hand2", bd=0)
        _sbtn = dict(bg=T["card_header"], fg=T["primary_lt"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary_dk"], activeforeground=T["text"],
                     cursor="hand2", bd=0)
        _lf   = dict(bg=T["card"], fg=T["primary"], font=("Segoe UI", 10, "bold"),
                     relief="groove", bd=1)
        _ent  = dict(bg=T["entry"], fg=T["text"], relief="flat",
                     highlightthickness=1, highlightcolor=T["border"],
                     highlightbackground=T["border"],
                     insertbackground=T["primary"], bd=0, font=("Segoe UI", 9))

        outer = ScrollableFrame(self.hydro_tab, bg=T["panel"])
        outer.pack(fill="both", expand=True)
        main = outer.scrollable_frame
        main.configure(bg=T["panel"])

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = tk.Frame(main, bg=T["panel"], pady=6)
        toolbar.pack(fill="x", padx=8)

        # Input mode selector
        tk.Label(toolbar, text="Input:", bg=T["panel"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(4, 2))
        for mode in ("Serial", "MQTT", "Manual"):
            tk.Radiobutton(toolbar, text=mode, variable=self._hydro_input_mode,
                           value=mode, command=self._hydro_mode_changed,
                           bg=T["panel"], fg=T["text_muted"],
                           selectcolor=T["primary_dk"],
                           activebackground=T["panel"],
                           activeforeground=T["primary_lt"],
                           font=("Segoe UI", 9)).pack(side="left", padx=(0, 6))

        self._hydro_connect_btn = tk.Button(toolbar, text="⚡ Connect",
                                             command=self._hydro_connect, **_btn)
        self._hydro_connect_btn.pack(side="left", padx=(4, 4))
        tk.Button(toolbar, text="🗑 Clear History",
                  command=self._hydro_clear_history, **_sbtn).pack(side="left", padx=(0, 4))
        tk.Button(toolbar, text="⬇ Export CSV",
                  command=self._hydro_export_csv, **_btn).pack(side="left", padx=(0, 4))

        self._hydro_status_lbl = tk.Label(toolbar, textvariable=self.hydro_status_var,
                                           bg=T["panel"], fg=T["error"],
                                           font=("Segoe UI", 10, "bold"))
        self._hydro_status_lbl.pack(side="left", padx=(12, 0))
        tk.Label(toolbar, text="Last reading:",
                 bg=T["panel"], fg=T["text_dim"], font=("Segoe UI", 8)).pack(side="right", padx=(0, 2))
        tk.Label(toolbar, textvariable=self.hydro_last_ts,
                 bg=T["panel"], fg=T["text_muted"], font=("Segoe UI", 8)).pack(side="right")

        # ── Connection config panel (mode-switched) ───────────────────────────
        self._hydro_conn_frame = tk.LabelFrame(main, text=" Connection ", **_lf)
        self._hydro_conn_frame.pack(fill="x", padx=8, pady=(0, 6))

        # Serial sub-panel
        self._hydro_serial_frame = tk.Frame(self._hydro_conn_frame, bg=T["card"])
        self._hydro_serial_frame.pack(fill="x")
        ports = self.get_serial_ports()
        tk.Label(self._hydro_serial_frame, text="Port:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        hydro_port_cb = ttk.Combobox(self._hydro_serial_frame, textvariable=self.hydro_port_var,
                                      values=ports, state="readonly", width=20, font=("Segoe UI", 9))
        hydro_port_cb.grid(row=0, column=1, sticky="ew", padx=4)
        if ports:
            hydro_port_cb.current(0)
        tk.Label(self._hydro_serial_frame, text="Baud:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w", padx=4)
        ttk.Combobox(self._hydro_serial_frame, textvariable=self.hydro_baud_var,
                     values=['9600','19200','38400','57600','115200'], state="readonly",
                     width=10, font=("Segoe UI", 9)).grid(row=0, column=3, padx=4)
        tk.Button(self._hydro_serial_frame, text="↺ Ports",
                  command=lambda: hydro_port_cb.configure(values=self.get_serial_ports()), **_sbtn
                  ).grid(row=0, column=4, padx=6)
        tk.Label(self._hydro_serial_frame,
                 text="Expected JSON per line: {\"ph\":6.1,\"ec\":1.8,\"tds\":1200,...}",
                 bg=T["card"], fg=T["text_dim"], font=("Segoe UI", 7, "italic")).grid(
                 row=1, column=0, columnspan=5, sticky="w", padx=8, pady=(0, 4))

        # MQTT sub-panel (hidden initially)
        self._hydro_mqtt_frame = tk.Frame(self._hydro_conn_frame, bg=T["card"])
        for ci, (lbl, var, w) in enumerate([
            ("Host",  self.hydro_mqtt_host_var,  16),
            ("Port",  self.hydro_mqtt_port_var,   6),
            ("Topic", self.hydro_mqtt_topic_var,  22),
            ("User",  self.hydro_mqtt_user_var,   12),
            ("Pass",  self.hydro_mqtt_pass_var,   12),
        ]):
            tk.Label(self._hydro_mqtt_frame, text=f"{lbl}:", bg=T["card"],
                     fg=T["text_muted"], font=("Segoe UI", 9)).grid(
                     row=0, column=ci*2, sticky="w", padx=(6 if ci==0 else 2, 2), pady=4)
            tk.Entry(self._hydro_mqtt_frame, textvariable=var, width=w, **_ent).grid(
                     row=0, column=ci*2+1, padx=(0,6), pady=4)
        tk.Label(self._hydro_mqtt_frame,
                 text="MQTT message payload: JSON object with channel keys",
                 bg=T["card"], fg=T["text_dim"], font=("Segoe UI", 7, "italic")).grid(
                 row=1, column=0, columnspan=10, sticky="w", padx=8, pady=(0, 4))

        # Manual entry sub-panel
        self._hydro_manual_frame = tk.Frame(self._hydro_conn_frame, bg=T["card"])
        manual_row = tk.Frame(self._hydro_manual_frame, bg=T["card"])
        manual_row.pack(fill="x", padx=6, pady=4)
        self._hydro_manual_vars = {}
        for ci, ch in enumerate(self.hydro_channels):
            meta = self._HYDRO_META[ch]
            tk.Label(manual_row, text=f"{meta[0]}:", bg=T["card"], fg=T["text_muted"],
                     font=("Segoe UI", 8)).grid(row=0, column=ci*2, padx=(4,1), pady=2)
            v = tk.StringVar(value="")
            self._hydro_manual_vars[ch] = v
            tk.Entry(manual_row, textvariable=v, width=7, **_ent).grid(
                row=0, column=ci*2+1, padx=(0,4), pady=2)
        tk.Button(self._hydro_manual_frame, text="Submit Reading",
                  command=self._hydro_submit_manual, **_btn).pack(pady=(0, 4))

        self._hydro_mode_changed()  # show correct sub-panel

        # ── Gauge row ─────────────────────────────────────────────────────────
        gauge_lf = tk.LabelFrame(main, text=" Current Nutrient Readings ", **_lf)
        gauge_lf.pack(fill="x", padx=8, pady=(0, 6))
        gauge_inner = tk.Frame(gauge_lf, bg=T["card"])
        gauge_inner.pack(fill="x")

        gauge_ranges = {
            'ph':          (0,   14),
            'ec':          (0,    5),
            'tds':         (0, 3000),
            'water_temp':  (0,   40),
            'do':          (0,   15),
            'orp':         (-200, 600),
            'nitrogen':    (0,  500),
            'phosphorus':  (0,  200),
            'potassium':   (0,  600),
        }
        for ci, ch in enumerate(self.hydro_channels):
            meta = self._HYDRO_META[ch]
            lo, hi = gauge_ranges[ch]
            cell = tk.Frame(gauge_inner, bg=T["card"])
            cell.grid(row=0, column=ci, padx=6, pady=4)
            g = CircularGauge(cell, f"{meta[0]}\n{meta[1]}", lo, hi, width=130, height=130)
            g.pack()
            self.hydro_gauges[ch] = g
            # Value label
            val_var = tk.StringVar(value="—")
            self.hydro_val_labels[ch] = val_var
            tk.Label(cell, textvariable=val_var, bg=T["card"], fg=T["accent"],
                     font=("Segoe UI", 9, "bold")).pack()
            # Target range label
            lo_t, hi_t = self.hydro_targets[ch]
            tk.Label(cell, text=f"Target: {lo_t}–{hi_t} {meta[1]}",
                     bg=T["card"], fg=T["text_dim"], font=("Segoe UI", 7)).pack()
            # Alert badge
            alert_lbl = tk.Label(cell, text="", bg=T["card"],
                                  font=("Segoe UI", 7, "bold"))
            alert_lbl.pack()
            self.hydro_alert_labels[ch] = alert_lbl
        gauge_inner.columnconfigure(tuple(range(len(self.hydro_channels))), weight=1)

        # ── Deficiency analysis panel ─────────────────────────────────────────
        defi_lf = tk.LabelFrame(main, text=" Deficiency / Toxicity Analysis ", **_lf)
        defi_lf.pack(fill="x", padx=8, pady=(0, 6))
        self._hydro_defi_text = tk.Text(
            defi_lf, bg=T["card"], fg=T["text"], height=4,
            font=("Segoe UI", 9), relief="flat", wrap="word", state="disabled")
        self._hydro_defi_text.pack(fill="x", padx=4, pady=4)
        self._hydro_defi_text.tag_configure("ok",    foreground=T["success"])
        self._hydro_defi_text.tag_configure("warn",  foreground=T["warning"])
        self._hydro_defi_text.tag_configure("error", foreground=T["error"])

        # ── Dosing pump controls ──────────────────────────────────────────────
        pump_lf = tk.LabelFrame(main, text=" Dosing Pump Controls ", **_lf)
        pump_lf.pack(fill="x", padx=8, pady=(0, 6))
        pump_inner = tk.Frame(pump_lf, bg=T["card"])
        pump_inner.pack(fill="x", padx=4, pady=4)
        self._hydro_pump_btns = []
        for pi in range(8):
            col = tk.Frame(pump_inner, bg=T["card"])
            col.grid(row=0, column=pi, padx=6)
            var = self.hydro_pump_vars[pi]
            btn = tk.Button(col, text=self.hydro_pump_names[pi],
                            bg=T["card_header"], fg=T["text_muted"],
                            relief="flat", font=("Segoe UI", 8), padx=6, pady=4,
                            activebackground=T["primary_dk"],
                            cursor="hand2", bd=0,
                            command=lambda p=pi: self._hydro_pump_toggle(p))
            btn.pack()
            dot = tk.Label(col, text="●", bg=T["card"], fg=T["text_dim"],
                           font=("Segoe UI", 10))
            dot.pack()
            self._hydro_pump_btns.append((btn, dot))
        pump_inner.columnconfigure(tuple(range(8)), weight=1)

        # ── Setpoints editor ──────────────────────────────────────────────────
        sp_lf = tk.LabelFrame(main, text=" Target Setpoints (Low — High) ", **_lf)
        sp_lf.pack(fill="x", padx=8, pady=(0, 6))
        sp_inner = tk.Frame(sp_lf, bg=T["card"])
        sp_inner.pack(fill="x", padx=4, pady=4)
        for ci, ch in enumerate(self.hydro_channels):
            meta = self._HYDRO_META[ch]
            lo_v, hi_v = self.hydro_target_vars[ch]
            tk.Label(sp_inner, text=f"{meta[0]}:", bg=T["card"],
                     fg=T["text_muted"], font=("Segoe UI", 8)).grid(
                     row=0, column=ci*3, sticky="e", padx=(8,2))
            tk.Entry(sp_inner, textvariable=lo_v, width=6, **_ent).grid(
                     row=0, column=ci*3+1, padx=1)
            tk.Entry(sp_inner, textvariable=hi_v, width=6, **_ent).grid(
                     row=0, column=ci*3+2, padx=(1,8))
        tk.Button(sp_lf, text="Apply Setpoints",
                  command=self._hydro_apply_setpoints, **_btn).pack(pady=(0, 4))

        # ── Historical chart ──────────────────────────────────────────────────
        chart_lf = tk.LabelFrame(main, text=" Historical Trend ", **_lf)
        chart_lf.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        chart_toolbar = tk.Frame(chart_lf, bg=T["card"])
        chart_toolbar.pack(fill="x")
        tk.Label(chart_toolbar, text="Channel:", bg=T["card"],
                 fg=T["text_muted"], font=("Segoe UI", 9)).pack(side="left", padx=6)
        ch_choices = [self._HYDRO_META[c][0] for c in self.hydro_channels]
        ch_cb = ttk.Combobox(chart_toolbar, textvariable=self.hydro_graph_ch_var,
                              values=self.hydro_channels, state="readonly", width=16,
                              font=("Segoe UI", 9))
        ch_cb.pack(side="left", padx=(0, 8))
        tk.Button(chart_toolbar, text="↻ Refresh Chart",
                  command=self._hydro_update_chart, **_sbtn).pack(side="left")

        self.hydro_fig = plt.figure(figsize=(10, 3.2), tight_layout=True)
        self.hydro_fig.patch.set_facecolor(T["graph_bg"])
        self.hydro_canvas = FigureCanvasTkAgg(self.hydro_fig, master=chart_lf)
        self.hydro_canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._hydro_draw_blank_chart("No data yet — connect a sensor or enter values manually.")

        # ── Audit log ─────────────────────────────────────────────────────────
        log_lf = tk.LabelFrame(main, text=" Reading Log ", **_lf)
        log_lf.pack(fill="both", expand=True, padx=8, pady=(0, 10))
        log_cols  = ["Timestamp"] + [self._HYDRO_META[c][0] for c in self.hydro_channels] + ["Alerts"]
        log_widths = [148] + [74]*9 + [260]
        log_vsb = ttk.Scrollbar(log_lf, orient="vertical")
        log_hsb = ttk.Scrollbar(log_lf, orient="horizontal")
        self.hydro_log_tree = ttk.Treeview(
            log_lf, columns=log_cols, show="headings",
            yscrollcommand=log_vsb.set, xscrollcommand=log_hsb.set,
            selectmode="browse", height=8)
        log_vsb.configure(command=self.hydro_log_tree.yview)
        log_hsb.configure(command=self.hydro_log_tree.xview)
        for col, w in zip(log_cols, log_widths):
            self.hydro_log_tree.heading(col, text=col)
            self.hydro_log_tree.column(col, width=w, minwidth=w, stretch=(col=="Alerts"))
        log_vsb.pack(side="right", fill="y")
        log_hsb.pack(side="bottom", fill="x")
        self.hydro_log_tree.pack(fill="both", expand=True)
        self.hydro_log_tree.tag_configure("alert", background="#1a0a00", foreground=THEME["warning"])

    # ── Hydroponics helpers ────────────────────────────────────────────────────

    def _hydro_mode_changed(self):
        """Show/hide connection sub-panels based on selected input mode."""
        mode = self._hydro_input_mode.get()
        self._hydro_serial_frame.pack_forget()
        self._hydro_mqtt_frame.pack_forget()
        self._hydro_manual_frame.pack_forget()
        if mode == "Serial":
            self._hydro_serial_frame.pack(fill="x")
            self._hydro_connect_btn.config(state="normal", text="⚡ Connect")
        elif mode == "MQTT":
            self._hydro_mqtt_frame.pack(fill="x")
            self._hydro_connect_btn.config(state="normal", text="⚡ Connect MQTT")
        else:
            self._hydro_manual_frame.pack(fill="x")
            self._hydro_connect_btn.config(state="disabled", text="—")

    def _hydro_connect(self):
        """Connect or disconnect based on current input mode."""
        mode = self._hydro_input_mode.get()
        if mode == "Serial":
            self._hydro_serial_toggle()
        elif mode == "MQTT":
            self._hydro_mqtt_toggle()

    def _hydro_serial_toggle(self):
        T = THEME
        if self.hydro_serial_running:
            self.hydro_serial_running = False
            if self.hydro_serial_thread and self.hydro_serial_thread.is_alive():
                self.hydro_serial_thread.join(timeout=1.5)
            if self.hydro_serial_port:
                try:
                    self.hydro_serial_port.close()
                except Exception:
                    pass
                self.hydro_serial_port = None
            self._hydro_set_status("● Disconnected", T["error"])
            self._hydro_connect_btn.config(text="⚡ Connect")
        else:
            port = self.hydro_port_var.get()
            baud = int(self.hydro_baud_var.get() or "115200")
            if not port:
                messagebox.showwarning("Hydroponics", "Select a serial port first.")
                return
            try:
                self.hydro_serial_port = serial.Serial(port, baud, timeout=1.0)
                self.hydro_serial_running = True
                self.hydro_serial_thread = threading.Thread(
                    target=self._hydro_serial_reader, daemon=True)
                self.hydro_serial_thread.start()
                self._hydro_set_status(f"● Serial {port}", T["success"])
                self._hydro_connect_btn.config(text="⏹ Disconnect")
            except Exception as exc:
                messagebox.showerror("Hydroponics Serial", str(exc))

    def _hydro_serial_reader(self):
        """Background thread: read JSON lines from the serial port."""
        while self.hydro_serial_running and self.hydro_serial_port:
            try:
                line = self.hydro_serial_port.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    self._hydro_ingest_json(line, source="serial")
            except Exception as exc:
                if self.hydro_serial_running:
                    logging.error(f"Hydro serial read error: {exc}")
                break

    def _hydro_mqtt_toggle(self):
        T = THEME
        if self._hydro_mqtt_client is not None:
            try:
                self._hydro_mqtt_client.loop_stop()
                self._hydro_mqtt_client.disconnect()
            except Exception:
                pass
            self._hydro_mqtt_client = None
            self._hydro_set_status("● Disconnected", T["error"])
            self._hydro_connect_btn.config(text="⚡ Connect MQTT")
            return
        try:
            import paho.mqtt.client as mqtt_client
            host  = self.hydro_mqtt_host_var.get() or "localhost"
            port  = int(self.hydro_mqtt_port_var.get() or "1883")
            topic = self.hydro_mqtt_topic_var.get() or "hydroponics/#"
            user  = self.hydro_mqtt_user_var.get() or None
            pw    = self.hydro_mqtt_pass_var.get() or None

            client = mqtt_client.Client(client_id="greensight_hydro")
            if user:
                client.username_pw_set(user, pw)

            def on_connect(c, userdata, flags, rc):
                if rc == 0:
                    c.subscribe(topic)
                    self.root.after(0, lambda: self._hydro_set_status(
                        f"● MQTT {host}:{port}", T["success"]))
                    self.root.after(0, lambda: self._hydro_connect_btn.config(
                        text="⏹ Disconnect MQTT"))
                else:
                    self.root.after(0, lambda: self._hydro_set_status(
                        f"MQTT error rc={rc}", T["error"]))

            def on_message(c, userdata, msg):
                try:
                    payload = msg.payload.decode("utf-8", errors="ignore")
                    self._hydro_ingest_json(payload, source="mqtt")
                except Exception as exc:
                    logging.error(f"Hydro MQTT message error: {exc}")

            client.on_connect = on_connect
            client.on_message = on_message
            client.connect_async(host, port, keepalive=60)
            client.loop_start()
            self._hydro_mqtt_client = client
            self._hydro_set_status(f"Connecting to {host}…", T["warning"])
        except ImportError:
            messagebox.showerror("MQTT",
                "paho-mqtt not installed.\nRun:  pip install paho-mqtt")
        except Exception as exc:
            messagebox.showerror("Hydroponics MQTT", str(exc))

    def _hydro_ingest_json(self, raw, source="serial"):
        """Parse a JSON nutrient reading and push it to the UI."""
        try:
            data = json.loads(raw)
        except Exception:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        reading = {}
        for ch in self.hydro_channels:
            val = data.get(ch) or data.get(ch.replace("_", "")) or data.get(ch.upper())
            if val is not None:
                try:
                    reading[ch] = float(val)
                except (TypeError, ValueError):
                    pass
        if not reading:
            return
        with self.hydro_lock:
            for ch, v in reading.items():
                self.hydro_history[ch].append(v)
                if len(self.hydro_history[ch]) > self.hydro_max_history:
                    self.hydro_history[ch].pop(0)
            self.hydro_time.append(ts)
            if len(self.hydro_time) > self.hydro_max_history:
                self.hydro_time.pop(0)
        if not self.is_closing:
            self.root.after(0, lambda r=reading, t=ts: self._hydro_push_to_ui(r, t))

    def _hydro_submit_manual(self):
        """Read manual entry fields and ingest as a sensor reading."""
        data = {}
        for ch, var in self._hydro_manual_vars.items():
            raw = var.get().strip()
            if raw:
                try:
                    data[ch] = float(raw)
                except ValueError:
                    pass
        if data:
            self._hydro_ingest_json(json.dumps(data), source="manual")

    def _hydro_push_to_ui(self, reading, ts):
        """Update all live widgets with the latest reading dict."""
        T = THEME
        alerts = []
        for ch, v in reading.items():
            meta    = self._HYDRO_META[ch]
            lo_t, hi_t = self.hydro_targets[ch]
            # Gauge
            if ch in self.hydro_gauges:
                self.hydro_gauges[ch].set_value(v)
            # Value label
            unit = meta[1]
            if ch in self.hydro_val_labels:
                self.hydro_val_labels[ch].set(f"{v:.2f} {unit}".strip())
            # Alert badge
            if ch in self.hydro_alert_labels:
                lbl = self.hydro_alert_labels[ch]
                if v < lo_t:
                    lbl.config(text="▼ LOW", fg=T["warning"])
                    alerts.append(f"{meta[0]} LOW")
                elif v > hi_t:
                    lbl.config(text="▲ HIGH", fg=T["error"])
                    alerts.append(f"{meta[0]} HIGH")
                else:
                    lbl.config(text="✔ OK", fg=T["success"])

        self.hydro_last_ts.set(ts)

        # Deficiency analysis
        self._hydro_run_deficiency_analysis(reading)

        # Append to log treeview
        row_vals = [ts]
        alert_tag = ""
        for ch in self.hydro_channels:
            v = reading.get(ch)
            row_vals.append(f"{v:.2f}" if v is not None else "—")
        row_vals.append(", ".join(alerts) if alerts else "OK")
        if alerts:
            alert_tag = "alert"
        self.hydro_log_tree.insert("", "end", values=row_vals, tags=(alert_tag,))
        self.hydro_log_tree.yview_moveto(1.0)

        # Update chart if visible channel has new data
        self._hydro_update_chart()

        # Feed automation sensor cache
        self._auto_update_sensor_cache(reading)

    def _hydro_run_deficiency_analysis(self, reading):
        """Evaluate current readings against targets and write advisory text."""
        lines = []
        for ch in self.hydro_channels:
            v = reading.get(ch)
            if v is None:
                continue
            lo_t, hi_t  = self.hydro_targets[ch]
            advice_map  = self._HYDRO_ADVICE.get(ch, {})
            if v < lo_t:
                lines.append(("warn", f"⚠ {self._HYDRO_META[ch][0]}: {v:.2f} < {lo_t}  →  {advice_map.get('low', 'Check low reading.')}"))
            elif v > hi_t:
                lines.append(("error", f"⛔ {self._HYDRO_META[ch][0]}: {v:.2f} > {hi_t}  →  {advice_map.get('high', 'Check high reading.')}"))
        if not lines:
            lines.append(("ok", "✔ All nutrient channels within target range."))

        txt = self._hydro_defi_text
        txt.config(state="normal")
        txt.delete("1.0", "end")
        txt.config(height=max(3, min(10, len(lines) + 1)))
        for tag, line in lines:
            txt.insert("end", line + "\n", tag)
        txt.config(state="disabled")

    def _hydro_update_chart(self):
        """Redraw the historical trend chart for the selected channel."""
        T   = THEME
        ch  = self.hydro_graph_ch_var.get()
        if ch not in self.hydro_history:
            return
        with self.hydro_lock:
            vals = list(self.hydro_history[ch])
        if not vals:
            self._hydro_draw_blank_chart("No data yet.")
            return
        fig = self.hydro_fig
        fig.clear()
        ax  = fig.add_subplot(111)
        ax.set_facecolor(T["graph_ax"])
        fig.patch.set_facecolor(T["graph_bg"])

        n    = len(vals)
        xs   = list(range(n))
        meta = self._HYDRO_META.get(ch, (ch, "", 0, 100, 0))
        lo_t, hi_t = self.hydro_targets.get(ch, (None, None))

        ax.plot(xs, vals, color=T["primary"], linewidth=1.6, marker="o",
                markersize=3, label=f"{meta[0]} ({meta[1]})", zorder=3)

        # Target range band
        if lo_t is not None and hi_t is not None:
            ax.axhspan(lo_t, hi_t, color=T["primary"], alpha=0.07, label="Target range")
            ax.axhline(lo_t, color=T["warning"], linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axhline(hi_t, color=T["warning"], linewidth=0.8, linestyle="--", alpha=0.6)

        ax.set_title(f"{meta[0]} — {n} readings",
                     color=T["text_muted"], fontsize=9, pad=4)
        ax.set_xlabel("Reading #", color=T["text_muted"], fontsize=8)
        ax.set_ylabel(f"{meta[0]} {meta[1]}", color=T["text_muted"], fontsize=8)
        ax.tick_params(colors=T["text_dim"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(T["border"])
        ax.legend(facecolor=T["card"], edgecolor=T["border"],
                  labelcolor=T["text_muted"], fontsize=7, loc="best")
        ax.grid(axis="y", color=T["border"], linewidth=0.4, alpha=0.5)
        self.hydro_canvas.draw()

    def _hydro_draw_blank_chart(self, message=""):
        T   = THEME
        fig = self.hydro_fig
        if fig is None:
            return
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor(T["graph_ax"])
        ax.text(0.5, 0.5, message, ha="center", va="center",
                color=T["text_muted"], fontsize=10, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(T["border"])
        if self.hydro_canvas:
            self.hydro_canvas.draw()

    def _hydro_apply_setpoints(self):
        """Read setpoint entries and update self.hydro_targets, then refresh gauges."""
        for ch in self.hydro_channels:
            lo_v, hi_v = self.hydro_target_vars[ch]
            try:
                lo = float(lo_v.get())
                hi = float(hi_v.get())
                self.hydro_targets[ch] = (lo, hi)
            except ValueError:
                pass
        messagebox.showinfo("Setpoints", "Target setpoints updated.")

    def _hydro_pump_toggle(self, pump_idx):
        """Toggle a dosing pump on/off (UI only — wire GPIO/serial for real actuation)."""
        T   = THEME
        var = self.hydro_pump_vars[pump_idx]
        var.set(not var.get())
        btn, dot = self._hydro_pump_btns[pump_idx]
        if var.get():
            btn.config(bg=T["primary_dk"], fg=T["text"])
            dot.config(fg=T["accent"])
            logging.info(f"Hydro pump '{self.hydro_pump_names[pump_idx]}' ACTIVATED")
        else:
            btn.config(bg=T["card_header"], fg=T["text_muted"])
            dot.config(fg=T["text_dim"])
            logging.info(f"Hydro pump '{self.hydro_pump_names[pump_idx]}' deactivated")

    def _hydro_clear_history(self):
        if not messagebox.askyesno("Clear History",
                "Delete all hydroponics reading history?"):
            return
        with self.hydro_lock:
            for ch in self.hydro_channels:
                self.hydro_history[ch].clear()
            self.hydro_time.clear()
        if self.hydro_log_tree:
            for iid in self.hydro_log_tree.get_children():
                self.hydro_log_tree.delete(iid)
        self._hydro_draw_blank_chart("History cleared.")

    def _hydro_set_status(self, text, color):
        T = THEME
        self.hydro_status_var.set(text)
        if hasattr(self, "_hydro_status_lbl"):
            self._hydro_status_lbl.config(fg=color)

    def _hydro_export_csv(self):
        import csv as _csv
        path = filedialog.asksaveasfilename(
            title="Save Hydroponics CSV",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not path:
            return
        try:
            headers = ["Timestamp"] + [self._HYDRO_META[c][0] for c in self.hydro_channels] + ["Alerts"]
            rows = [self.hydro_log_tree.item(iid, "values")
                    for iid in self.hydro_log_tree.get_children()]
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = _csv.writer(fh)
                w.writerow(headers)
                w.writerows(rows)
            messagebox.showinfo("Export CSV", f"Saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export CSV", str(exc))

    # ── Automations tab ────────────────────────────────────────────────────────

    # Channel display names for the trigger channel combobox
    _AUTO_CH_LABELS = {
        # Hydro
        'ph': 'pH', 'ec': 'EC (mS/cm)', 'tds': 'TDS (ppm)',
        'water_temp': 'Water Temp (°C)', 'do': 'DO (mg/L)', 'orp': 'ORP (mV)',
        'nitrogen': 'Nitrogen (ppm)', 'phosphorus': 'Phosphorus (ppm)',
        'potassium': 'Potassium (ppm)',
        # Env
        'temperature': 'Air Temp (°C)', 'humidity': 'Humidity (%)',
        'co2': 'CO₂ (ppm)', 'light': 'Light (lux)', 'soil_moisture': 'Soil Moisture (%)',
        # Health
        'health_score': 'Plant Health Score (0-1)', 'ndvi': 'NDVI~',
    }
    _AUTO_ACTION_LABELS = {
        'pump_on':      'Pump — Turn ON',
        'pump_off':     'Pump — Turn OFF',
        'pump_pulse':   'Pump — Pulse (timed)',
        'alert_popup':  'Show Alert Popup',
        'log_event':    'Log Event',
        'analyze_panel': 'Trigger Camera Analysis',
    }

    def create_automations_widgets(self):
        T = THEME
        _btn  = dict(bg=T["primary_dk"], fg=T["text"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary"], activeforeground=T["text"],
                     cursor="hand2", bd=0)
        _sbtn = dict(bg=T["card_header"], fg=T["primary_lt"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary_dk"], activeforeground=T["text"],
                     cursor="hand2", bd=0)
        _lf   = dict(bg=T["card"], fg=T["primary"], font=("Segoe UI", 10, "bold"),
                     relief="groove", bd=1)
        _ent  = dict(bg=T["entry"], fg=T["text"], relief="flat",
                     highlightthickness=1, highlightcolor=T["border"],
                     highlightbackground=T["border"],
                     insertbackground=T["primary"], bd=0, font=("Segoe UI", 9))

        outer = tk.Frame(self.auto_tab, bg=T["panel"])
        outer.pack(fill="both", expand=True)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = tk.Frame(outer, bg=T["panel"], pady=6)
        toolbar.pack(fill="x", padx=8)

        tk.Button(toolbar, text="＋ New Rule",    command=self._auto_new_rule,     **_btn).pack(side="left", padx=(4, 4))
        tk.Button(toolbar, text="✔ Enable All",  command=self._auto_enable_all,   **_sbtn).pack(side="left", padx=(0, 4))
        tk.Button(toolbar, text="⏸ Disable All", command=self._auto_disable_all,  **_sbtn).pack(side="left", padx=(0, 4))
        tk.Button(toolbar, text="✕ Delete",      command=self._auto_delete_selected, **_sbtn).pack(side="left", padx=(0, 12))
        tk.Button(toolbar, text="⬇ Export JSON", command=self._auto_export,       **_btn).pack(side="left", padx=(0, 4))
        tk.Button(toolbar, text="⬆ Import JSON", command=self._auto_import,       **_btn).pack(side="left", padx=(0, 12))

        # Engine toggle
        self._auto_engine_btn = tk.Button(
            toolbar, text="▶ Start Engine", command=self._auto_engine_toggle, **_btn)
        self._auto_engine_btn.pack(side="right", padx=(0, 4))
        self._auto_engine_lbl = tk.Label(
            toolbar, text="Engine: stopped", bg=T["panel"],
            fg=T["text_dim"], font=("Segoe UI", 8, "italic"))
        self._auto_engine_lbl.pack(side="right", padx=(0, 8))

        # ── Rule list ─────────────────────────────────────────────────────────
        rule_lf = tk.LabelFrame(outer, text=" Automation Rules ", **_lf)
        rule_lf.pack(fill="x", padx=8, pady=(0, 6))

        rule_cols  = ("#", "Name", "Enabled", "Trigger", "Operator", "Value/Time",
                      "Action", "Target", "Cooldown(s)", "Last Triggered", "Runs")
        rule_widths = (32, 140, 58, 180, 68, 90, 160, 120, 84, 140, 42)

        r_vsb = ttk.Scrollbar(rule_lf, orient="vertical")
        r_hsb = ttk.Scrollbar(rule_lf, orient="horizontal")
        self._auto_rule_tree = ttk.Treeview(
            rule_lf, columns=rule_cols, show="headings",
            yscrollcommand=r_vsb.set, xscrollcommand=r_hsb.set,
            selectmode="browse", height=7)
        r_vsb.configure(command=self._auto_rule_tree.yview)
        r_hsb.configure(command=self._auto_rule_tree.xview)
        for col, w in zip(rule_cols, rule_widths):
            self._auto_rule_tree.heading(col, text=col,
                                         command=lambda c=col: self._auto_sort_rules(c))
            self._auto_rule_tree.column(col, width=w, minwidth=w,
                                         stretch=(col == "Name"))
        self._auto_rule_tree.tag_configure("enabled",  foreground=T["success"])
        self._auto_rule_tree.tag_configure("disabled", foreground=T["text_dim"])
        self._auto_rule_tree.tag_configure("fired",    foreground=T["accent"])
        r_vsb.pack(side="right", fill="y")
        r_hsb.pack(side="bottom", fill="x")
        self._auto_rule_tree.pack(fill="both", expand=True)
        self._auto_rule_tree.bind("<<TreeviewSelect>>", self._auto_on_select)

        # ── Rule editor ───────────────────────────────────────────────────────
        ed_lf = tk.LabelFrame(outer, text=" Rule Editor ", **_lf)
        ed_lf.pack(fill="x", padx=8, pady=(0, 6))

        # We store all editor widgets' StringVars in self._ae
        ae = self._ae

        # Build tk vars
        for key in ("name", "trigger_type", "trigger_channel", "trigger_op",
                    "trigger_value", "trigger_time",
                    "action_type", "action_target", "action_duration",
                    "cooldown_s", "notes"):
            ae[key] = tk.StringVar(value="")
        ae["enabled"] = tk.BooleanVar(value=True)

        # Row 0 — name + enabled
        r0 = tk.Frame(ed_lf, bg=T["card"]); r0.pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(r0, text="Name:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Entry(r0, textvariable=ae["name"], width=26, **_ent).pack(side="left", padx=(4, 12))
        tk.Checkbutton(r0, text="Enabled", variable=ae["enabled"],
                       bg=T["card"], fg=T["text"], selectcolor=T["primary_dk"],
                       activebackground=T["card"], activeforeground=T["primary_lt"],
                       font=("Segoe UI", 9)).pack(side="left")

        # Row 1 — trigger
        r1 = tk.Frame(ed_lf, bg=T["card"]); r1.pack(fill="x", padx=8, pady=2)
        tk.Label(r1, text="Trigger:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 6))

        tk.Label(r1, text="Type:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._ae_ttype_cb = ttk.Combobox(
            r1, textvariable=ae["trigger_type"],
            values=self.AUTO_TRIGGER_TYPES, state="readonly", width=18,
            font=("Segoe UI", 9))
        self._ae_ttype_cb.pack(side="left", padx=(2, 8))
        self._ae_ttype_cb.bind("<<ComboboxSelected>>", self._auto_editor_ttype_changed)

        tk.Label(r1, text="Channel:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left")
        all_channels = (self._hydro_ch_keys + self._env_ch_keys + self._health_ch_keys)
        self._ae_channel_cb = ttk.Combobox(
            r1, textvariable=ae["trigger_channel"],
            values=all_channels, state="readonly", width=20,
            font=("Segoe UI", 9))
        self._ae_channel_cb.pack(side="left", padx=(2, 8))

        tk.Label(r1, text="Op:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left")
        ttk.Combobox(r1, textvariable=ae["trigger_op"],
                     values=self.AUTO_OPS, state="readonly", width=5,
                     font=("Segoe UI", 9)).pack(side="left", padx=(2, 8))

        tk.Label(r1, text="Value:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Entry(r1, textvariable=ae["trigger_value"], width=8, **_ent).pack(
            side="left", padx=(2, 12))

        tk.Label(r1, text="Time (HH:MM):", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Entry(r1, textvariable=ae["trigger_time"], width=7, **_ent).pack(
            side="left", padx=(2, 0))

        # Row 2 — action
        r2 = tk.Frame(ed_lf, bg=T["card"]); r2.pack(fill="x", padx=8, pady=2)
        tk.Label(r2, text="Action:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 6))

        tk.Label(r2, text="Type:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left")
        ttk.Combobox(r2, textvariable=ae["action_type"],
                     values=self.AUTO_ACTION_TYPES, state="readonly", width=20,
                     font=("Segoe UI", 9)).pack(side="left", padx=(2, 8))

        tk.Label(r2, text="Target:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left")
        pump_choices = [f"Pump {i+1}: {self.hydro_pump_names[i]}"
                        for i in range(len(self.hydro_pump_names))]
        panel_choices = [f"Panel {i+1}" for i in range(MAX_CAMERA_SLOTS)]
        self._ae_target_cb = ttk.Combobox(
            r2, textvariable=ae["action_target"],
            values=pump_choices + panel_choices + ["(message / custom)"],
            state="normal", width=26, font=("Segoe UI", 9))
        self._ae_target_cb.pack(side="left", padx=(2, 8))

        tk.Label(r2, text="Duration(s):", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Entry(r2, textvariable=ae["action_duration"], width=6, **_ent).pack(
            side="left", padx=(2, 12))

        tk.Label(r2, text="Cooldown(s):", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Entry(r2, textvariable=ae["cooldown_s"], width=6, **_ent).pack(
            side="left", padx=(2, 0))

        # Row 3 — notes + save
        r3 = tk.Frame(ed_lf, bg=T["card"]); r3.pack(fill="x", padx=8, pady=(2, 6))
        tk.Label(r3, text="Notes:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Entry(r3, textvariable=ae["notes"], width=50, **_ent).pack(
            side="left", padx=(4, 12))
        tk.Button(r3, text="💾 Save Rule", command=self._auto_save_rule, **_btn).pack(side="left", padx=(0, 4))
        tk.Button(r3, text="✕ Clear",      command=self._auto_clear_editor, **_sbtn).pack(side="left")
        self._ae["_editing_id"] = None  # id of rule being edited, or None for new

        # ── Execution log ─────────────────────────────────────────────────────
        log_lf = tk.LabelFrame(outer, text=" Execution Log ", **_lf)
        log_lf.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        log_bar = tk.Frame(log_lf, bg=T["card"])
        log_bar.pack(fill="x")
        tk.Button(log_bar, text="🗑 Clear Log",
                  command=self._auto_clear_log, **_sbtn).pack(side="left", padx=4, pady=2)
        self._auto_log_count_lbl = tk.Label(log_bar, text="0 events",
                                             bg=T["card"], fg=T["text_dim"],
                                             font=("Segoe UI", 8))
        self._auto_log_count_lbl.pack(side="left", padx=6)

        log_cols2  = ("Timestamp", "Rule", "Trigger", "Action", "Result")
        log_widths2 = (148, 160, 220, 180, 300)
        l_vsb = ttk.Scrollbar(log_lf, orient="vertical")
        l_hsb = ttk.Scrollbar(log_lf, orient="horizontal")
        self._auto_log_tree = ttk.Treeview(
            log_lf, columns=log_cols2, show="headings",
            yscrollcommand=l_vsb.set, xscrollcommand=l_hsb.set,
            selectmode="browse", height=8)
        l_vsb.configure(command=self._auto_log_tree.yview)
        l_hsb.configure(command=self._auto_log_tree.xview)
        for col, w in zip(log_cols2, log_widths2):
            self._auto_log_tree.heading(col, text=col)
            self._auto_log_tree.column(col, width=w, minwidth=w, stretch=(col == "Result"))
        self._auto_log_tree.tag_configure("ok",  foreground=T["success"])
        self._auto_log_tree.tag_configure("err", foreground=T["error"])
        l_vsb.pack(side="right", fill="y")
        l_hsb.pack(side="bottom", fill="x")
        self._auto_log_tree.pack(fill="both", expand=True)

    # ── Automation helpers ─────────────────────────────────────────────────────

    def _auto_next_id(self):
        self._auto_id_counter += 1
        return self._auto_id_counter

    def _auto_clear_editor(self):
        ae = self._ae
        for key in ("name", "trigger_type", "trigger_channel", "trigger_op",
                    "trigger_value", "trigger_time", "action_type",
                    "action_target", "action_duration", "cooldown_s", "notes"):
            ae[key].set("")
        ae["enabled"].set(True)
        ae["_editing_id"] = None

    def _auto_new_rule(self):
        self._auto_clear_editor()
        ae = self._ae
        ae["name"].set(f"Rule {self._auto_id_counter + 1}")
        ae["trigger_type"].set("hydro_threshold")
        ae["trigger_channel"].set("ph")
        ae["trigger_op"].set("<")
        ae["trigger_value"].set("5.5")
        ae["action_type"].set("pump_pulse")
        ae["action_target"].set("Pump 1: pH Up")
        ae["action_duration"].set("5")
        ae["cooldown_s"].set("300")

    def _auto_editor_ttype_changed(self, event=None):
        pass  # Future: show/hide time vs threshold fields

    def _auto_save_rule(self):
        ae = self._ae
        name = ae["name"].get().strip()
        if not name:
            messagebox.showwarning("Automation", "Rule name is required.")
            return
        rule = {
            "id":               ae["_editing_id"] or self._auto_next_id(),
            "name":             name,
            "enabled":          ae["enabled"].get(),
            "trigger_type":     ae["trigger_type"].get(),
            "trigger_channel":  ae["trigger_channel"].get(),
            "trigger_op":       ae["trigger_op"].get(),
            "trigger_value":    ae["trigger_value"].get(),
            "trigger_time":     ae["trigger_time"].get().strip(),
            "action_type":      ae["action_type"].get(),
            "action_target":    ae["action_target"].get(),
            "action_duration":  ae["action_duration"].get() or "5",
            "cooldown_s":       ae["cooldown_s"].get() or "300",
            "last_triggered":   "",
            "trigger_count":    0,
            "notes":            ae["notes"].get(),
        }
        # Update existing or append new
        editing_id = ae["_editing_id"]
        if editing_id is not None:
            for i, r in enumerate(self.automations):
                if r["id"] == editing_id:
                    rule["last_triggered"] = r.get("last_triggered", "")
                    rule["trigger_count"]  = r.get("trigger_count", 0)
                    self.automations[i] = rule
                    break
        else:
            rule["id"] = self._auto_next_id()
            self.automations.append(rule)
        ae["_editing_id"] = rule["id"]
        self._auto_refresh_rule_tree()

    def _auto_on_select(self, event=None):
        sel = self._auto_rule_tree.selection()
        if not sel:
            return
        iid = sel[0]
        rule = next((r for r in self.automations
                     if str(r["id"]) == iid), None)
        if rule is None:
            return
        ae = self._ae
        ae["name"].set(rule.get("name", ""))
        ae["enabled"].set(bool(rule.get("enabled", True)))
        ae["trigger_type"].set(rule.get("trigger_type", ""))
        ae["trigger_channel"].set(rule.get("trigger_channel", ""))
        ae["trigger_op"].set(rule.get("trigger_op", ""))
        ae["trigger_value"].set(str(rule.get("trigger_value", "")))
        ae["trigger_time"].set(rule.get("trigger_time", ""))
        ae["action_type"].set(rule.get("action_type", ""))
        ae["action_target"].set(rule.get("action_target", ""))
        ae["action_duration"].set(str(rule.get("action_duration", "")))
        ae["cooldown_s"].set(str(rule.get("cooldown_s", "")))
        ae["notes"].set(rule.get("notes", ""))
        ae["_editing_id"] = rule["id"]

    def _auto_delete_selected(self):
        sel = self._auto_rule_tree.selection()
        if not sel:
            return
        iid = sel[0]
        self.automations = [r for r in self.automations if str(r["id"]) != iid]
        self._auto_refresh_rule_tree()
        self._auto_clear_editor()

    def _auto_enable_all(self):
        for r in self.automations:
            r["enabled"] = True
        self._auto_refresh_rule_tree()

    def _auto_disable_all(self):
        for r in self.automations:
            r["enabled"] = False
        self._auto_refresh_rule_tree()

    def _auto_sort_rules(self, col):
        pass  # simple alphabetical sort could be added later

    def _auto_refresh_rule_tree(self):
        tree = self._auto_rule_tree
        if tree is None:
            return
        for iid in tree.get_children():
            tree.delete(iid)
        for r in self.automations:
            ttype = r.get("trigger_type", "")
            ch    = r.get("trigger_channel", "")
            op    = r.get("trigger_op", "")
            val   = r.get("trigger_value", "")
            ttime = r.get("trigger_time", "")
            trigger_str = (f"{self._AUTO_CH_LABELS.get(ch, ch)} {op} {val}"
                           if ttype != "schedule" else f"Schedule @ {ttime}")
            action_lbl  = self._AUTO_ACTION_LABELS.get(r.get("action_type", ""), r.get("action_type", ""))
            tag = "enabled" if r.get("enabled") else "disabled"
            tree.insert("", "end", iid=str(r["id"]), tags=(tag,), values=(
                r["id"],
                r.get("name", ""),
                "✔" if r.get("enabled") else "✕",
                ttype,
                op,
                val or ttime,
                action_lbl,
                r.get("action_target", ""),
                r.get("cooldown_s", ""),
                r.get("last_triggered", "—"),
                r.get("trigger_count", 0),
            ))

    def _auto_clear_log(self):
        if self._auto_log_tree:
            for iid in self._auto_log_tree.get_children():
                self._auto_log_tree.delete(iid)
            self._auto_log_count_lbl.config(text="0 events")

    def _auto_log_event(self, rule_name, trigger_desc, action_desc, result, ok=True):
        """Append one row to the execution log (main-thread safe via root.after)."""
        if self.is_closing:
            return
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tag = "ok" if ok else "err"
        def _insert():
            if self._auto_log_tree is None:
                return
            self._auto_log_tree.insert("", "end", tags=(tag,), values=(
                ts, rule_name, trigger_desc, action_desc, result))
            self._auto_log_tree.yview_moveto(1.0)
            n = len(self._auto_log_tree.get_children())
            self._auto_log_count_lbl.config(text=f"{n} events")
        self.root.after(0, _insert)

    def _auto_engine_toggle(self):
        T = THEME
        if self._auto_engine_running:
            self._auto_engine_running = False
            self._auto_engine_btn.config(text="▶ Start Engine")
            self._auto_engine_lbl.config(text="Engine: stopped", fg=T["text_dim"])
        else:
            self._auto_engine_running = True
            self._auto_engine_btn.config(text="⏹ Stop Engine")
            self._auto_engine_lbl.config(text="Engine: running", fg=T["success"])
            self._auto_engine_thread = threading.Thread(
                target=self._auto_engine_loop, daemon=True)
            self._auto_engine_thread.start()

    def _auto_engine_loop(self):
        """
        Background evaluation loop.  Every _auto_engine_interval seconds it reads
        the shared _auto_sensor_cache and evaluates all enabled rules.
        """
        while self._auto_engine_running and not self.is_closing:
            now = datetime.now()
            with self._auto_sensor_lock:
                cache = dict(self._auto_sensor_cache)

            for rule in list(self.automations):
                if not rule.get("enabled", False):
                    continue
                try:
                    fired, trigger_desc = self._auto_evaluate_trigger(rule, cache, now)
                except Exception as exc:
                    logging.error(f"Automation trigger eval error [{rule.get('name')}]: {exc}")
                    continue
                if not fired:
                    continue

                # Respect cooldown
                last_ts = rule.get("last_triggered", "")
                if last_ts:
                    try:
                        last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
                        cooldown = int(rule.get("cooldown_s", 300) or 300)
                        if (now - last_dt).total_seconds() < cooldown:
                            continue
                    except ValueError:
                        pass

                # Execute action
                action_desc, result, ok = self._auto_execute_action(rule)

                # Update rule metadata
                rule["last_triggered"]  = now.strftime("%Y-%m-%d %H:%M:%S")
                rule["trigger_count"]   = rule.get("trigger_count", 0) + 1

                self._auto_log_event(
                    rule.get("name", "—"),
                    trigger_desc, action_desc, result, ok)

                # Refresh rule tree row
                if not self.is_closing:
                    self.root.after(0, self._auto_refresh_rule_tree)

            time.sleep(self._auto_engine_interval)

    def _auto_evaluate_trigger(self, rule, cache, now):
        """
        Evaluate whether a rule's trigger condition is met.
        Returns (fired: bool, description: str).
        """
        ttype = rule.get("trigger_type", "")
        ch    = rule.get("trigger_channel", "")
        op    = rule.get("trigger_op", "<")
        try:
            threshold = float(rule.get("trigger_value", 0))
        except (TypeError, ValueError):
            threshold = 0.0

        if ttype in ("hydro_threshold", "env_threshold", "health_threshold"):
            val = cache.get(ch)
            if val is None:
                return False, f"{ch} not in cache"
            ops_map = {
                "<":  val <  threshold,
                "<=": val <= threshold,
                ">":  val >  threshold,
                ">=": val >= threshold,
                "==": abs(val - threshold) < 1e-9,
                "!=": abs(val - threshold) >= 1e-9,
            }
            fired = ops_map.get(op, False)
            desc  = f"{self._AUTO_CH_LABELS.get(ch, ch)} = {val:.3f} {op} {threshold}"
            return fired, desc

        elif ttype == "schedule":
            ttime = rule.get("trigger_time", "").strip()
            if not ttime:
                return False, "No time set"
            try:
                h, m = map(int, ttime.split(":"))
                fired = (now.hour == h and now.minute == m and now.second < self._auto_engine_interval)
                return fired, f"Schedule @ {ttime}"
            except ValueError:
                return False, f"Bad time '{ttime}'"

        return False, f"Unknown trigger type '{ttype}'"

    def _auto_execute_action(self, rule):
        """
        Execute the rule's action.  Returns (action_desc, result_str, ok_bool).
        All Tkinter mutations dispatched via root.after() from the engine thread.
        """
        atype  = rule.get("action_type", "")
        target = rule.get("action_target", "")

        try:
            duration = float(rule.get("action_duration", 5) or 5)
        except (TypeError, ValueError):
            duration = 5.0
        action_label = self._AUTO_ACTION_LABELS.get(atype, atype)

        if atype in ("pump_on", "pump_off", "pump_pulse"):
            # Parse pump index from target string "Pump N: name"
            pump_idx = None
            try:
                pump_idx = int(target.split(":")[0].replace("Pump", "").strip()) - 1
            except (IndexError, ValueError):
                pass
            if pump_idx is None or pump_idx < 0 or pump_idx >= len(self.hydro_pump_vars):
                return action_label, f"Invalid pump target '{target}'", False

            if atype == "pump_on":
                self.root.after(0, lambda p=pump_idx: self._hydro_pump_set(p, True))
                return action_label, f"{target} turned ON", True

            elif atype == "pump_off":
                self.root.after(0, lambda p=pump_idx: self._hydro_pump_set(p, False))
                return action_label, f"{target} turned OFF", True

            else:  # pump_pulse
                def _pulse(p=pump_idx, d=duration):
                    self._hydro_pump_set(p, True)
                    self.root.after(int(d * 1000), lambda: self._hydro_pump_set(p, False))
                self.root.after(0, _pulse)
                return action_label, f"{target} pulsed for {duration}s", True

        elif atype == "alert_popup":
            msg = target or rule.get("notes", "Automation triggered.")
            rule_name = rule.get("name", "")
            self.root.after(0, lambda m=msg, n=rule_name:
                messagebox.showwarning(f"Automation: {n}", m))
            return action_label, f"Popup shown: {msg[:60]}", True

        elif atype == "log_event":
            msg = target or rule.get("notes", "Automation event logged.")
            logging.info(f"[Automation] {rule.get('name', '')}: {msg}")
            return action_label, msg[:80], True

        elif atype == "analyze_panel":
            panel_idx = None
            try:
                panel_idx = int(target.replace("Panel", "").strip()) - 1
            except (IndexError, ValueError):
                pass
            if panel_idx is None or panel_idx < 0 or panel_idx >= self.max_camera_slots:
                return action_label, f"Invalid panel '{target}'", False
            self.root.after(0, lambda p=panel_idx: self._auto_trigger_analysis(p))
            return action_label, f"Analysis triggered on Panel {panel_idx+1}", True

        return action_label, f"Unknown action '{atype}'", False

    def _hydro_pump_set(self, pump_idx, state):
        """Turn a dosing pump on (state=True) or off (state=False) — UI + log."""
        if pump_idx < 0 or pump_idx >= len(self.hydro_pump_vars):
            return
        T   = THEME
        var = self.hydro_pump_vars[pump_idx]
        var.set(state)
        if pump_idx < len(self._hydro_pump_btns):
            btn, dot = self._hydro_pump_btns[pump_idx]
            if state:
                btn.config(bg=T["primary_dk"], fg=T["text"])
                dot.config(fg=T["accent"])
            else:
                btn.config(bg=T["card_header"], fg=T["text_muted"])
                dot.config(fg=T["text_dim"])
        verb = "ACTIVATED" if state else "deactivated"
        logging.info(f"Hydro pump '{self.hydro_pump_names[pump_idx]}' {verb} by automation")

    def _auto_trigger_analysis(self, panel_idx):
        """Trigger a camera analysis on the given panel from the main thread."""
        if panel_idx < len(self.grids) and self.current_frames[panel_idx] is not None:
            analysis_type = self.grids[panel_idx]["analysis_dropdown"].get()
            self.start_analysis_job(panel_idx, self.current_frames[panel_idx].copy(),
                                    analysis_type)

    def _auto_update_sensor_cache(self, updates: dict):
        """Thread-safe update of the sensor cache from any pipeline."""
        with self._auto_sensor_lock:
            self._auto_sensor_cache.update(updates)

    def _auto_export(self):
        path = filedialog.asksaveasfilename(
            title="Export Automations",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.automations, fh, indent=2)
            messagebox.showinfo("Export", f"Automations saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export", str(exc))

    def _auto_import(self):
        path = filedialog.askopenfilename(
            title="Import Automations",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                rules = json.load(fh)
            if not isinstance(rules, list):
                raise ValueError("Expected a JSON array of rules.")
            self.automations = rules
            # Resync ID counter
            self._auto_id_counter = max((r.get("id", 0) for r in rules), default=0)
            self._auto_refresh_rule_tree()
            messagebox.showinfo("Import", f"Loaded {len(rules)} rule(s).")
        except Exception as exc:
            messagebox.showerror("Import", str(exc))

    # ── 3D Morphology Viewer ──────────────────────────────────────────────────

    def create_morphology_widgets(self):
        T = THEME
        _btn  = dict(bg=T["primary_dk"], fg=T["text"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary"], activeforeground=T["text"],
                     cursor="hand2", bd=0)
        _sbtn = dict(bg=T["card_header"], fg=T["primary_lt"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary_dk"], activeforeground=T["text"],
                     cursor="hand2", bd=0)

        outer = tk.Frame(self.morph_tab, bg=T["panel"])
        outer.pack(fill="both", expand=True)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = tk.Frame(outer, bg=T["panel"], pady=6)
        toolbar.pack(fill="x", padx=8)

        tk.Button(toolbar, text="📂 Load OBJ / STL",
                  command=self._morph_load_file, **_btn).pack(side="left", padx=(4, 4))
        tk.Button(toolbar, text="🔄 Rebuild from Live Data",
                  command=self._morph_rebuild_live, **_btn).pack(side="left", padx=(0, 12))
        tk.Button(toolbar, text="⬇ Export OBJ",
                  command=self._morph_export_obj, **_sbtn).pack(side="left", padx=(0, 4))
        tk.Button(toolbar, text="🗑 Clear",
                  command=self._morph_clear, **_sbtn).pack(side="left", padx=(0, 12))

        tk.Label(toolbar, text="Section:", bg=T["panel"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left")
        section_choices = ["All Sections"] + [f"Section {i}" for i in range(1, (MAX_CAMERA_SLOTS // 2) + 1)]
        ttk.Combobox(toolbar, textvariable=self._morph_section_var,
                     values=section_choices, state="readonly", width=14,
                     font=("Segoe UI", 9)).pack(side="left", padx=(4, 12))

        tk.Checkbutton(toolbar, text="Axes", variable=self._morph_show_axes,
                       command=self._morph_refresh_view,
                       bg=T["panel"], fg=T["text"], selectcolor=T["primary_dk"],
                       activebackground=T["panel"], activeforeground=T["primary_lt"],
                       font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        tk.Checkbutton(toolbar, text="Grid", variable=self._morph_show_grid,
                       command=self._morph_refresh_view,
                       bg=T["panel"], fg=T["text"], selectcolor=T["primary_dk"],
                       activebackground=T["panel"], activeforeground=T["primary_lt"],
                       font=("Segoe UI", 9)).pack(side="left", padx=(0, 12))

        self._morph_status_lbl = tk.Label(
            toolbar, text="No model loaded — showing blank plane",
            bg=T["panel"], fg=T["text_dim"], font=("Segoe UI", 8, "italic"))
        self._morph_status_lbl.pack(side="right", padx=8)

        # ── View controls strip ───────────────────────────────────────────────
        ctrl = tk.Frame(outer, bg=T["card"], pady=4)
        ctrl.pack(fill="x", padx=8, pady=(0, 4))

        tk.Label(ctrl, text="Azimuth:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 8)).pack(side="left", padx=(8, 2))
        tk.Scale(ctrl, variable=self._morph_azim, from_=-180, to=180,
                 orient="horizontal", length=160, resolution=1,
                 command=lambda _: self._morph_refresh_view(),
                 bg=T["card"], fg=T["text"], troughcolor=T["border"],
                 highlightthickness=0, sliderrelief="flat",
                 font=("Segoe UI", 8), showvalue=True,
                 activebackground=T["primary"]).pack(side="left")

        tk.Label(ctrl, text="Elevation:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 8)).pack(side="left", padx=(16, 2))
        tk.Scale(ctrl, variable=self._morph_elev, from_=-90, to=90,
                 orient="horizontal", length=160, resolution=1,
                 command=lambda _: self._morph_refresh_view(),
                 bg=T["card"], fg=T["text"], troughcolor=T["border"],
                 highlightthickness=0, sliderrelief="flat",
                 font=("Segoe UI", 8), showvalue=True,
                 activebackground=T["primary"]).pack(side="left")

        tk.Button(ctrl, text="⤢ Reset View",
                  command=self._morph_reset_view, **_sbtn).pack(side="left", padx=(16, 0))
        tk.Button(ctrl, text="⊕ Top", command=lambda: self._morph_set_view(90, 0), **_sbtn).pack(side="left", padx=(4, 0))
        tk.Button(ctrl, text="⊙ Front", command=lambda: self._morph_set_view(0, 0), **_sbtn).pack(side="left", padx=(4, 0))
        tk.Button(ctrl, text="⊘ Side", command=lambda: self._morph_set_view(0, 90), **_sbtn).pack(side="left", padx=(4, 0))

        self._morph_file_lbl = tk.Label(
            ctrl, textvariable=self._morph_loaded_file,
            bg=T["card"], fg=T["text_dim"], font=("Segoe UI", 8, "italic"))
        self._morph_file_lbl.pack(side="right", padx=8)

        # ── Main pane: 3D canvas + side panel ────────────────────────────────
        pane = tk.Frame(outer, bg=T["bg"])
        pane.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # 3D matplotlib canvas
        from mpl_toolkits.mplot3d import Axes3D  # ensure 3D toolkit registered
        self._morph_fig = plt.Figure(figsize=(8, 6), dpi=96,
                                     facecolor=T["graph_bg"])
        self._morph_ax  = self._morph_fig.add_subplot(111, projection="3d")
        self._morph_ax.set_facecolor(T["graph_ax"])
        self._morph_canvas = FigureCanvasTkAgg(self._morph_fig, master=pane)
        widget = self._morph_canvas.get_tk_widget()
        widget.configure(bg=T["graph_bg"], highlightthickness=0)
        widget.pack(side="left", fill="both", expand=True)

        # Mouse-drag rotation wired directly to matplotlib
        widget.bind("<ButtonPress-1>",   self._morph_mouse_press)
        widget.bind("<B1-Motion>",       self._morph_mouse_drag)
        widget.bind("<MouseWheel>",      self._morph_mouse_wheel)
        widget.bind("<Button-4>",        self._morph_mouse_wheel)
        widget.bind("<Button-5>",        self._morph_mouse_wheel)

        # Side info panel
        side = tk.Frame(pane, bg=T["card"], width=200)
        side.pack(side="right", fill="y", padx=(6, 0))
        side.pack_propagate(False)

        tk.Label(side, text="Model Info", bg=T["card"], fg=T["primary_lt"],
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(10, 4))
        self._morph_info_txt = tk.Text(
            side, bg=T["graph_bg"], fg=T["text"], relief="flat", bd=0,
            font=("Courier New", 8), state="disabled",
            highlightthickness=0, padx=8, pady=6, wrap="word")
        self._morph_info_txt.pack(fill="both", expand=True, padx=6, pady=(0, 8))

        # Draw initial blank plane
        self._morph_draw_blank_plane()

    # ── 3D view helpers ────────────────────────────────────────────────────────

    def _morph_draw_blank_plane(self):
        """Render a green reference ground plane so the viewer is never empty."""
        T  = THEME
        ax = self._morph_ax
        ax.cla()
        # Ground plane grid
        xs = np.linspace(-1, 1, 11)
        zs = np.linspace(-1, 1, 11)
        X, Z = np.meshgrid(xs, zs)
        Y    = np.zeros_like(X)
        ax.plot_surface(X, Y, Z, alpha=0.18, color=T["primary_dk"], zorder=0)
        for xi in xs:
            ax.plot([xi, xi], [0, 0], [-1, 1], color=T["border"], lw=0.5, alpha=0.5)
        for zi in zs:
            ax.plot([-1, 1], [0, 0], [zi, zi], color=T["border"], lw=0.5, alpha=0.5)
        # Axis arrows
        ax.quiver(0, 0, 0, 1.2, 0, 0, color="#ef5350", lw=1.5, arrow_length_ratio=0.15)
        ax.quiver(0, 0, 0, 0, 1.2, 0, color="#66bb6a", lw=1.5, arrow_length_ratio=0.15)
        ax.quiver(0, 0, 0, 0, 0, 1.2, color="#42a5f5", lw=1.5, arrow_length_ratio=0.15)
        ax.text(1.3, 0,   0,   "X", color="#ef5350", fontsize=8)
        ax.text(0,   1.3, 0,   "Y", color="#66bb6a", fontsize=8)
        ax.text(0,   0,   1.3, "Z", color="#42a5f5", fontsize=8)
        self._morph_style_ax()
        self._morph_canvas.draw_idle()
        self._morph_set_info("No model loaded.\n\nLoad an OBJ/STL file or run\nPlant Morphology Analysis\nthen click 'Rebuild from Live Data'.")

    def _morph_style_ax(self):
        T  = THEME
        ax = self._morph_ax
        ax.set_facecolor(T["graph_ax"])
        ax.tick_params(colors=T["text_dim"], labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(T["border"])
        ax.yaxis.pane.set_edgecolor(T["border"])
        ax.zaxis.pane.set_edgecolor(T["border"])
        if not self._morph_show_grid.get():
            ax.grid(False)
        else:
            ax.grid(True, color=T["border"], linewidth=0.4)
        if not self._morph_show_axes.get():
            ax.set_axis_off()
        else:
            ax.set_axis_on()
        ax.view_init(elev=self._morph_elev.get(), azim=self._morph_azim.get())

    def _morph_set_info(self, text):
        txt = self._morph_info_txt
        txt.config(state="normal")
        txt.delete("1.0", "end")
        txt.insert("end", text)
        txt.config(state="disabled")

    def _morph_refresh_view(self):
        ax = self._morph_ax
        ax.view_init(elev=self._morph_elev.get(), azim=self._morph_azim.get())
        self._morph_style_ax()
        self._morph_canvas.draw_idle()

    def _morph_reset_view(self):
        self._morph_azim.set(-60)
        self._morph_elev.set(25)
        self._morph_refresh_view()

    def _morph_set_view(self, elev, azim):
        self._morph_elev.set(elev)
        self._morph_azim.set(azim)
        self._morph_refresh_view()

    # ── Mouse interaction ──────────────────────────────────────────────────────

    def _morph_mouse_press(self, event):
        self._morph_dragging   = True
        self._morph_drag_start = (event.x, event.y,
                                   self._morph_azim.get(),
                                   self._morph_elev.get())

    def _morph_mouse_drag(self, event):
        if not self._morph_dragging or self._morph_drag_start is None:
            return
        x0, y0, az0, el0 = self._morph_drag_start
        da = (event.x - x0) * 0.5
        de = (y0 - event.y) * 0.5
        self._morph_azim.set(az0 + da)
        self._morph_elev.set(max(-90, min(90, el0 + de)))
        self._morph_refresh_view()

    def _morph_mouse_wheel(self, event):
        ax = self._morph_ax
        try:
            # event.delta is ±120 on Windows; event.num 4/5 on Linux
            delta = event.delta if event.delta else (-120 if event.num == 5 else 120)
            scale = 1.1 if delta < 0 else 0.9
            xlim  = ax.get_xlim3d()
            ylim  = ax.get_ylim3d()
            zlim  = ax.get_zlim3d()
            xc    = (xlim[0] + xlim[1]) / 2
            yc    = (ylim[0] + ylim[1]) / 2
            zc    = (zlim[0] + zlim[1]) / 2
            xr    = (xlim[1] - xlim[0]) / 2 * scale
            yr    = (ylim[1] - ylim[0]) / 2 * scale
            zr    = (zlim[1] - zlim[0]) / 2 * scale
            ax.set_xlim3d([xc - xr, xc + xr])
            ax.set_ylim3d([yc - yr, yc + yr])
            ax.set_zlim3d([zc - zr, zc + zr])
            self._morph_canvas.draw_idle()
        except Exception:
            pass

    # ── File I/O ───────────────────────────────────────────────────────────────

    def _morph_load_file(self):
        path = filedialog.askopenfilename(
            title="Load 3D Model",
            filetypes=[
                ("3D Files", "*.obj *.stl *.ply"),
                ("OBJ Files", "*.obj"),
                ("STL Files", "*.stl"),
                ("PLY Files", "*.ply"),
                ("All Files", "*.*"),
            ])
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".obj":
                verts, faces, normals = self._morph_parse_obj(path)
            elif ext == ".stl":
                verts, faces, normals = self._morph_parse_stl(path)
            elif ext == ".ply":
                verts, faces, normals = self._morph_parse_ply(path)
            else:
                messagebox.showerror("Load Model", f"Unsupported format: {ext}")
                return
            self._morph_render_mesh(verts, faces, normals)
            fname = os.path.basename(path)
            self._morph_loaded_file.set(f"File: {fname}")
            self._morph_status_lbl.config(
                text=f"Loaded: {fname}  ({len(verts)} verts, {len(faces)} faces)")
            self._morph_set_info(
                f"File: {fname}\nVertices: {len(verts)}\nFaces:    {len(faces)}\n"
                f"Format: {ext.upper()[1:]}\n\nDrag to rotate\nScroll to zoom")
        except Exception as exc:
            messagebox.showerror("Load Model", f"Could not load file:\n{exc}")
            logging.error(f"Morph load error: {exc}")

    def _morph_parse_obj(self, path):
        """Minimal OBJ parser — returns (vertices, faces, normals)."""
        verts   = []
        normals = []
        faces   = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("vn "):
                    parts = line.split()
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    # Support v, v/t, v/t/n, v//n
                    idx = [int(p.split("/")[0]) - 1 for p in parts]
                    # Triangulate polygon fans
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
        return np.array(verts, dtype=float), faces, np.array(normals, dtype=float)

    def _morph_parse_stl(self, path):
        """Parse ASCII or binary STL."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                first = fh.read(80)
            is_ascii = "facet" in first.lower() or first.strip().startswith("solid")
        except Exception:
            is_ascii = False

        verts   = []
        normals = []
        faces   = []

        if is_ascii:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                current_normal = [0, 0, 1]
                tri_verts      = []
                for line in fh:
                    line = line.strip()
                    if line.startswith("facet normal"):
                        parts = line.split()
                        current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
                    elif line.startswith("vertex"):
                        parts = line.split()
                        tri_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith("endfacet"):
                        if len(tri_verts) == 3:
                            base = len(verts)
                            verts.extend(tri_verts)
                            normals.extend([current_normal] * 3)
                            faces.append([base, base + 1, base + 2])
                        tri_verts = []
        else:
            import struct
            with open(path, "rb") as fh:
                fh.read(80)  # header
                n_tri = struct.unpack("<I", fh.read(4))[0]
                for _ in range(n_tri):
                    nx, ny, nz = struct.unpack("<fff", fh.read(12))
                    tri_vs = []
                    for __ in range(3):
                        x, y, z = struct.unpack("<fff", fh.read(12))
                        tri_vs.append([x, y, z])
                    fh.read(2)  # attrib
                    base = len(verts)
                    verts.extend(tri_vs)
                    normals.extend([[nx, ny, nz]] * 3)
                    faces.append([base, base + 1, base + 2])

        return np.array(verts, dtype=float), faces, np.array(normals, dtype=float)

    def _morph_parse_ply(self, path):
        """Parse ASCII PLY files (common subset)."""
        verts   = []
        faces   = []
        normals = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        i        = 0
        n_verts  = 0
        n_faces  = 0
        in_header = True
        while i < len(lines):
            line = lines[i].strip()
            if in_header:
                if line.startswith("element vertex"):
                    n_verts = int(line.split()[-1])
                elif line.startswith("element face"):
                    n_faces = int(line.split()[-1])
                elif line == "end_header":
                    in_header = False
            else:
                if len(verts) < n_verts:
                    parts = line.split()
                    verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                elif len(faces) < n_faces:
                    parts = line.split()
                    n = int(parts[0])
                    idx = [int(parts[j + 1]) for j in range(n)]
                    for k in range(1, n - 1):
                        faces.append([idx[0], idx[k], idx[k + 1]])
            i += 1
        return np.array(verts, dtype=float), faces, np.array(normals, dtype=float)

    def _morph_render_mesh(self, verts, faces, normals=None):
        """Render a mesh (vertices + triangle face list) into the 3D axes."""
        T  = THEME
        ax = self._morph_ax
        ax.cla()

        if len(verts) == 0:
            self._morph_draw_blank_plane()
            return

        # Normalise to [-1, 1]^3
        centre = verts.mean(axis=0)
        verts  = verts - centre
        scale  = np.abs(verts).max()
        if scale > 0:
            verts = verts / scale

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        poly_list = []
        for tri in faces:
            if max(tri) < len(verts):
                poly_list.append(verts[tri])

        # Split into chunks for performance (matplotlib 3D renders all at once)
        MAX_FACES = 12000
        if len(poly_list) > MAX_FACES:
            step      = max(1, len(poly_list) // MAX_FACES)
            poly_list = poly_list[::step]

        if poly_list:
            collection = Poly3DCollection(
                poly_list,
                alpha=0.82,
                facecolor=T["primary_dk"],
                edgecolor=T["border"] if len(poly_list) < 3000 else "none",
                linewidth=0.2)
            ax.add_collection3d(collection)

        # Draw reference ground plane
        r = 1.2
        xs = np.linspace(-r, r, 6)
        zs = np.linspace(-r, r, 6)
        X, Z = np.meshgrid(xs, zs)
        Y    = np.full_like(X, verts[:, 1].min() - 0.05)
        ax.plot_surface(X, Y, Z, alpha=0.1, color=T["primary_dk"])

        ax.set_xlim3d([-1.5, 1.5])
        ax.set_ylim3d([-1.5, 1.5])
        ax.set_zlim3d([-1.5, 1.5])
        self._morph_style_ax()
        self._morph_canvas.draw_idle()

    # ── Live data rebuild ──────────────────────────────────────────────────────

    def _morph_rebuild_live(self):
        """
        Build a 3D point-cloud / stem skeleton from morph_section_data collected
        during Plant Morphology Analysis runs.
        """
        T   = THEME
        ax  = self._morph_ax
        sec_filter = self._morph_section_var.get()

        if not self.morph_section_data:
            self._morph_draw_blank_plane()
            self._morph_status_lbl.config(
                text="No live morphology data yet — run Plant Morphology Analysis first")
            return

        ax.cla()
        total_points = 0
        info_lines   = []

        for section, data in sorted(self.morph_section_data.items()):
            if sec_filter != "All Sections" and sec_filter != f"Section {section}":
                continue

            for role in ("overhead", "side"):
                df = data.get(role)
                if df is None or df.empty:
                    continue

                # Extract numeric height/area metrics and treat them as pseudo-3D coords
                heights = []
                widths  = []
                depths  = []

                def _col(name):
                    for col in df.columns:
                        if name.lower() in col.lower():
                            try:
                                return pd.to_numeric(df[col], errors="coerce").dropna().tolist()
                            except Exception:
                                return []
                    return []

                stem_heights = _col("stem_height")
                leaf_lengths = _col("path_length") or _col("euclidean_length")
                widths_raw   = _col("segment_width") or _col("mean_segment_width")
                angles       = _col("angle") or _col("insertion_angle")

                n = max(len(stem_heights), len(leaf_lengths), 1)
                xs, ys, zs = [], [], []

                # Synthesise point positions from available metrics
                for j in range(n):
                    h = stem_heights[j] if j < len(stem_heights) else 0.5
                    l = leaf_lengths[j] if j < len(leaf_lengths) else 0.3
                    a = math.radians(angles[j]) if j < len(angles) else (j * 2.0)
                    w = widths_raw[j] if j < len(widths_raw) else 0.1
                    # Overhead view drives X/Z spread; side view drives Y depth
                    if role == "overhead":
                        xs.append(l * math.cos(a) + (section - 1) * 2)
                        ys.append(h)
                        zs.append(l * math.sin(a))
                    else:
                        xs.append((section - 1) * 2 + w * 0.5)
                        ys.append(h)
                        zs.append(l)

                if xs:
                    colour = T["primary"] if role == "overhead" else T["primary_lt"]
                    ax.scatter(xs, ys, zs, c=colour, s=18, alpha=0.75, depthshade=True)
                    # Connect stem points
                    ax.plot(xs, ys, zs, color=colour, lw=0.8, alpha=0.4)
                    total_points += len(xs)
                    ts = data.get("timestamp", "?")
                    info_lines.append(
                        f"S{section} {role[:4]}: {len(xs)} pts  @{ts[-8:] if len(ts) > 8 else ts}")

        # Ground plane
        r  = max(2, len(self.morph_section_data) * 2)
        xs2 = np.linspace(-1, r, 8)
        zs2 = np.linspace(-1, r, 8)
        X2, Z2 = np.meshgrid(xs2, zs2)
        Y2      = np.zeros_like(X2)
        ax.plot_surface(X2, Y2, Z2, alpha=0.12, color=T["primary_dk"])

        self._morph_style_ax()
        self._morph_canvas.draw_idle()
        self._morph_status_lbl.config(
            text=f"Live data: {total_points} points across {len(self.morph_section_data)} section(s)")
        self._morph_set_info(
            f"Live morphology data\n\n" + "\n".join(info_lines) +
            f"\n\nTotal points: {total_points}\nSections: {len(self.morph_section_data)}")

    def _morph_clear(self):
        self._morph_loaded_file.set("")
        self._morph_status_lbl.config(text="No model loaded — showing blank plane")
        self._morph_draw_blank_plane()

    def _morph_export_obj(self):
        """Export the current 3D plot data as a simple OBJ file."""
        ax = self._morph_ax
        path = filedialog.asksaveasfilename(
            title="Export OBJ",
            defaultextension=".obj",
            filetypes=[("OBJ Files", "*.obj"), ("All Files", "*.*")])
        if not path:
            return
        try:
            lines = ["# GreenSight 3D Morphology Export", f"# {datetime.now().isoformat()}", ""]
            v_idx = 1
            face_lines = []
            # Collect scatter points from the axes
            for col in ax.collections:
                try:
                    data = col._offsets3d
                    xs, ys, zs = data
                    for x, y, z in zip(xs, ys, zs):
                        lines.append(f"v {float(x):.6f} {float(y):.6f} {float(z):.6f}")
                except Exception:
                    pass
            if len(lines) <= 3:
                messagebox.showwarning("Export OBJ", "Nothing to export — load a model or rebuild first.")
                return
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            messagebox.showinfo("Export OBJ", f"Saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export OBJ", str(exc))

    # ── Leafy AI Copilot ──────────────────────────────────────────────────────

    def _leafy_open(self):
        """Open or raise the Leafy floating panel."""
        if self._leafy_window is None or not self._leafy_window.winfo_exists():
            self._leafy_build_window()
        else:
            self._leafy_window.deiconify()
            self._leafy_window.lift()

    def _leafy_build_window(self):
        T = THEME
        win = tk.Toplevel(self.root)
        win.title("\U0001f331 Leafy \u2014 Crop Steering Copilot")
        win.geometry("520x740")
        win.configure(bg=T["bg"])
        win.resizable(True, True)
        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        self._leafy_window = win

        _btn  = dict(bg=T["primary_dk"], fg=T["text"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary"], activeforeground=T["text"],
                     cursor="hand2", bd=0)
        _qbtn = dict(bg=T["card_header"], fg=T["primary_lt"], relief="flat",
                     font=("Segoe UI", 8), padx=8, pady=3,
                     activebackground=T["primary_dk"], activeforeground=T["text"],
                     cursor="hand2", bd=0)

        # ── Title bar ─────────────────────────────────────────────────────────
        title_bar = tk.Frame(win, bg=T["card_header"])
        title_bar.pack(fill="x")
        tk.Label(title_bar, text="\U0001f331", bg=T["card_header"], fg=T["accent"],
                 font=("Segoe UI", 24)).pack(side="left", padx=12, pady=8)
        name_f = tk.Frame(title_bar, bg=T["card_header"])
        name_f.pack(side="left", pady=8)
        tk.Label(name_f, text="Leafy", bg=T["card_header"], fg=T["accent"],
                 font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(name_f, text="Crop Steering Copilot  \u2014  GreenSight AI",
                 bg=T["card_header"], fg=T["text_muted"],
                 font=("Segoe UI", 8)).pack(anchor="w")
        self._leafy_status_lbl = tk.Label(
            title_bar, text="\u25cf Ready", bg=T["card_header"],
            fg=T["success"], font=("Segoe UI", 8))
        self._leafy_status_lbl.pack(side="right", padx=14)

        # ── API key row ───────────────────────────────────────────────────────
        key_row = tk.Frame(win, bg=T["card"], pady=5)
        key_row.pack(fill="x")
        tk.Label(key_row, text="OpenAI key:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 8)).pack(side="left", padx=(10, 4))
        tk.Entry(key_row, textvariable=self._leafy_api_key, show="\u2022", width=32,
                 bg=T["entry"], fg=T["text"], relief="flat", bd=0,
                 font=("Segoe UI", 8), insertbackground=T["primary"],
                 highlightthickness=1, highlightcolor=T["border"],
                 highlightbackground=T["border"]).pack(side="left", padx=(0, 6))
        tk.Label(key_row, text="Model:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 8)).pack(side="left", padx=(4, 2))
        ttk.Combobox(key_row, textvariable=self._leafy_model,
                     values=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                     state="readonly", width=13,
                     font=("Segoe UI", 8)).pack(side="left", padx=(0, 6))
        tk.Button(key_row, text="Save", command=self._leafy_save_key,
                  **{**_qbtn, "padx": 6}).pack(side="left")

        # ── Quick-action row ──────────────────────────────────────────────────
        qa_row = tk.Frame(win, bg=T["panel"], pady=5)
        qa_row.pack(fill="x")
        tk.Button(qa_row, text="\U0001f4ca Analyze Now",
                  command=self._leafy_analyze_now, **_btn).pack(side="left", padx=(8, 4))
        tk.Button(qa_row, text="\U0001f4a7 Hydro",
                  command=lambda: self._leafy_quick("hydro"), **_qbtn).pack(side="left", padx=(0, 4))
        tk.Button(qa_row, text="\U0001f321 VPD",
                  command=lambda: self._leafy_quick("vpd"), **_qbtn).pack(side="left", padx=(0, 4))
        tk.Button(qa_row, text="\U0001f33f Health",
                  command=lambda: self._leafy_quick("health"), **_qbtn).pack(side="left", padx=(0, 4))
        tk.Button(qa_row, text="\U0001f9ea Steer",
                  command=lambda: self._leafy_quick("steer"), **_qbtn).pack(side="left", padx=(0, 4))
        tk.Button(qa_row, text="\U0001f5d1 Clear",
                  command=self._leafy_clear_chat, **_qbtn).pack(side="right", padx=8)

        # ── Chat area ─────────────────────────────────────────────────────────
        chat_frame = tk.Frame(win, bg=T["bg"])
        chat_frame.pack(fill="both", expand=True, padx=6, pady=(4, 0))
        chat_vsb = ttk.Scrollbar(chat_frame, orient="vertical")
        self._leafy_chat_txt = tk.Text(
            chat_frame, bg=T["graph_bg"], fg=T["text"], relief="flat",
            bd=0, wrap="word", state="disabled",
            font=("Segoe UI", 9), yscrollcommand=chat_vsb.set,
            cursor="arrow", highlightthickness=0,
            padx=12, pady=10, spacing1=4, spacing3=4)
        chat_vsb.configure(command=self._leafy_chat_txt.yview)
        chat_vsb.pack(side="right", fill="y")
        self._leafy_chat_txt.pack(fill="both", expand=True)
        # Text tags
        self._leafy_chat_txt.tag_configure(
            "leafy_name", foreground=T["accent"], font=("Segoe UI", 9, "bold"))
        self._leafy_chat_txt.tag_configure(
            "user_name",  foreground=T["primary_lt"], font=("Segoe UI", 9, "bold"))
        self._leafy_chat_txt.tag_configure(
            "leafy_msg",  foreground=T["text"], font=("Segoe UI", 9))
        self._leafy_chat_txt.tag_configure(
            "user_msg",   foreground=T["text_muted"], font=("Segoe UI", 9))
        self._leafy_chat_txt.tag_configure(
            "thinking",   foreground=T["text_dim"], font=("Segoe UI", 9, "italic"))

        # ── Input row ─────────────────────────────────────────────────────────
        input_row = tk.Frame(win, bg=T["card"], pady=7)
        input_row.pack(fill="x")
        self._leafy_input_var = tk.StringVar()
        inp = tk.Entry(input_row, textvariable=self._leafy_input_var,
                       bg=T["entry"], fg=T["text"], relief="flat", bd=0,
                       font=("Segoe UI", 10), insertbackground=T["primary"],
                       highlightthickness=1, highlightcolor=T["primary"],
                       highlightbackground=T["border"])
        inp.pack(side="left", fill="x", expand=True, padx=(10, 6), ipady=6)
        inp.bind("<Return>", lambda e: self._leafy_send())
        tk.Button(input_row, text="Send \u25b6",
                  command=self._leafy_send, **_btn).pack(side="right", padx=(0, 8))

        # Post greeting
        self._leafy_post_msg("leafy", _LEAFY_GREETING)

    def _leafy_save_key(self):
        """Persist API key + model preference to a local JSON config file."""
        key = self._leafy_api_key.get().strip()
        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".leafy_config.json")
        try:
            cfg = {}
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
            cfg["api_key"] = key
            cfg["model"]   = self._leafy_model.get()
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            self._leafy_set_status("Key saved \u2714", ok=True)
        except Exception as exc:
            self._leafy_set_status(f"Save failed: {exc}", ok=False)

    def _leafy_load_config(self):
        """Load saved API key / model preference on boot."""
        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".leafy_config.json")
        try:
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                self._leafy_api_key.set(cfg.get("api_key", ""))
                self._leafy_model.set(cfg.get("model", "gpt-4o"))
        except Exception:
            pass

    def _leafy_set_status(self, text, ok=True):
        T = THEME
        if self._leafy_status_lbl and self._leafy_status_lbl.winfo_exists():
            self._leafy_status_lbl.config(
                text=f"\u25cf {text}",
                fg=T["success"] if ok else T["error"])

    def _leafy_post_msg(self, role, text):
        """Append a styled message to the chat widget (always on main thread)."""
        def _insert():
            if self._leafy_chat_txt is None:
                return
            try:
                if not self._leafy_chat_txt.winfo_exists():
                    return
            except tk.TclError:
                return
            txt = self._leafy_chat_txt
            txt.config(state="normal")
            if role == "leafy":
                txt.insert("end", "\U0001f331 Leafy\n", "leafy_name")
                txt.insert("end", text + "\n\n", "leafy_msg")
            elif role == "user":
                txt.insert("end", "\U0001f464 You\n", "user_name")
                txt.insert("end", text + "\n\n", "user_msg")
            elif role == "thinking":
                txt.insert("end", text + "\n", "thinking")
            txt.config(state="disabled")
            txt.yview_moveto(1.0)
        self.root.after(0, _insert)

    def _leafy_clear_chat(self):
        if self._leafy_chat_txt:
            try:
                self._leafy_chat_txt.config(state="normal")
                self._leafy_chat_txt.delete("1.0", "end")
                self._leafy_chat_txt.config(state="disabled")
            except tk.TclError:
                pass
        self._leafy_chat = []
        self._leafy_post_msg("leafy", _LEAFY_GREETING)

    # ── Context builder ───────────────────────────────────────────────────────

    def _leafy_build_context(self):
        """Assemble a concise snapshot of ALL live sensor data for the LLM prompt."""
        lines = ["=== GreenSight Live Sensor Snapshot ==="]

        with self._auto_sensor_lock:
            cache = dict(self._auto_sensor_cache)

        # Environmental
        env_lines = []
        for k in self._env_ch_keys:
            v     = cache.get(k)
            label = self._AUTO_CH_LABELS.get(k, k)
            env_lines.append(f"  {label}: {f'{v:.2f}' if v is not None else 'N/A'}")
        # VPD
        temp = cache.get("temperature")
        rh   = cache.get("humidity")
        if temp is not None and rh is not None:
            try:
                svp = 0.6108 * math.exp(17.27 * temp / (temp + 237.3))
                vpd = svp * (1.0 - rh / 100.0)
                env_lines.append(f"  VPD (calculated): {vpd:.3f} kPa")
            except Exception:
                pass
        if env_lines:
            lines.append("--- Environmental ---")
            lines.extend(env_lines)

        # Hydroponics
        hydro_lines = []
        for k in self._hydro_ch_keys:
            v     = cache.get(k)
            label = self._AUTO_CH_LABELS.get(k, k)
            tgt   = ""
            if hasattr(self, "hydro_targets") and k in self.hydro_targets:
                lo, hi = self.hydro_targets[k]
                status = " \u2714" if (v is not None and lo <= v <= hi) else " \u26a0"
                tgt = f"  [target {lo:.1f}\u2013{hi:.1f}]{status}"
            hydro_lines.append(
                f"  {label}: {f'{v:.2f}' if v is not None else 'N/A'}{tgt}")
        if hydro_lines:
            lines.append("--- Hydroponics ---")
            lines.extend(hydro_lines)

        # Plant health
        active = [(i, d) for i, d in enumerate(self.comm_data) if d is not None]
        if active:
            lines.append("--- Plant Health (per panel) ---")
            for slot, snap in active:
                score  = snap.get("health_score", "N/A")
                hlabel = snap.get("health_label", "?")
                ndvi   = snap.get("ndvi", "N/A")
                flags  = snap.get("nutrient_flag", [])
                s_str  = f"{score:.2f}" if isinstance(score, float) else str(score)
                n_str  = f"{ndvi:.3f}"  if isinstance(ndvi,  float) else str(ndvi)
                lines.append(f"  Panel {slot+1}: Health={s_str} ({hlabel}), NDVI\u2248{n_str}")
                for fl in flags:
                    lines.append(f"    {fl}")

        # Pump states
        if hasattr(self, "hydro_pump_vars") and hasattr(self, "hydro_pump_names"):
            states = [f"{self.hydro_pump_names[i]}: {'ON' if self.hydro_pump_vars[i].get() else 'off'}"
                      for i in range(len(self.hydro_pump_vars))]
            lines.append("--- Dosing Pumps ---")
            lines.append("  " + "  |  ".join(states))

        lines.append("=== End Snapshot ===")
        return "\n".join(lines)

    # ── Dispatch helpers ──────────────────────────────────────────────────────

    def _leafy_analyze_now(self):
        ctx = self._leafy_build_context()
        prompt = (
            "Please analyse the current greenhouse sensor snapshot and give me:\n"
            "1. \U0001f534 Critical issues (need immediate action)\n"
            "2. \U0001f7e1 Warnings (monitor closely)\n"
            "3. \u2705 What\'s looking good\n"
            "4. \U0001f4cb Top 3 action items prioritised by urgency\n\n"
            + ctx
        )
        self._leafy_post_msg("user", "\U0001f4ca Analyze all sensors now")
        self._leafy_dispatch(prompt)

    def _leafy_quick(self, topic):
        ctx = self._leafy_build_context()
        prompts = {
            "hydro":  (f"Focus on the hydroponic readings. What pH, EC, and nutrient adjustments "
                       f"are needed? Give specific dose recommendations where possible.\n\n{ctx}"),
            "vpd":    (f"Evaluate my VPD and climate conditions for crop steering. Am I in a "
                       f"vegetative or generative range? What should I adjust?\n\n{ctx}"),
            "health": (f"Diagnose any plant health or nutrient deficiency issues from the metrics. "
                       f"What corrective steps do you recommend?\n\n{ctx}"),
            "steer":  (f"Based on everything you can see, am I steering vegetative or generative "
                       f"right now? What one change would have the biggest positive impact?\n\n{ctx}"),
        }
        labels = {
            "hydro":  "\U0001f4a7 Check hydroponics",
            "vpd":    "\U0001f321 VPD & climate check",
            "health": "\U0001f33f Plant health check",
            "steer":  "\U0001f9ea Crop steering check",
        }
        self._leafy_post_msg("user", labels.get(topic, topic))
        self._leafy_dispatch(prompts.get(topic, ctx))

    def _leafy_send(self):
        if self._leafy_input_var is None:
            return
        user_text = self._leafy_input_var.get().strip()
        if not user_text:
            return
        self._leafy_input_var.set("")
        self._leafy_post_msg("user", user_text)
        # Attach live sensor data if the message mentions sensor-related words
        sensor_kws = {"sensor", "reading", "ph", "ec", "temp", "vpd", "humidity",
                       "co2", "water", "plant", "health", "defici", "nutrient",
                       "ndvi", "pump", "ec", "tds", "do ", "orp"}
        if any(kw in user_text.lower() for kw in sensor_kws):
            ctx     = self._leafy_build_context()
            payload = user_text + "\n\n[Current sensor data attached:]\n" + ctx
        else:
            payload = user_text
        self._leafy_dispatch(payload)

    def _leafy_dispatch(self, prompt):
        """Route to OpenAI if a key is set, otherwise use the rule-based advisor."""
        if self._leafy_thinking:
            self._leafy_post_msg("leafy", "\u23f3 Still thinking\u2026 please wait.")
            return
        key = self._leafy_api_key.get().strip()
        if key:
            self._leafy_thinking = True
            self._leafy_set_status("Thinking\u2026", ok=True)
            self._leafy_post_msg("thinking", "\U0001f331 Leafy is thinking\u2026")
            threading.Thread(
                target=self._leafy_call_openai,
                args=(prompt, key), daemon=True).start()
        else:
            reply = self._leafy_rule_based_advice()
            self._leafy_post_msg("leafy", reply)

    def _leafy_call_openai(self, prompt, api_key):
        """Background thread — calls OpenAI chat completion and posts the reply."""
        try:
            try:
                import openai as _openai
            except ImportError:
                fallback = self._leafy_rule_based_advice()
                self.root.after(0, lambda: self._leafy_post_msg(
                    "leafy",
                    "\u26a0 The `openai` package is not installed.\n"
                    "Run:  pip install openai\nthen restart.\n\n"
                    "Here is rule-based advice instead:\n\n" + fallback))
                return

            client = _openai.OpenAI(api_key=api_key)
            messages = [{"role": "system", "content": _LEAFY_SYSTEM}]
            # Keep the last 12 exchanges for multi-turn context
            for entry in self._leafy_chat[-12:]:
                messages.append(entry)
            messages.append({"role": "user", "content": prompt})

            resp  = client.chat.completions.create(
                model=self._leafy_model.get(),
                messages=messages,
                max_tokens=1000,
                temperature=0.55,
            )
            reply = resp.choices[0].message.content.strip()
            # Save to conversation history
            self._leafy_chat.append({"role": "user",      "content": prompt})
            self._leafy_chat.append({"role": "assistant",  "content": reply})
            self.root.after(0, lambda: self._leafy_post_msg("leafy", reply))
            self.root.after(0, lambda: self._leafy_set_status("Ready", ok=True))

        except Exception as exc:
            err = f"\u26a0 OpenAI error: {exc}\n\nFalling back to rule-based advice:\n\n"
            fallback = self._leafy_rule_based_advice()
            self.root.after(0, lambda e=err, f=fallback:
                self._leafy_post_msg("leafy", e + f))
            self.root.after(0, lambda: self._leafy_set_status("Error (fallback used)", ok=False))
        finally:
            self._leafy_thinking = False

    # ── Rule-based offline advisor ────────────────────────────────────────────

    def _leafy_rule_based_advice(self):
        """
        Evaluates the live sensor cache with domain rules and returns a
        formatted crop-steering advisory — no external API required.
        """
        with self._auto_sensor_lock:
            cache = dict(self._auto_sensor_cache)

        issues   = []
        warnings = []
        good     = []
        actions  = []

        # ── pH ────────────────────────────────────────────────────────────────
        ph = cache.get("ph")
        if ph is not None:
            if ph < 5.3:
                issues.append(f"\U0001f534 pH {ph:.2f} \u2014 critically LOW \u2192 Dose pH Up (KOH/KHCO\u2083), target 5.8\u20136.2")
                actions.append("Raise pH immediately; flush if >0.5 below target")
            elif ph < 5.8:
                warnings.append(f"\U0001f7e1 pH {ph:.2f} slightly low \u2192 Add small pH Up dose")
            elif ph > 6.8:
                issues.append(f"\U0001f534 pH {ph:.2f} \u2014 critically HIGH \u2192 Dose pH Down (H\u2083PO\u2084/citric acid)")
                actions.append("pH >6.8 locks out iron/manganese \u2014 flush reservoir if needed")
            elif ph > 6.3:
                warnings.append(f"\U0001f7e1 pH {ph:.2f} slightly high \u2192 Small pH Down dose recommended")
            else:
                good.append(f"\u2705 pH {ph:.2f} is in the optimal range (5.8\u20136.2)")

        # ── EC ────────────────────────────────────────────────────────────────
        ec = cache.get("ec")
        if ec is not None:
            if ec < 0.8:
                issues.append(f"\U0001f534 EC {ec:.2f}\u00a0mS/cm \u2014 critically LOW (starvation risk) \u2192 Mix stock A+B")
                actions.append("Raise EC by 0.2\u00a0mS/cm per feed cycle; check runoff EC")
            elif ec < 1.2:
                warnings.append(f"\U0001f7e1 EC {ec:.2f}\u00a0mS/cm low for vegetative stage \u2192 Increase nutrient concentration")
            elif ec > 4.0:
                issues.append(f"\U0001f534 EC {ec:.2f}\u00a0mS/cm \u2014 very HIGH (salt stress) \u2192 Flush with plain water")
                actions.append("High EC causes tip burn & wilting \u2014 flush immediately")
            elif ec > 3.0:
                warnings.append(f"\U0001f7e1 EC {ec:.2f}\u00a0mS/cm elevated \u2192 Consider a clean-water feed or partial flush")
            else:
                good.append(f"\u2705 EC {ec:.2f}\u00a0mS/cm is within target range")

        # ── VPD ───────────────────────────────────────────────────────────────
        temp = cache.get("temperature")
        rh   = cache.get("humidity")
        if temp is not None and rh is not None:
            try:
                svp = 0.6108 * math.exp(17.27 * temp / (temp + 237.3))
                vpd = svp * (1.0 - rh / 100.0)
                if vpd < 0.4:
                    warnings.append(f"\U0001f7e1 VPD {vpd:.2f}\u00a0kPa \u2014 too low (disease risk) \u2192 Reduce RH or raise air temp")
                elif vpd < 0.8:
                    warnings.append(f"\U0001f7e1 VPD {vpd:.2f}\u00a0kPa \u2014 low-vegetative zone; watch for Botrytis")
                elif vpd > 1.6:
                    issues.append(f"\U0001f534 VPD {vpd:.2f}\u00a0kPa \u2014 HIGH (heat/drought stress) \u2192 Raise humidity or lower temp")
                    actions.append("High VPD forces stomata closed \u2014 nutrient uptake & photosynthesis drop")
                elif vpd > 1.2:
                    warnings.append(f"\U0001f7e1 VPD {vpd:.2f}\u00a0kPa \u2014 generative zone; monitor plant stress")
                else:
                    good.append(f"\u2705 VPD {vpd:.2f}\u00a0kPa is in the optimal range (0.8\u20131.2\u00a0kPa)")
            except Exception:
                pass

        # ── CO\u2082 ──────────────────────────────────────────────────────────────────
        co2 = cache.get("co2")
        if co2 is not None:
            if co2 < 350:
                warnings.append(f"\U0001f7e1 CO\u2082 {co2:.0f}\u00a0ppm \u2014 below ambient; check ventilation or CO\u2082 injection")
            elif co2 > 1500:
                warnings.append(f"\U0001f7e1 CO\u2082 {co2:.0f}\u00a0ppm \u2014 very high; ensure adequate air exchange")
            elif co2 >= 800:
                good.append(f"\u2705 CO\u2082 {co2:.0f}\u00a0ppm \u2014 enriched atmosphere (great for photosynthesis)")
            else:
                good.append(f"\u2705 CO\u2082 {co2:.0f}\u00a0ppm \u2014 within normal range")

        # ── Water temperature ─────────────────────────────────────────────────
        wtemp = cache.get("water_temp")
        if wtemp is not None:
            if wtemp < 16:
                warnings.append(f"\U0001f7e1 Water temp {wtemp:.1f}\u00b0C \u2014 cold; root activity reduced below 18\u00b0C")
            elif wtemp > 26:
                warnings.append(f"\U0001f7e1 Water temp {wtemp:.1f}\u00b0C \u2014 warm; DO drops and pythium risk rises above 24\u00b0C")
            else:
                good.append(f"\u2705 Water temp {wtemp:.1f}\u00b0C is in the ideal range (18\u201324\u00b0C)")

        # ── DO ────────────────────────────────────────────────────────────────
        do = cache.get("do")
        if do is not None:
            if do < 5:
                issues.append(f"\U0001f534 DO {do:.1f}\u00a0mg/L \u2014 critically LOW (root hypoxia) \u2192 Add air stones or chiller")
                actions.append("Dissolved oxygen <5\u00a0mg/L starves roots \u2014 oxygenate reservoir immediately")
            elif do < 7:
                warnings.append(f"\U0001f7e1 DO {do:.1f}\u00a0mg/L \u2014 suboptimal \u2192 Increase aeration")
            else:
                good.append(f"\u2705 DO {do:.1f}\u00a0mg/L \u2014 well-oxygenated reservoir")

        # ── Plant health ──────────────────────────────────────────────────────
        active = [(i, d) for i, d in enumerate(self.comm_data) if d is not None]
        for slot, snap in active:
            score = snap.get("health_score")
            label = snap.get("health_label", "?")
            if score is not None:
                if score < 0.35:
                    issues.append(f"\U0001f534 Panel {slot+1} health score {score:.2f} ({label}) \u2014 urgent attention needed")
                elif score < 0.55:
                    warnings.append(f"\U0001f7e1 Panel {slot+1} health score {score:.2f} ({label}) \u2014 monitor closely")
                else:
                    good.append(f"\u2705 Panel {slot+1} health score {score:.2f} ({label})")
            for fl in snap.get("nutrient_flag", []):
                warnings.append(f"\U0001f7e1 Panel {slot+1}: {fl}")

        # ── Assemble ──────────────────────────────────────────────────────────
        if not any([issues, warnings, good]):
            return (
                "\U0001f331 No live data yet.\nConnect your sensors and press "
                "\"Analyze Now\" once readings are flowing!"
            )
        parts = []
        if issues:
            parts.append("**\U0001f534 Critical Issues**\n" +
                         "\n".join(f"\u2022 {i}" for i in issues))
        if warnings:
            parts.append("**\U0001f7e1 Warnings**\n" +
                         "\n".join(f"\u2022 {w}" for w in warnings))
        if good:
            parts.append("**\u2705 Looking Good**\n" +
                         "\n".join(f"\u2022 {g}" for g in good))
        if actions:
            parts.append("**\U0001f4cb Action Items**\n" +
                         "\n".join(f"{j+1}. {a}" for j, a in enumerate(actions[:4])))
        parts.append("_\u2500 Leafy rule-based advisor (add OpenAI key for AI-powered advice) \u2500_")
        return "\n\n".join(parts)

    def _apply_theme(self):
        """Configure ttk styles, tk option defaults, and matplotlib rcParams for the greenhouse SaaS palette."""
        T = THEME
        self.root.configure(bg=T["bg"])

        # ── Global tk widget defaults (applied before widget creation) ──────
        self.root.option_add("*Font",                   "Segoe\\ UI 9")
        self.root.option_add("*Background",             T["bg"])
        self.root.option_add("*Foreground",             T["text"])
        self.root.option_add("*Button.Background",      T["primary_dk"])
        self.root.option_add("*Button.Foreground",      T["text"])
        self.root.option_add("*Button.Relief",          "flat")
        self.root.option_add("*Button.BorderWidth",     "1")
        self.root.option_add("*Button.ActiveBackground", T["primary"])
        self.root.option_add("*Button.ActiveForeground", T["text"])
        self.root.option_add("*Label.Background",       T["bg"])
        self.root.option_add("*Label.Foreground",       T["text"])
        self.root.option_add("*Frame.Background",       T["bg"])
        self.root.option_add("*Canvas.Background",      T["bg"])
        self.root.option_add("*Scrollbar.Background",   T["card"])
        self.root.option_add("*Scrollbar.TroughColor",  T["scrollbar"])
        self.root.option_add("*Scale.Background",       T["card"])
        self.root.option_add("*Scale.Foreground",       T["text_muted"])
        self.root.option_add("*Scale.ActiveBackground", T["primary"])
        self.root.option_add("*Scale.TroughColor",      T["scrollbar"])
        self.root.option_add("*Scale.SliderRelief",     "flat")
        self.root.option_add("*Spinbox.Background",     T["entry"])
        self.root.option_add("*Spinbox.Foreground",     T["text"])
        self.root.option_add("*Spinbox.InsertBackground", T["primary"])
        self.root.option_add("*Entry.Background",       T["entry"])
        self.root.option_add("*Entry.Foreground",       T["text"])
        self.root.option_add("*Entry.InsertBackground", T["primary"])
        self.root.option_add("*Checkbutton.Background",       T["card"])
        self.root.option_add("*Checkbutton.Foreground",       T["text"])
        self.root.option_add("*Checkbutton.ActiveBackground", T["card"])
        self.root.option_add("*Checkbutton.ActiveForeground", T["primary_lt"])
        self.root.option_add("*Checkbutton.SelectColor",      T["primary_dk"])

        # ── ttk styles ───────────────────────────────────────────────────────
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("TNotebook",
            background=T["panel"], borderwidth=0, tabmargins=[0, 0, 0, 0])
        style.configure("TNotebook.Tab",
            background=T["panel"], foreground=T["text_muted"],
            padding=[18, 8], font=("Segoe UI", 10),
            borderwidth=0, focuscolor=T["panel"])
        style.map("TNotebook.Tab",
            background=[("selected", T["card_header"]), ("active", T["card"])],
            foreground=[("selected", T["accent"]),      ("active", T["primary_lt"])],
            expand=[("selected", [1, 1, 1, 0])],
        )

        style.configure("TFrame", background=T["panel"])

        style.configure("TLabelframe",
            background=T["card"], foreground=T["primary_lt"],
            bordercolor=T["border"], relief="groove", borderwidth=1)
        style.configure("TLabelframe.Label",
            background=T["card"], foreground=T["primary"],
            font=("Segoe UI", 9, "bold"))

        style.configure("TScrollbar",
            background=T["card_header"], troughcolor=T["scrollbar"],
            arrowcolor=T["primary_lt"], borderwidth=0, relief="flat", gripcount=0)
        style.map("TScrollbar",
            background=[("active", T["primary_dk"])],
        )

        style.configure("Treeview",
            background=T["treeview_row"], foreground=T["text"],
            fieldbackground=T["treeview_row"],
            rowheight=26, font=("Segoe UI", 9), relief="flat", borderwidth=0)
        style.configure("Treeview.Heading",
            background=T["card_header"], foreground=T["primary_lt"],
            font=("Segoe UI", 9, "bold"), relief="flat", padding=[6, 4])
        style.map("Treeview",
            background=[("selected", T["treeview_sel"])],
            foreground=[("selected", T["text"])],
        )
        style.map("Treeview.Heading",
            background=[("active", T["border"])],
        )

        style.configure("TCombobox",
            fieldbackground=T["entry"], background=T["card_header"],
            foreground=T["text"], selectbackground=T["treeview_sel"],
            selectforeground=T["text"], arrowcolor=T["primary_lt"],
            bordercolor=T["border"], lightcolor=T["border"],
            darkcolor=T["border"], relief="flat")
        style.map("TCombobox",
            fieldbackground=[("readonly", T["entry"]),    ("disabled", T["panel"])],
            foreground=[("readonly", T["text"]),           ("disabled", T["text_dim"])],
            selectbackground=[("readonly", T["treeview_sel"])],
            background=[("active", T["card_header"])],
        )

        # ── Matplotlib global theme ──────────────────────────────────────────
        plt.rcParams.update({
            "figure.facecolor":  T["graph_bg"],
            "axes.facecolor":    T["graph_ax"],
            "axes.edgecolor":    T["border"],
            "axes.labelcolor":   T["text_muted"],
            "axes.titlecolor":   T["primary_lt"],
            "xtick.color":       T["text_muted"],
            "ytick.color":       T["text_muted"],
            "text.color":        T["text"],
            "grid.color":        T["border"],
            "grid.alpha":        0.4,
            "lines.color":       T["primary"],
            "patch.facecolor":   T["primary_dk"],
            "legend.facecolor":  T["card"],
            "legend.edgecolor":  T["border"],
            "legend.labelcolor": T["text"],
            "figure.edgecolor":  T["border"],
        })

    def _create_header(self):
        """Branded sticky header bar shown above the notebook."""
        T = THEME
        header = tk.Frame(self.root, bg=T["card_header"], height=50)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text="🌱  GreenSight",
                 bg=T["card_header"], fg=T["accent"],
                 font=("Segoe UI", 15, "bold")).pack(side="left", padx=18, pady=10)

        tk.Label(header, text="Plant Intelligence Platform",
                 bg=T["card_header"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left", pady=14)

        self._header_time = tk.Label(header, text="",
                                     bg=T["card_header"], fg=T["text_dim"],
                                     font=("Segoe UI", 8))
        self._header_time.pack(side="right", padx=18, pady=16)

        tk.Label(header, text="v1.0",
                 bg=T["card_header"], fg=T["text_dim"],
                 font=("Segoe UI", 8)).pack(side="right", padx=(0, 4), pady=16)

        tk.Button(header, text="\U0001f331 Ask Leafy",
                  command=self._leafy_open,
                  bg=T["primary_dk"], fg=T["accent"],
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  padx=12, pady=4, bd=0, cursor="hand2",
                  activebackground=T["primary"],
                  activeforeground=T["text"]).pack(side="right", padx=(0, 16), pady=10)

        self._tick_header_clock()

    def _tick_header_clock(self):
        self._header_time.config(text=datetime.now().strftime("%Y-%m-%d   %H:%M:%S"))
        if not self.is_closing:
            self.root.after(1000, self._tick_header_clock)

    def create_plant_detection_widgets(self):
        T = THEME
        _btn = dict(bg=T["primary_dk"], fg=T["text"], relief="flat",
                    font=("Segoe UI", 9), padx=10, pady=4,
                    activebackground=T["primary"], activeforeground=T["text"],
                    cursor="hand2", bd=0)
        _sbtn = dict(bg=T["card_header"], fg=T["primary_lt"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary_dk"], activeforeground=T["text"],
                     cursor="hand2", bd=0)

        self.main_frame = ScrollableFrame(self.plant_tab, bg=T["panel"])
        self.main_frame.pack(fill="both", expand=True)

        self.scrollable_frame = self.main_frame.scrollable_frame
        self.scrollable_frame.configure(bg=T["panel"])
        for column in range(self.camera_panel_columns):
            self.scrollable_frame.columnconfigure(column, weight=1)

        # ── Toolbar ───────────────────────────────────────────────────────────
        controls_frame = tk.Frame(self.plant_tab, bg=T["panel"])
        controls_frame.pack(fill="x", padx=10, pady=(6, 2))

        tk.Button(controls_frame, text="⊕ Zoom In",  command=lambda: self.main_frame.zoom(1.1), **_btn).pack(side="left", padx=(0, 4))
        tk.Button(controls_frame, text="⊖ Zoom Out", command=lambda: self.main_frame.zoom(0.9), **_btn).pack(side="left", padx=(0, 4))
        tk.Button(controls_frame, text="↺ Cameras",  command=self.refresh_camera_inventory, **_btn).pack(side="left", padx=(0, 4))
        tk.Button(controls_frame, text="🚁 DJI Drone",  command=self._open_drone_manager, **_btn).pack(side="left", padx=(0, 12))

        # Section selector — each section = 1 overhead + 1 side camera
        tk.Label(controls_frame, text="Sections:", bg=T["panel"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        section_values = [str(v) for v in range(1, (self.max_camera_slots // 2) + 1)]
        self.visible_section_dropdown = ttk.Combobox(
            controls_frame, values=section_values, state="readonly", width=4)
        n_sections_default = max(1, self.visible_camera_slots // 2)
        self.visible_section_dropdown.set(str(n_sections_default))
        self.visible_section_dropdown.bind(
            "<<ComboboxSelected>>",
            lambda e: self.set_visible_sections(int(self.visible_section_dropdown.get())))
        self.visible_section_dropdown.pack(side="left")

        # Hidden legacy dropdown kept for internal compat (not shown to user)
        panel_values = [str(v) for v in range(1, self.max_camera_slots + 1)]
        self.visible_panel_dropdown = ttk.Combobox(
            controls_frame, values=panel_values, state="readonly", width=1)
        self.visible_panel_dropdown.set(str(self.visible_camera_slots))
        # deliberately not packed

        # Section legend
        self._section_pair_label = tk.Label(
            controls_frame,
            text=self._section_legend_text(n_sections_default),
            bg=T["panel"], fg=T["text_dim"], font=("Segoe UI", 8))
        self._section_pair_label.pack(side="left", padx=(10, 0))

        tk.Button(controls_frame, text="↻ Refresh Analyses", command=self.refresh_all, **_btn).pack(side="right")

        ml_frame = tk.Frame(self.plant_tab, bg=T["panel"])
        ml_frame.pack(fill="x", padx=10, pady=(2, 6))
        tk.Button(ml_frame, text="📊  Create Dataset",    command=self.create_dataset,  **_sbtn).pack(side="left", padx=(0, 6))
        tk.Button(ml_frame, text="🤖  Train / Load Model", command=self.train_model,     **_sbtn).pack(side="left")

        # ── Camera panel cards ────────────────────────────────────────────────
        _sl = dict(bg=T["card"], fg=T["text_muted"], troughcolor=T["scrollbar"],
                   activebackground=T["primary"], highlightthickness=0,
                   font=("Segoe UI", 8), relief="flat", sliderlength=12, bd=0)
        _spn = dict(bg=T["entry"], fg=T["text"], relief="flat",
                    buttonbackground=T["card_header"],
                    highlightthickness=1, highlightcolor=T["border"],
                    highlightbackground=T["border"],
                    insertbackground=T["primary"], bd=0,
                    font=("Segoe UI", 9))
        _cbtn = dict(bg=T["primary_dk"], fg=T["text"], relief="flat",
                     font=("Segoe UI", 9), padx=8, pady=3,
                     activebackground=T["primary"], activeforeground=T["text"],
                     cursor="hand2", bd=0)

        for i in range(self.max_camera_slots):
            frame = tk.Frame(self.scrollable_frame, bg=T["card"],
                             highlightbackground=T["border"], highlightthickness=1)
            for col in range(2):
                frame.columnconfigure(col, weight=1)
            for row in range(10):
                frame.rowconfigure(row, weight=1)

            # Derive section and role from slot index
            section_num  = (i // 2) + 1
            is_overhead  = (i % 2 == 0)
            role_label   = "🔝 Overhead" if is_overhead else "↕ Side/Canopy"
            role_color   = T["accent"]   if is_overhead else T["warning"]

            # Card header strip
            hdr = tk.Frame(frame, bg=T["card_header"], height=28)
            hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
            hdr.grid_propagate(False)
            tk.Label(hdr, text=f"  📷  Panel {i + 1}",
                     bg=T["card_header"], fg=T["primary_lt"],
                     font=("Segoe UI", 9, "bold")).pack(side="left", padx=4, pady=4)
            # Section badge
            tk.Label(hdr, text=f"Section {section_num}",
                     bg=T["card_header"], fg=T["text_dim"],
                     font=("Segoe UI", 8)).pack(side="left", padx=(0, 6))
            # Role badge
            tk.Label(hdr, text=role_label,
                     bg=T["card_header"], fg=role_color,
                     font=("Segoe UI", 8, "bold")).pack(side="left")

            # Camera select
            tk.Label(frame, text="Camera", bg=T["card"], fg=T["text_muted"],
                     font=("Segoe UI", 8)).grid(row=1, column=0, sticky="w", padx=(8, 4), pady=(6, 2))
            camera_dropdown = ttk.Combobox(frame, values=self.get_camera_labels(), state="readonly")
            camera_dropdown.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(6, 2))
            camera_dropdown.bind("<<ComboboxSelected>>", lambda e, idx=i: self.select_camera(idx))

            # Stream controls row
            scf = tk.Frame(frame, bg=T["card"])
            scf.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=4)

            stream_toggle  = tk.Button(scf, text="▶ Stream",  command=lambda idx=i: self.toggle_stream(idx), **_cbtn)
            capture_button = tk.Button(scf, text="📷 Capture", command=lambda idx=i: self.capture_single_frame(idx), **_cbtn)
            stream_toggle.pack(side="left", padx=(0, 4))
            capture_button.pack(side="left", padx=(0, 10))

            tk.Label(scf, text="Quality", bg=T["card"], fg=T["text_dim"],
                     font=("Segoe UI", 8)).pack(side="left", padx=(0, 2))
            quality_slider = tk.Scale(scf, from_=10, to=100, orient="horizontal", length=80,
                                      command=lambda val, idx=i: self.set_stream_quality(idx, val), **_sl)
            quality_slider.set(self.stream_quality[i])
            quality_slider.pack(side="left", padx=(0, 8))

            tk.Label(scf, text="FPS", bg=T["card"], fg=T["text_dim"],
                     font=("Segoe UI", 8)).pack(side="left", padx=(0, 2))
            fps_slider = tk.Scale(scf, from_=1, to=30, orient="horizontal", length=70,
                                  command=lambda val, idx=i: self.set_stream_fps(idx, val), **_sl)
            fps_slider.set(self.stream_fps[i])
            fps_slider.pack(side="left")

            # DJI / RTMP quick-connect row
            drf = tk.Frame(frame, bg=T["card"])
            drf.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 2))
            drf.grid_remove()   # hidden initially; shown when drone mode active
            tk.Label(drf, text="🚁 RTMP/URL:", bg=T["card"], fg=T["accent"],
                     font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 4))
            _drone_ent = dict(bg=T["entry"], fg=T["text"], relief="flat",
                              highlightthickness=1, highlightcolor=T["border"],
                              highlightbackground=T["border"],
                              insertbackground=T["primary"], bd=0,
                              font=("Segoe UI", 8))
            drone_url_var = tk.StringVar(value="")
            drone_entry = tk.Entry(drf, textvariable=drone_url_var, width=28, **_drone_ent)
            drone_entry.pack(side="left", padx=(0, 4))
            drone_status_lbl = tk.Label(drf, text="", bg=T["card"],
                                        fg=T["text_muted"], font=("Segoe UI", 8))
            drone_status_lbl.pack(side="left")
            tk.Button(drf, text="Connect",
                      command=lambda idx=i: self._drone_quick_connect(idx),
                      **_cbtn).pack(side="left", padx=(4, 0))

            # Pipeline select
            tk.Label(frame, text="Pipeline", bg=T["card"], fg=T["text_muted"],
                     font=("Segoe UI", 8)).grid(row=3, column=0, sticky="w", padx=(8, 4), pady=(4, 2))
            analysis_dropdown = ttk.Combobox(frame, values=self.analysis_options, state="readonly")
            analysis_dropdown.grid(row=3, column=1, sticky="ew", padx=(0, 8), pady=(4, 2))
            analysis_dropdown.current(0)

            # Analysis controls row
            acf = tk.Frame(frame, bg=T["card"])
            acf.grid(row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=4)

            analyze_button = tk.Button(acf, text="⚡ Analyze",
                                       command=lambda idx=i: self.analyze_current_frame(idx), **_cbtn)
            analyze_button.pack(side="left", padx=(0, 8))

            auto_analyze_var = tk.BooleanVar(value=False)
            tk.Checkbutton(acf, text="Auto",
                           variable=auto_analyze_var,
                           command=lambda idx=i: self.update_camera_runtime_status(idx),
                           bg=T["card"], fg=T["text"], selectcolor=T["primary_dk"],
                           activebackground=T["card"], activeforeground=T["primary_lt"],
                           font=("Segoe UI", 9)).pack(side="left", padx=(0, 6))

            tk.Label(acf, text="Every (s):", bg=T["card"], fg=T["text_dim"],
                     font=("Segoe UI", 8)).pack(side="left", padx=(0, 2))
            analysis_interval_var = tk.StringVar(value=str(DEFAULT_AUTO_ANALYSIS_INTERVAL_SECONDS))
            tk.Spinbox(acf, from_=1, to=60, width=4, textvariable=analysis_interval_var,
                       **_spn).pack(side="left")

            # Camera feed display
            image_display = tk.Label(frame, text="  No Signal  ",
                                     bg="#0a120a", fg=T["text_dim"], font=("Segoe UI", 9))
            image_display.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=6, pady=(2, 0))

            # Treeview
            tk.Label(frame, text="Analysis Results", bg=T["card"], fg=T["text_muted"],
                     font=("Segoe UI", 8, "bold")).grid(row=6, column=0, sticky="w", padx=8, pady=(6, 2))
            tree = ttk.Treeview(frame, show="headings", height=5)
            tree.grid(row=7, column=0, columnspan=2, sticky="nsew", padx=6, pady=(0, 4))

            # Graph frame
            graph_frame = tk.Frame(frame, bg=T["card"])
            graph_frame.grid(row=8, column=0, columnspan=2, sticky="nsew", padx=6, pady=4)

            # Status bar
            camera_status_label = tk.Label(frame, text="Idle",
                                           fg=T["text_dim"], bg=T["card"],
                                           font=("Segoe UI", 8))
            camera_status_label.grid(row=9, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 6))

            self.grids.append({
                "frame":                frame,
                "camera_dropdown":      camera_dropdown,
                "analysis_dropdown":    analysis_dropdown,
                "image_display":        image_display,
                "tree":                 tree,
                "graph_frame":          graph_frame,
                "stream_toggle":        stream_toggle,
                "capture_button":       capture_button,
                "quality_slider":       quality_slider,
                "fps_slider":           fps_slider,
                "analyze_button":       analyze_button,
                "auto_analyze_var":     auto_analyze_var,
                "analysis_interval_var": analysis_interval_var,
                "camera_status_label":  camera_status_label,
                # Section / role meta-data (static, derived from slot index)
                "section":              section_num,
                "role":                 "overhead" if is_overhead else "side",
                # DJI / RTMP drone stream widgets
                "drone_url_var":        drone_url_var,
                "drone_entry":          drone_entry,
                "drone_status_lbl":     drone_status_lbl,
                "drone_row_frame":      drf,
            })

        self.update_camera_grid_layout()

    def update_camera_grid_layout(self):
        for idx, grid in enumerate(self.grids):
            if idx < self.visible_camera_slots:
                row = idx // self.camera_panel_columns
                column = idx % self.camera_panel_columns
                grid["frame"].grid(row=row, column=column, padx=10, pady=10, sticky="nsew")
            else:
                grid["frame"].grid_remove()

    def set_visible_camera_slots(self, count):
        self.visible_camera_slots = max(1, min(self.max_camera_slots, count))
        self.visible_panel_dropdown.set(str(self.visible_camera_slots))
        self.update_camera_grid_layout()
        if hasattr(self, "comm_panels") and self.comm_panels:
            self._comm_update_visibility()

    def set_visible_sections(self, n_sections):
        """Show exactly *n_sections* section-pairs (2 cameras each: overhead + side)."""
        n_sections = max(1, min(self.max_camera_slots // 2, n_sections))
        # Keep the dropdown in sync
        if hasattr(self, "visible_section_dropdown"):
            self.visible_section_dropdown.set(str(n_sections))
        # Update legend label
        if self._section_pair_label is not None:
            self._section_pair_label.config(
                text=self._section_legend_text(n_sections))
        self.set_visible_camera_slots(n_sections * 2)

    @staticmethod
    def _section_legend_text(n_sections):
        """Return a short one-line description of the active section pairing."""
        pairs = []
        for s in range(1, n_sections + 1):
            overhead = (s - 1) * 2 + 1   # 1-based panel number
            side     = overhead + 1
            pairs.append(f"S{s}: P{overhead}\u2191 P{side}\u2194")
        return "  |  ".join(pairs)

    def get_camera_labels(self):
        return [camera["label"] for camera in self.cameras]

    def refresh_camera_inventory(self):
        local_cameras = self.get_available_cameras()
        # Merge with any registered drone streams
        drone_entries = [
            {"label": d["label"], "index": d["url"],
             "backend": cv2.CAP_FFMPEG, "type": d["type"]}
            for d in self.drone_streams
        ]
        self.cameras = local_cameras + drone_entries
        camera_labels = self.get_camera_labels()
        for idx, grid in enumerate(self.grids):
            selected_label = grid["camera_dropdown"].get()
            grid["camera_dropdown"]["values"] = camera_labels
            if selected_label in camera_labels:
                grid["camera_dropdown"].set(selected_label)
            else:
                if selected_label:
                    self.stop_stream(idx)
                grid["camera_dropdown"].set("")
                self.current_frames[idx] = None
                self.update_camera_runtime_status(idx)

        logging.info(f"Refreshed camera inventory: {len(camera_labels)} camera(s) available")

    def create_environmental_widgets(self):
        T = THEME
        _btn = dict(bg=T["primary_dk"], fg=T["text"], relief="flat",
                    font=("Segoe UI", 9), padx=10, pady=4,
                    activebackground=T["primary"], activeforeground=T["text"],
                    cursor="hand2", bd=0)
        _spn = dict(bg=T["entry"], fg=T["text"], relief="flat",
                    buttonbackground=T["card_header"],
                    highlightthickness=1, highlightcolor=T["border"],
                    highlightbackground=T["border"],
                    insertbackground=T["primary"], bd=0,
                    font=("Segoe UI", 9))
        _lf  = dict(bg=T["card"], fg=T["primary"], font=("Segoe UI", 10, "bold"),
                    relief="groove", bd=1)

        env_main = tk.Frame(self.env_tab, bg=T["panel"])
        env_main.pack(fill='both', expand=True, padx=12, pady=10)
        env_main.columnconfigure(0, weight=1)
        env_main.columnconfigure(1, weight=1)
        env_main.rowconfigure(0, weight=0)
        env_main.rowconfigure(1, weight=0)
        env_main.rowconfigure(2, weight=1)

        # ── Serial connection ──────────────────────────────────────────────────
        serial_frame = tk.LabelFrame(env_main, text=" Serial Connection ", padx=10, pady=8, **_lf)
        serial_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 8))

        ports = self.get_serial_ports()

        tk.Label(serial_frame, text="Port:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.port_dropdown = ttk.Combobox(serial_frame, values=ports, state="readonly", width=28)
        self.port_dropdown.grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        if ports:
            self.port_dropdown.current(0)

        tk.Label(serial_frame, text="Baud:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w", padx=4)
        baud_rates = ['9600', '19200', '38400', '57600', '115200']
        self.baud_dropdown = ttk.Combobox(serial_frame, values=baud_rates, state="readonly", width=10)
        self.baud_dropdown.grid(row=0, column=3, sticky="ew", padx=4)
        self.baud_dropdown.current(0)

        self.serial_connect_btn = tk.Button(serial_frame, text="⚡  Connect",
                                            command=self.toggle_serial_connection, **_btn)
        self.serial_connect_btn.grid(row=0, column=4, padx=8)
        tk.Button(serial_frame, text="↺  Ports",
                  command=self.refresh_serial_ports, **_btn).grid(row=0, column=5, padx=4)
        tk.Button(serial_frame, text="🗑  Clear",
                  command=self.clear_env_data, **_btn).grid(row=0, column=6, padx=4)

        tk.Label(serial_frame, text="History:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", padx=4, pady=(8, 2))
        tk.Spinbox(serial_frame, from_=10, to=5000, width=10,
                   textvariable=self.history_window_var,
                   **_spn).grid(row=1, column=1, sticky="w", padx=4, pady=(8, 2))
        tk.Button(serial_frame, text="Apply",
                  command=self.set_history_window, **_btn).grid(row=1, column=2, padx=4, pady=(8, 2), sticky="w")

        # ── Gauge panel ────────────────────────────────────────────────────────
        gauge_frame = tk.LabelFrame(env_main, text=" Current Readings ", padx=10, pady=8, **_lf)
        gauge_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=4, pady=(0, 8))
        for col in range(5):
            gauge_frame.columnconfigure(col, weight=1)

        self.gauges = {
            'temperature':  CircularGauge(gauge_frame, "Temp °C",    0,     50,    width=160, height=160),
            'humidity':     CircularGauge(gauge_frame, "Humidity %",  0,     100,   width=160, height=160),
            'co2':          CircularGauge(gauge_frame, "CO₂ ppm",    0,     2000,  width=160, height=160),
            'light':        CircularGauge(gauge_frame, "Light lux",  0,     10000, width=160, height=160),
            'soil_moisture': CircularGauge(gauge_frame, "Soil %",    0,     100,   width=160, height=160),
        }
        for col, key in enumerate(['temperature', 'humidity', 'co2', 'light', 'soil_moisture']):
            self.gauges[key].grid(row=0, column=col, padx=8, pady=8)

        self.last_updated_label = tk.Label(gauge_frame, text="Last Updated: Never",
                                           bg=T["card"], fg=T["text_dim"], font=("Segoe UI", 8))
        self.last_updated_label.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(0, 4))

        # ── Historical graphs ──────────────────────────────────────────────────
        graph_frame = tk.LabelFrame(env_main, text=" Historical Data ", padx=10, pady=8, **_lf)
        graph_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=4, pady=(0, 8))

        self.env_fig = plt.figure(figsize=(11, 5))
        self.env_canvas = FigureCanvasTkAgg(self.env_fig, master=graph_frame)
        self.env_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        # Status bar
        self.status_label = tk.Label(env_main, text="● Disconnected",
                                     fg=T["error"], bg=T["panel"],
                                     font=("Segoe UI", 10, "bold"))
        self.status_label.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        self.update_env_graphs()

    def get_serial_ports(self):
        """Get list of available serial ports"""
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append(port.device)
        return ports

    def refresh_serial_ports(self):
        """Refresh the list of available serial ports"""
        ports = self.get_serial_ports()
        self.port_dropdown['values'] = ports
        if ports:
            self.port_dropdown.current(0)
        logging.info(f"Refreshed serial ports: {len(ports)} ports found")

    def toggle_serial_connection(self):
        """Connect to or disconnect from the selected serial port"""
        if self.is_serial_running:
            # Disconnect
            self.is_serial_running = False
            if self.serial_thread and self.serial_thread.is_alive():
                self.serial_thread.join(timeout=1.0)
            if self.serial_port:
                self.serial_port.close()
                self.serial_port = None
            
            self.serial_connect_btn.config(text="⚡  Connect")
            self.status_label.config(text="● Disconnected", fg=THEME["error"])
            logging.info("Disconnected from serial port")
        else:
            # Connect
            try:
                port = self.port_dropdown.get()
                baud = int(self.baud_dropdown.get())
                
                if not port:
                    logging.error("No serial port selected")
                    return
                    
                self.serial_port = serial.Serial(port, baud, timeout=0.5)
                self.is_serial_running = True
                
                # Start reading thread
                self.serial_thread = threading.Thread(target=self.read_serial_data)
                self.serial_thread.daemon = True
                self.serial_thread.start()
                
                self.serial_connect_btn.config(text="⏹  Disconnect")
                self.status_label.config(text=f"● Connected to {port} at {baud} baud", fg=THEME["success"])
                logging.info(f"Connected to {port} at {baud} baud")
                
            except Exception as e:
                logging.error(f"Error connecting to serial port: {e}")
                self.status_label.config(text=f"Connection error: {str(e)}", fg="red")

    def read_serial_data(self):
        """Read and process data from the serial port"""
        buffer = ""
        
        while self.is_serial_running and self.serial_port:
            try:
                # Read from serial port
                data = self.serial_port.read(100).decode('utf-8', errors='replace')
                
                if data:
                    buffer += data
                    
                    # Process complete messages
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        self.process_serial_message(line.strip())
                        
                # Small delay to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                logging.error(f"Error reading from serial port: {e}")
                self.is_serial_running = False
                self.root.after(0, lambda: self.status_label.config(
                    text=f"● Serial error: {str(e)}", fg=THEME["error"]))
                break

    def process_serial_message(self, message):
        """Process a complete message from the serial port"""
        try:
            if not message:
                return

            data = self.parse_serial_message(message)
            if not data:
                logging.warning(f"Ignoring unsupported serial payload: {message}")
                return
            
            # Check for the environmental data values we're interested in
            now = datetime.now()
            did_update = False
            
            with self.env_data_lock:
                if 'temp' in data or 'temperature' in data:
                    temp = data.get('temp', data.get('temperature', 0))
                    self.env_data['temperature'].append(temp)
                    self.root.after(0, lambda: self.gauges['temperature'].set_value(temp))
                    did_update = True
                
                if 'humidity' in data:
                    humidity = data.get('humidity', 0)
                    self.env_data['humidity'].append(humidity)
                    self.root.after(0, lambda: self.gauges['humidity'].set_value(humidity))
                    did_update = True
                
                if 'co2' in data:
                    co2 = data.get('co2', 0)
                    self.env_data['co2'].append(co2)
                    self.root.after(0, lambda: self.gauges['co2'].set_value(co2))
                    did_update = True
                
                if 'light' in data:
                    light = data.get('light', 0)
                    self.env_data['light'].append(light)
                    self.root.after(0, lambda: self.gauges['light'].set_value(light))
                    did_update = True
                
                if 'moisture' in data or 'soil_moisture' in data:
                    moisture = data.get('moisture', data.get('soil_moisture', 0))
                    self.env_data['soil_moisture'].append(moisture)
                    self.root.after(0, lambda: self.gauges['soil_moisture'].set_value(moisture))
                    did_update = True

                if did_update:
                    self.env_data['time'].append(now)

                    if len(self.env_data['time']) > self.max_history_points:
                        for key in self.env_data:
                            self.env_data[key] = self.env_data[key][-self.max_history_points:]

            if not did_update:
                return
            
            # Update the timestamp and graphs in the main thread
            self.root.after(0, lambda: self.last_updated_label.config(
                text=f"Last Updated: {now.strftime('%Y-%m-%d %H:%M:%S')}"))
            self.root.after(0, self.schedule_env_graph_update)
            
            logging.debug(f"Processed environmental data: {data}")
            
        except Exception as e:
            logging.error(f"Error processing serial message '{message}': {e}")

    def parse_serial_message(self, message):
        """Parse either JSON payloads or comma-separated key=value readings."""
        try:
            if message.startswith('{'):
                payload = json.loads(message)
                return {
                    key: float(value) if isinstance(value, (int, float, str)) and str(value).replace('.', '', 1).replace('-', '', 1).isdigit() else value
                    for key, value in payload.items()
                }
        except json.JSONDecodeError as exc:
            logging.warning(f"Invalid JSON payload received: {exc}")
            return {}

        data = {}
        for part in message.split(','):
            if '=' not in part:
                continue

            key, value = part.split('=', 1)
            key = key.strip()
            raw_value = value.strip()

            try:
                data[key] = float(raw_value)
            except ValueError:
                data[key] = raw_value

        return data

    def schedule_env_graph_update(self):
        """Coalesce graph redraw requests so serial bursts do not flood the UI thread."""
        if self.env_graph_update_pending:
            return

        self.env_graph_update_pending = True
        self.root.after(250, self._flush_env_graph_update)

    def _flush_env_graph_update(self):
        self.env_graph_update_pending = False
        self.update_env_graphs()

    def update_env_graphs(self):
        T = THEME
        _COLORS = [T["primary"], T["accent"], T["warning"], "#64b5f6", "#a5d6a7"]
        _KEYS   = ['temperature', 'humidity', 'co2', 'light', 'soil_moisture']
        _TITLES = ['Temperature', 'Humidity', 'CO₂', 'Light', 'Soil Moisture']
        try:
            self.env_fig.clear()
            self.env_fig.patch.set_facecolor(T["graph_bg"])

            with self.env_data_lock:
                axes = self.env_fig.subplots(2, 3)
                axes = axes.flatten()

                for i, (key, title) in enumerate(zip(_KEYS, _TITLES)):
                    ax = axes[i]
                    ax.set_facecolor(T["graph_ax"])
                    for spine in ax.spines.values():
                        spine.set_edgecolor(T["border"])
                    ax.tick_params(colors=T["text_muted"], labelsize=7)
                    ax.set_title(title, color=T["primary_lt"], fontsize=9, fontweight="bold", pad=4)
                    ax.set_ylabel("Value", color=T["text_muted"], fontsize=8)
                    ax.grid(True, color=T["border"], alpha=0.4, linewidth=0.6)

                    data = self.env_data.get(key, [])
                    if data:
                        x = list(range(len(data)))
                        times = [t.strftime('%H:%M:%S') for t in self.env_data['time']]
                        color = _COLORS[i]
                        ax.plot(x, data, '-', linewidth=2, color=color)
                        ax.plot(x[-1:], data[-1:], 'o', markersize=5, color=color)
                        ax.fill_between(x, data, alpha=0.12, color=color)
                        skip = max(1, len(times) // 6)
                        ax.set_xticks(x[::skip])
                        ax.set_xticklabels(times[::skip], rotation=35, ha='right', fontsize=6,
                                           color=T["text_muted"])

                if len(axes) > 5:
                    axes[5].set_visible(False)

            self.env_fig.tight_layout(pad=1.2)
            self.env_canvas.draw()
        except Exception as e:
            logging.error(f"Error updating environmental graphs: {e}")

    def clear_env_data(self):
        """Clear all environmental data"""
        with self.env_data_lock:
            for key in self.env_data:
                self.env_data[key] = []
        
        # Reset gauges
        for gauge in self.gauges.values():
            gauge.set_value(0)
            
        # Reset last updated label
        self.last_updated_label.config(text="Last Updated: Never")
        
        # Update graphs
        self.update_env_graphs()
        logging.info("Environmental data cleared")

    def set_history_window(self):
        """Apply a new maximum history size for environmental data."""
        try:
            self.max_history_points = max(10, int(self.history_window_var.get()))
        except ValueError:
            self.history_window_var.set(str(self.max_history_points))
            logging.error("History window must be a valid integer")
            return

        with self.env_data_lock:
            for key in self.env_data:
                self.env_data[key] = self.env_data[key][-self.max_history_points:]

        self.update_env_graphs()
        logging.info(f"Environmental history window set to {self.max_history_points} points")

    def on_close(self):
        """Handle application close"""
        self.is_closing = True
        for i in range(len(self.grids)):
            self.stop_stream(i)
            
        # Stop serial connection
        self.is_serial_running = False
        if self.serial_port:
            self.serial_port.close()
            
        # Destroy root window
        self.root.destroy()

    # The rest of your existing methods remain unchanged
    # (get_available_cameras, select_camera, toggle_stream, capture_single_frame, etc.)

    def _start_camera_probe(self):
        """Launch background camera discovery so the UI is never blocked."""
        for grid in self.grids:
            lbl = grid.get("camera_status_label")
            if lbl:
                lbl.config(text="◌ Searching for cameras...", fg=THEME["primary_lt"])
                break
        t = threading.Thread(target=self._probe_cameras_thread, daemon=True)
        t.start()

    def _probe_cameras_thread(self):
        cameras = self.get_available_cameras()
        if not self.is_closing:
            self.root.after(0, lambda: self._on_cameras_found(cameras))

    def _on_cameras_found(self, cameras):
        self.cameras = cameras
        camera_labels = self.get_camera_labels()
        for grid in self.grids:
            grid["camera_dropdown"]["values"] = camera_labels
        if cameras:
            logging.info(f"Camera probe complete: {len(cameras)} camera(s) found")
        else:
            logging.warning("Camera probe complete: no cameras detected")
        self.update_camera_runtime_status(0)

    def get_available_cameras(self):
        cameras = []
        prev_log_level = cv2.getLogLevel()
        cv2.setLogLevel(0)  # Suppress C++ backend errors during probing
        consecutive_failures = 0
        try:
            for i in range(self.camera_probe_limit):
                found = False
                for backend in _CV2_CAM_BACKENDS:
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        cameras.append({"label": f"Camera {i}", "index": i, "backend": backend, "type": "local"})
                        cap.release()
                        found = True
                        break
                    cap.release()
                if found:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        break  # No camera at last 3 indices; stop probing
        finally:
            cv2.setLogLevel(prev_log_level)
        return cameras

    def get_selected_camera_index(self, idx):
        """Return the camera index (int) or URL (str) for the selected camera in slot *idx*."""
        selected_label = self.grids[idx]["camera_dropdown"].get()
        for camera in self.cameras:
            if camera["label"] == selected_label:
                # URL-based streams (RTMP / YouTube / custom) store index as the URL string
                return camera["index"]
        return None

    def update_camera_runtime_status(self, idx):
        if idx >= len(self.grids):
            return

        label = self.grids[idx].get("camera_status_label")
        if label is None:
            return

        if self.analysis_in_progress[idx]:
            label.config(text="⚙ Analyzing frame", fg=THEME["warning"])
        elif self.stream_active[idx] and self.grids[idx]["auto_analyze_var"].get():
            label.config(text="● Streaming + Auto Analyze", fg=THEME["accent"])
        elif self.stream_active[idx]:
            label.config(text="● Streaming", fg=THEME["success"])
        elif self.get_selected_camera_index(idx) is not None:
            label.config(text="◉ Camera ready", fg=THEME["primary_lt"])
        else:
            label.config(text="Idle", fg=THEME["text_dim"])

    def select_camera(self, idx):
        camera_idx = self.get_selected_camera_index(idx)
        if camera_idx is not None:
            self.stop_stream(idx)
            self.analysis_completed[idx] = False
            self.last_analysis_time[idx] = 0.0
            self.current_frames[idx] = None
            self.update_camera_runtime_status(idx)

            logging.info(f"Camera {camera_idx} selected for Grid {idx + 1}. Click 'Start Stream' to begin streaming.")

    def ensure_camera_ready(self, idx):
        """Open the selected camera for the grid if it is not already available.
        Supports integer device indices (USB/MSMF) and URL strings (RTMP/YouTube/custom).
        """
        camera_idx = self.get_selected_camera_index(idx)
        if camera_idx is None:
            logging.error(f"No camera selected for Grid {idx + 1}")
            return False

        if self.camera_streams[idx] is not None and self.camera_streams[idx].isOpened():
            return True

        # URL-based stream (RTMP, YouTube live, or custom URL)
        if isinstance(camera_idx, str):
            url = camera_idx
            # OpenCV reads RTMP/HLS via the FFmpeg backend (CAP_FFMPEG) — no extra deps needed
            camera_stream = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not camera_stream.isOpened():
                logging.error(f"Failed to open stream URL for Grid {idx + 1}: {url}")
                camera_stream.release()
                return False
            self.camera_streams[idx] = camera_stream
            return True

        # Integer device index (local camera)
        backend = next((c["backend"] for c in self.cameras if c["index"] == camera_idx), _CV2_CAM_BACKENDS[0])
        camera_stream = cv2.VideoCapture(camera_idx, backend)
        if not camera_stream.isOpened():
            logging.error(f"Failed to open Camera {camera_idx} for Grid {idx + 1}")
            camera_stream.release()
            return False

        self.camera_streams[idx] = camera_stream
        return True

    # ── DJI / RTMP drone stream helpers ───────────────────────────────────────

    @staticmethod
    def _normalise_stream_url(raw_url):
        """
        Accept various DJI-compatible free streaming URL forms and return the
        OpenCV-readable URL string:

        Supported formats (all free / no paid service required):
          • RTMP server  : rtmp://192.168.1.1/live/drone1
          • RTSP         : rtsp://192.168.1.1:554/stream
          • HLS          : http://…/playlist.m3u8
          • YouTube Live : https://www.youtube.com/watch?v=<ID>
                           → automatically converted to an HLS manifest URL
                             via yt-dlp if available, or ffplay-style URL otherwise

        DJI GO / DJI Fly RTMP workflow (free):
          1. Open DJI GO 4 / DJI Fly → Me → Live Broadcast → Custom RTMP
          2. Enter your local RTMP server address (e.g. rtmp://<PC_IP>/live/drone)
          3. Paste the same address here — OpenCV reads it via the bundled FFmpeg.
        """
        url = raw_url.strip()
        if not url:
            return None
        # YouTube → try yt-dlp for best stream URL, fall back to direct
        if "youtube.com" in url or "youtu.be" in url:
            try:
                import subprocess
                result = subprocess.run(
                    ["yt-dlp", "-g", "-f", "best[ext=mp4]/best", url],
                    capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    direct = result.stdout.strip().splitlines()[0]
                    if direct:
                        return direct
            except Exception:
                pass  # yt-dlp not installed — return raw URL and let OpenCV try
        return url

    def _drone_add(self, label, raw_url, stream_type="rtmp"):
        """Register a new drone stream and refresh all camera dropdowns."""
        url = self._normalise_stream_url(raw_url)
        if not url:
            return False, "Empty URL"
        # Avoid duplicates by label
        for d in self.drone_streams:
            if d["label"] == label:
                return False, f"A drone named '{label}' already exists."
        entry = {"label": label, "url": url, "raw_url": raw_url, "type": stream_type}
        self.drone_streams.append(entry)
        self._rebuild_camera_list_with_drones()
        return True, "OK"

    def _drone_remove(self, label):
        """Remove a drone stream by label and refresh dropdowns."""
        self.drone_streams = [d for d in self.drone_streams if d["label"] != label]
        self._rebuild_camera_list_with_drones()

    def _rebuild_camera_list_with_drones(self):
        """
        Merge local cameras + drone_streams into self.cameras and refresh
        every camera-slot dropdown.  Drone entries use the resolved URL as
        their 'index' so ensure_camera_ready can open them with cv2.VideoCapture(url).
        """
        local = [c for c in self.cameras if c.get("type", "local") == "local"]
        drone_entries = [
            {"label": d["label"], "index": d["url"],
             "backend": cv2.CAP_FFMPEG, "type": d["type"]}
            for d in self.drone_streams
        ]
        self.cameras = local + drone_entries
        camera_labels = self.get_camera_labels()
        for idx, grid in enumerate(self.grids):
            current = grid["camera_dropdown"].get()
            grid["camera_dropdown"]["values"] = camera_labels
            if current in camera_labels:
                grid["camera_dropdown"].set(current)

    def _drone_quick_connect(self, idx):
        """
        Called by the per-card 'Connect' button.  Reads the URL entry in
        the card, registers the drone (if new), selects it in the dropdown,
        and arms the stream so the user can press ▶.
        """
        T   = THEME
        url = self.grids[idx]["drone_url_var"].get().strip()
        lbl = self.grids[idx]["drone_status_lbl"]
        if not url:
            lbl.config(text="Enter URL first", fg=T["warning"])
            return
        name = f"Drone-P{idx + 1}"
        # Detect type from URL
        if "rtmp://" in url:
            stype = "rtmp"
        elif "rtsp://" in url:
            stype = "rtsp"
        elif "youtube.com" in url or "youtu.be" in url:
            stype = "youtube"
        else:
            stype = "custom"
        ok, msg = self._drone_add(name, url, stype)
        if not ok and "already exists" not in msg:
            lbl.config(text=f"Error: {msg}", fg=T["error"])
            return
        # Select the drone in this slot's dropdown
        self.grids[idx]["camera_dropdown"].set(name)
        self.select_camera(idx)
        lbl.config(text=f"Ready ({stype.upper()})", fg=T["success"])

    def _open_drone_manager(self):
        """Open a modal Drone Stream Manager window."""
        T   = THEME
        win = tk.Toplevel(self.root)
        win.title("DJI / RTMP Drone Stream Manager")
        win.configure(bg=T["panel"])
        win.geometry("700x480")
        win.resizable(True, True)
        win.grab_set()

        _btn  = dict(bg=T["primary_dk"], fg=T["text"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary"], activeforeground=T["text"],
                     cursor="hand2", bd=0)
        _sbtn = dict(bg=T["card_header"], fg=T["primary_lt"], relief="flat",
                     font=("Segoe UI", 9), padx=10, pady=4,
                     activebackground=T["primary_dk"], activeforeground=T["text"],
                     cursor="hand2", bd=0)
        _ent  = dict(bg=T["entry"], fg=T["text"], relief="flat",
                     highlightthickness=1, highlightcolor=T["border"],
                     highlightbackground=T["border"],
                     insertbackground=T["primary"], bd=0, font=("Segoe UI", 9))

        # ── Instructions ──────────────────────────────────────────────────────
        info_text = (
            "DJI Free Streaming Methods — no subscription needed:\n\n"
            "  RTMP (recommended)  rtmp://<PC_IP>:1935/live/<key>\n"
            "    → In DJI GO 4 / DJI Fly: Me ▶ Live Broadcast ▶ Custom RTMP\n"
            "    → Run a free RTMP server (e.g. nginx-rtmp, MediaMTX) on this PC\n\n"
            "  RTSP   rtsp://<drone_IP>:554/stream  (DJI Mini 4 Pro, Avata 2 etc.)\n\n"
            "  YouTube Live  https://www.youtube.com/watch?v=<LIVE_ID>\n"
            "    → Requires yt-dlp installed:  pip install yt-dlp\n"
        )
        info = tk.Text(win, bg=T["card"], fg=T["text_muted"], font=("Segoe UI", 8),
                       height=8, relief="flat", wrap="word", state="normal")
        info.insert("1.0", info_text)
        info.config(state="disabled")
        info.pack(fill="x", padx=10, pady=(10, 4))

        # ── Add new stream form ───────────────────────────────────────────────
        add_frame = tk.LabelFrame(win, text=" Add Stream ", bg=T["card"],
                                  fg=T["primary"], font=("Segoe UI", 9, "bold"),
                                  relief="groove", bd=1)
        add_frame.pack(fill="x", padx=10, pady=4)

        tk.Label(add_frame, text="Name:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        name_var = tk.StringVar(value=f"Drone-{len(self.drone_streams)+1}")
        tk.Entry(add_frame, textvariable=name_var, width=18, **_ent).grid(
            row=0, column=1, sticky="ew", padx=(0, 12), pady=4)

        tk.Label(add_frame, text="URL:", bg=T["card"], fg=T["text_muted"],
                 font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w", padx=(0, 4))
        url_var = tk.StringVar(value="rtmp://")
        url_entry = tk.Entry(add_frame, textvariable=url_var, width=38, **_ent)
        url_entry.grid(row=0, column=3, sticky="ew", padx=(0, 8), pady=4)
        add_frame.columnconfigure(3, weight=1)

        status_lbl = tk.Label(add_frame, text="", bg=T["card"], fg=T["text_muted"],
                               font=("Segoe UI", 8, "italic"))
        status_lbl.grid(row=1, column=0, columnspan=3, sticky="w", padx=6)

        def _test_url():
            raw = url_var.get().strip()
            resolved = self._normalise_stream_url(raw)
            if not resolved:
                status_lbl.config(text="Empty URL", fg=T["warning"])
                return
            status_lbl.config(text="Testing connection…", fg=T["text_muted"])
            win.update_idletasks()

            def _probe():
                cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
                ok  = cap.isOpened()
                cap.release()
                msg  = f"OK — stream opened ({resolved[:50]})" if ok else f"Failed to open stream"
                color = T["success"] if ok else T["error"]
                win.after(0, lambda: status_lbl.config(text=msg, fg=color))

            threading.Thread(target=_probe, daemon=True).start()

        def _add():
            name = name_var.get().strip()
            raw  = url_var.get().strip()
            if not name:
                status_lbl.config(text="Name required", fg=T["warning"])
                return
            stype = ("youtube" if ("youtube.com" in raw or "youtu.be" in raw)
                     else "rtsp" if raw.startswith("rtsp://")
                          else "rtmp" if raw.startswith("rtmp://")
                               else "custom")
            ok, msg = self._drone_add(name, raw, stype)
            if ok:
                status_lbl.config(text=f"Added '{name}'", fg=T["success"])
                _refresh_list()
                name_var.set(f"Drone-{len(self.drone_streams)+1}")
                url_var.set("rtmp://")
            else:
                status_lbl.config(text=msg, fg=T["warning"])

        btn_row = tk.Frame(add_frame, bg=T["card"])
        btn_row.grid(row=1, column=3, sticky="e", padx=4, pady=4)
        tk.Button(btn_row, text="Test URL", command=_test_url, **_sbtn).pack(side="left", padx=(0, 4))
        tk.Button(btn_row, text="+ Add", command=_add, **_btn).pack(side="left")

        # ── Active drone list ─────────────────────────────────────────────────
        list_frame = tk.LabelFrame(win, text=" Active Drone Streams ",
                                   bg=T["card"], fg=T["primary"],
                                   font=("Segoe UI", 9, "bold"),
                                   relief="groove", bd=1)
        list_frame.pack(fill="both", expand=True, padx=10, pady=4)

        cols = ("Name", "Type", "URL")
        vsb  = ttk.Scrollbar(list_frame, orient="vertical")
        drone_tree = ttk.Treeview(list_frame, columns=cols, show="headings",
                                  selectmode="browse", height=6,
                                  yscrollcommand=vsb.set)
        vsb.configure(command=drone_tree.yview)
        for col, w in zip(cols, (140, 70, 420)):
            drone_tree.heading(col, text=col)
            drone_tree.column(col, width=w, minwidth=w, stretch=(col == "URL"))
        vsb.pack(side="right", fill="y")
        drone_tree.pack(fill="both", expand=True)

        def _refresh_list():
            for iid in drone_tree.get_children():
                drone_tree.delete(iid)
            for d in self.drone_streams:
                drone_tree.insert("", "end", values=(
                    d["label"], d["type"].upper(), d["raw_url"]))

        def _remove_selected():
            sel = drone_tree.selection()
            if not sel:
                return
            iid = sel[0]
            vals = drone_tree.item(iid, "values")
            if vals:
                self._drone_remove(vals[0])
                _refresh_list()

        btn_bar = tk.Frame(list_frame, bg=T["card"])
        btn_bar.pack(fill="x", pady=(4, 2))
        tk.Button(btn_bar, text="✕ Remove Selected",
                  command=_remove_selected, **_sbtn).pack(side="left", padx=6)

        _refresh_list()

        # ── Show drone row on all cards ───────────────────────────────────────
        def _toggle_drone_rows():
            show = len(self.drone_streams) > 0
            for g in self.grids:
                if show:
                    g["drone_row_frame"].grid()
                else:
                    g["drone_row_frame"].grid_remove()

        win.protocol("WM_DELETE_WINDOW", lambda: (_toggle_drone_rows(), win.destroy()))

    def toggle_stream(self, idx):
        if self.stream_active[idx]:
            self.stream_active[idx] = False
            self.is_running[idx] = False
            self.grids[idx]["stream_toggle"].config(text="▶ Stream")
            self.update_camera_runtime_status(idx)
            logging.info(f"Streaming stopped for Grid {idx + 1}")
        else:
            if not self.ensure_camera_ready(idx):
                return

            self.is_running[idx] = True
            self.stream_active[idx] = True
            self.grids[idx]["stream_toggle"].config(text="⏹ Stop")
            self.update_camera_runtime_status(idx)
            logging.info(f"Streaming started for Grid {idx + 1}")

            if self.update_threads[idx] is None or not self.update_threads[idx].is_alive():
                self.update_threads[idx] = threading.Thread(target=self.stream_frames, args=(idx,))
                self.update_threads[idx].daemon = True
                self.update_threads[idx].start()

            if self.display_threads[idx] is None or not self.display_threads[idx].is_alive():
                self.display_threads[idx] = threading.Thread(target=self.display_stream, args=(idx,))
                self.display_threads[idx].daemon = True
                self.display_threads[idx].start()

    def capture_single_frame(self, idx):
        if not self.ensure_camera_ready(idx):
            return
            
        # Capture a single frame without streaming
        ret, frame = self.camera_streams[idx].read()
        if not ret:
            logging.error(f"Failed to capture frame from Camera {idx}")
            return
            
        # Store the captured frame
        self.current_frames[idx] = frame
        
        # Display the frame (without analysis overlay)
        self.display_image(frame, frame.copy(), idx)
        logging.info(f"Frame captured for Grid {idx + 1}")

    def stream_frames(self, idx):
        """Thread function to capture frames continuously at specified FPS"""
        while self.is_running[idx] and self.stream_active[idx]:
            if self.camera_streams[idx] is None or not self.camera_streams[idx].isOpened():
                break
                
            # Control frame rate
            current_time = time.time()
            time_diff = current_time - self.last_frame_time[idx]
            target_diff = 1.0 / self.stream_fps[idx]
            
            if time_diff < target_diff:
                time.sleep(target_diff - time_diff)
                
            # Capture new frame
            ret, frame = self.camera_streams[idx].read()
            self.last_frame_time[idx] = time.time()
            
            if not ret:
                logging.error(f"Failed to capture frame from Camera {idx}")
                self.stream_active[idx] = False
                break
                
            # Store the frame for analysis
            self.current_frames[idx] = frame.copy()
            self.maybe_schedule_auto_analysis(idx)
            
            # Resize for display based on quality setting
            # Lower quality = smaller size = better performance
            scale_factor = self.stream_quality[idx] / 100.0
            if scale_factor < 1.0:
                width = int(frame.shape[1] * scale_factor)
                height = int(frame.shape[0] * scale_factor)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            
            # Put in queue, remove old frames if queue is full
            if self.frame_queues[idx].full():
                try:
                    self.frame_queues[idx].get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.frame_queues[idx].put_nowait(frame)
            except queue.Full:
                pass  # Skip frame if queue is still full

        self.stream_active[idx] = False
        self.is_running[idx] = False
        if not self.is_closing:
            self.root.after(0, lambda idx=idx: self.update_camera_runtime_status(idx))
    
    def display_stream(self, idx):
        """Thread function to display frames from the queue"""
        while self.is_running[idx] and self.stream_active[idx]:
            try:
                # Get frame with timeout to avoid blocking forever
                frame = self.frame_queues[idx].get(timeout=0.5)
                
                # Use PIL for more efficient image conversion and display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(image=img)
                
                # Update UI in main thread
                self.root.after(0, lambda img=photo, idx=idx: self.update_display(img, idx))
                
                # Short sleep to prevent UI updates from overwhelming the system
                time.sleep(0.01)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in display stream for Grid {idx + 1}: {e}")
                break
    
    def update_display(self, photo, idx):
        """Update the image display label (called from main thread)"""
        try:
            display = self.grids[idx]["image_display"]
            display.configure(image=photo)
            display.image = photo  # Keep reference to prevent garbage collection
        except Exception as e:
            logging.error(f"Error updating display for Grid {idx + 1}: {e}")

    def analyze_current_frame(self, idx):
        """Analyze the currently displayed frame"""
        if self.current_frames[idx] is None:
            logging.error(f"No frame available for analysis in Grid {idx + 1}")
            return
            
        frame = self.current_frames[idx].copy()
        analysis_type = self.grids[idx]['analysis_dropdown'].get()
        self.start_analysis_job(idx, frame, analysis_type)

    def maybe_schedule_auto_analysis(self, idx):
        if not self.grids[idx]["auto_analyze_var"].get():
            return

        if self.analysis_in_progress[idx] or self.current_frames[idx] is None:
            return

        interval_seconds = self.get_analysis_interval_seconds(idx)
        if time.time() - self.last_analysis_time[idx] < interval_seconds:
            return

        analysis_type = self.grids[idx]['analysis_dropdown'].get()
        self.start_analysis_job(idx, self.current_frames[idx].copy(), analysis_type)

    def get_analysis_interval_seconds(self, idx):
        try:
            return max(1, int(self.grids[idx]["analysis_interval_var"].get()))
        except (TypeError, ValueError):
            self.grids[idx]["analysis_interval_var"].set(str(DEFAULT_AUTO_ANALYSIS_INTERVAL_SECONDS))
            return DEFAULT_AUTO_ANALYSIS_INTERVAL_SECONDS

    def start_analysis_job(self, idx, frame, analysis_type):
        if self.analysis_in_progress[idx]:
            logging.info(f"Analysis already in progress for Grid {idx + 1}")
            return False

        try:
            self.analysis_in_progress[idx] = True
            self.last_analysis_time[idx] = time.time()
            self.update_camera_runtime_status(idx)
            self.grids[idx]["analyze_button"].config(text="⏳ Processing...", state="disabled")

            analysis_thread = threading.Thread(
                target=self.perform_analysis,
                args=(idx, frame, analysis_type),
            )
            analysis_thread.daemon = True
            self.analysis_threads[idx] = analysis_thread
            analysis_thread.start()
            return True
        except Exception as e:
            self.analysis_in_progress[idx] = False
            logging.error(f"Error starting analysis for Grid {idx + 1}: {e}")
            self.grids[idx]["analyze_button"].config(text="Analyze Current Frame", state="normal")
            self.update_camera_runtime_status(idx)
            return False

    def perform_analysis(self, idx, frame, analysis_type):
        """Run analysis in background thread"""
        try:
            processed_img, analysis_data, photo_data, overlay_img, comm_metrics, comm_meas = \
                self.process_image(frame, analysis_type, camera_idx=idx)
            
            if not self.is_closing:
                self.root.after(0, lambda: self.display_image(frame, overlay_img, idx))
                self.root.after(0, lambda: self.update_table(idx, analysis_data, photo_data))
                self.root.after(0, lambda: self.update_graph(idx, analysis_data, photo_data))

            # Feed Plant Communication tab
            if comm_metrics and not self.is_closing:
                self.push_plant_comm_update(idx, comm_metrics, comm_meas)

            # Store morphology data for 3D reconstruction
            if analysis_type == "Plant Morphology Analysis" and not self.is_closing:
                section  = (idx // 2) + 1
                role_key = "overhead" if (idx % 2 == 0) else "side"
                ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                entry    = self.morph_section_data.get(section, {})
                entry[role_key]    = analysis_data.copy()
                entry["timestamp"] = ts
                self.morph_section_data[section] = entry

            # Append to Plant History audit log
            if not self.is_closing:
                self._hist_append_record(idx, analysis_type)

            self.analysis_completed[idx] = True
        except Exception as e:
            logging.error(f"Error in analysis for Grid {idx + 1}: {e}")
        finally:
            self.analysis_in_progress[idx] = False
            self.analysis_threads[idx] = None
            if not self.is_closing:
                self.root.after(0, lambda idx=idx: self.grids[idx]["analyze_button"].config(
                    text="⚡ Analyze", state="normal"))
                self.root.after(0, lambda idx=idx: self.update_camera_runtime_status(idx))

    def set_stream_quality(self, idx, value):
        """Set the quality of the stream (resolution scaling)"""
        self.stream_quality[idx] = int(value)
        logging.info(f"Stream quality for Grid {idx + 1} set to {value}%")

    def set_stream_fps(self, idx, value):
        """Set the target FPS for the stream"""
        self.stream_fps[idx] = int(value)
        logging.info(f"Stream FPS for Grid {idx + 1} set to {value}")

    def stop_stream(self, idx):
        """Stop streaming and release resources"""
        self.stream_active[idx] = False
        self.is_running[idx] = False
        
        # Clear the frame queue
        while not self.frame_queues[idx].empty():
            try:
                self.frame_queues[idx].get_nowait()
            except queue.Empty:
                break
        
        # Join thread if it exists
        if self.update_threads[idx] is not None and self.update_threads[idx].is_alive():
            self.update_threads[idx].join(timeout=1.0)
        self.update_threads[idx] = None

        if self.display_threads[idx] is not None and self.display_threads[idx].is_alive():
            self.display_threads[idx].join(timeout=1.0)
        self.display_threads[idx] = None
        
        # Release camera
        if self.camera_streams[idx] is not None:
            self.camera_streams[idx].release()
            self.camera_streams[idx] = None
        
        # Update button text
        if idx < len(self.grids) and self.grids[idx].get("stream_toggle"):
            self.grids[idx]["stream_toggle"].config(text="▶ Stream")
            self.update_camera_runtime_status(idx)

    def refresh_all(self):
        """Refresh all analysis without restarting streams"""
        for i in range(self.visible_camera_slots):
            if self.current_frames[i] is not None:
                self.analyze_current_frame(i)

    def process_image(self, frame, analysis_type, camera_idx=None):
        overlay_img = frame.copy()
        analysis_data = []
        photo_data = []
        # Parallel lists for Plant Communication tab
        comm_metrics_list      = []
        comm_measurements_list = []

        # Section / role metadata derived from slot index
        if camera_idx is not None:
            cam_section = (camera_idx // 2) + 1
            cam_role    = "overhead" if (camera_idx % 2 == 0) else "side"
        else:
            cam_section = 1
            cam_role    = "overhead"

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)

            with self.plantcv_lock:
                lab_a = pcv.rgb2gray_lab(rgb_img=rgb_frame, channel="a")
                a_thresh = pcv.threshold.binary(lab_a, 134, object_type="light")
                a_fill = pcv.fill(a_thresh, size=200)
                labeled_mask, num_rois = pcv.create_labels(a_fill)

                total_area = 0.0

                for roi_idx in range(1, num_rois + 1):
                    mask = labeled_mask == roi_idx
                    mask_uint8 = (mask.astype(np.uint8)) * 255
                    roi_metrics = self.extract_roi_metrics(mask, rgb_frame, hsv_frame)
                    if roi_metrics is None:
                        continue

                    overlay_img[mask] = [0, 255, 0]

                    if analysis_type == 'Get ROI Info':
                        measurements = self.collect_plantcv_measurements(frame, rgb_frame, mask_uint8, roi_idx, analysis_type)
                        analysis_data.extend(self.build_roi_info_rows(roi_idx, roi_metrics, measurements))

                    elif analysis_type == 'Photosynthetic Analysis':
                        measurements = self.collect_plantcv_measurements(frame, rgb_frame, mask_uint8, roi_idx, analysis_type)
                        analysis_data.extend(self.build_photosynthetic_rows(roi_idx, measurements))

                    elif analysis_type == 'Health Status Analysis':
                        measurements = self.collect_plantcv_measurements(frame, rgb_frame, mask_uint8, roi_idx, analysis_type)
                        analysis_data.extend(self.build_health_rows(roi_idx, roi_metrics, measurements))

                    elif analysis_type == 'Growth Rate Analysis':
                        measurements = self.collect_plantcv_measurements(frame, rgb_frame, mask_uint8, roi_idx, analysis_type)
                        analysis_data.extend(self.build_growth_rows(roi_idx, roi_metrics, measurements))
                        size_area = self.get_observation_value(measurements.get("size", {}), "area", roi_metrics["area_px"])
                        total_area += float(size_area)

                    elif analysis_type == 'Nutrient Deficiency Detection':
                        measurements = self.collect_plantcv_measurements(frame, rgb_frame, mask_uint8, roi_idx, analysis_type)
                        analysis_data.extend(self.build_nutrient_rows(roi_idx, measurements))

                    elif analysis_type == 'Machine Learning Detection':
                        measurements = {}
                        analysis_data.extend(self.build_ml_rows(roi_idx, roi_metrics))

                    elif analysis_type == 'Plant Morphology Analysis':
                        measurements = {}
                        try:
                            self.process_plant_morphology(
                                roi_idx, mask, analysis_data,
                                section=cam_section, role=cam_role)
                        except Exception as e:
                            logging.error(f"Error in Plant Morphology Analysis for ROI {roi_idx}: {e}")
                            analysis_data.append({"ROI": f"ROI {roi_idx}", "Parameter": "Error", "Value": str(e)})
                    else:
                        measurements = {}

                    # Always accumulate comm data regardless of pipeline
                    comm_metrics_list.append(roi_metrics)
                    comm_measurements_list.append(measurements)

                if analysis_type == 'Growth Rate Analysis' and camera_idx is not None and total_area > 0:
                    analysis_data.extend(self.build_growth_summary_rows(camera_idx, total_area))
        except Exception as e:
            logging.error(f"Error in process_image: {e}")
            # Add a default entry if analysis failed
            analysis_data.append({"ROI": "Error", "Parameter": "Processing Error", "Value": str(e)})
        
        # Convert to DataFrame (even if empty)
        if not analysis_data:
            analysis_data.append({"ROI": "None", "Parameter": "No Data", "Value": "N/A"})
        
        return frame, pd.DataFrame(analysis_data), pd.DataFrame(photo_data), overlay_img, comm_metrics_list, comm_measurements_list

    def extract_roi_metrics(self, mask, rgb_frame, hsv_frame):
        mask_uint8 = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area <= 0:
            return None

        perimeter = float(cv2.arcLength(contour, True))
        x, y, width, height = cv2.boundingRect(contour)
        bounding_area = max(width * height, 1)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) if hull is not None else area
        solidity = area / hull_area if hull_area else 0.0
        extent = area / bounding_area
        aspect_ratio = width / max(height, 1)

        roi_pixels = rgb_frame[mask]
        roi_hsv_pixels = hsv_frame[mask]
        if roi_pixels.size == 0 or roi_hsv_pixels.size == 0:
            return None

        red_mean = float(np.mean(roi_pixels[:, 0]))
        green_mean = float(np.mean(roi_pixels[:, 1]))
        blue_mean = float(np.mean(roi_pixels[:, 2]))
        channel_total = max(red_mean + green_mean + blue_mean, 1.0)
        greenness = green_mean / channel_total
        exg_raw = float(np.mean((2 * roi_pixels[:, 1]) - roi_pixels[:, 0] - roi_pixels[:, 2]))
        exg_index = max(0.0, min(1.0, (exg_raw + 255.0) / 510.0))

        hue = roi_hsv_pixels[:, 0]
        sat = roi_hsv_pixels[:, 1]
        val = roi_hsv_pixels[:, 2]
        yellow_ratio = float(np.mean((hue >= 18) & (hue <= 40) & (sat >= 40) & (val >= 60)))
        brown_ratio = float(np.mean((hue >= 5) & (hue <= 25) & (sat >= 40) & (val >= 20) & (val <= 180)))
        purple_ratio = float(np.mean((hue >= 125) & (hue <= 165) & (sat >= 40) & (val >= 40)))

        return {
            "area_px": area,
            "perimeter_px": perimeter,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "extent": extent,
            "greenness": greenness,
            "exg_index": exg_index,
            "yellow_ratio": yellow_ratio,
            "brown_ratio": brown_ratio,
            "purple_ratio": purple_ratio,
            "red_mean": red_mean,
            "green_mean": green_mean,
            "blue_mean": blue_mean,
            "bbox_width": width,
            "bbox_height": height,
        }

    def run_iterative_plantcv_analysis(self, sample_label, analysis_callable):
        try:
            pcv.outputs.clear()
            analysis_callable(sample_label)
            observations = pcv.outputs.observations.get(f"{sample_label}_1", {})
            return copy.deepcopy(observations)
        except Exception as e:
            logging.error(f"PlantCV iterative analysis failed for {sample_label}: {e}")
            return {}

    def run_direct_plantcv_analysis(self, sample_label, analysis_callable):
        try:
            pcv.outputs.clear()
            analysis_callable(sample_label)
            observations = pcv.outputs.observations.get(sample_label, {})
            return copy.deepcopy(observations)
        except Exception as e:
            logging.error(f"PlantCV direct analysis failed for {sample_label}: {e}")
            return {}

    def collect_plantcv_measurements(self, bgr_frame, rgb_frame, mask_uint8, roi_idx, analysis_type):
        label_base = f"roi_{roi_idx}"
        measurements = {}

        if analysis_type in {'Get ROI Info', 'Health Status Analysis', 'Growth Rate Analysis'}:
            measurements["size"] = self.run_iterative_plantcv_analysis(
                f"{label_base}_size",
                lambda sample_label: pcv.analyze.size(img=bgr_frame, labeled_mask=mask_uint8, n_labels=1, label=sample_label),
            )

        if analysis_type in {'Photosynthetic Analysis', 'Health Status Analysis', 'Nutrient Deficiency Detection'}:
            measurements["color"] = self.run_iterative_plantcv_analysis(
                f"{label_base}_color",
                lambda sample_label: pcv.analyze.color(rgb_img=bgr_frame, labeled_mask=mask_uint8, n_labels=1, colorspaces="all", label=sample_label),
            )

        if analysis_type in {'Photosynthetic Analysis', 'Health Status Analysis', 'Nutrient Deficiency Detection'}:
            sat = pcv.rgb2gray_hsv(rgb_img=rgb_frame, channel="s")
            measurements["saturation"] = self.run_iterative_plantcv_analysis(
                f"{label_base}_sat",
                lambda sample_label: pcv.analyze.grayscale(gray_img=sat, labeled_mask=mask_uint8, n_labels=1, bins=64, label=sample_label),
            )

        if analysis_type in {'Photosynthetic Analysis', 'Nutrient Deficiency Detection'}:
            lab_a = pcv.rgb2gray_lab(rgb_img=rgb_frame, channel="a")
            measurements["lab_a"] = self.run_iterative_plantcv_analysis(
                f"{label_base}_lab_a",
                lambda sample_label: pcv.analyze.grayscale(gray_img=lab_a, labeled_mask=mask_uint8, n_labels=1, bins=64, label=sample_label),
            )

        if analysis_type == 'Photosynthetic Analysis':
            val = pcv.rgb2gray_hsv(rgb_img=rgb_frame, channel="v")
            measurements["value"] = self.run_iterative_plantcv_analysis(
                f"{label_base}_val",
                lambda sample_label: pcv.analyze.grayscale(gray_img=val, labeled_mask=mask_uint8, n_labels=1, bins=64, label=sample_label),
            )

        if analysis_type == 'Health Status Analysis':
            lightness = pcv.rgb2gray_lab(rgb_img=rgb_frame, channel="l")
            measurements["lightness"] = self.run_iterative_plantcv_analysis(
                f"{label_base}_lightness",
                lambda sample_label: pcv.analyze.grayscale(gray_img=lightness, labeled_mask=mask_uint8, n_labels=1, bins=64, label=sample_label),
            )
            bin_size = self.get_distribution_bin_size(mask_uint8)
            measurements["distribution_down"] = self.run_iterative_plantcv_analysis(
                f"{label_base}_dist_down",
                lambda sample_label: pcv.analyze.distribution(labeled_mask=mask_uint8, n_labels=1, direction="down", bin_size=bin_size, hist_range="relative", label=sample_label),
            )
            measurements["distribution_across"] = self.run_iterative_plantcv_analysis(
                f"{label_base}_dist_across",
                lambda sample_label: pcv.analyze.distribution(labeled_mask=mask_uint8, n_labels=1, direction="across", bin_size=bin_size, hist_range="relative", label=sample_label),
            )

        if analysis_type == 'Growth Rate Analysis':
            bin_size = self.get_distribution_bin_size(mask_uint8)
            measurements["distribution_down"] = self.run_iterative_plantcv_analysis(
                f"{label_base}_growth_dist",
                lambda sample_label: pcv.analyze.distribution(labeled_mask=mask_uint8, n_labels=1, direction="down", bin_size=bin_size, hist_range="relative", label=sample_label),
            )

        if analysis_type == 'Nutrient Deficiency Detection':
            lab_b = pcv.rgb2gray_lab(rgb_img=rgb_frame, channel="b")
            measurements["lab_b"] = self.run_iterative_plantcv_analysis(
                f"{label_base}_lab_b",
                lambda sample_label: pcv.analyze.grayscale(gray_img=lab_b, labeled_mask=mask_uint8, n_labels=1, bins=64, label=sample_label),
            )
            value = pcv.rgb2gray_hsv(rgb_img=rgb_frame, channel="v")
            measurements["value"] = self.run_iterative_plantcv_analysis(
                f"{label_base}_nutrient_val",
                lambda sample_label: pcv.analyze.grayscale(gray_img=value, labeled_mask=mask_uint8, n_labels=1, bins=64, label=sample_label),
            )

        return measurements

    def get_distribution_bin_size(self, mask_uint8):
        rows, cols = np.where(mask_uint8 > 0)
        if rows.size == 0 or cols.size == 0:
            return 10

        roi_height = int(rows.max() - rows.min() + 1)
        roi_width = int(cols.max() - cols.min() + 1)
        shortest_dimension = max(1, min(roi_height, roi_width))
        return max(1, shortest_dimension // 10)

    def get_observation_value(self, observations, variable, default=None):
        return observations.get(variable, {}).get("value", default)

    def is_missing_value(self, value):
        return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))

    def format_measurement_value(self, value, decimals=3):
        if self.is_missing_value(value):
            return "N/A"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return round(float(value), decimals)
        if isinstance(value, tuple):
            return ", ".join(str(self.format_measurement_value(item, decimals)) for item in value)
        return value

    def make_analysis_row(self, roi_label, parameter, value, decimals=3):
        return {
            "ROI": roi_label,
            "Parameter": parameter,
            "Value": self.format_measurement_value(value, decimals),
        }

    def summarize_numeric_list(self, values, reducer="mean"):
        numeric_values = [float(value) for value in values if isinstance(value, (int, float, np.integer, np.floating))]
        if not numeric_values:
            return None

        if reducer == "max":
            return max(numeric_values)
        if reducer == "min":
            return min(numeric_values)
        return float(np.mean(numeric_values))

    def interpret_health_status(self, hue_median, saturation_mean, solidity, lightness_median):
        if any(self.is_missing_value(value) for value in [hue_median, saturation_mean, solidity, lightness_median]):
            return "Insufficient PlantCV measurements for interpretation"

        if hue_median < 75 and lightness_median > 140:
            return "Canopy is shifted toward yellow and bright tissue"
        if saturation_mean < 60 and solidity < 0.85:
            return "Low color saturation with reduced compactness"
        if 75 <= hue_median <= 170 and saturation_mean >= 60 and solidity >= 0.85:
            return "Color and structure are consistent with vigorous green tissue"
        return "Intermediate RGB phenotype; monitor color and canopy structure over time"

    def interpret_nutrient_screen(self, lab_a_median, lab_b_median, hue_median, saturation_mean, value_mean):
        if any(self.is_missing_value(value) for value in [lab_a_median, lab_b_median, hue_median, saturation_mean, value_mean]):
            return "Insufficient PlantCV measurements for screening"

        if lab_b_median > 150 and hue_median < 90:
            return "Yellow-blue channel is elevated, consistent with chlorosis-like yellowing"
        if hue_median > 250 and saturation_mean >= 70:
            return "Hue distribution is shifted toward purple tissue; inspect for stress pigmentation"
        if saturation_mean < 45 and value_mean > 150:
            return "Low saturation with high brightness suggests pale or bleached tissue"
        if lab_a_median < 120 and 90 <= hue_median <= 170:
            return "Green-magenta channel remains plant-like with no strong RGB deficiency signature"
        return "Mixed RGB symptom pattern; review tissue color directly before diagnosis"

    def build_roi_info_rows(self, roi_idx, metrics, measurements):
        size_obs = measurements.get("size", {})
        roi_label = f"ROI {roi_idx}"
        return [
            self.make_analysis_row(roi_label, "Detected", "Yes"),
            self.make_analysis_row(roi_label, "Area", self.get_observation_value(size_obs, "area", metrics["area_px"]), 2),
            self.make_analysis_row(roi_label, "Perimeter", self.get_observation_value(size_obs, "perimeter", metrics["perimeter_px"]), 2),
            self.make_analysis_row(roi_label, "Width", self.get_observation_value(size_obs, "width", metrics["bbox_width"]), 2),
            self.make_analysis_row(roi_label, "Height", self.get_observation_value(size_obs, "height", metrics["bbox_height"]), 2),
            self.make_analysis_row(roi_label, "Solidity", self.get_observation_value(size_obs, "solidity", metrics["solidity"])),
            self.make_analysis_row(roi_label, "Longest Path", self.get_observation_value(size_obs, "longest_path")),
            self.make_analysis_row(roi_label, "Ellipse Eccentricity", self.get_observation_value(size_obs, "ellipse_eccentricity")),
            self.make_analysis_row(roi_label, "Object In Frame", self.get_observation_value(size_obs, "object_in_frame")),
        ]

    def build_photosynthetic_rows(self, roi_idx, measurements):
        color_obs = measurements.get("color", {})
        lab_a_obs = measurements.get("lab_a", {})
        sat_obs = measurements.get("saturation", {})
        value_obs = measurements.get("value", {})
        roi_label = f"ROI {roi_idx}"
        return [
            self.make_analysis_row(roi_label, "Measurement Basis", "PlantCV RGB color phenotyping"),
            self.make_analysis_row(roi_label, "Fluorescence Status", "NPQ/YII require fluorescence imaging, not RGB webcam frames"),
            self.make_analysis_row(roi_label, "Hue Median (deg)", self.get_observation_value(color_obs, "hue_median")),
            self.make_analysis_row(roi_label, "Hue Circular Mean (deg)", self.get_observation_value(color_obs, "hue_circular_mean")),
            self.make_analysis_row(roi_label, "Hue Circular Std (deg)", self.get_observation_value(color_obs, "hue_circular_std")),
            self.make_analysis_row(roi_label, "LAB a Mean", self.get_observation_value(lab_a_obs, "gray_mean")),
            self.make_analysis_row(roi_label, "LAB a Median", self.get_observation_value(lab_a_obs, "gray_median")),
            self.make_analysis_row(roi_label, "Saturation Mean", self.get_observation_value(sat_obs, "gray_mean")),
            self.make_analysis_row(roi_label, "Value Mean", self.get_observation_value(value_obs, "gray_mean")),
        ]

    def build_health_rows(self, roi_idx, metrics, measurements):
        size_obs = measurements.get("size", {})
        color_obs = measurements.get("color", {})
        lightness_obs = measurements.get("lightness", {})
        sat_obs = measurements.get("saturation", {})
        dist_down_obs = measurements.get("distribution_down", {})
        dist_across_obs = measurements.get("distribution_across", {})
        hue_median = self.get_observation_value(color_obs, "hue_median")
        saturation_mean = self.get_observation_value(sat_obs, "gray_mean")
        lightness_median = self.get_observation_value(lightness_obs, "gray_median")
        solidity = self.get_observation_value(size_obs, "solidity", metrics["solidity"])
        roi_label = f"ROI {roi_idx}"
        return [
            self.make_analysis_row(roi_label, "Measurement Basis", "PlantCV canopy structure and color traits"),
            self.make_analysis_row(roi_label, "Area", self.get_observation_value(size_obs, "area", metrics["area_px"]), 2),
            self.make_analysis_row(roi_label, "Solidity", solidity),
            self.make_analysis_row(roi_label, "Ellipse Eccentricity", self.get_observation_value(size_obs, "ellipse_eccentricity")),
            self.make_analysis_row(roi_label, "Hue Median (deg)", hue_median),
            self.make_analysis_row(roi_label, "Lightness Median", lightness_median),
            self.make_analysis_row(roi_label, "Vertical Distribution Mean", self.get_observation_value(dist_down_obs, "y_distribution_mean")),
            self.make_analysis_row(roi_label, "Vertical Distribution Std", self.get_observation_value(dist_down_obs, "y_distribution_std")),
            self.make_analysis_row(roi_label, "Horizontal Distribution Mean", self.get_observation_value(dist_across_obs, "x_distribution_mean")),
            self.make_analysis_row(roi_label, "Observation", self.interpret_health_status(hue_median, saturation_mean, solidity, lightness_median)),
        ]

    def build_growth_rows(self, roi_idx, metrics, measurements):
        size_obs = measurements.get("size", {})
        dist_down_obs = measurements.get("distribution_down", {})
        roi_label = f"ROI {roi_idx}"
        return [
            self.make_analysis_row(roi_label, "Current Area", self.get_observation_value(size_obs, "area", metrics["area_px"]), 2),
            self.make_analysis_row(roi_label, "Width", self.get_observation_value(size_obs, "width", metrics["bbox_width"]), 2),
            self.make_analysis_row(roi_label, "Height", self.get_observation_value(size_obs, "height", metrics["bbox_height"]), 2),
            self.make_analysis_row(roi_label, "Longest Path", self.get_observation_value(size_obs, "longest_path")),
            self.make_analysis_row(roi_label, "Vertical Distribution Mean", self.get_observation_value(dist_down_obs, "y_distribution_mean")),
            self.make_analysis_row(roi_label, "Vertical Distribution Std", self.get_observation_value(dist_down_obs, "y_distribution_std")),
        ]

    def build_growth_summary_rows(self, camera_idx, total_area):
        history = self.growth_history[camera_idx]
        timestamp = time.time()
        history.append((timestamp, total_area))
        if len(history) > 25:
            self.growth_history[camera_idx] = history[-25:]
            history = self.growth_history[camera_idx]

        if len(history) < 2:
            return [
                {"ROI": "Camera Summary", "Parameter": "Total Plant Area (px)", "Value": round(total_area, 2)},
                {"ROI": "Camera Summary", "Parameter": "Growth Rate", "Value": "Collecting baseline"},
            ]

        previous_timestamp, previous_area = history[-2]
        elapsed_seconds = max(timestamp - previous_timestamp, 1e-6)
        area_delta = total_area - previous_area
        percent_change = (area_delta / previous_area * 100.0) if previous_area else 0.0
        growth_per_minute = area_delta / elapsed_seconds * 60.0
        trend = "Increasing" if area_delta > 0 else "Stable" if abs(area_delta) < 1e-3 else "Decreasing"

        return [
            {"ROI": "Camera Summary", "Parameter": "Total Plant Area (px)", "Value": round(total_area, 2)},
            {"ROI": "Camera Summary", "Parameter": "Area Change (%)", "Value": round(percent_change, 3)},
            {"ROI": "Camera Summary", "Parameter": "Growth Rate (px/min)", "Value": round(growth_per_minute, 3)},
            {"ROI": "Camera Summary", "Parameter": "Trend", "Value": trend},
        ]

    def build_nutrient_rows(self, roi_idx, measurements):
        color_obs = measurements.get("color", {})
        lab_a_obs = measurements.get("lab_a", {})
        lab_b_obs = measurements.get("lab_b", {})
        sat_obs = measurements.get("saturation", {})
        value_obs = measurements.get("value", {})
        lab_a_median = self.get_observation_value(lab_a_obs, "gray_median")
        lab_b_median = self.get_observation_value(lab_b_obs, "gray_median")
        hue_median = self.get_observation_value(color_obs, "hue_median")
        saturation_mean = self.get_observation_value(sat_obs, "gray_mean")
        value_mean = self.get_observation_value(value_obs, "gray_mean")
        roi_label = f"ROI {roi_idx}"
        return [
            self.make_analysis_row(roi_label, "Measurement Basis", "PlantCV LAB and HSV symptom screening"),
            self.make_analysis_row(roi_label, "Hue Median (deg)", hue_median),
            self.make_analysis_row(roi_label, "LAB a Median", lab_a_median),
            self.make_analysis_row(roi_label, "LAB b Median", lab_b_median),
            self.make_analysis_row(roi_label, "Saturation Mean", saturation_mean),
            self.make_analysis_row(roi_label, "Value Mean", value_mean),
            self.make_analysis_row(roi_label, "Screening Result", self.interpret_nutrient_screen(lab_a_median, lab_b_median, hue_median, saturation_mean, value_mean)),
        ]

    def build_ml_rows(self, roi_idx, metrics):
        if self.ml_model is None:
            return [
                {"ROI": f"ROI {roi_idx}", "Parameter": "ML Status", "Value": "No model loaded"},
                {"ROI": f"ROI {roi_idx}", "Parameter": "Action", "Value": "Use Train/Load Model to load or train a classifier"},
            ]

        feature_frame = pd.DataFrame([self.metrics_to_model_features(metrics)])
        try:
            prediction = self.ml_model.predict(feature_frame)[0]
        except Exception:
            prediction = self.ml_model.predict(feature_frame[self.ml_feature_columns].to_numpy())[0]

        confidence = None
        if hasattr(self.ml_model, "predict_proba"):
            try:
                probabilities = self.ml_model.predict_proba(feature_frame)
                confidence = float(np.max(probabilities[0]))
            except Exception:
                try:
                    probabilities = self.ml_model.predict_proba(feature_frame[self.ml_feature_columns].to_numpy())
                    confidence = float(np.max(probabilities[0]))
                except Exception:
                    confidence = None

        rows = [
            {"ROI": f"ROI {roi_idx}", "Parameter": "ML Prediction", "Value": prediction},
            {"ROI": f"ROI {roi_idx}", "Parameter": "Model Source", "Value": os.path.basename(self.ml_model_path) if self.ml_model_path else "In-memory model"},
        ]
        if confidence is not None:
            rows.append({"ROI": f"ROI {roi_idx}", "Parameter": "Prediction Confidence", "Value": round(confidence, 3)})
        return rows

    def metrics_to_model_features(self, metrics):
        return {column: metrics[column] for column in self.ml_feature_columns}

    def process_plant_morphology(self, roi_idx, mask, analysis_data, section=1, role="overhead"):
        role_tag = "Overhead" if role == "overhead" else "Side/Canopy"
        mask_binary = mask.astype(np.uint8)
        mask_uint8 = mask_binary * 255

        # Tag every result row with section and camera role for 3D reconstruction
        analysis_data.append(self.make_analysis_row(f"ROI {roi_idx}", "Section",     section))
        analysis_data.append(self.make_analysis_row(f"ROI {roi_idx}", "Camera Role", role_tag))

        skeleton = pcv.morphology.skeletonize(mask=mask_binary)
        pruned_skel, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=50, mask=mask_binary)

        if len(edge_objects) == 0:
            analysis_data.append(self.make_analysis_row(f"ROI {roi_idx}", "Status", "No plant segments detected"))
            return

        try:
            leaf_obj, stem_obj = pcv.morphology.segment_sort(skel_img=pruned_skel, objects=edge_objects, mask=mask_uint8)
            stem_count = len(stem_obj) if stem_obj is not None else 0
            leaf_count = len(leaf_obj)
            segment_count = len(edge_objects)
            label_base = f"roi_{roi_idx}_morph"
            branch_obs = self.run_direct_plantcv_analysis(
                f"{label_base}_branch",
                lambda sample_label: pcv.morphology.find_branch_pts(skel_img=pruned_skel, mask=mask_binary, label=sample_label),
            )
            tip_obs = self.run_direct_plantcv_analysis(
                f"{label_base}_tips",
                lambda sample_label: pcv.morphology.find_tips(skel_img=pruned_skel, mask=None, label=sample_label),
            )

            analysis_data.append(self.make_analysis_row(f"ROI {roi_idx}", "Stem Count", stem_count))
            analysis_data.append(self.make_analysis_row(f"ROI {roi_idx}", "Leaf Count", leaf_count))
            analysis_data.append(self.make_analysis_row(f"ROI {roi_idx}", "Segment Count", segment_count))
            analysis_data.append(self.make_analysis_row(f"ROI {roi_idx}", "Branch Points", len(self.get_observation_value(branch_obs, "branch_pts", []))))
            analysis_data.append(self.make_analysis_row(f"ROI {roi_idx}", "Tip Points", len(self.get_observation_value(tip_obs, "tips", []))))

            if stem_count > 0:
                stem_obs = self.run_direct_plantcv_analysis(
                    f"{label_base}_stem",
                    lambda sample_label: pcv.morphology.analyze_stem(rgb_img=cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR), stem_objects=stem_obj, label=sample_label),
                )
                analysis_data.append(self.make_analysis_row(f"ROI {roi_idx}", "Stem Height", self.get_observation_value(stem_obs, "stem_height")))
                analysis_data.append(self.make_analysis_row(f"ROI {roi_idx}", "Stem Angle", self.get_observation_value(stem_obs, "stem_angle")))
                analysis_data.append(self.make_analysis_row(f"ROI {roi_idx}", "Stem Length", self.get_observation_value(stem_obs, "stem_length")))

            width_obs = self.run_direct_plantcv_analysis(
                f"{label_base}_width",
                lambda sample_label: pcv.morphology.segment_width(segmented_img=seg_img, skel_img=pruned_skel, labeled_mask=np.where(mask_binary > 0, 1, 0).astype(np.uint8), n_labels=1, label=sample_label),
            )
            mean_width = self.get_observation_value(width_obs, "mean_segment_width", [])
            max_width = self.get_observation_value(width_obs, "segment_width_max", [])
            analysis_data.append(self.make_analysis_row(f"ROI {roi_idx}", "Mean Segment Width", self.summarize_numeric_list(mean_width)))
            analysis_data.append(self.make_analysis_row(f"ROI {roi_idx}", "Max Segment Width", self.summarize_numeric_list(max_width, reducer="max")))

            if leaf_count > 0:
                path_obs = self.run_direct_plantcv_analysis(
                    f"{label_base}_path",
                    lambda sample_label: pcv.morphology.segment_path_length(segmented_img=seg_img, objects=leaf_obj, label=sample_label),
                )
                eu_obs = self.run_direct_plantcv_analysis(
                    f"{label_base}_eu",
                    lambda sample_label: pcv.morphology.segment_euclidean_length(segmented_img=seg_img, objects=leaf_obj, label=sample_label),
                )
                curvature_obs = self.run_direct_plantcv_analysis(
                    f"{label_base}_curve",
                    lambda sample_label: pcv.morphology.segment_curvature(segmented_img=seg_img, objects=leaf_obj, label=sample_label),
                )
                angle_obs = self.run_direct_plantcv_analysis(
                    f"{label_base}_angle",
                    lambda sample_label: pcv.morphology.segment_angle(segmented_img=seg_img, objects=leaf_obj, label=sample_label),
                )

                insertion_values = []
                if stem_count > 0:
                    insertion_obs = self.run_direct_plantcv_analysis(
                        f"{label_base}_insert",
                        lambda sample_label: pcv.morphology.segment_insertion_angle(
                            skel_img=pruned_skel,
                            segmented_img=seg_img,
                            leaf_objects=leaf_obj,
                            stem_objects=stem_obj,
                            size=20,
                            label=sample_label,
                        ),
                    )
                    insertion_values = self.get_observation_value(insertion_obs, "segment_insertion_angle", [])

                path_values = self.get_observation_value(path_obs, "segment_path_length", [])
                eu_values = self.get_observation_value(eu_obs, "segment_eu_length", [])
                curvature_values = self.get_observation_value(curvature_obs, "segment_curvature", [])
                angle_values = self.get_observation_value(angle_obs, "segment_angle", [])

                for idx, _ in enumerate(leaf_obj, start=1):
                    leaf_label = f"ROI {roi_idx} Leaf {idx}"
                    path_value = path_values[idx - 1] if idx - 1 < len(path_values) else None
                    eu_value = eu_values[idx - 1] if idx - 1 < len(eu_values) else None
                    curvature_value = curvature_values[idx - 1] if idx - 1 < len(curvature_values) else None
                    angle_value = angle_values[idx - 1] if idx - 1 < len(angle_values) else None
                    insertion_value = insertion_values[idx - 1] if idx - 1 < len(insertion_values) else "N/A"

                    analysis_data.append(self.make_analysis_row(leaf_label, "Path Length", path_value))
                    analysis_data.append(self.make_analysis_row(leaf_label, "Euclidean Length", eu_value))
                    analysis_data.append(self.make_analysis_row(leaf_label, "Curvature", curvature_value))
                    analysis_data.append(self.make_analysis_row(leaf_label, "Angle", angle_value))
                    analysis_data.append(self.make_analysis_row(leaf_label, "Insertion Angle", insertion_value))
        except Exception as e:
            logging.error(f"Error during plant morphology processing for ROI {roi_idx}: {e}")
            analysis_data.append(self.make_analysis_row(f"ROI {roi_idx}", "Error", str(e)))

    def display_image(self, frame, overlay_img, grid_idx):
        blended = cv2.addWeighted(frame, 0.6, overlay_img, 0.4, 0)
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(blended)
        img = ImageTk.PhotoImage(img)
        display = self.grids[grid_idx]["image_display"]
        display.configure(image=img)
        display.image = img

    def update_table(self, grid_idx, analysis_data, photo_data):
        if analysis_data.empty:
            return

        self.latest_analysis_results[grid_idx] = analysis_data.copy()
            
        tree = self.grids[grid_idx]["tree"]
        for row in tree.get_children():
            tree.delete(row)

        columns = tuple(analysis_data.columns.tolist())
        tree["columns"] = columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center")

        for _, row in analysis_data.iterrows():
            tree.insert("", "end", values=tuple(row.get(col, "N/A") for col in columns))

    def update_graph(self, grid_idx, analysis_data, photo_data):
        T = THEME
        for widget in self.grids[grid_idx]["graph_frame"].winfo_children():
            widget.destroy()

        if analysis_data.empty:
            return

        analysis_type = self.grids[grid_idx]['analysis_dropdown'].get()
        fig, ax = plt.subplots(figsize=(5, 2.5))
        fig.patch.set_facecolor(T["graph_bg"])
        ax.set_facecolor(T["graph_ax"])
        for spine in ax.spines.values():
            spine.set_edgecolor(T["border"])
        ax.tick_params(colors=T["text_muted"], labelsize=7)

        bar_palette = [T["primary"], T["accent"], T["warning"], T["primary_lt"], "#64b5f6"]

        try:
            if 'Parameter' in analysis_data.columns and 'Value' in analysis_data.columns:
                rois = analysis_data["ROI"].unique()
                for roi_i, roi in enumerate(rois):
                    roi_data = analysis_data[analysis_data["ROI"] == roi]
                    parameters = roi_data["Parameter"].tolist()
                    values = []
                    for v in roi_data["Value"].tolist():
                        try:
                            if isinstance(v, (int, float)):
                                values.append(float(v))
                            elif isinstance(v, str) and v.replace('.', '', 1).replace('-', '', 1).isdigit():
                                values.append(float(v))
                            else:
                                values.append(0)
                        except (ValueError, TypeError):
                            values.append(0)

                    if len(parameters) == len(values) and len(parameters) > 0:
                        color = bar_palette[roi_i % len(bar_palette)]
                        ax.bar(parameters, values, label=roi,
                               color=color, alpha=0.85, edgecolor=T["border"])

            ax.set_title(analysis_type, color=T["primary_lt"], fontsize=9, fontweight="bold", pad=4)
            ax.set_ylabel("Value", color=T["text_muted"], fontsize=8)
            ax.grid(True, color=T["border"], alpha=0.4, axis="y", linewidth=0.6)
            if len(ax.get_legend_handles_labels()[0]) > 0:
                ax.legend(facecolor=T["card"], edgecolor=T["border"],
                          labelcolor=T["text"], fontsize=7)

            plt.tight_layout()
            fig.subplots_adjust(bottom=0.32)
            plt.xticks(rotation=40, ha='right', color=T["text_muted"], fontsize=7)

            canvas = FigureCanvasTkAgg(fig, master=self.grids[grid_idx]["graph_frame"])
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            logging.error(f"Error updating graph: {e}")
        finally:
            plt.close(fig)

    def create_dataset(self):
        dataset_directory = filedialog.askdirectory(title="Select Directory to Save New Dataset")
        if dataset_directory:
            rows = []
            for idx in range(self.visible_camera_slots):
                analysis_data = self.latest_analysis_results[idx]
                if analysis_data.empty:
                    continue

                frame_copy = self.current_frames[idx].copy() if self.current_frames[idx] is not None else None
                if frame_copy is None:
                    continue

                rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
                lab_a = pcv.rgb2gray_lab(rgb_img=rgb_frame, channel="a")
                a_thresh = pcv.threshold.binary(lab_a, 134, object_type="light")
                a_fill = pcv.fill(a_thresh, size=200)
                labeled_mask, num_rois = pcv.create_labels(a_fill)

                for roi_idx in range(1, num_rois + 1):
                    metrics = self.extract_roi_metrics(labeled_mask == roi_idx, rgb_frame, hsv_frame)
                    if metrics is None:
                        continue
                    row = self.metrics_to_model_features(metrics)
                    row["camera_panel"] = idx + 1
                    row["roi"] = f"ROI {roi_idx}"
                    row["analysis_type"] = self.grids[idx]["analysis_dropdown"].get()
                    rows.append(row)

            if not rows:
                logging.warning("No analysis data available to export")
                messagebox.showwarning("Dataset Export", "No analyzed ROI data is available to export yet.")
                return

            dataset_path = os.path.join(dataset_directory, 'dataset.csv')
            pd.DataFrame(rows).to_csv(dataset_path, index=False)
            logging.info(f"Dataset created successfully at {dataset_path}")
            messagebox.showinfo("Dataset Export", f"Saved dataset to {dataset_path}")

    def train_model(self):
        training_file = filedialog.askopenfilename(
            title="Select Training File or Model",
            filetypes=[
                ("Supported Files", "*.csv *.pkl *.pickle *.joblib"),
                ("CSV Files", "*.csv"),
                ("Pickle Files", "*.pkl *.pickle"),
                ("Joblib Files", "*.joblib"),
            ],
        )
        if training_file:
            extension = os.path.splitext(training_file)[1].lower()

            if extension in {'.pkl', '.pickle'}:
                with open(training_file, 'rb') as model_file:
                    self.ml_model = pickle.load(model_file)
                self.ml_model_path = training_file
                logging.info(f"Loaded ML model from {training_file}")
                messagebox.showinfo("ML Model", f"Loaded model from {training_file}")
                return

            if extension == '.joblib':
                try:
                    import joblib
                except ImportError:
                    messagebox.showerror("ML Model", "joblib is not installed. Install it to load .joblib models.")
                    return

                self.ml_model = joblib.load(training_file)
                self.ml_model_path = training_file
                logging.info(f"Loaded ML model from {training_file}")
                messagebox.showinfo("ML Model", f"Loaded model from {training_file}")
                return

            if extension == '.csv':
                try:
                    from sklearn.ensemble import RandomForestClassifier
                except ImportError:
                    messagebox.showerror("ML Training", "scikit-learn is not installed. Install it to train a model from CSV.")
                    return

                dataframe = pd.read_csv(training_file)
                target_column = 'label' if 'label' in dataframe.columns else None
                if target_column is None:
                    messagebox.showerror("ML Training", "Training CSV must contain a 'label' column.")
                    return

                missing_columns = [column for column in self.ml_feature_columns if column not in dataframe.columns]
                if missing_columns:
                    messagebox.showerror("ML Training", f"Training CSV is missing required feature columns: {', '.join(missing_columns)}")
                    return

                classifier = RandomForestClassifier(n_estimators=150, random_state=42)
                classifier.fit(dataframe[self.ml_feature_columns], dataframe[target_column])
                self.ml_model = classifier
                self.ml_model_path = training_file
                logging.info(f"Trained ML model from dataset {training_file}")
                messagebox.showinfo("ML Training", f"Model trained from {training_file}")
                return

            messagebox.showerror("ML Model", "Unsupported file type selected.")

    def mouse_wheel_zoom(self, event):
        if event.state & 0x0004:  # If Ctrl key is held down
            if event.delta > 0:
                self.main_frame.zoom(1.1)
            elif event.delta < 0:
                self.main_frame.zoom(0.9)


if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDetectionDashboard(root)
    root.geometry("1440x900")
    root.minsize(1100, 700)
    root.mainloop()
