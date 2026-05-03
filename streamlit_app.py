from __future__ import annotations

import html
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from functions.frontend_backend import DEMO_SCENARIOS, MODEL_NAMES, run_scenario


st.set_page_config(
    page_title="Indoor Positioning Demo",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


APP_CSS = """
<style>
    :root {
        --bg: #f6f8fb;
        --panel: #ffffff;
        --text: #13202e;
        --muted: #607084;
        --line: #d9e1ea;
        --wifi: #176b87;
        --wifi-soft: #dff4f7;
        --fused: #d97706;
        --fused-soft: #fff3d6;
        --good: #12805c;
        --shadow: 0 16px 40px rgba(23, 38, 54, 0.08);
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    /* ── Navbar / header visibility fix ── */
    header[data-testid="stHeader"] {
        background: rgba(246, 248, 251, 0.97) !important;
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--line);
    }
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a,
    header[data-testid="stHeader"] span {
        color: var(--text) !important;
    }
    header[data-testid="stHeader"] svg {
        fill: var(--text) !important;
    }
    [data-testid="stToolbar"] {
        right: 0;
    }
    [data-testid="stDecoration"] {
        display: none;
    }
    /* controls card */
    .controls-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 14px 20px 18px;
        margin-bottom: 20px;
        box-shadow: var(--shadow);
    }
    .controls-card-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--muted);
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 14px;
    }
    .ctrl-label {
        font-weight: 700;
        font-size: 0.93rem;
        color: var(--text);
        margin-bottom: 2px;
    }
    .ctrl-desc {
        font-size: 0.77rem;
        color: var(--muted);
        margin-bottom: 6px;
        line-height: 1.35;
    }
    /* expander visibility */
    details summary {
        color: var(--text) !important;
        font-weight: 600;
    }
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }

    .block-container {
        padding-top: 4.5rem;
        padding-bottom: 2.5rem;
        max-width: 1380px;
    }

    h1, h2, h3, p {
        letter-spacing: 0;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: var(--shadow);
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        color: #607084 !important;
        font-size: 0.82rem;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #13202e !important;
        font-weight: 760;
    }

    .app-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 24px;
        padding: 8px 2px 18px;
    }

    .app-title {
        font-size: 2rem;
        line-height: 1.1;
        font-weight: 780;
        margin: 0;
        color: var(--text);
    }

    .app-subtitle {
        margin: 8px 0 0;
        color: var(--muted);
        font-size: 0.98rem;
        max-width: 820px;
    }

    .status-chip {
        border: 1px solid #b7e0d4;
        background: #e9f8f2;
        color: var(--good);
        font-weight: 700;
        border-radius: 999px;
        padding: 8px 12px;
        white-space: nowrap;
        font-size: 0.84rem;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow);
        padding: 18px;
    }

    .panel-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        gap: 16px;
    }

    .panel-title h3 {
        margin: 0;
        font-size: 1.03rem;
        color: var(--text);
    }

    .badge {
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 0.78rem;
        font-weight: 760;
    }

    .badge-wifi {
        background: var(--wifi-soft);
        color: var(--wifi);
    }

    .badge-fused {
        background: var(--fused-soft);
        color: var(--fused);
    }

    .room-svg {
        width: 100%;
        height: auto;
        display: block;
        border-radius: 8px;
        background: #fbfdff;
        border: 1px solid #e3eaf1;
    }

    .legend-row {
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
        color: var(--muted);
        font-size: 0.84rem;
        margin-top: 10px;
    }

    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 99px;
        display: inline-block;
        margin-right: 6px;
    }

    .insight {
        border-left: 4px solid var(--good);
        background: #f2fbf7;
        border-radius: 8px;
        padding: 14px 16px;
        color: #1b4336;
        font-size: 0.94rem;
        line-height: 1.55;
    }

    .small-note {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.45;
    }

    .bar-row {
        display: grid;
        grid-template-columns: 112px 1fr 64px;
        align-items: center;
        gap: 12px;
        margin: 12px 0;
    }

    .bar-label {
        color: var(--text);
        font-weight: 700;
        font-size: 0.88rem;
    }

    .bar-track {
        display: grid;
        gap: 5px;
    }

    .bar {
        height: 12px;
        border-radius: 99px;
    }

    .bar-value {
        color: var(--muted);
        font-variant-numeric: tabular-nums;
        font-size: 0.82rem;
    }

    @keyframes pulse {
        0% { r: 7; opacity: 0.52; }
        70% { r: 25; opacity: 0; }
        100% { r: 25; opacity: 0; }
    }

    .pulse {
        animation: pulse 1.8s ease-out infinite;
    }

    @media (max-width: 900px) {
        .app-header {
            display: block;
        }

        .status-chip {
            display: inline-block;
            margin-top: 12px;
        }
    }
</style>
"""


def load_demo_data(scenario: int) -> dict:
    return run_scenario(scenario)


@st.cache_data(show_spinner=False)
def cached_demo_data(scenario: int) -> dict:
    return load_demo_data(scenario)


def format_delta(value: float) -> str:
    return f"{value:+.3f} m"


def pick_metric(metrics: list[dict], model_name: str) -> dict:
    return next(row for row in metrics if row["model"] == model_name)


def prediction_frame(data: dict, model_name: str, test_point: int) -> pd.DataFrame:
    frame = pd.DataFrame(data["predictions"])
    return frame[(frame["model"] == model_name) & (frame["test_point"] == test_point)].copy()


def scale_point(x: float, y: float, room: dict, width: int = 560, height: int = 360) -> tuple[float, float]:
    pad = 42
    usable_w = width - 2 * pad
    usable_h = height - 2 * pad
    x_span = room["max_x"] - room["min_x"] or 1
    y_span = room["max_y"] - room["min_y"] or 1
    sx = pad + ((x - room["min_x"]) / x_span) * usable_w
    sy = height - pad - ((y - room["min_y"]) / y_span) * usable_h
    return sx, sy


def room_panel(title: str, approach: str, row: pd.Series, room: dict, color: str, soft_color: str) -> str:
    width = 560
    height = 360
    actual_x, actual_y = scale_point(row["actual_x"], row["actual_y"], room, width, height)
    pred_x, pred_y = scale_point(row["predicted_x"], row["predicted_y"], room, width, height)
    escaped_title = html.escape(title)
    escaped_approach = html.escape(approach)

    grid_lines = []
    for i in range(6):
        x = 42 + i * ((width - 84) / 5)
        grid_lines.append(f'<line x1="{x:.1f}" y1="42" x2="{x:.1f}" y2="{height - 42}" stroke="#e8eef5" />')
    for i in range(5):
        y = 42 + i * ((height - 84) / 4)
        grid_lines.append(f'<line x1="42" y1="{y:.1f}" x2="{width - 42}" y2="{y:.1f}" stroke="#e8eef5" />')

    return f"""
    <div class="panel">
        <div class="panel-title">
            <h3>{escaped_title}</h3>
            <span class="badge {'badge-wifi' if approach == 'WiFi-Only' else 'badge-fused'}">{row['error_m']:.3f} m error</span>
        </div>
        <svg viewBox="0 0 {width} {height}" class="room-svg" role="img" aria-label="{escaped_approach} actual versus predicted position">
            <defs>
                <pattern id="dots-{escaped_approach.replace(' ', '-')}" width="18" height="18" patternUnits="userSpaceOnUse">
                    <circle cx="1.5" cy="1.5" r="1" fill="#dbe4ed" />
                </pattern>
            </defs>
            <rect x="20" y="20" width="{width - 40}" height="{height - 40}" rx="10" fill="url(#dots-{escaped_approach.replace(' ', '-')})" stroke="#d6e0ea" />
            {''.join(grid_lines)}
            <line x1="{actual_x:.1f}" y1="{actual_y:.1f}" x2="{pred_x:.1f}" y2="{pred_y:.1f}" stroke="{color}" stroke-width="3" stroke-dasharray="8 7" opacity="0.72" />
            <circle class="pulse" cx="{actual_x:.1f}" cy="{actual_y:.1f}" r="8" fill="{soft_color}" stroke="{color}" stroke-width="2" />
            <circle cx="{actual_x:.1f}" cy="{actual_y:.1f}" r="7" fill="{color}" />
            <circle cx="{pred_x:.1f}" cy="{pred_y:.1f}" r="11" fill="#ffffff" stroke="{color}" stroke-width="4" />
            <circle cx="{pred_x:.1f}" cy="{pred_y:.1f}" r="4" fill="{color}" />
            <text x="{actual_x + 12:.1f}" y="{actual_y - 10:.1f}" fill="#13202e" font-size="13" font-weight="700">Actual</text>
            <text x="{pred_x + 12:.1f}" y="{pred_y + 22:.1f}" fill="#13202e" font-size="13" font-weight="700">Predicted</text>
            <rect x="34" y="{height - 72}" width="230" height="38" rx="8" fill="#ffffff" stroke="#dce5ee" />
            <text x="48" y="{height - 49}" fill="#607084" font-size="13">Actual ({row['actual_x']:.2f}, {row['actual_y']:.2f}) → Predicted ({row['predicted_x']:.2f}, {row['predicted_y']:.2f})</text>
        </svg>
        <div class="legend-row">
            <span><span class="legend-dot" style="background:{color};"></span>Actual location</span>
            <span><span class="legend-dot" style="background:#fff; border:3px solid {color}; box-sizing:border-box;"></span>Predicted location</span>
            <span>Dashed line = localization error</span>
        </div>
    </div>
    """


def bar_comparison_chart(metrics: list[dict]) -> plt.Figure:
    """Matplotlib grouped bar chart comparing WiFi-only vs Fused mean errors."""
    models = [row["model"] for row in metrics]
    wifi_errors = [row["wifi_error_m"] for row in metrics]
    fused_errors = [row["fused_error_m"] for row in metrics]

    x = np.arange(len(models))
    w = 0.34

    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f9fbfd")

    b1 = ax.bar(x - w / 2, wifi_errors, w, label="WiFi-Only",     color="#176b87", alpha=0.90, zorder=3)
    b2 = ax.bar(x + w / 2, fused_errors, w, label="WiFi+BLE Fused", color="#d97706", alpha=0.90, zorder=3)

    for bar, val in zip(b1, wifi_errors):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f} m", ha="center", va="bottom", fontsize=8.5,
                color="#176b87", fontweight="bold")
    for bar, val in zip(b2, fused_errors):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f} m", ha="center", va="bottom", fontsize=8.5,
                color="#d97706", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, color="#13202e")
    ax.set_ylabel("Mean Distance Error (m)", fontsize=9.5, color="#607084")
    ax.set_title("Model Error Comparison — Lower is Better",
                 fontsize=11, color="#13202e", fontweight="bold", pad=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#d9e1ea")
    ax.spines["bottom"].set_color("#d9e1ea")
    ax.tick_params(colors="#607084")
    ax.grid(axis="y", color="#e8eef5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=True, fontsize=9)
    leg.get_frame().set_edgecolor("#d9e1ea")
    leg.get_frame().set_facecolor("#ffffff")
    fig.tight_layout()
    return fig


st.markdown(APP_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
        <div>
            <h1 class="app-title">Indoor Positioning Demo</h1>
            <p class="app-subtitle">
                Interactive comparison of WiFi-only fingerprinting and WiFi+BLE sensor fusion.
                The system predicts indoor coordinates from RSSI fingerprints and measures localization error.
            </p>
        </div>
        <div class="status-chip">Final demo scope: Scenario 1 & 2</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="controls-card"><div class="controls-card-title">⚙️ Configure Demo</div>', unsafe_allow_html=True)
control_col_1, control_col_2, control_col_3 = st.columns([1, 1, 1.2])

with control_col_1:
    st.markdown('<div class="ctrl-label">📡 Scenario</div><div class="ctrl-desc">The physical indoor environment being tested.</div>', unsafe_allow_html=True)
    scenario = st.selectbox(
        "Scenario",
        options=list(DEMO_SCENARIOS),
        format_func=lambda item: f"Scenario {item}",
        help="Each scenario is a different room layout. Scenario 1 & 2 are included in this demo.",
        label_visibility="collapsed",
    )

with st.spinner("Running positioning models..."):
    data = cached_demo_data(scenario)

with control_col_2:
    st.markdown('<div class="ctrl-label">🤖 Model</div><div class="ctrl-desc">The ML algorithm used to predict position.</div>', unsafe_allow_html=True)
    model_name = st.selectbox(
        "Model",
        options=list(MODEL_NAMES),
        help="KNN = K-Nearest Neighbors · SVM = Support Vector Machine · Random Forest = ensemble of decision trees.",
        label_visibility="collapsed",
    )

test_points = sorted({row["test_point"] for row in data["predictions"]})
with control_col_3:
    st.markdown('<div class="ctrl-label">🔢 Test Point</div><div class="ctrl-desc">Scroll through individual RSSI samples to see predicted vs actual location.</div>', unsafe_allow_html=True)
    test_point = st.slider(
        "Test point",
        min_value=min(test_points),
        max_value=max(test_points),
        value=min(test_points),
        step=1,
        help="Each test point is one RSSI fingerprint measurement. The model predicts the (x, y) room coordinate from signal strengths.",
        label_visibility="collapsed",
    )

st.markdown('</div>', unsafe_allow_html=True)

metric = pick_metric(data["metrics"], model_name)
prediction_rows = prediction_frame(data, model_name, test_point)
wifi_row = prediction_rows[prediction_rows["approach"] == "WiFi-Only"].iloc[0]
fused_row = prediction_rows[prediction_rows["approach"] == "WiFi+BLE Fused"].iloc[0]

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
metric_col_1.metric(
    "WiFi Mean Error",
    f"{metric['wifi_error_m']:.3f} m",
    help="Average physical distance (in metres) between the true location and the WiFi-only predicted location, across all test points. Lower = better accuracy.",
)
metric_col_2.metric(
    "Fused Mean Error",
    f"{metric['fused_error_m']:.3f} m",
    delta=format_delta(-metric["improvement_m"]),
    delta_color="inverse",
    help="Same error metric but when WiFi and BLE signals are combined (fused). Compare with WiFi Mean Error to see how much fusion helps.",
)
metric_col_3.metric(
    "Error Reduction",
    f"{metric['improvement_pct']:.1f}%",
    delta=f"{metric['improvement_m']:.3f} m",
    help="How much the fused approach reduced the localization error vs WiFi-only — shown as a percentage and absolute metres saved.",
)
metric_col_4.metric(
    "Fused Positioning Score",
    f"{metric['fused_positioning_score']:.1f}/100",
    help="A 0–100 score derived from room size and fused mean error. Higher = more accurate relative to room size. Useful for presentations; mean error is the primary technical metric.",
)

# Metric description row
st.markdown(
    f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:6px;margin-bottom:4px;">
        <div style="background:#dff4f7;border-left:4px solid #176b87;border-radius:6px;padding:10px 12px;font-size:0.8rem;color:#0d4a5e;line-height:1.45;">
            <strong>📶 WiFi-only mean distance error</strong><br>
            How far off the WiFi-only model is on average, measured in metres. Calculated across all test fingerprints in this scenario.
        </div>
        <div style="background:#fff3d6;border-left:4px solid #d97706;border-radius:6px;padding:10px 12px;font-size:0.8rem;color:#7a4a00;line-height:1.45;">
            <strong>📡 WiFi + BLE fused mean distance error</strong><br>
            Same error metric after combining WiFi and BLE signals. A lower value than WiFi Mean Error confirms that fusion improves accuracy.
        </div>
        <div style="background:#e9f8f2;border-left:4px solid #12805c;border-radius:6px;padding:10px 12px;font-size:0.8rem;color:#1b4336;line-height:1.45;">
            <strong>📉 Localization improvement from fusion</strong><br>
            Percentage and absolute-metre reduction in error when BLE is added. Positive values mean fusion helps; negative would mean it hurts.
        </div>
        <div style="background:#f0eeff;border-left:4px solid #6c47d4;border-radius:6px;padding:10px 12px;font-size:0.8rem;color:#3b2280;line-height:1.45;">
            <strong>🏆 Normalised positioning accuracy score</strong><br>
            A 0–100 score scaled by room size and fused error. Higher = more accurate relative to the environment. Use for quick comparisons; mean error is the rigorous metric.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tab layout ────────────────────────────────────────────────────────────────
st.write("")
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎬 Live Demo", "🏆 Model Race"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard (existing view)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns(2)
    with left:
        st.markdown(
            room_panel(
                "WiFi-Only Positioning",
                "WiFi-Only",
                wifi_row,
                data["room"],
                "#176b87",
                "#dff4f7",
            ),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            room_panel(
                "WiFi + BLE Fusion",
                "WiFi+BLE Fused",
                fused_row,
                data["room"],
                "#d97706",
                "#fff3d6",
            ),
            unsafe_allow_html=True,
        )

    st.write("")
    chart_col, insight_col = st.columns([1.25, 0.75])
    with chart_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig = bar_comparison_chart(data["metrics"])
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with insight_col:
        best_model = min(data["metrics"], key=lambda row: row["fused_error_m"])
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">
                    <h3>Scenario Insight</h3>
                    <span class="badge badge-fused">{html.escape(best_model['model'])} leads</span>
                </div>
                <div class="insight">
                    In Scenario {scenario}, WiFi+BLE fusion reduces localization error for the selected final-demo models.
                    For <strong>{html.escape(model_name)}</strong>, the fused estimate improves error by
                    <strong>{metric['improvement_m']:.3f} m</strong> compared with WiFi-only positioning.
                </div>
                <p class="small-note">
                    The Positioning Score is derived from room size and mean distance error. It is useful for presentation,
                    while mean distance error remains the technical evaluation metric.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("📋 Show Evaluation Table", expanded=False):
        st.caption("Full per-model comparison of WiFi-only vs fused positioning metrics for the selected scenario.")
        raw = pd.DataFrame(data["metrics"])[
            ["model", "wifi_error_m", "fused_error_m", "improvement_m", "improvement_pct",
             "wifi_positioning_score", "fused_positioning_score"]
        ].rename(columns={
            "model": "Model", "wifi_error_m": "WiFi Error (m)", "fused_error_m": "Fused Error (m)",
            "improvement_m": "Improvement (m)", "improvement_pct": "Improvement (%)",
            "wifi_positioning_score": "WiFi Score", "fused_positioning_score": "Fused Score",
        })
        for col in ["WiFi Error (m)", "Fused Error (m)", "Improvement (m)"]:
            raw[col] = raw[col].round(3)
        for col in ["Improvement (%)", "WiFi Score", "Fused Score"]:
            raw[col] = raw[col].round(1)
        st.dataframe(raw, width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Demo  (auto-play through test points)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    # Session-state bookkeeping
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "play_idx" not in st.session_state:
        st.session_state.play_idx = 0

    all_tps = sorted({row["test_point"] for row in data["predictions"]})

    # Controls bar
    ctrl_a, ctrl_b, ctrl_c, ctrl_d = st.columns([1, 1, 1, 3])
    with ctrl_a:
        if st.button("▶ Play" if not st.session_state.playing else "⏸ Pause",
                     use_container_width=True):
            st.session_state.playing = not st.session_state.playing
    with ctrl_b:
        if st.button("↩ Reset", use_container_width=True):
            st.session_state.play_idx = 0
            st.session_state.playing = False
    with ctrl_c:
        speed = st.selectbox("Speed", ["Slow", "Normal", "Fast"],
                             index=1, label_visibility="collapsed")
    speed_map = {"Slow": 1.5, "Normal": 0.8, "Fast": 0.3}

    live_idx = st.session_state.play_idx
    live_tp  = all_tps[live_idx]

    # Progress bar
    st.progress(live_idx / max(len(all_tps) - 1, 1),
                text=f"Test point {live_tp} of {all_tps[-1]}  ({live_idx + 1}/{len(all_tps)})")

    # Build per-frame data
    frame_all  = pd.DataFrame(data["predictions"])
    live_rows  = frame_all[(frame_all["model"] == model_name) &
                           (frame_all["test_point"] == live_tp)]
    live_wifi  = live_rows[live_rows["approach"] == "WiFi-Only"].iloc[0]
    live_fused = live_rows[live_rows["approach"] == "WiFi+BLE Fused"].iloc[0]

    # Trajectory history (all visited points so far)
    visited_tps  = all_tps[: live_idx + 1]
    hist_rows    = frame_all[(frame_all["model"] == model_name) &
                             (frame_all["test_point"].isin(visited_tps)) &
                             (frame_all["approach"] == "WiFi+BLE Fused")].sort_values("test_point")

    room = data["room"]
    W, H = 700, 420
    PAD  = 48
    x_span = (room["max_x"] - room["min_x"]) or 1
    y_span = (room["max_y"] - room["min_y"]) or 1

    def sp(x, y):
        sx = PAD + ((x - room["min_x"]) / x_span) * (W - 2 * PAD)
        sy = H - PAD - ((y - room["min_y"]) / y_span) * (H - 2 * PAD)
        return sx, sy

    # Build trajectory SVG path
    traj_pts = [sp(r["actual_x"], r["actual_y"]) for _, r in hist_rows.iterrows()]
    traj_path = " L".join(f"{px:.1f},{py:.1f}" for px, py in traj_pts)
    traj_svg  = f'<polyline points="{" ".join(f"{px:.1f},{py:.1f}" for px, py in traj_pts)}" fill="none" stroke="#12805c" stroke-width="2" stroke-dasharray="5 3" opacity="0.55" />' if len(traj_pts) > 1 else ""

    ax_w, ay = sp(live_wifi["actual_x"],  live_wifi["actual_y"])
    px_w, py_w = sp(live_wifi["predicted_x"], live_wifi["predicted_y"])
    px_f, py_f = sp(live_fused["predicted_x"], live_fused["predicted_y"])

    grid_lines = ""
    for i in range(7):
        x = PAD + i * ((W - 2 * PAD) / 6)
        grid_lines += f'<line x1="{x:.1f}" y1="{PAD}" x2="{x:.1f}" y2="{H - PAD}" stroke="#e4ecf3" />'
    for i in range(5):
        y = PAD + i * ((H - 2 * PAD) / 4)
        grid_lines += f'<line x1="{PAD}" y1="{y:.1f}" x2="{W - PAD}" y2="{y:.1f}" stroke="#e4ecf3" />'

    live_svg = f"""
    <svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;border-radius:10px;border:1px solid #d6e0ea;background:#fbfdff;" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <pattern id="ldots" width="18" height="18" patternUnits="userSpaceOnUse">
          <circle cx="1.5" cy="1.5" r="1" fill="#dbe4ed"/>
        </pattern>
        <radialGradient id="glow-actual" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#12805c" stop-opacity="0.35"/>
          <stop offset="100%" stop-color="#12805c" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <rect x="10" y="10" width="{W-20}" height="{H-20}" rx="10" fill="url(#ldots)" stroke="#d0dce8"/>
      {grid_lines}
      <!-- trajectory trail -->
      {traj_svg}
      <!-- WiFi predicted -->
      <line x1="{ax_w:.1f}" y1="{ay:.1f}" x2="{px_w:.1f}" y2="{py_w:.1f}" stroke="#176b87" stroke-width="2" stroke-dasharray="7 5" opacity="0.6"/>
      <circle cx="{px_w:.1f}" cy="{py_w:.1f}" r="10" fill="#fff" stroke="#176b87" stroke-width="3"/>
      <circle cx="{px_w:.1f}" cy="{py_w:.1f}" r="4" fill="#176b87"/>
      <text x="{px_w+13:.1f}" y="{py_w-6:.1f}" fill="#176b87" font-size="11" font-weight="700">WiFi pred ({live_wifi['error_m']:.2f} m)</text>
      <!-- Fused predicted -->
      <line x1="{ax_w:.1f}" y1="{ay:.1f}" x2="{px_f:.1f}" y2="{py_f:.1f}" stroke="#d97706" stroke-width="2" stroke-dasharray="7 5" opacity="0.6"/>
      <circle cx="{px_f:.1f}" cy="{py_f:.1f}" r="10" fill="#fff" stroke="#d97706" stroke-width="3"/>
      <circle cx="{px_f:.1f}" cy="{py_f:.1f}" r="4" fill="#d97706"/>
      <text x="{px_f+13:.1f}" y="{py_f+18:.1f}" fill="#d97706" font-size="11" font-weight="700">Fused pred ({live_fused['error_m']:.2f} m)</text>
      <!-- Actual glow + dot -->
      <circle cx="{ax_w:.1f}" cy="{ay:.1f}" r="22" fill="url(#glow-actual)"/>
      <circle cx="{ax_w:.1f}" cy="{ay:.1f}" r="9" fill="#12805c"/>
      <circle cx="{ax_w:.1f}" cy="{ay:.1f}" r="5" fill="#fff"/>
      <text x="{ax_w+12:.1f}" y="{ay-12:.1f}" fill="#12805c" font-size="12" font-weight="800">Actual position</text>
      <!-- Legend -->
      <rect x="14" y="{H-58}" width="300" height="44" rx="8" fill="#fff" stroke="#d9e1ea" opacity="0.92"/>
      <circle cx="30" cy="{H-42}" r="5" fill="#12805c"/><text x="40" y="{H-38}" fill="#13202e" font-size="11">Actual</text>
      <circle cx="95" cy="{H-42}" r="5" fill="none" stroke="#176b87" stroke-width="2.5"/><text x="105" y="{H-38}" fill="#13202e" font-size="11">WiFi-Only pred</text>
      <circle cx="200" cy="{H-42}" r="5" fill="none" stroke="#d97706" stroke-width="2.5"/><text x="210" y="{H-38}" fill="#13202e" font-size="11">Fused pred</text>
      <polyline points="30,{H-24} 40,{H-24}" fill="none" stroke="#12805c" stroke-width="2" stroke-dasharray="4 2"/><text x="46" y="{H-20}" fill="#607084" font-size="10">Trajectory trail</text>
    </svg>
    """

    st.markdown(live_svg, unsafe_allow_html=True)

    # Live error scoreboard
    s1, s2, s3 = st.columns(3)
    winner = "WiFi-Only" if live_wifi["error_m"] < live_fused["error_m"] else "WiFi+BLE Fused"
    s1.metric("Actual Position",
              f"({live_wifi['actual_x']:.2f}, {live_wifi['actual_y']:.2f}) m")
    s2.metric("WiFi-Only Error",  f"{live_wifi['error_m']:.3f} m")
    s3.metric("Fused Error",      f"{live_fused['error_m']:.3f} m",
              delta=f"{live_fused['error_m'] - live_wifi['error_m']:+.3f} m",
              delta_color="inverse")
    st.markdown(
        f'<p style="font-size:0.85rem;color:#607084;margin-top:4px;">'
        f'✅ <strong>{html.escape(winner)}</strong> wins this point &nbsp;·&nbsp; '
        f'Model: <strong>{html.escape(model_name)}</strong></p>',
        unsafe_allow_html=True,
    )

    # Auto-advance
    if st.session_state.playing:
        time.sleep(speed_map[speed])
        if live_idx < len(all_tps) - 1:
            st.session_state.play_idx += 1
        else:
            st.session_state.playing = False
            st.session_state.play_idx = 0
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Race  (all 3 models head-to-head for the same test point)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        "<p style='color:#607084;font-size:0.9rem;margin-bottom:8px;'>'"
        "All three models predict the <strong>same test point</strong> simultaneously — see which one is closest to the actual location.</p>",
        unsafe_allow_html=True,
    )

    frame_all2 = pd.DataFrame(data["predictions"])
    race_rows  = frame_all2[
        (frame_all2["test_point"] == test_point) &
        (frame_all2["approach"]   == "WiFi+BLE Fused")
    ].copy()

    # ── matplotlib head-to-head floor plan ──
    model_colors = {"KNN": "#176b87", "SVM": "#d97706", "Random Forest": "#7c3aed"}
    room = data["room"]
    x_span2 = (room["max_x"] - room["min_x"]) or 1
    y_span2 = (room["max_y"] - room["min_y"]) or 1

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    fig2.patch.set_facecolor("#fbfdff")
    ax2.set_facecolor("#f4f8fc")
    ax2.set_xlim(room["min_x"] - 0.3, room["max_x"] + 0.3)
    ax2.set_ylim(room["min_y"] - 0.3, room["max_y"] + 0.3)
    ax2.set_xlabel("X coordinate (m)", color="#607084", fontsize=9)
    ax2.set_ylabel("Y coordinate (m)", color="#607084", fontsize=9)
    ax2.set_title(f"Model Race — Test Point {test_point} (Fused)",
                  fontsize=12, color="#13202e", fontweight="bold", pad=10)
    ax2.tick_params(colors="#607084")
    ax2.grid(color="#e4ecf3", linewidth=0.7, zorder=0)
    ax2.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    ax2.spines["left"].set_color("#d9e1ea")
    ax2.spines["bottom"].set_color("#d9e1ea")

    if not race_rows.empty:
        actual_x = race_rows.iloc[0]["actual_x"]
        actual_y = race_rows.iloc[0]["actual_y"]
        # Glow ring around actual
        ax2.scatter(actual_x, actual_y, s=600, color="#12805c", alpha=0.18, zorder=2)
        ax2.scatter(actual_x, actual_y, s=180, color="#12805c", zorder=3,
                    label="Actual position", edgecolors="white", linewidths=2)

        scoreboard = []
        for _, row in race_rows.iterrows():
            m   = row["model"]
            col = model_colors.get(m, "#aaa")
            ax2.plot([actual_x, row["predicted_x"]], [actual_y, row["predicted_y"]],
                     color=col, linewidth=1.8, linestyle="--", alpha=0.6, zorder=2)
            ax2.scatter(row["predicted_x"], row["predicted_y"], s=130, color=col,
                        zorder=4, edgecolors="white", linewidths=1.8, label=f"{m} ({row['error_m']:.3f} m)")
            scoreboard.append({"Model": m, "Error (m)": round(row["error_m"], 3), "Color": col})

        ax2.legend(frameon=True, fontsize=9, loc="upper right").get_frame().set_edgecolor("#d9e1ea")
        fig2.tight_layout()
        st.pyplot(fig2, width="stretch")
        plt.close(fig2)

        # Scoreboard
        scoreboard.sort(key=lambda r: r["Error (m)"])
        st.markdown("<h4 style='color:#13202e;margin:16px 0 8px;'>🏅 Leaderboard — Fused predictions (lower error wins)</h4>",
                    unsafe_allow_html=True)
        medal = ["🥇", "🥈", "🥉"]
        for i, entry in enumerate(scoreboard):
            winner_badge = (
                "<span style='margin-left:auto;background:#e9f8f2;color:#12805c;"
                "padding:3px 10px;border-radius:99px;font-size:0.8rem;font-weight:700;'>WINNER</span>"
                if i == 0 else ""
            )
            card_html = (
                "<div style='display:flex;align-items:center;gap:12px;padding:10px 14px;"
                "margin-bottom:6px;border-radius:8px;background:#fff;border:1px solid #d9e1ea;"
                "box-shadow:0 2px 8px rgba(0,0,0,0.04);'>"
                f"<span style='font-size:1.4rem;'>{medal[i]}</span>"
                f"<span style='font-weight:700;color:{entry['Color']};min-width:130px;'>{html.escape(entry['Model'])}</span>"
                f"<span style='color:#13202e;font-size:1rem;font-weight:600;'>{entry['Error (m)']:.3f} m error</span>"
                f"{winner_badge}"
                "</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)

        # All-points winner tally
        st.markdown("<h4 style='color:#13202e;margin:20px 0 8px;'>📈 Win count across all test points (Fused)</h4>",
                    unsafe_allow_html=True)
        fused_all = frame_all2[frame_all2["approach"] == "WiFi+BLE Fused"].copy()
        winners = fused_all.loc[fused_all.groupby("test_point")["error_m"].idxmin(), "model"].value_counts()
        for m, count in winners.items():
            col  = model_colors.get(m, "#aaa")
            pct  = int(count / len(all_tps) * 100)
            st.markdown(
                f"""<div style='display:grid;grid-template-columns:140px 1fr 50px;align-items:center;gap:10px;margin:6px 0;'>
                <span style='font-weight:700;color:{col};font-size:0.9rem;'>{html.escape(m)}</span>
                <div style='background:#f0f4f8;border-radius:99px;height:14px;overflow:hidden;'>
                  <div style='height:100%;width:{pct}%;background:{col};border-radius:99px;'></div>
                </div>
                <span style='color:#607084;font-size:0.85rem;'>{count} pts</span>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("No race data available for this test point.")

st.caption(
    "Dataset: RSSI Dataset for Indoor Localization Fingerprinting. "
    "Final demo uses Scenario 1 and Scenario 2."
)
