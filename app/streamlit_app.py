import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_PROCESS_ID = os.getpid()

import streamlit as st
import cv2
import plotly.graph_objects as go
from collections import deque
from datetime import datetime

from src.inference.post_processor import STATE_COLORS, DISCLAIMER
from config.constants import PSYCHOLOGICAL_STATES
from app.session_logger import SessionLogger


def _load_realtime_predictor(**kwargs):
    from src.inference.realtime_predictor import RealtimePredictor
    return RealtimePredictor(**kwargs)


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Affect-Cue Estimation (research prototype)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
def init_session():
    if st.session_state.get("_pid") != _PROCESS_ID:
        for key in ("predictor", "session_logger"):
            obj = st.session_state.get(key)
            if obj is not None:
                try: obj.stop() if key == "predictor" else obj.close()
                except Exception: pass
        st.session_state.clear()

    defaults = {
        "_pid":           _PROCESS_ID,
        "predictor":      None,
        "running":        False,
        "session_logger": None,
        "session_id":     None,
        "latest_result":  None,
        "prob_history":   {s: deque(maxlen=60) for s in PSYCHOLOGICAL_STATES},
        "state_history":  deque(maxlen=200),
        "frame_count":    0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("🧠 Affect-Cue Estimator")
st.sidebar.caption(DISCLAIMER)
st.sidebar.markdown("---")

st.sidebar.markdown("**Trained checkpoints** (a modality runs only if its checkpoint is given)")
face_model   = st.sidebar.text_input("Face model path",   value="", placeholder="models/face_agent/face_agent_best.pth")
body_model   = st.sidebar.text_input("Body model path",   value="", placeholder="models/body_agent/body_lstm_best.pth")
speech_model = st.sidebar.text_input("Speech model path", value="", placeholder="models/speech_agent/speech_cnn_lstm_best.pth")
allow_rules  = st.sidebar.checkbox(
    "Use experimental rule-based body heuristics",
    value=False,
    help="Hand-written posture rules with no validation. Only used when no body checkpoint is given.",
)
camera_idx   = st.sidebar.number_input("Camera index", min_value=0, max_value=4, value=0, step=1)

st.sidebar.markdown("---")
consent = st.sidebar.checkbox(
    "Everyone in view and earshot has agreed to be analysed",
    value=False,
    help="Camera and microphone are analysed locally. Only per-frame probabilities are logged "
         "to CSV; no video or audio is saved.",
)
start_btn = st.sidebar.button("▶ Start", use_container_width=True, disabled=not consent)
stop_btn  = st.sidebar.button("⏹ Stop",  use_container_width=True)
reset_btn = st.sidebar.button("🔄 Reset Session", use_container_width=True)

st.sidebar.markdown("---")
_log_path = st.session_state.get("last_log_path")
if _log_path and os.path.isfile(_log_path):
    with open(_log_path, "rb") as _f:
        st.sidebar.download_button(
            label="📥 Download Session Log (CSV)",
            data=_f.read(),
            file_name=os.path.basename(_log_path),
            mime="text/csv",
            use_container_width=True,
        )


# ------------------------------------------------------------------
# Start / Stop / Reset handlers  (these trigger a full rerun once)
# ------------------------------------------------------------------
def _close_logger():
    sl = st.session_state.get("session_logger")
    if sl:
        st.session_state.last_log_path = sl.filepath
        sl.close()


if start_btn and consent and not st.session_state.running:
    try:
        pred = _load_realtime_predictor(
            face_model_path       = face_model   or None,
            body_model_path       = body_model   or None,
            speech_model_path     = speech_model or None,
            camera_idx            = int(camera_idx),
            allow_rule_based_body = allow_rules,
        )
    except (FileNotFoundError, ValueError) as e:
        st.sidebar.error(str(e))
    except Exception as e:
        st.sidebar.error(f"Failed to start: {e}")
    else:
        sid = datetime.now().strftime("%H%M%S")
        st.session_state.session_id     = sid
        st.session_state.session_logger = SessionLogger(sid)
        st.session_state.frame_count    = 0
        pred.start()
        st.session_state.predictor = pred
        st.session_state.running   = True
        label = ", ".join(pred.plan.active)
        if pred.plan.body_is_rule_based:
            label += " (body = unvalidated rules)"
        st.sidebar.success(f"Running: {label}")

if stop_btn and st.session_state.running:
    pred = st.session_state.get("predictor")
    if pred:
        pred.stop()
        st.session_state.predictor = None
    _close_logger()
    st.session_state.running = False
    st.sidebar.info("Stopped.")

if reset_btn:
    pred = st.session_state.get("predictor")
    if pred:
        try: pred.reset_session()
        except Exception: pass
    _close_logger()
    st.session_state.latest_result = None
    st.session_state.prob_history  = {s: deque(maxlen=60) for s in PSYCHOLOGICAL_STATES}
    st.session_state.state_history.clear()
    st.session_state.frame_count   = 0
    if st.session_state.running:
        sid = datetime.now().strftime("%H%M%S")
        st.session_state.session_id     = sid
        st.session_state.session_logger = SessionLogger(sid)
    st.sidebar.success("Session reset.")


# ------------------------------------------------------------------
# Static page headers + placeholders  (rendered ONCE, outside fragment)
# ------------------------------------------------------------------
st.title("Affect-Cue Estimation")
st.warning(DISCLAIMER)
st.markdown("---")

col_v, col_s = st.columns([3, 2])
with col_v:
    st.subheader("Live Feed")
    video_ph = st.empty()
with col_s:
    st.subheader("Current Cue")
    state_ph = st.empty()
    conf_ph  = st.empty()
    msg_ph   = st.empty()
    mods_ph  = st.empty()
    micro_ph = st.empty()

st.markdown("---")
col_probs, col_hist = st.columns(2)
with col_probs:
    st.subheader("Cue Probabilities")
    bar_ph = st.empty()
with col_hist:
    st.subheader("Cues Over Time")
    line_ph = st.empty()

st.markdown("---")
st.subheader("Session Statistics")
stats_ph = st.empty()


# ------------------------------------------------------------------
# Live detection fragment — only updates the placeholders above
# ------------------------------------------------------------------
@st.fragment(run_every=0.1)
def live_detection():
    if not st.session_state.running or not st.session_state.predictor:
        return

    def _hex(color):
        return "#{:02x}{:02x}{:02x}".format(*color)

    def render_bar(result):
        probs  = result["probs"]
        states = list(probs.keys())
        values = [probs[s] * 100 for s in states]
        colors = [_hex(STATE_COLORS.get(s, (150, 150, 150))) for s in states]
        fig = go.Figure(go.Bar(
            x=states, y=values, marker_color=colors,
            text=[f"{v:.1f}%" for v in values], textposition="outside",
        ))
        fig.update_layout(
            yaxis=dict(range=[0, 105], title="Probability (%)"),
            margin=dict(t=20, b=20, l=20, r=20), height=280,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            uirevision="bar",
        )
        bar_ph.plotly_chart(fig, use_container_width=True, key="bar_chart")

    def render_history(prob_history):
        fig = go.Figure()
        for s in PSYCHOLOGICAL_STATES:
            color = _hex(STATE_COLORS.get(s, (150, 150, 150)))
            fig.add_trace(go.Scatter(
                y=list(prob_history[s]), name=s, mode="lines",
                line=dict(color=color, width=2),
            ))
        fig.update_layout(
            yaxis=dict(range=[0, 1], title="Probability"),
            xaxis=dict(title="Frames"),
            margin=dict(t=20, b=20, l=20, r=20), height=280,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            uirevision="history",
        )
        line_ph.plotly_chart(fig, use_container_width=True, key="history_chart")

    def render_stats(stats):
        cols = st.columns(len(PSYCHOLOGICAL_STATES))
        for col, s in zip(cols, PSYCHOLOGICAL_STATES):
            pct   = stats.get(s, 0.0) * 100
            color = _hex(STATE_COLORS.get(s, (150, 150, 150)))
            col.markdown(
                f'<div style="text-align:center;padding:10px;border-radius:8px;'
                f'background:{color}22;border:1px solid {color};">'
                f'<b style="color:{color}">{s.capitalize()}</b><br>'
                f'<span style="font-size:1.4rem;">{pct:.1f}%</span></div>',
                unsafe_allow_html=True,
            )

    def render_state_card(result):
        state = result["state"]
        conf  = result["confidence"]
        color = _hex(result.get("color", (150, 150, 150)))
        state_ph.markdown(
            f'<div style="background:{color}22;border-left:6px solid {color};'
            f'padding:16px;border-radius:8px;text-align:center;">'
            f'<h1 style="color:{color};margin:0;font-size:2.5rem;">{state.upper()}</h1></div>',
            unsafe_allow_html=True,
        )
        conf_ph.metric("Confidence", f"{conf*100:.1f}%")
        msg_ph.info(result.get("message", ""))
        mods = result.get("modalities")
        if mods:
            mods_ph.caption("Contributing this frame: " + ", ".join(mods))
        if result.get("micro_event"):
            micro_ph.warning("⚡ Micro-expression detected")
        else:
            micro_ph.empty()

    result = st.session_state.predictor.get_result(timeout=0.05)
    if result:
        st.session_state.latest_result  = result
        st.session_state.frame_count   += 1
        sl = st.session_state.get("session_logger")
        if sl and sl.is_open:
            sl.log(st.session_state.frame_count, 1, result)
        for s in PSYCHOLOGICAL_STATES:
            st.session_state.prob_history[s].append(result["probs"].get(s, 0.0))
        st.session_state.state_history.append(result["state"])

        frame_rgb = result.get("frame_rgb")
        if frame_rgb is not None:
            lf = frame_rgb.copy()
            color = result.get("color", (150, 150, 150))
            cv2.putText(lf, f"{result['state'].upper()}  {result['confidence']*100:.0f}%",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (color[0], color[1], color[2]), 2)
            video_ph.image(lf, channels="RGB", use_container_width=True)

    if st.session_state.latest_result:
        r = st.session_state.latest_result
        render_state_card(r)
        render_bar(r)
        render_history(st.session_state.prob_history)
        with stats_ph.container():
            render_stats(r["session_stats"])


live_detection()
