import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os as _os
_PROCESS_ID = _os.getpid()

import streamlit as st
import numpy as np
import cv2
import random
import plotly.graph_objects as go
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.inference.post_processor import STATE_COLORS, STATE_MESSAGES
from config.constants import PSYCHOLOGICAL_STATES
from app.multi_person_tracker import PersonTracker, TrackedPerson
from app.session_logger import SessionLogger

def _load_realtime_predictor(**kwargs):
    from src.inference.realtime_predictor import RealtimePredictor
    return RealtimePredictor(**kwargs)


# ------------------------------------------------------------------
# Face detection (Haar cascade)
# ------------------------------------------------------------------
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def _detect_faces(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


# ------------------------------------------------------------------
# Demo helpers
# ------------------------------------------------------------------
def _demo_result_for_person(person: TrackedPerson, frame_rgb: np.ndarray) -> Dict:
    state = person.next_demo_state()
    probs = {s: round(random.uniform(0.02, 0.15), 3) for s in PSYCHOLOGICAL_STATES}
    probs[state] = round(random.uniform(0.55, 0.85), 3)
    total = sum(probs.values())
    probs = {s: round(v / total, 3) for s, v in probs.items()}
    return {
        "state":         state,
        "raw_state":     state,
        "confidence":    probs[state],
        "probs":         probs,
        "message":       STATE_MESSAGES.get(state, ""),
        "color":         STATE_COLORS.get(state, (150, 150, 150)),
        "micro_event":   random.random() < 0.05,
        "frame_rgb":     frame_rgb,
        "session_stats": {s: round(1 / len(PSYCHOLOGICAL_STATES), 3)
                          for s in PSYCHOLOGICAL_STATES},
    }


def _blank_person_history() -> Dict:
    return {
        "prob_history":  {s: deque(maxlen=60) for s in PSYCHOLOGICAL_STATES},
        "state_history": deque(maxlen=200),
        "frame_count":   0,
    }


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Psychological State Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
def init_session():
    if st.session_state.get("_pid") != _PROCESS_ID:
        for key in ("cap", "predictor", "session_logger"):
            obj = st.session_state.get(key)
            if obj is not None:
                try: obj.release() if key == "cap" else obj.stop() if key == "predictor" else obj.close()
                except Exception: pass
        st.session_state.clear()

    defaults = {
        "_pid":             _PROCESS_ID,
        "predictor":        None,
        "running":          False,
        "demo_mode":        True,
        "cap":              None,
        "tracker":          None,
        "session_logger":   None,
        "session_id":       None,
        "person_results":   {},
        "person_histories": {},
        "latest_result":    None,
        "prob_history":     {s: deque(maxlen=60) for s in PSYCHOLOGICAL_STATES},
        "state_history":    deque(maxlen=200),
        "frame_count":      0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("🧠 Psych State Detector")
st.sidebar.markdown("---")

face_model   = st.sidebar.text_input("Face model path",   value="", placeholder="models/face_agent/best.pth")
body_model   = st.sidebar.text_input("Body model path",   value="", placeholder="models/body_agent/best.pth")
speech_model = st.sidebar.text_input("Speech model path", value="", placeholder="models/speech_agent/best.pth")
camera_idx   = st.sidebar.number_input("Camera index", min_value=0, max_value=4, value=0, step=1)

st.sidebar.markdown("---")
demo_mode = st.sidebar.toggle("Demo Mode (no models needed)", value=st.session_state.get("demo_mode", True))
start_btn = st.sidebar.button("▶ Start Detection",  use_container_width=True)
stop_btn  = st.sidebar.button("⏹ Stop Detection",   use_container_width=True)
reset_btn = st.sidebar.button("🔄 Reset Session",    use_container_width=True)

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

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown(
    "Detects psychological states (calm, stressed, anxious, confident, frustrated) "
    "from facial expressions, body language, and voice in real time."
)


# ------------------------------------------------------------------
# Start / Stop / Reset handlers  (these trigger a full rerun once)
# ------------------------------------------------------------------
if start_btn and not st.session_state.running:
    st.session_state.demo_mode = demo_mode
    sid = datetime.now().strftime("%H%M%S")
    st.session_state.session_id       = sid
    st.session_state.tracker          = PersonTracker()
    st.session_state.session_logger   = SessionLogger(sid)
    st.session_state.person_results   = {}
    st.session_state.person_histories = {}
    st.session_state.frame_count      = 0

    if demo_mode:
        cap = cv2.VideoCapture(int(camera_idx))
        st.session_state.cap     = cap
        st.session_state.running = True
        st.sidebar.success(
            f"Demo mode started  |  Log: "
            f"`{os.path.basename(st.session_state.session_logger.filepath)}`"
        )
    else:
        try:
            pred = _load_realtime_predictor(
                face_model_path   = face_model   or None,
                body_model_path   = body_model   or None,
                speech_model_path = speech_model or None,
                camera_idx        = int(camera_idx),
            )
            pred.start()
            st.session_state.predictor = pred
            st.session_state.running   = True
            st.sidebar.success("Detection started!")
        except Exception as e:
            st.sidebar.error(f"Failed to start: {e}")

if stop_btn and st.session_state.running:
    cap = st.session_state.get("cap")
    if cap:
        cap.release()
        st.session_state.cap = None
    pred = st.session_state.get("predictor")
    if pred:
        pred.stop()
        st.session_state.predictor = None
    sl = st.session_state.get("session_logger")
    if sl:
        st.session_state.last_log_path = sl.filepath
        sl.close()
    st.session_state.running = False
    st.sidebar.info("Detection stopped.")

if reset_btn:
    pred = st.session_state.get("predictor")
    if pred:
        try: pred.reset_session()
        except Exception: pass
    tracker = st.session_state.get("tracker")
    if tracker:
        tracker.reset()
    sl = st.session_state.get("session_logger")
    if sl:
        st.session_state.last_log_path = sl.filepath
        sl.close()
    st.session_state.person_results   = {}
    st.session_state.person_histories = {}
    st.session_state.prob_history = {s: deque(maxlen=60) for s in PSYCHOLOGICAL_STATES}
    st.session_state.state_history.clear()
    st.session_state.frame_count = 0
    if st.session_state.running:
        sid = datetime.now().strftime("%H%M%S")
        st.session_state.session_id     = sid
        st.session_state.session_logger = SessionLogger(sid)
    st.sidebar.success("Session reset.")


# ------------------------------------------------------------------
# Static page headers + placeholders  (rendered ONCE, outside fragment)
# ------------------------------------------------------------------
st.title("Psychological State Detection")
st.markdown("Real-time multimodal analysis: face · body · voice")
st.markdown("---")

# All st.empty() / column placeholders live OUTSIDE the fragment so they
# are created once and only their .image/.plotly_chart contents change.
_is_demo_persistent = st.session_state.get("demo_mode", True)

if _is_demo_persistent:
    video_ph   = st.empty()
    st.markdown("---")
    persons_ph = st.empty()
    state_ph = conf_ph = msg_ph = micro_ph = None
else:
    col_v, col_s = st.columns([3, 2])
    with col_v:
        st.subheader("Live Feed")
        video_ph = st.empty()
    with col_s:
        st.subheader("Current State")
        state_ph = st.empty()
        conf_ph  = st.empty()
        msg_ph   = st.empty()
        micro_ph = st.empty()
    persons_ph = None

st.markdown("---")
col_probs, col_hist = st.columns(2)
with col_probs:
    st.subheader("State Probabilities")
    bar_ph = st.empty()
with col_hist:
    st.subheader("State Over Time")
    line_ph = st.empty()

st.markdown("---")
st.subheader("Session Statistics")
stats_ph = st.empty()


# ------------------------------------------------------------------
# Live detection fragment — only updates the placeholders above
# ------------------------------------------------------------------
@st.fragment(run_every=0.1)
def live_detection():
    if not st.session_state.running:
        return

    is_demo = st.session_state.get("demo_mode", True)

    # ── Helpers (inline to use local placeholders) ───────────────────

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
        if result.get("micro_event"):
            micro_ph.warning("⚡ Micro-expression detected")
        else:
            micro_ph.empty()

    def render_person_cards(person_results):
        if not person_results:
            persons_ph.info("No faces detected in frame.")
            return
        n = len(person_results)
        with persons_ph.container():
            st.markdown(f"**{n} person{'s' if n > 1 else ''} detected**")
            cols = st.columns(min(n, 4))
            for i, (pid, result) in enumerate(person_results.items()):
                state = result["state"]
                conf  = result["confidence"]
                color = _hex(result.get("color", (150, 150, 150)))
                probs = result["probs"]
                fc    = st.session_state.person_histories.get(pid, {}).get("frame_count", 0)
                with cols[i % 4]:
                    st.markdown(
                        f'<div style="background:{color}22;border-left:5px solid {color};'
                        f'padding:12px;border-radius:8px;text-align:center;margin-bottom:6px;">'
                        f'<span style="font-size:0.85rem;color:#888;">Person {pid}</span><br>'
                        f'<span style="font-size:1.8rem;font-weight:bold;color:{color};">{state.upper()}</span><br>'
                        f'<span style="font-size:0.9rem;color:#aaa;">{conf*100:.1f}% confidence</span></div>',
                        unsafe_allow_html=True,
                    )
                    if result.get("micro_event"):
                        st.warning("⚡ Micro-expression")
                    st.caption(f"Frames analysed: {fc}")
                    fig = go.Figure(go.Bar(
                        x=PSYCHOLOGICAL_STATES,
                        y=[probs.get(s, 0) * 100 for s in PSYCHOLOGICAL_STATES],
                        marker_color=[_hex(STATE_COLORS.get(s, (150,150,150))) for s in PSYCHOLOGICAL_STATES],
                    ))
                    fig.update_layout(
                        height=180, margin=dict(t=5,b=5,l=5,r=5),
                        yaxis=dict(range=[0, 105], title=""),
                        xaxis=dict(tickfont=dict(size=10)),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                        uirevision=f"person_{pid}",
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"person_chart_{pid}")

    # ── DEMO MODE ────────────────────────────────────────────────────
    if is_demo:
        cap = st.session_state.get("cap")
        if cap and cap.isOpened():
            ret, frame_bgr = cap.read()
            if ret:
                frame_rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                tracker     = st.session_state.tracker
                session_log = st.session_state.session_logger
                frame_no    = st.session_state.frame_count + 1
                bboxes      = _detect_faces(frame_bgr)
                tracked     = tracker.update(bboxes) if tracker else []
                annotated   = frame_rgb.copy()

                if tracked:
                    person_results = {}
                    for pid, bbox in tracked:
                        person = tracker.get_person(pid)
                        result = _demo_result_for_person(person, frame_rgb)
                        person_results[pid] = result

                        if session_log and session_log.is_open:
                            session_log.log(frame_no, pid, result)

                        if pid not in st.session_state.person_histories:
                            st.session_state.person_histories[pid] = _blank_person_history()
                        hist = st.session_state.person_histories[pid]
                        for s in PSYCHOLOGICAL_STATES:
                            hist["prob_history"][s].append(result["probs"].get(s, 0.0))
                        hist["state_history"].append(result["state"])
                        hist["frame_count"] += 1

                        x, y, w, h = bbox
                        r, g, b = result["color"]
                        cv2.rectangle(annotated, (x, y), (x+w, y+h), (r, g, b), 2)
                        cv2.putText(
                            annotated,
                            f"P{pid}: {result['state'].upper()} {result['confidence']*100:.0f}%",
                            (x, max(20, y-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (r, g, b), 2,
                        )

                    st.session_state.person_results  = person_results
                    st.session_state.frame_count    += 1

                    first_pid  = min(person_results.keys())
                    first_hist = st.session_state.person_histories[first_pid]
                    render_bar(person_results[first_pid])
                    render_history(first_hist["prob_history"])
                    with stats_ph.container():
                        render_stats(person_results[first_pid]["session_stats"])

                video_ph.image(annotated, channels="RGB", use_container_width=True)
                render_person_cards(st.session_state.person_results)

    # ── REAL MODE ────────────────────────────────────────────────────
    elif st.session_state.predictor:
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
                state = result["state"]
                color = result.get("color", (150, 150, 150))
                cv2.putText(lf, f"{state.upper()}  {result['confidence']*100:.0f}%",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (color[2], color[1], color[0]), 2)
                video_ph.image(lf, channels="RGB", use_container_width=True)

        if st.session_state.latest_result:
            r = st.session_state.latest_result
            render_state_card(r)
            render_bar(r)
            render_history(st.session_state.prob_history)
            with stats_ph.container():
                render_stats(r["session_stats"])


live_detection()
