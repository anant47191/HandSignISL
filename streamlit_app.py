import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
from collections import Counter
import html

from ML import MudraRecognizer

st.set_page_config(page_title="MUDRA: ISL Detection", layout="centered")

CSS = '''
<style>
/* page */
html, body, [data-testid='stAppViewContainer'] {
    background: radial-gradient(ellipse at 20% 10%, rgba(0,255,200,0.05), rgba(0,0,0,0) 20%), linear-gradient(120deg, #061021 0%, #07111f 40%, #0b0816 100%);
    color: #e6f7f3;
}
.neon-title{ font-size:3.6rem; font-weight:900; color:#00ffd2; text-shadow:0 0 20px rgba(0,255,210,0.22),0 0 40px rgba(0,170,200,0.12); text-align:center; margin-bottom:0.25rem }
.subtitle{ font-size:1rem; color:#cfdde2; text-align:center; margin-top:-10px }
.card{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:1rem; border:1px solid rgba(255,255,255,0.03); box-shadow: 0 6px 30px rgba(0,0,0,0.6) }
.camera-box{ border-radius:12px; padding:0.75rem; border:1px solid rgba(255,255,255,0.03); box-shadow: 0 6px 20px rgba(0,0,0,0.65); background: linear-gradient(135deg, rgba(0,170,255,0.03), rgba(255,0,150,0.02)); }
.neon-output{ font-size:1.5rem; font-weight:800; color:#ffffff; border-radius:8px; padding:0.6rem 0.85rem; text-align:center; background: linear-gradient(90deg, rgba(0,255,200,0.08), rgba(255,0,120,0.02)); border:1px solid rgba(255,255,255,0.04)}
.muted{ color:#a9bcc1; font-size:0.95rem }
.controls{ display:flex; gap:0.6rem; align-items:center }
.btn-start{ background:linear-gradient(90deg,#00d5a5,#00ffd2); border-radius:8px; color:#04121a; font-weight:800; padding:0.5rem 0.9rem; border:none }
.btn-stop{ background:linear-gradient(90deg,#ff5a9e,#ffb3d1); border-radius:8px; color:#1b0418; font-weight:800; padding:0.5rem 0.9rem; border:none }
.status-pill{ padding:0.35rem 0.65rem; border-radius:999px; font-weight:700; font-size:0.9rem }
.status-running{ background:linear-gradient(90deg,#22c55e,#168944); color:white }
.status-idle{ background:linear-gradient(90deg,#374151,#1f2937); color:#e6eef0 }
.history-item{ padding:0.45rem 0.6rem; border-radius:8px; margin-bottom:0.5rem; background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(0,0,0,0.02)); border: 1px solid rgba(255,255,255,0.02)}
.confidence-bar{ height:10px; border-radius:999px; background:linear-gradient(90deg,#111827,#0b1220); border:1px solid rgba(255,255,255,0.03) }
.confidence-fill{ height:100%; border-radius:999px; background:linear-gradient(90deg,#00ffd2,#00b0a0); }
.small-muted{ font-size:0.85rem; color:#9fb4b6 }
.branding{ color:#a0fff0; font-weight:700; opacity:0.9 }
</style>
'''

st.markdown(CSS, unsafe_allow_html=True)

st.markdown('<div class="neon-title">MUDRA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Indian Sign Language Interpreter — live webcam mudra detection</div>', unsafe_allow_html=True)
st.write('---')

prediction_placeholder = st.empty()
# prettier 'idle' state — helpful instructions and a subtle animated pulse
idle_html = '''
<div style="display:flex;flex-direction:column;align-items:center;gap:0.45rem;">
    <div class="neon-output" style="min-width:320px;padding:1rem 1.25rem;">
        <div style="font-size:1.15rem;font-weight:800;">Ready to detect</div>
        <div class="small-muted" style="margin-top:0.25rem;font-weight:600;color:rgba(255,255,255,0.75)">Press Start Live to begin — show your hand clearly near the camera</div>
    </div>
    <div style="display:flex;gap:0.6rem;align-items:center;">
        <div style="width:12px;height:12px;border-radius:50%;background:#00ffd2;opacity:0.2;animation: pulse 1.55s infinite;box-shadow:0 0 12px rgba(0,255,210,0.12)"></div>
        <div style="width:12px;height:12px;border-radius:50%;background:#ff7aa3;opacity:0.15;animation: pulse 1.8s infinite;box-shadow:0 0 10px rgba(255,122,163,0.08)"></div>
        <div class="small-muted">Tip: use good lighting and face the camera</div>
    </div>
</div>
<style>@keyframes pulse { 0% { transform: scale(0.85); opacity: 0.2 } 50% { transform: scale(1.15); opacity: 0.8 } 100% { transform: scale(0.85); opacity: 0.2 } }</style>
'''
prediction_box = prediction_placeholder.markdown(idle_html, unsafe_allow_html=True)

st.write('---')

st.markdown('### Live Camera — instant mudra recognition')

col1, col2 = st.columns([3,2])

with col1:
    st.markdown('<div class="camera-box card">', unsafe_allow_html=True)
    st.write('')
    # Live detection controls — only Start / Stop and camera index
    if 'live' not in st.session_state:
        # default to NOT running; user must press Start
        st.session_state.live = False
    live_col1, live_col2 = st.columns([1, 2])
    # prettier start / stop controls
    with live_col1:
        c1, c2 = st.columns([1, 1])
        with c1:
            if not st.session_state.live:
                if st.button('Start Live', key='start_live'):
                    st.session_state.live = True
            else:
                if st.button('Stop Live', key='stop_live'):
                    st.session_state.live = False
    with live_col2:
        device_index = st.number_input('Camera index (if needed)', min_value=0, max_value=10, value=0, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.write('')
    st.markdown('<div class="card small-muted">', unsafe_allow_html=True)
    st.markdown('<strong class="branding">Live-only</strong> — press <strong>Start Live</strong> to begin. Use the Camera Index selector if your webcam is not the default device.')
    st.markdown('</div>', unsafe_allow_html=True)


@st.cache_resource
def get_recognizer():
    recognizer = MudraRecognizer()
    # Attempt to load existing model if available
    recognizer.load_model('mudra_model.pkl')
    return recognizer

recognizer = get_recognizer()

def extract_landmarks_from_image(pil_image):
    # Convert PIL -> BGR numpy
    image = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    mp_hands = recognizer.mp_hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    hands.close()

    vector = [0.0] * (21 * 3 * 2)
    if results and results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = None
            if results.multi_handedness and len(results.multi_handedness) > i:
                try:
                    label = results.multi_handedness[i].classification[0].label
                except Exception:
                    label = None

            base = 0
            if label and label.lower() == 'right':
                base = 63

            for j, lm in enumerate(hand_landmarks.landmark):
                vector[base + j*3] = lm.x
                vector[base + j*3 + 1] = lm.y
                vector[base + j*3 + 2] = lm.z

    return vector


def do_predict_and_show(pil_image):
    """Predict on a PIL image and update the UI + TTS session_state."""
    if recognizer.model is None:
        st.warning("No trained model found. Train a model using ML.py --train before using the web UI.")
        return

    vector = extract_landmarks_from_image(pil_image)
    if sum(abs(x) for x in vector) == 0:
        st.info('No hands detected in the image — try again or upload a clearer image.')
        return

    label, conf = recognizer.predict(vector)
    # Update prediction box visually
    color = '#00ffd2' if conf >= 0.5 else '#ff7aa3'
    prediction_html = f'<div class="neon-output" style="border-color:{color}; color:{color};">{label} ({conf:.2f})</div>'
    prediction_placeholder.markdown(prediction_html, unsafe_allow_html=True)

    # maintain prediction history for UI
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, (label, conf, time.time()))
    # keep last 6
    st.session_state.history = st.session_state.history[:6]

    # optionally speak prediction
    try:
        if st.session_state.get('speech_enabled') and conf >= st.session_state.get('speech_threshold', 0.5):
            now = time.time()
            last = st.session_state.get('speech_last', {'label': None, 'ts': 0})
            if label != last.get('label') or (now - last.get('ts', 0)) > 1.8:
                st.session_state.speech_text = label
                st.session_state.speech_last = {'label': label, 'ts': now}
    except Exception:
        pass


def run_live_detection(device_index=0):
    """Open webcam, predict continuously and update UI placeholders."""
    if recognizer.model is None:
        st.warning("No trained model found. Train a model using ML.py --train before using live detection.")
        st.session_state.live = False
        return

    cap = cv2.VideoCapture(device_index)
    if not cap or not cap.isOpened():
        st.error(f"Unable to open camera at index {device_index}. Try a different index or run diagnostics.")
        st.session_state.live = False
        return

    frame_placeholder = st.empty()
    fps_placeholder = st.empty()
    conf_placeholder = st.empty()

    try:
        last_time = time.time()
        while st.session_state.live:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            # process frame
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_h = recognizer.mp_hands
            with mp_h.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6) as hands_live:
                results = hands_live.process(img_rgb)

            vector = [0.0] * (21 * 3 * 2)
            if results and results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = None
                    if results.multi_handedness and len(results.multi_handedness) > i:
                        try:
                            label = results.multi_handedness[i].classification[0].label
                        except Exception:
                            label = None

                    base = 0
                    if label and label.lower() == 'right':
                        base = 63

                    for j, lm in enumerate(hand_landmarks.landmark):
                        vector[base + j*3] = lm.x
                        vector[base + j*3 + 1] = lm.y
                        vector[base + j*3 + 2] = lm.z

            # Predict and annotate frame
            if sum(abs(x) for x in vector) > 0:
                try:
                    lbl, conf = recognizer.predict(vector)
                    text = f"{lbl} ({conf:.2f})"
                except Exception as e:
                    text = f"Error: {e}"
            else:
                text = "No hands"

            annotated = frame.copy()
            box_color = (12, 255, 200) if 'No hands' not in text and 'Error' not in text else (180, 180, 180)
            cv2.rectangle(annotated, (6,6), (annotated.shape[1]-6, 62), (12,12,12), -1)
            cv2.putText(annotated, text, (12, 44), cv2.FONT_HERSHEY_DUPLEX, 1.0, box_color, 2, cv2.LINE_AA)

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, channels='RGB', use_container_width=True)

            # compute fps
            now = time.time()
            fps = 1.0 / max((now - last_time), 1e-6)
            last_time = now
            fps_placeholder.markdown(f"<div class='small-muted'>FPS: <strong>{fps:.1f}</strong></div>", unsafe_allow_html=True)

            # update prediction buffer & smoothing
            if sum(abs(x) for x in vector) > 0 and 'lbl' in locals():
                # initialize buffer
                if 'pred_buffer' not in st.session_state:
                    st.session_state.pred_buffer = []

                # Add only meaningful labels (skip 'No hands' and errors)
                if isinstance(lbl, str) and lbl and not lbl.lower().startswith('error') and lbl.lower() != 'no hands':
                    st.session_state.pred_buffer.append((lbl, conf, time.time()))
                    # limit buffer to window
                    st.session_state.pred_buffer = st.session_state.pred_buffer[-st.session_state.get('smooth_window', 5):]
                else:
                    # if no meaningful label detected, we clear short-term buffer so consensus resets
                    st.session_state.pred_buffer = []

                # Determine final stabilized label if smoothing enabled
                final_label = lbl
                final_conf = conf
                stable = True
                if st.session_state.get('smoothing', True) and len(st.session_state.pred_buffer) > 0:
                    labels = [p[0] for p in st.session_state.pred_buffer]
                    counts = Counter(labels)
                    top_label, top_count = counts.most_common(1)[0]
                    frac = top_count / max(1, len(labels))
                    if frac >= st.session_state.get('smooth_consensus', 0.6):
                        # accepted consensus
                        final_label = top_label
                        confs = [p[1] for p in st.session_state.pred_buffer if p[0] == top_label]
                        final_conf = float(sum(confs)/len(confs)) if confs else final_conf
                        stable = True
                    else:
                        # not stable yet — don't commit final label
                        stable = False
                else:
                    # smoothing disabled or no buffer: immediate label
                    final_label = lbl
                    final_conf = conf
                    stable = True
                # display the stabilized value (final_label/final_conf) for the UI
                color_hex = '#00ffd2' if final_conf >= 0.5 else '#ff7aa3'
                pct = min(max(final_conf, 0.0), 1.0)
                fill_w = int(pct * 100)
                conf_html = f"<div class='small-muted'>Confidence: <strong>{final_conf:.2f}</strong></div><div class='confidence-bar'><div class='confidence-fill' style='width:{fill_w}%; background:linear-gradient(90deg,{color_hex}, #00a090)'></div></div>"
                conf_placeholder.markdown(conf_html, unsafe_allow_html=True)

                # speak live detection if enabled and above threshold
                try:
                    if st.session_state.get('speech_enabled') and final_conf >= st.session_state.get('speech_threshold', 0.5) and stable:
                        now = time.time()
                        last = st.session_state.get('speech_last', {'label': None, 'ts': 0})
                        if final_label != last.get('label') or (now - last.get('ts', 0)) > 1.8:
                            st.session_state.speech_text = final_label
                            st.session_state.speech_last = {'label': final_label, 'ts': now}
                except Exception:
                    pass

    finally:
        try:
            cap.release()
        except Exception:
            pass

# If live mode was turned on, run the loop
if st.session_state.live:
    run_live_detection(int(device_index))

# right side: prediction history and model information
st.write('---')
side_col1, side_col2 = st.columns([2, 1])
with side_col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex;justify-content:space-between;align-items:center"><div class="small-muted">Model status</div><div class="status-pill ' + ('status-running' if st.session_state.live else 'status-idle') + '">' + ('Running' if st.session_state.live else 'Idle') + '</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
    # show a brief model hint
    if recognizer.model is None:
        st.warning('No trained model found. Run `python ML.py --train` to create one.')
    else:
        st.markdown('<div class="small-muted">Model loaded</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with side_col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex;justify-content:space-between;align-items:center"><div><strong>Recent predictions</strong></div><div class="small-muted">(most recent)</div></div>', unsafe_allow_html=True)
    if 'history' not in st.session_state or len(st.session_state.history) == 0:
        st.markdown('<div class="muted" style="padding:0.75rem;margin-top:0.5rem">No predictions yet — start live detection to populate history.</div>', unsafe_allow_html=True)
    else:
        for lbl, conf, ts in st.session_state.history:
            t = time.strftime('%H:%M:%S', time.localtime(ts))
            st.markdown(f'<div class="history-item"><div style="display:flex;justify-content:space-between"><div><strong>{lbl}</strong></div><div class="small-muted">{t}</div></div><div class="small-muted">confidence {conf:.2f}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# speech controls for the right panel (browser TTS)
if 'speech_enabled' not in st.session_state:
    st.session_state.speech_enabled = False
if 'speech_threshold' not in st.session_state:
    st.session_state.speech_threshold = 0.5
if 'speech_last' not in st.session_state:
    st.session_state.speech_last = {'label': None, 'ts': 0}

with side_col2:
    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<strong>Speech output</strong>', unsafe_allow_html=True)
    # Start / Stop buttons
    start_col, stop_col = st.columns([1,1])
    with start_col:
        if st.button('Start Speaking', key='start_speech'):
            st.session_state.speech_enabled = True
    with stop_col:
        if st.button('Stop Speaking', key='stop_speech'):
            st.session_state.speech_enabled = False

    st.markdown('<div style="height:0.4rem"></div>', unsafe_allow_html=True)
    thr = st.slider('Speak when confidence ≥', min_value=0.0, max_value=1.0, value=float(st.session_state.speech_threshold), step=0.05)
    st.session_state.speech_threshold = float(thr)
    if st.session_state.speech_enabled:
        st.success('Speech output: ON')
    else:
        st.info('Speech output: OFF')
    st.markdown('<div style="height:0.45rem"></div>', unsafe_allow_html=True)
    # quick speak / test controls
    tcol1, tcol2 = st.columns([3,1])
    with tcol1:
        test_phrase = st.text_input('Test phrase', value='Hello from MUDRA')
    with tcol2:
        if st.button('Speak now', key='speak_now'):
            st.session_state.speech_text = test_phrase
    st.markdown('</div>', unsafe_allow_html=True)

    # Smoothing / stabilization options
    st.markdown('<div style="height:0.35rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<strong>Stabilization</strong>', unsafe_allow_html=True)
    if 'smoothing' not in st.session_state:
        st.session_state.smoothing = True
    st.session_state.smoothing = st.checkbox('Enable smoothing / stabilization', value=st.session_state.smoothing)
    if 'smooth_window' not in st.session_state:
        st.session_state.smooth_window = 5
    if 'smooth_consensus' not in st.session_state:
        st.session_state.smooth_consensus = 0.6

    sw = st.slider('Smoothing window (frames)', min_value=1, max_value=20, value=int(st.session_state.smooth_window), step=1)
    st.session_state.smooth_window = int(sw)
    sc = st.slider('Consensus to accept label (fraction)', min_value=0.0, max_value=1.0, value=float(st.session_state.smooth_consensus), step=0.05)
    st.session_state.smooth_consensus = float(sc)
    st.markdown('<div class="small-muted">How smoothing works: the detector aggregates the last N predictions and only commits a label if the top label reaches the consensus fraction.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# client-side TTS player: if a new text is present in session_state we'll inject JS that uses speechSynthesis
def _maybe_render_tts():
    if 'speech_text' in st.session_state and st.session_state.speech_text:
        safe_text = html.escape(str(st.session_state.speech_text), quote=True)
        # minimal JS that speaks once when the DOM snippet is rendered (browser-side)
        js = f"""
        <div id='st_tts' data-tts='{safe_text}'></div>
        <script>
        (function(){{
            try {{
                const t = document.getElementById('st_tts').getAttribute('data-tts');
                if (!t) return;
                const utter = new SpeechSynthesisUtterance(t);
                utter.rate = 1.0; utter.pitch = 1.0; utter.volume = 1.0;
                speechSynthesis.cancel(); // stop any prior speech
                speechSynthesis.speak(utter);
            }} catch (e) {{ console.warn('TTS error', e); }}
        }})();
        </script>
        """
        st.markdown(js, unsafe_allow_html=True)
        # clear so it doesn't replay on every rerun
        try:
            st.session_state.speech_text = None
        except Exception:
            pass

# call once at the end of the file so TTS runs in the browser when session_state.speech_text is set
_maybe_render_tts()

st.write('---')
st.markdown('''
### Integration Guide

Use `predict()` via the ML model. This UI calls `recognizer.predict(vector)` using landmarks extracted with MediaPipe hands. If your model is missing or outdated, run `python ML.py --train` to (re)train on `dataset/`.
''')
