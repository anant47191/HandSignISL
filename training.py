"""
Training / Dataset capture tool for ISL.

Features:
- Camera diagnostics: --diagnose to probe indices and report which indices can return frames
- Force a device index: --index
- Robust camera acquire with retries and helpful debug output
- Saves images to dataset/<mudra>/images and CSV rows to dataset/<mudra>/<mudra>.csv

Usage examples:
  python training.py                # interactive capture (asks mudra name)
  python training.py --index 0      # force device index 0
  python training.py --diagnose     # probe indices and report availability
"""
import argparse
import csv
import os
import time
import cv2
import mediapipe as mp


def parse_args():
    p = argparse.ArgumentParser(description="Create dataset by capturing hand landmarks with MediaPipe")
    p.add_argument("--diagnose", action="store_true", help="Probe camera indices and report which open/read frames")
    p.add_argument("--index", type=int, help="Force a specific camera device index to open")
    p.add_argument("--max-index", type=int, default=15, help="Max device index to probe when diagnosing (default 15)")
    return p.parse_args()


def _try_open_camera(idx, backend=None, warmup_frames=8, timeout_s=2.0):
    """Attempt to open a camera (index, optional backend) and verify we can read a frame.

    Returns (cap) when successful or None.
    """
    try:
        if backend is None:
            cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(idx, backend)
    except Exception:
        return None

    if not cap or not cap.isOpened():
        try:
            if cap:
                cap.release()
        except Exception:
            pass
        return None

    start = time.time()
    # try warmup reads until we get a frame or timeout
    while time.time() - start < timeout_s:
        ret, frame = cap.read()
        if ret and frame is not None and getattr(frame, 'shape', None):
            return cap
        time.sleep(0.05)

    try:
        cap.release()
    except Exception:
        pass
    return None


def open_camera_auto(max_index=8, verbose=True):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    # try indices first in ascending order; many laptops use index 0
    for idx in range(0, max_index + 1):
        for backend in backends:
            if verbose:
                print(f"Trying index={idx} backend={backend} ...", end=' ')
            cap = _try_open_camera(idx, backend=backend)
            if cap is not None:
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                except Exception:
                    pass
                if verbose:
                    print("OK")
                return cap, idx
            if verbose:
                print("no frame")
    return None, None


def probe_cameras(max_index=15):
    print(f"Probing camera indices 0..{max_index} (this may take a few seconds)...")
    available = []
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for idx in range(0, max_index + 1):
        ok = False
        for backend in backends:
            cap = _try_open_camera(idx, backend=backend)
            if cap is not None:
                # read a sample frame for reporting
                ret, frame = cap.read()
                if ret and frame is not None and getattr(frame, 'shape', None):
                    h, w = frame.shape[:2]
                    print(f"Index {idx} -> OK (backend={backend}) frame={w}x{h}")
                    available.append(idx)
                    ok = True
                else:
                    print(f"Index {idx} -> opened but no readable frame (backend={backend})")
                try:
                    cap.release()
                except Exception:
                    pass
                if ok:
                    break
        if not ok:
            print(f"Index {idx} -> NOT AVAILABLE")

    if available:
        print("Working indices:", available)
    else:
        print("No working camera indices were found.")
    return available


def prepare_csv(img_folder, csv_file, label_name):
    os.makedirs(img_folder, exist_ok=True)
    # header: two hands (LEFT, RIGHT) × 21 × (x,y,z) and class
    header = []
    for hand_id in ["LEFT", "RIGHT"]:
        for i in range(21):
            header += [f"{hand_id}_x{i}", f"{hand_id}_y{i}", f"{hand_id}_z{i}"]
    header.append("class")

    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def capture_loop(cap, img_path, csv_path, label_name, max_samples=None):
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    print("Camera opened successfully — you can press 'q' to stop recording at any time.")

    # show camera properties
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 0)
        print(f"Camera properties: width={w} height={h} fps={fps}")
    except Exception:
        pass

    img_count = 0
    print("\nRecording Started... Press 'q' to stop.\n")

    MAX_READ_RETRIES = 10
    while True:
        # Try reading a frame with retries
        success = False
        frame = None
        for attempt in range(MAX_READ_RETRIES):
            success, frame = cap.read()
            if success and frame is not None and getattr(frame, 'shape', None):
                if attempt > 0:
                    print(f"Frame read succeeded after {attempt+1} attempts")
                break
            time.sleep(0.03)

        if not success:
            print("Warning: no frames received from camera; ending capture loop.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        row = [0.0] * (21 * 3 * 2)
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # determine whether this is left/right when available
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
                    row[base + j*3] = lm.x
                    row[base + j*3 + 1] = lm.y
                    row[base + j*3 + 2] = lm.z

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Save frame image
        img_file = os.path.join(img_path, f"{label_name}_{img_count}.jpg")
        saved = False
        try:
            saved = cv2.imwrite(img_file, frame)
        except Exception as e:
            print(f"Warning: failed to save image: {e}")

        if saved:
            img_count += 1

        # write CSV row (append label)
        try:
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(row + [label_name])
        except Exception as e:
            print(f"Warning: failed to append CSV row: {e}")

        # show frame if GUI available
        try:
            cv2.imshow('Dataset Creator - Press Q to Quit', frame)
        except cv2.error:
            # headless environment - don't show
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if max_samples and img_count >= max_samples:
            print(f"Reached max_samples={max_samples}")
            break

    try:
        cap.release()
        cv2.destroyAllWindows()
    except Exception:
        pass

    print(f"\nDataset capture finished. Saved {img_count} images to {img_path} and CSV rows to {csv_path}")


def main():
    args = parse_args()

    if args.diagnose:
        probe_cameras(args.max_index)
        return

    # choose mudra label
    label_name = input("Enter Mudra Name: ").strip() or 'mudra'
    base_path = 'dataset'
    img_path = os.path.join(base_path, label_name, 'images')
    csv_path = os.path.join(base_path, label_name, f"{label_name}.csv")

    prepare_csv(img_path, csv_path, label_name)

    cap = None
    used_index = None

    # If user forced an index, try it
    if args.index is not None:
        print(f"Trying camera index {args.index} as requested...")
        cap = _try_open_camera(args.index)
        if cap is not None:
            used_index = args.index
        else:
            print(f"Unable to open camera index {args.index}. Falling back to auto-detect.")

    if cap is None:
        cap, used_index = open_camera_auto(max_index=8)

    if cap is None:
        print("Could not open any camera automatically. Run 'python training.py --diagnose' to probe device indices, check that no other app uses the camera, and verify Windows privacy/driver settings.")
        return

    capture_loop(cap, img_path, csv_path, label_name)


if __name__ == '__main__':
    main()
import cv2
import mediapipe as mp
import csv
import os
import time

# Ask mudra name
mudra_name = input("Enter Mudra Name: ").strip()

# Create dataset folders
base_path = "dataset"
img_path = os.path.join(base_path, mudra_name, "images")
csv_path = os.path.join(base_path, mudra_name, f"{mudra_name}.csv")

os.makedirs(img_path, exist_ok=True)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# CSV header (42 landmarks × 3 coords per hand = 126 values per hand)
header = []
for hand_id in ["LEFT", "RIGHT"]:
    for i in range(21):
        header += [
            f"{hand_id}_x{i}", 
            f"{hand_id}_y{i}", 
            f"{hand_id}_z{i}"
        ]

# Create CSV file
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

cap = cv2.VideoCapture(0)
img_count = 0

print("\nRecording Started... Press 'q' to stop.\n")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    row = [0] * (21 * 3 * 2)  # default values if only 1 hand is visible

    if results.multi_hand_landmarks and results.multi_handedness:
        handedness = {  
            h.classification[0].label: i  
            for i, h in enumerate(results.multi_handedness)
        }

        for label, idx in handedness.items():
            hand_landmarks = results.multi_hand_landmarks[idx]

            for i, lm in enumerate(hand_landmarks.landmark):
                base = 0 if label == "Left" else 63
                row[base + i*3] = lm.x
                row[base + i*3 + 1] = lm.y
                row[base + i*3 + 2] = lm.z

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Save frame image
    img_file = os.path.join(img_path, f"{mudra_name}_{img_count}.jpg")
    cv2.imwrite(img_file, frame)
    img_count += 1

    # Save CSV row
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    cv2.imshow("Dataset Creator - Press Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nDataset Created Successfully!")
print(f"Saved in: dataset/{mudra_name}/")
