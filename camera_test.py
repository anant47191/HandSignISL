"""
Camera test helper.
Attempts indices 0..5 and tries to grab a single frame from each. Saves a JPG if successful.
Usage:
  python camera_test.py
Outputs which indices are usable.
"""
import argparse
import cv2
import time

def try_index(idx, backends=None, save_prefix='test'):
    if backends is None:
        backends = [None, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for backend in backends:
        try:
            if backend is None:
                cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx, backend)

            if not cap or not cap.isOpened():
                # try next backend
                # ensure released if possible
                try:
                    if cap:
                        cap.release()
                except Exception:
                    pass
                continue

            # warm up
            for _ in range(6):
                ret, frame = cap.read()
                time.sleep(0.03)

            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                out = f"{save_prefix}_{idx}.jpg"
                cv2.imwrite(out, frame)
                try:
                    cap.release()
                except Exception:
                    pass
                return True, f"OK (backend={backend}) frame={w}x{h} saved={out}"

            try:
                cap.release()
            except Exception:
                pass

        except Exception as e:
            return False, f"error: {e}"

    return False, "no backend opened or readable frame"


def main():
    parser = argparse.ArgumentParser(description='Probe camera indices and save one test frame.')
    parser.add_argument('--max-index', type=int, default=15)
    parser.add_argument('--save-prefix', type=str, default='test')
    args = parser.parse_args()

    found = []
    for idx in range(args.max_index + 1):
        print(f"Trying index {idx}...")
        ok, msg = try_index(idx, save_prefix=args.save_prefix)
        print(f"  index {idx}: {msg}")
        if ok:
            found.append(idx)

    print("Done. Found indices:", found)


if __name__ == '__main__':
    main()
