import time
import requests
from typing import Optional

import cv2
import numpy as np
from picamera2 import Picamera2
from pyzbar.pyzbar import decode as zbar_decode
from libcamera import controls

API_BASE = "http://192.168.50.1:8000/api"
USERNAME = "Petr"
PASSWORD = "auto1111"

ORGANIZER_ID = 1
TARGET_POSITION = 1
DEBOUNCE_SECONDS = 1.0

DECODE_EVERY_N_FRAMES = 3
ROI_REL = (0.15, 0.15, 0.70, 0.70)

WIDTH, HEIGHT = 1920, 1080


def get_auth() -> tuple[str, str]:
    return (USERNAME, PASSWORD)


def find_bin_id_by_code(bin_code: str) -> Optional[int]:
    r = requests.get(
        f"{API_BASE}/bins/",
        params={"bin_code": bin_code},
        auth=get_auth(),
        timeout=3,
    )
    r.raise_for_status()
    data = r.json()
    items = data["results"] if isinstance(data, dict) and "results" in data else data

    for item in items:
        if item.get("bin_code") == bin_code:
            return item.get("id")
    return None


def upsert_slot_state(organizer_id: int, position: int, bin_id: int) -> None:
    r = requests.get(
        f"{API_BASE}/organizer-slot-states/",
        params={"organizer": organizer_id, "position": position},
        auth=get_auth(),
        timeout=3,
    )
    r.raise_for_status()
    data = r.json()
    items = data["results"] if isinstance(data, dict) and "results" in data else data

    payload = {
        "organizer": organizer_id,
        "position": position,
        "bin_id": bin_id,
        "is_present": True,
        "is_empty": None,
        "last_seen": None,
    }

    if len(items) > 0:
        slot_id = items[0]["id"]
        pr = requests.patch(
            f"{API_BASE}/organizer-slot-states/{slot_id}/",
            json=payload,
            auth=get_auth(),
            timeout=3,
        )
        pr.raise_for_status()
    else:
        pr = requests.post(
            f"{API_BASE}/organizer-slot-states/",
            json=payload,
            auth=get_auth(),
            timeout=3,
        )
        pr.raise_for_status()


def main():
    # --- Picamera2 setup ---
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    picam2.set_controls({
        "AfMode": controls.AfModeEnum.Continuous,
        "Sharpness": 1.5,
        "Contrast": 1.1,
        })

    use_roi = False
    frame_count = 0
    t_prev = time.time()

    last_written_code = None
    last_write_time = 0.0
    last_data = None

    while True:
        frame_rgb = picam2.capture_array()  # RGB888 -> (H,W,3), uint8
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        frame_count += 1
        t_now = time.time()
        fps = 1.0 / max(1e-6, (t_now - t_prev))
        t_prev = t_now

        h, w = frame_bgr.shape[:2]

        # ROI crop
        if use_roi:
            rx, ry, rw, rh = ROI_REL
            x = int(rx * w); y = int(ry * h)
            ww = int(rw * w); hh = int(rh * h)
            roi = frame_bgr[y:y+hh, x:x+ww]
            cv2.rectangle(frame_bgr, (x, y), (x+ww, y+hh), (0, 255, 0), 2)
        else:
            x = y = 0
            roi = frame_bgr

        data = ""

        # decode only every N frames
        if frame_count % DECODE_EVERY_N_FRAMES == 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            results = zbar_decode(gray)

            if results:
                # take first QR code found
                r0 = results[0]
                data = r0.data.decode("utf-8", errors="ignore").strip()

                # draw bbox
                pts = np.array([(p.x, p.y) for p in r0.polygon], dtype=np.int32)
                pts[:, 0] += x
                pts[:, 1] += y
                cv2.polylines(frame_bgr, [pts], True, (0, 0, 255), 2)

        if data:
            if data != last_data:
                print(f"[QR] {data}")
                last_data = data

            cv2.putText(frame_bgr, data, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            now = time.time()
            if (data != last_written_code) or (now - last_write_time > DEBOUNCE_SECONDS):
                try:
                    bin_id = find_bin_id_by_code(data)
                    if bin_id is not None:
                        upsert_slot_state(ORGANIZER_ID, TARGET_POSITION, bin_id)
                        print(f"[DB] Stored {data} (bin_id={bin_id}) into organizer={ORGANIZER_ID} position={TARGET_POSITION}")
                    else:
                        print(f"[DB] Ignored unknown code: {data}")

                    last_written_code = data
                    last_write_time = now

                except requests.RequestException as e:
                    print(f"[DB] API error: {e}")

        cv2.putText(frame_bgr, f"FPS {fps:.1f} | ROI {use_roi}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("QR Live (ESC quit, R toggle ROI)", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('r'), ord('R')):
            use_roi = not use_roi
            print(f"ROI enabled: {use_roi}")

    cv2.destroyAllWindows()
    picam2.stop()


if __name__ == "__main__":
    main()
