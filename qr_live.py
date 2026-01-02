import cv2
import time
import requests
from typing import Optional

API_BASE = "http://127.0.0.1:8000/api"
USERNAME = "Petr"
PASSWORD = "auto1111"

ORGANIZER_ID = 1
TARGET_POSITION = 1
DEBOUNCE_SECONDS = 1.0

CAM_INDEX = 1

DECODE_EVERY_N_FRAMES = 3
ROI_REL = (0.15, 0.15, 0.70, 0.70)

WIDTH, HEIGHT, FPS = 1280, 720, 30


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
    use_roi = False

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    detector = cv2.QRCodeDetector()


    last_written_code = None
    last_write_time = 0.0

    frame_count = 0
    last_data = None
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_count += 1

        t_now = time.time()
        fps = 1.0 / max(1e-6, (t_now - t_prev))
        t_prev = t_now

        h, w = frame.shape[:2]

        if use_roi:
            rx, ry, rw, rh = ROI_REL
            x = int(rx * w)
            y = int(ry * h)
            ww = int(rw * w)
            hh = int(rh * h)
            roi = frame[y:y + hh, x:x + ww]
            cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        else:
            x = y = 0
            roi = frame

        data, points = "", None
        if frame_count % DECODE_EVERY_N_FRAMES == 0:
            data, points, _ = detector.detectAndDecode(roi)

        if points is not None and len(points) > 0:
            pts = points.astype(int).reshape(-1, 2)
            pts[:, 0] += x
            pts[:, 1] += y
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

        if data:
            data = data.strip()

            if data != last_data:
                print(f"[QR] {data}")
                last_data = data

            cv2.putText(frame, data, (20, 80),
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

        cv2.putText(frame,
                    f"Cam {CAM_INDEX} | FPS {fps:.1f} | ROI {use_roi}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("QR Live (ESC quit, R toggle ROI)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('r'), ord('R')):
            use_roi = not use_roi
            print(f"ROI enabled: {use_roi}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
