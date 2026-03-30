import time
from typing import Optional, Dict, Tuple

import cv2
import numpy as np
import requests
from picamera2 import Picamera2
from pyzbar.pyzbar import decode as zbar_decode
from libcamera import controls

# ---------- KONFIGURACE KAMERY ----------

# Které kamery použít (Picamera2 indexy)
CAMERA_INDICES = [0, 1]  # 0 = první port kamery, 1 = druhý

# Rozlišení
WIDTH, HEIGHT = 1920, 1080

# FPS limit
TARGET_FPS = 5
FRAME_TIME = 1.0 / TARGET_FPS

# Časový limit pro považování slotu za prázdný
LAST_SEEN_TIMEOUT = 2.0 
slot_last_seen: Dict[str, float] = {}

# Počet vodorovných sekcí (slotů) na kameru
NUM_SECTIONS = 3

# Ladění výkonu dekódování
DECODE_EVERY_N_FRAMES = 3

# ROI (relativní) – ořez v ose Y, šířka zůstane plná
# (rx, ry, rw, rh) hodnoty 0..1 relativní k celému snímku
ROI_REL = (0.0, 0.15, 1.0, 0.70)

# Mapování (index_kamery, index_sekce) → logický název slotu
SLOT_MAPPING: Dict[Tuple[int, int], str] = {
    (0, 0): "R1C1",
    (0, 1): "R1C2",
    (0, 2): "R1C3",
    (1, 0): "R2C1",
    (1, 1): "R2C2",
    (1, 2): "R2C3",
}

# Mapování logického názvu slotu → pozice organizéru v Django
SLOT_TO_POSITION: Dict[str, int] = {
    "R1C1": 1,
    "R1C2": 2,
    "R1C3": 3,
    "R2C1": 4,
    "R2C2": 5,
    "R2C3": 6,
}

# ---------- KONFIGURACE API ----------

API_BASE = "http://192.168.137.1:8000/api"
USERNAME = "Petr"
PASSWORD = "auto1111"

ORGANIZER_ID = 1

# Jak často synchronizovat stav do backendu
SYNC_INTERVAL_SECONDS = 1.0

# Časový limit požadavku
API_TIMEOUT = 3


def get_auth() -> tuple[str, str]:
    return (USERNAME, PASSWORD)


def create_camera(cam_idx: int) -> Picamera2:
    """Vytvoří a nakonfiguruje instanci Picamera2 pro daný index."""
    picam = Picamera2(camera_num=cam_idx)
    config = picam.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "BGR888"}
    )
    picam.configure(config)
    picam.start()

    # Nastavení manuálního ostření
    picam.set_controls(
        {
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": 5.0,
        }
    )

    time.sleep(0.5) 
    return picam


def find_bin_id_by_code(bin_code: str) -> Optional[int]:
    """Převede skenovaný kód BIN-... na Bin.id v Django."""
    r = requests.get(
        f"{API_BASE}/bins/",
        params={"bin_code": bin_code},
        auth=get_auth(),
        timeout=API_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    items = data["results"] if isinstance(data, dict) and "results" in data else data

    for item in items:
        if item.get("bin_code") == bin_code:
            return item.get("id")
    return None


def upsert_slot_state(
    organizer_id: int,
    position: int,
    bin_id: Optional[int],
    is_present: bool,
    is_empty: Optional[bool] = None,
) -> None:
    """
    Vytvoří nebo aktualizuje OrganizerSlotState pro jednu pozici organizéru.
    Používá pole bin_id, protože serializer mapuje bin_id → model.bin.
    """
    r = requests.get(
        f"{API_BASE}/organizer-slot-states/",
        params={"organizer": organizer_id, "position": position},
        auth=get_auth(),
        timeout=API_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    items = data["results"] if isinstance(data, dict) and "results" in data else data

    payload = {
        "organizer": organizer_id,
        "position": position,
        "bin_id": bin_id,
        "is_present": is_present,
        "is_empty": is_empty,
        "last_seen": None,
    }

    if len(items) > 0:
        slot_id = items[0]["id"]
        pr = requests.patch(
            f"{API_BASE}/organizer-slot-states/{slot_id}/",
            json=payload,
            auth=get_auth(),
            timeout=API_TIMEOUT,
        )
        pr.raise_for_status()
    else:
        pr = requests.post(
            f"{API_BASE}/organizer-slot-states/",
            json=payload,
            auth=get_auth(),
            timeout=API_TIMEOUT,
        )
        pr.raise_for_status()


def sync_slot_state_to_api(
    organizer_id: int,
    slot_state: Dict[str, Optional[str]],
    last_sent_state: Dict[str, Optional[str]],
) -> None:
    """
    Odešle aktuální stav slotů do API.
    Zapisuje pouze sloty, které se změnily od poslední synchronizace.
    """
    for slot_name in sorted(slot_state.keys()):
        current_code = slot_state[slot_name]
        previous_code = last_sent_state.get(slot_name)

        if current_code == previous_code:
            continue

        position = SLOT_TO_POSITION.get(slot_name)
        if position is None:
            print(f"[API] No position mapping for slot {slot_name}, skipping.")
            continue

        try:
            if current_code is None:
                upsert_slot_state(
                    organizer_id=organizer_id,
                    position=position,
                    bin_id=None,
                    is_present=False,
                    is_empty=None,
                )
                print(f"[API] Cleared slot {slot_name} (position={position})")
            else:
                bin_id = find_bin_id_by_code(current_code)
                if bin_id is None:
                    print(f"[API] Unknown bin_code for {slot_name}: {current_code}")
                    continue

                upsert_slot_state(
                    organizer_id=organizer_id,
                    position=position,
                    bin_id=bin_id,
                    is_present=True,
                    is_empty=None,
                )
                print(
                    f"[API] Stored {current_code} -> {slot_name} "
                    f"(position={position}, bin_id={bin_id})"
                )

            last_sent_state[slot_name] = current_code

        except requests.RequestException as e:
            print(f"[API ERROR] slot={slot_name} code={current_code} error={e}")


def main():
    # Inicializace kamer
    cams: Dict[int, Picamera2] = {}
    for idx in CAMERA_INDICES:
        print(f"[INIT] Starting camera {idx}...")
        cams[idx] = create_camera(idx)

    # Aktuální stav snímku: slot_name → kód skenu nebo None
    slot_state: Dict[str, Optional[str]] = {
        slot_name: None for slot_name in SLOT_MAPPING.values()
    }

    # Poslední stav úspěšně odeslán do API
    last_sent_state: Dict[str, Optional[str]] = dict(slot_state)

    use_roi = True
    frame_count = 0
    t_prev = time.time()
    last_sync_time = 0.0

    print("[INFO] Running dual-camera QR scanner with API sync.")
    print("       Press 'q' or ESC in any window to quit.")
    print("       Press 'r' in any window to toggle ROI.")

    while True:
        loop_start = time.time()
        frame_count += 1
        t_now = time.time()
        fps = 1.0 / max(1e-6, (t_now - t_prev))
        t_prev = t_now

        # Vynulování – předpokládáme prázdné sloty
        for s in slot_state:
            slot_state[s] = None

        # Pro každou kameru: ziska snímek, detekuje QR kódy a přiřadí sloty
        for cam_idx, picam in cams.items():
            frame_rgb = picam.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            h, w = frame_bgr.shape[:2]

            # Aplikace ROI
            if use_roi:
                rx, ry, rw, rh = ROI_REL
                x0 = int(rx * w)
                y0 = int(ry * h)
                ww = int(rw * w)
                hh = int(rh * h)
                roi = frame_bgr[y0:y0 + hh, x0:x0 + ww]
                cv2.rectangle(frame_bgr, (x0, y0), (x0 + ww, y0 + hh), (0, 255, 0), 2)
            else:
                x0 = 0
                y0 = 0
                roi = frame_bgr

            # Vykreslí hranice sekcí
            third = w // NUM_SECTIONS
            for s in range(1, NUM_SECTIONS):
                x_line = s * third
                cv2.line(frame_bgr, (x_line, 0), (x_line, h), (0, 255, 0), 1)

            if frame_count % DECODE_EVERY_N_FRAMES == 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                results = zbar_decode(gray)

                for r in results:
                    data = r.data.decode("utf-8", errors="ignore").strip()
                    if not data:
                        continue

                    pts = np.array([(p.x, p.y) for p in r.polygon], dtype=np.int32)
                    pts[:, 0] += x0
                    pts[:, 1] += y0

                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))

                    section = cx // third
                    if section < 0:
                        section = 0
                    if section > NUM_SECTIONS - 1:
                        section = NUM_SECTIONS - 1

                    slot_name = SLOT_MAPPING.get((cam_idx, section))

                    cv2.polylines(frame_bgr, [pts], True, (0, 0, 255), 2)
                    cv2.circle(frame_bgr, (cx, cy), 5, (0, 0, 255), -1)

                    if slot_name is not None:
                        slot_state[slot_name] = data
                        slot_last_seen[slot_name] = time.time()
                        label = f"{slot_name}: {data}"
                    else:
                        label = f"cam{cam_idx} sec{section}: {data}"

                    cv2.putText(
                        frame_bgr,
                        label,
                        (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    print(
                        f"[QR] cam={cam_idx}, section={section}, "
                        f"slot={slot_name}, code='{data}'"
                    )

            hud = f"CAM {cam_idx} | FPS {fps:.1f} | ROI {use_roi}"
            cv2.putText(
                frame_bgr,
                hud,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            win_name = f"QR Live - Camera {cam_idx}"
            cv2.imshow(win_name, frame_bgr)

        now = time.time()
        for slot_name in slot_state.keys():
            last = slot_last_seen.get(slot_name, 0)
            if now - last > LAST_SEEN_TIMEOUT:
                slot_state[slot_name] = None

        # Periodická synchronizace API
        if time.time() - last_sync_time >= SYNC_INTERVAL_SECONDS:
            sync_slot_state_to_api(
                organizer_id=ORGANIZER_ID,
                slot_state=slot_state,
                last_sent_state=last_sent_state,
            )
            last_sync_time = time.time()
        
            # Souhrn jednotlivých snímků
            summary_parts = []
            for slot_name in sorted(slot_state.keys()):
                val = slot_state[slot_name]
                if val is None:
                    summary_parts.append(f"{slot_name}: EMPTY")
                else:
                    summary_parts.append(f"{slot_name}: {val}")
            print(" | ".join(summary_parts))

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("r"), ord("R")):
            use_roi = not use_roi
            print(f"[INFO] ROI enabled: {use_roi}")

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("r"), ord("R")):
            use_roi = not use_roi
            print(f"[INFO] ROI enabled: {use_roi}")

        elapsed = time.time() - loop_start
        sleep_time = FRAME_TIME - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cv2.destroyAllWindows()
    for picam in cams.values():
        picam.stop()


if __name__ == "__main__":
    main()
