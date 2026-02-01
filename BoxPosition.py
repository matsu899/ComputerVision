import time
from typing import Optional, Dict, Tuple

import cv2
import numpy as np
from picamera2 import Picamera2
from pyzbar.pyzbar import decode as zbar_decode
from libcamera import controls

# ---------- CONFIG ----------

# Which cameras to use (Picamera2 indices)
CAMERA_INDICES = [0, 1]  # 0 = first camera port, 1 = second

# Resolution
WIDTH, HEIGHT = 1920, 1080

# Number of horizontal sections (slots) per camera
NUM_SECTIONS = 3

# Decode performance tuning
DECODE_EVERY_N_FRAMES = 3

# ROI (relative) – here we only crop vertically so width remains full
# (rx, ry, rw, rh) where values are 0..1 relative to full frame
ROI_REL = (0.0, 0.15, 1.0, 0.70)  # full width, crop top/bottom

# Mapping (camera_index, section_index) -> logical slot name
# Adjust these names to whatever makes sense for your organizer
SLOT_MAPPING: Dict[Tuple[int, int], str] = {
    (0, 0): "R1C1",
    (0, 1): "R1C2",
    (0, 2): "R1C3",
    (1, 0): "R2C1",
    (1, 1): "R2C2",
    (1, 2): "R2C3",
}

# ----------------------------


def create_camera(cam_idx: int) -> Picamera2:
    """Create and configure a Picamera2 instance for given index."""
    picam = Picamera2(camera_num=cam_idx)
    config = picam.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()

    # Basic autofocus and sharpening (like in your original code)
    picam.set_controls(
        {
            "AfMode": controls.AfModeEnum.Continuous,
            "Sharpness": 1.5,
            "Contrast": 1.1,
        }
    )

    time.sleep(0.5)  # small warm-up
    return picam


def main():
    # --- Init cameras ---
    cams: Dict[int, Picamera2] = {}
    for idx in CAMERA_INDICES:
        print(f"[INIT] Starting camera {idx}...")
        cams[idx] = create_camera(idx)

    # Pre-build a slot -> state dict
    # Example: {"R1C1": None, "R1C2": None, ...}
    slot_state: Dict[str, Optional[str]] = {
        slot_name: None for slot_name in SLOT_MAPPING.values()
    }

    use_roi = True  # start with ROI on
    frame_count = 0
    t_prev = time.time()

    print("[INFO] Running dual-camera QR scanner.")
    print("       Press 'q' or ESC in any window to quit.")
    print("       Press 'r' in any window to toggle ROI.")

    while True:
        frame_count += 1
        t_now = time.time()
        fps = 1.0 / max(1e-6, (t_now - t_prev))
        t_prev = t_now

        # Reset occupancy for this frame: assume all slots empty
        for s in slot_state:
            slot_state[s] = None

        # For each camera: grab frame, detect QRs, assign slots
        for cam_idx, picam in cams.items():
            frame_rgb = picam.capture_array()  # RGB888
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            h, w = frame_bgr.shape[:2]

            # Decide ROI
            if use_roi:
                rx, ry, rw, rh = ROI_REL
                # NOTE: rx=0, rw=1.0 so we keep full width for section mapping
                x0 = int(rx * w)
                y0 = int(ry * h)
                ww = int(rw * w)
                hh = int(rh * h)
                roi = frame_bgr[y0 : y0 + hh, x0 : x0 + ww]
                cv2.rectangle(frame_bgr, (x0, y0), (x0 + ww, y0 + hh), (0, 255, 0), 2)
            else:
                x0 = 0
                y0 = 0
                roi = frame_bgr

            # Draw section borders (on full frame)
            third = w // NUM_SECTIONS
            for s in range(1, NUM_SECTIONS):
                x_line = s * third
                cv2.line(frame_bgr, (x_line, 0), (x_line, h), (0, 255, 0), 1)

            # Decode only on every N-th frame (global)
            if frame_count % DECODE_EVERY_N_FRAMES == 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                results = zbar_decode(gray)

                for r in results:
                    # QR payload
                    data = r.data.decode("utf-8", errors="ignore").strip()
                    if not data:
                        continue

                    # Polygon points from pyzbar (relative to ROI)
                    pts = np.array([(p.x, p.y) for p in r.polygon], dtype=np.int32)
                    # Shift back to full-frame coordinates
                    pts[:, 0] += x0
                    pts[:, 1] += y0

                    # Center of QR
                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))

                    # Which section horizontally?
                    section = cx // third
                    if section < 0:
                        section = 0
                    if section > NUM_SECTIONS - 1:
                        section = NUM_SECTIONS - 1

                    # Map to logical slot
                    slot_name = SLOT_MAPPING.get((cam_idx, section))

                    # Draw QR bbox and label on frame
                    cv2.polylines(frame_bgr, [pts], True, (0, 0, 255), 2)
                    cv2.circle(frame_bgr, (cx, cy), 5, (0, 0, 255), -1)

                    if slot_name is not None:
                        slot_state[slot_name] = data
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
                        f"[QR] cam={cam_idx}, section={section}, slot={slot_name}, code='{data}'"
                    )

            # Put basic HUD info
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

            # Show each camera in its own window
            win_name = f"QR Live - Camera {cam_idx}"
            cv2.imshow(win_name, frame_bgr)

        # Print simple occupied / empty overview per frame
        summary_parts = []
        # Iterate in a stable order: sorted by slot name
        for slot_name in sorted(slot_state.keys()):
            val = slot_state[slot_name]
            if val is None:
                summary_parts.append(f"{slot_name}: EMPTY")
            else:
                summary_parts.append(f"{slot_name}: {val}")
        print(" | ".join(summary_parts))

        # Key handling: any window
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):  # ESC or 'q'
            break
        elif key in (ord("r"), ord("R")):
            use_roi = not use_roi
            print(f"[INFO] ROI enabled: {use_roi}")

    # Cleanup
    cv2.destroyAllWindows()
    for picam in cams.values():
        picam.stop()


if __name__ == "__main__":
    main()