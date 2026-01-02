import cv2
import time
import mediapipe as mp

CAM_INDEX = 1

CAP_W, CAP_H, CAP_FPS = 1280, 720, 30
PROCESS_W = 960  # lower to 640 if slow

MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.6

def open_camera(index: int):
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
    ]
    for name, backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            # Request settings (camera may choose closest)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
            cap.set(cv2.CAP_PROP_FPS, CAP_FPS)

            # Warm-up: throw away first frames (important for some webcams)
            ok_any = False
            for _ in range(20):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    ok_any = True
                    break
                time.sleep(0.02)

            if ok_any:
                print(f"Using backend: {name}")
                return cap
            cap.release()

    raise RuntimeError("Could not get stable frames from the camera. Try closing other apps or changing index/backend.")

def resize_keep_aspect(frame, target_w: int):
    h, w = frame.shape[:2]
    if w == target_w:
        return frame, 1.0
    scale = target_w / w
    new_h = int(h * scale)
    small = cv2.resize(frame, (target_w, new_h), interpolation=cv2.INTER_AREA)
    return small, scale

def main():
    cap = open_camera(CAM_INDEX)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    t_prev = time.time()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=1,  # set 0 if you need more speed
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                print("Frame grab failed (camera returned empty frame).")
                time.sleep(0.05)
                continue

            # FPS
            t_now = time.time()
            fps = 1.0 / max(1e-6, (t_now - t_prev))
            t_prev = t_now

            small, scale = resize_keep_aspect(frame, PROCESS_W)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            rgb.flags.writeable = True

            if result.multi_hand_landmarks:
                for hand_lms in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        small,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                # show annotated small image in top-left
                ph, pw = small.shape[:2]
                frame[0:ph, 0:pw] = small

            cv2.putText(frame, f"Cam {CAM_INDEX} | FPS {fps:.1f} | MP width {PROCESS_W}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Hands Live (ESC quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
