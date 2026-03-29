"""
Бенчмарк: вимірює FPS, latency та якість сегментації.
Виводить детальний звіт для порівняння на різних пристроях.
"""
import cv2
import time
import argparse
import platform
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def benchmark(
    source: str,
    model_path: str,
    imgsz: int = 640,
    conf: float = 0.3,
    max_frames: int = 200,
    warmup: int = 10
):
    print("=" * 50)
    print("  BENCHMARK: Military Vehicle Detection")
    print("=" * 50)
    print(f"  Пристрій  : {platform.node()} ({platform.machine()})")
    print(f"  Модель    : {model_path}")
    print(f"  imgsz     : {imgsz}")
    print(f"  conf      : {conf}")
    print(f"  Кадрів    : {max_frames}")
    print("=" * 50)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Не вдалося відкрити: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n  Відео: {w}x{h}")

    # Прогрів
    print(f"\n[1/3] Прогрів ({warmup} кадрів)...")
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(warmup):
        model.predict(dummy, imgsz=imgsz, verbose=False)

    # Бенчмарк
    print(f"[2/3] Вимірювання ({max_frames} кадрів)...")
    latencies = []
    detections_per_frame = []
    frames_with_detections = 0
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            ret, frame = cap.read()
            if not ret:
                break

        t0 = time.perf_counter()
        results = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)
        latency_ms = (time.perf_counter() - t0) * 1000

        latencies.append(latency_ms)
        n_det = len(results[0].boxes) if results[0].boxes else 0
        detections_per_frame.append(n_det)
        if n_det > 0:
            frames_with_detections += 1

        frame_count += 1

    cap.release()

    # Звіт
    print(f"[3/3] Результати:\n")
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        avg_fps = 1000.0 / avg_lat
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
        min_fps = 1000.0 / max(latencies)
        max_fps = 1000.0 / min(latencies)
        avg_det = sum(detections_per_frame) / len(detections_per_frame)
        det_rate = frames_with_detections / frame_count * 100

        print(f"  {'Метрика':<30} {'Значення':>12}")
        print(f"  {'-'*42}")
        print(f"  {'Середній FPS':<30} {avg_fps:>11.1f}")
        print(f"  {'Мінімальний FPS':<30} {min_fps:>11.1f}")
        print(f"  {'Максимальний FPS':<30} {max_fps:>11.1f}")
        print(f"  {'Середня latency (мс)':<30} {avg_lat:>11.1f}")
        print(f"  {'95-й перцентиль latency':<30} {p95_lat:>11.1f}")
        print(f"  {'Середньо детекцій/кадр':<30} {avg_det:>11.2f}")
        print(f"  {'Кадри з детекціями (%)':<30} {det_rate:>10.1f}%")
        print(f"  {'Оброблено кадрів':<30} {frame_count:>12}")
        print("=" * 50)

        # Оцінка придатності для RPi
        print("\n  Оцінка для Raspberry Pi 4:")
        if avg_fps >= 15:
            print("  [OK] Придатно для реального часу (>= 15 FPS)")
        elif avg_fps >= 5:
            print("  [!]  Повільно, але використовне (5-15 FPS)")
        else:
            print("  [X]  Замало для реального часу (< 5 FPS)")
            print("       Спробуй: --imgsz 160 або конвертацію в ONNX")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/test_videos/test21.MP4")
    parser.add_argument("--model",      default="../models/yolov8n-seg_tuned.pt")
    parser.add_argument("--imgsz",      type=int, default=320)
    parser.add_argument("--conf",       type=float, default=0.3)
    parser.add_argument("--max_frames", type=int, default=200)
    args = parser.parse_args()

    benchmark(
        source=args.source,
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        max_frames=args.max_frames
    )