"""
Скрипт для запуску на Raspberry Pi.
Оптимізований для мінімального використання пам'яті і максимального FPS.
"""
import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO


def run_rpi(
    source: str,
    model_path: str,
    conf: float = 0.35,
    imgsz: int = 320,       # Зменшений розмір для RPi
    output_path: str = None,
    warmup_frames: int = 5
):
    print("[RPi] Завантаження моделі...")
    model = YOLO(model_path)

    # Прогрів моделі (перші кадри завжди повільніші)
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(warmup_frames):
        model.predict(dummy, imgsz=imgsz, verbose=False)
    print(f"[RPi] Прогрів завершено ({warmup_frames} кадрів)")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Не вдалося відкрити відео: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))

    fps_log = []
    frame_idx = 0

    print(f"[RPi] Запуск. imgsz={imgsz}, conf={conf}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=imgsz,
            conf=conf,
            verbose=False,
            half=True    
        )

        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_log.append(fps)
        frame_idx += 1

        # Мінімальна візуалізація (легша за plot())
        annotated = frame.copy()
        for r in results:
            if r.masks is not None:
                for mask, box in zip(r.masks.data, r.boxes):
                    if float(box.conf) < conf:
                        continue
                    m = cv2.resize(
                        mask.cpu().numpy().astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_LINEAR
                    )
                    annotated[m > 0] = (
                        annotated[m > 0] * 0.55 +
                        np.array([255, 80, 40]) * 0.45
                    ).astype(np.uint8)

        cv2.putText(
            annotated,
            f"FPS:{fps:.1f}  f:{frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2
        )

        if writer:
            writer.write(annotated)

        # Лог кожні 30 кадрів
        if frame_idx % 30 == 0:
            avg30 = sum(fps_log[-30:]) / 30
            print(f"  кадр {frame_idx:4d} | FPS: {fps:.1f} | avg30: {avg30:.1f}")

    cap.release()
    if writer:
        writer.release()

    print("\n=== Підсумок RPi ===")
    if fps_log:
        # Відкидаємо перші 10 кадрів (розігрів)
        steady = fps_log[10:]
        if steady:
            print(f"Кадрів (стабільних) : {len(steady)}")
            print(f"Середній FPS        : {sum(steady)/len(steady):.2f}")
            print(f"Мін FPS             : {min(steady):.2f}")
            print(f"Макс FPS            : {max(steady):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8-seg на Raspberry Pi")
    parser.add_argument("--source",  required=True)
    parser.add_argument("--model",   default="../models/yolov8n-seg_tuned.pt")
    parser.add_argument("--output",  default=None)
    parser.add_argument("--conf",    type=float, default=0.35)
    parser.add_argument("--imgsz",   type=int, default=320)
    args = parser.parse_args()

    run_rpi(
        source=args.source,
        model_path=args.model,
        output_path=args.output,
        conf=args.conf,
        imgsz=args.imgsz
    )