import cv2
import time
import argparse
from pathlib import Path
from ultralytics import YOLO
from utils import draw_overlay, draw_hud


def run_detection(
    source: str,
    model_path: str,
    output_path: str = None,
    conf: float = 0.3,
    imgsz: int = 640,
    show: bool = False
):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Не вдалося відкрити: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))

    fps_values = []
    frame_count = 0

    print(f"[INFO] Запуск детекції: {source}")
    print(f"[INFO] Модель: {model_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # Трекінг з ByteTrack
        results = model.track(
            frame,
            imgsz=imgsz,
            conf=conf,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml"
        )

        elapsed = time.perf_counter() - t0
        curr_fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_values.append(curr_fps)
        frame_count += 1

        # Візуалізація
        annotated = draw_overlay(frame, results)
        annotated = draw_hud(annotated, curr_fps)

        if writer:
            writer.write(annotated)
        if show:
            cv2.imshow("Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if fps_values:
        avg = sum(fps_values) / len(fps_values)
        print(f"\n=== Результати ===")
        print(f"Оброблено кадрів : {frame_count}")
        print(f"Середній FPS     : {avg:.1f}")
        print(f"Мін FPS          : {min(fps_values):.1f}")
        print(f"Макс FPS         : {max(fps_values):.1f}")
        if output_path:
            print(f"Збережено        : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  required=True, help="Шлях до відео або 0 для камери")
    parser.add_argument("--model",   default="models/yolov8n-seg_tuned.pt")
    parser.add_argument("--output",  default=None, help="Шлях для збереження відео")
    parser.add_argument("--conf",    type=float, default=0.3)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--show",    action="store_true")
    args = parser.parse_args()

    run_detection(
        source=args.source,
        model_path=args.model,
        output_path=args.output,
        conf=args.conf,
        imgsz=args.imgsz,
        show=args.show
    )