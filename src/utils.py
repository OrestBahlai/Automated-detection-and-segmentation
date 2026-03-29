import cv2
import time
import numpy as np


def smooth_mask(mask, frame_shape):
    """Білінійна інтерполяція + бінаризація маски для плавних контурів."""
    if mask is None:
        return None
    mask_resized = cv2.resize(
        mask.astype(np.uint8),
        (frame_shape[1], frame_shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    _, binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
    return binary


def draw_overlay(frame, results, show_labels=False):
    """
    Накладає маски і confidence score на кадр.
    show_labels=False — приховує назви класів (як у дипломі).
    """
    annotated = frame.copy()

    for r in results:
        if r.masks is None:
            continue
        for i, (mask, box) in enumerate(zip(r.masks.data, r.boxes)):
            conf = float(box.conf)
            if conf < 0.3:
                continue

            # Плавна маска
            binary_mask = smooth_mask(
                mask.cpu().numpy(),
                frame.shape[:2]
            )

            # Синій напівпрозорий оверлей
            color_mask = np.zeros_like(frame)
            color_mask[binary_mask > 0] = [255, 100, 50]
            annotated = cv2.addWeighted(annotated, 1.0, color_mask, 0.45, 0)

            # Контур
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated, contours, -1, (255, 100, 50), 2)

            # Тільки confidence score (без назви класу)
            x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
            label = f"{conf:.2f}"
            cv2.rectangle(annotated, (x1, y1 - 22), (x1 + 55, y1), (30, 30, 30), -1)
            cv2.putText(
                annotated, label,
                (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 100), 1, cv2.LINE_AA
            )

    return annotated


def draw_hud(frame, fps, track_id=None):
    """HUD оверлей: заголовок + FPS."""
    h, w = frame.shape[:2]
    cv2.putText(
        frame, "MILITARY VEHICLE DETECTION",
        (24, 44), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (24, 72), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    return frame