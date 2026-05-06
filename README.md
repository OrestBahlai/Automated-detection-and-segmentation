# Automated Detection and Segmentation of Military Vehicles

Бакалаврська кваліфікаційна робота — НУ "Львівська політехніка", 2026.  
Студент: Орест Баглай | Керівник: Наталія Шаховська

Система автоматизованого виявлення та інстанс-сегментації військової техніки
на потокових зображеннях із використанням методів комп'ютерного зору.
Оптимізована для розгортання на бортових системах БПЛА (Raspberry Pi).

---

## Результати

### Метрики точності (тестова вибірка, 101 зображення)

| Метрика | Значення |
|---|:---:|
| mAP@50 (Box) | 0.499 |
| mAP@50-95 (Box) | 0.342 |
| mAP@50 (Mask) | 0.491 |
| mAP@50-95 (Mask) | 0.353 |
| Precision | 0.783 |
| Recall | 0.532 |

### Швидкість інференсу

| Пристрій | imgsz | Середній FPS | Latency (мс) |
|---|:---:|:---:|:---:|
| GPU (сервер, тренування) | 640 | ~128 | ~7.8 |
| Raspberry Pi 5 — CPU | 320 | **7.5** | **133.4** |

> RPi 5: Ubuntu 24.04 LTS (aarch64), PyTorch 2.11.0 CPU.  
> FPS діапазон: 6.9 – 7.6. Протестовано на 200 кадрах.

---

## Структура
.
├── models/                       # Ваги моделей (.pt файли)
├── data/
│   ├── test_videos/              # Тестові відео
│   └── sample_images/            # Тестові зображення
├── src/
│   ├── detect.py                 # Детекція на відео (ПК)
│   ├── rpi_detect.py             # Оптимізовано для Raspberry Pi
│   ├── benchmark.py              # Вимірювання FPS та якості
│   └── utils.py                  # Допоміжні функції
├── notebooks/                    # Jupyter ноутбуки (тренування, аналіз)
├── Mil_Vechicle_Segmentation-2/  # Датасет (Roboflow)
├── requirements.txt
└── requirements_rpi.txt

---

## Швидкий старт

### На ПК

```bash
pip install -r requirements.txt

python src/detect.py \
  --source data/test_videos/test21.MP4 \
  --model  models/yolov8n-seg_tuned.pt \
  --output output_result.mp4 \
  --show
```

### Бенчмарк

```bash
python src/benchmark.py \
  --source data/test_videos/test21.MP4 \
  --model  models/yolov8n-seg_tuned.pt \
  --imgsz  320 \
  --max_frames 200
```

### На Raspberry Pi 5

```bash
# Встановлення PyTorch для ARM64
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_rpi.txt

# Детекція
python src/rpi_detect.py \
  --source data/test_videos/test21.MP4 \
  --model  models/yolov8n-seg_tuned.pt \
  --imgsz  320 \
  --output output_rpi.mp4

# Бенчмарк на RPi
python src/benchmark.py \
  --source data/test_videos/test21.MP4 \
  --model  models/yolov8n-seg_tuned.pt \
  --imgsz  320 \
  --max_frames 200
```

---

## Датасет

Власний датасет `Mil_Vechicle_Segmentation` (2359 зображень, полігональна розмітка).  
Доступний на [Roboflow Universe](https://universe.roboflow.com/detection-fe3wj/mil_vechicle_segmentation-69ull).

| Вибірка | Зображень |
|---|:---:|
| Train | 70% |
| Val | 20% |
| Test | 101 |

Класи: `Tank`, `Military-vehicles`

---

## Завантаження моделей

Натреновані ваги доступні на сторінці [Releases](../../releases).  
Завантаж `yolov8n-seg_tuned.pt` і помісти в папку `models/`.

---

## Ключові технології

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/) — обробка відеопотоку
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — трекінг об'єктів
- [Roboflow](https://roboflow.com/) — анотація датасету