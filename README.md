# Automated Detection and Segmentation of Military Vehicles

Бакалаврська кваліфікаційна робота — НУ "Львівська політехніка", 2026.
Студент: Орест Баглай | Керівник: Наталія Шаховська

Система автоматизованого виявлення та інстанс-сегментації військової техніки
на потокових зображеннях із використанням методів комп'ютерного зору.
Оптимізована для розгортання на бортових системах БПЛА (Raspberry Pi, Jetson).

---

## Результати

| Модель | mAP@50 (Mask) | FPS (GPU) | FPS (RPi5) |
|--------|:---:|:---:|:---:|
| YOLOv8n-seg (tuned) | 0.748 | ~128 | - |
| YOLOv8s-seg         | 0.750 | ~120 | -  |
| YOLOv9c-seg         | 0.726 | ~75  | -  |

Класи: `Tank`, `Military-vehicle`

---

## Структура
```
.
├── models/                 # Ваги моделей (.pt файли)
├── data/
│   ├── test_videos/        # Тестові відео
│   └── sample_images/      # Тестові зображення
├── src/
│   ├── detect.py           # Детекція на відео (ПК)
│   ├── rpi_detect.py       # Оптимізовано для Raspberry Pi
│   ├── benchmark.py        # Вимірювання FPS та якості
│   └── utils.py            # Допоміжні функції
├── notebooks/              # Jupyter ноутбуки (тренування, аналіз)
├── Mil_Vechicle_Segmentation-2/  # Датасет (Roboflow)
├── requirements.txt
└── requirements_rpi.txt
```

---

## Швидкий старт

### На ПК
```bash
pip install -r requirements.txt

python src/detect.py \
  --source data/test_videos/tank_test.mp4 \
  --model  models/yolov8n-seg_tuned.pt \
  --output output_result.mp4 \
  --show
```

### Бенчмарк
```bash
python src/benchmark.py \
  --source data/test_videos/tank_test.mp4 \
  --model  models/yolov8n-seg_tuned.pt \
  --imgsz  320 \
  --max_frames 200
```

### На Raspberry Pi
```bash
# Встановлення PyTorch для ARM (RPi 4, Python 3.11)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_rpi.txt

python src/rpi_detect.py \
  --source data/test_videos/tank_test.mp4 \
  --model  models/yolov8n-seg_tuned.pt \
  --imgsz  320 \
  --output output_rpi.mp4
```

---

## Датасет

Власний датасет `Mil_Vechicle_Segmentation` (2359 зображень, полігональна розмітка).
Доступний на [Roboflow Universe](https://universe.roboflow.com/detection-fe3wj/mil_vechicle_segmentation-69ull).

Розподіл: Train 70% / Val 20% / Test 10%

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