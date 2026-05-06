# Automated Detection and Segmentation of Military Vehicles

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n--seg-success)](https://github.com/ultralytics/ultralytics)
[![Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-red)](https://www.raspberrypi.com/)

Бакалаврська кваліфікаційна робота — **НУ "Львівська політехніка" (2026)**.  
**Студент:** Орест Баглай | **Керівник:** Наталія Шаховська

## 📝 Опис проєкту
Система автоматизованого виявлення та інстанс-сегментації військової техніки на потокових зображеннях. Проєкт оптимізовано для розгортання на бортових системах БПЛА з обмеженими ресурсами (зокрема **Raspberry Pi 5**).

---

## 📊 Результати та метрики

### Точність (Тестова вибірка: 101 зображення)
| Метрика | Box (Детекція) | Mask (Сегментація) |
|:--- |:---:|:---:|
| **mAP@50** | 0.499 | 0.491 |
| **mAP@50-95** | 0.342 | 0.353 |
| **Precision** | 0.783 | — |
| **Recall** | 0.532 | — |

### Швидкість інференсу (Performance)
| Пристрій | Роздільна здатність (imgsz) | Середній FPS | Latency (мс) |
|:--- |:---:|:---:|:---:|
| GPU (Server) | 640 | ~128 | ~7.8 |
| **Raspberry Pi 5 (CPU)** | 320 | **7.5** | **133.4** |

> **Примітка:** На RPi 5 використовувалася ОС Ubuntu 24.04 LTS (aarch64) та PyTorch 2.11.0 CPU. FPS стабільний у діапазоні 6.9 – 7.6.

---

## 📂 Структура проєкту
```text
.
├── models/               # Натреновані ваги моделей (.pt)
├── data/
│   ├── test_videos/      # Тестові відеоматеріали
│   └── sample_images/    # Приклади зображень для тестів
├── src/                  # Вихідний код системи
│   ├── detect.py         # Детекція та сегментація (Desktop/GPU)
│   ├── rpi_detect.py     # Скрипт, оптимізований для Raspberry Pi
│   ├── benchmark.py      # Вимірювання продуктивності (FPS, Latency)
│   └── utils.py          # Допоміжні функції та візуалізація
├── notebooks/            # Jupyter ноутбуки для тренування та аналізу
├── requirements.txt      # Залежності для стандартних систем
└── requirements_rpi.txt  # Специфічні залежності для Raspberry Pi (ARM64)
```

---

## 🚀 Швидкий старт

### 🖥️ Використання на ПК
```bash
# Встановлення залежностей
pip install -r requirements.txt

# Запуск детекції на відео
python src/detect.py \
  --source data/test_videos/test21.MP4 \
  --model models/yolov8n-seg_tuned.pt \
  --output output_result.mp4 \
  --show
```

### 🍓 Використання на Raspberry Pi 5
```bash
# Встановлення PyTorch для ARM64
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install -r requirements_rpi.txt

# Запуск оптимізованої детекції
python src/rpi_detect.py \
  --source data/test_videos/test21.MP4 \
  --model models/yolov8n-seg_tuned.pt \
  --imgsz 320 \
  --output output_rpi.mp4
```

### 📈 Тестування продуктивності
```bash
python src/benchmark.py \
  --source data/test_videos/test21.MP4 \
  --model models/yolov8n-seg_tuned.pt \
  --imgsz 320 \
  --max_frames 200
```

---

## 🖼️ Датасет
Використовується власний набір даних **Mil_Vechicle_Segmentation**:
* **Обсяг:** 2359 зображень з полігональною розміткою.
* **Розподіл:** Train (70%), Val (20%), Test (101 зображення).
* **Класи:** `Tank`, `Military-vehicles`.
* **Доступ:** [Roboflow Universe](https://universe.roboflow.com/detection-fe3wj/mil_vechicle_segmentation-69ull).

---

## 🛠️ Технологічний стек
* **Архітектура:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (Instance Segmentation).
* **Відеообробка:** OpenCV.
* **Трекінг:** ByteTrack.
* **Анотування:** Roboflow.

---

## 📥 Завантаження моделей
Готові ваги `yolov8n-seg_tuned.pt` доступні в розділі **[Releases](../../releases)**. Завантажте їх та помістіть у папку `models/` перед стартом.