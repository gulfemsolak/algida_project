# System Architecture

## Overview

The Ice Cream Shelf Detector is a computer-vision pipeline that ingests vending machine shelf photos, runs YOLOv8m object detection, aligns detections to the shelf's physical slots via homography, performs shelf-level analysis, and surfaces results through a Streamlit dashboard ("IceVision").

## Pipeline Diagram

```
Raw Photos
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                        │
│                                                         │
│  scraper.py ──► data/raw/       (prototype images)      │
│  augmentor.py ──► data/augmented/  (5× per image)       │
│  validator.py ──► validation report (JSON)              │
│  dataset_builder.py ──► data/splits/ (70/20/10)         │
│                    + config/dataset.yaml                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                  TRAINING PIPELINE                      │
│                                                         │
│  config/config.yaml ──► trainer.py ──► runs/<exp>/      │
│                             │           weights/best.pt │
│                             │                           │
│                         evaluator.py                    │
│                             │                           │
│                    mAP, counting accuracy               │
│                    confusion matrix (PNG)               │
│                    evaluation_report.json               │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                 INFERENCE PIPELINE                      │
│                                                         │
│  Image ──► orientation.py (EXIF 90° normalize)          │
│         ──► skew_corrector.py (canonical orientation)   │
│         ──► predictor.predict_shelf() [YOLOv8m]          │
│                │                                        │
│                ▼                                        │
│   { detections, summary, fill_rate, annotated_image }   │
│                │                                        │
│                ▼                                        │
│   shelf_grid.py / product_anchored_grid.py               │
│   (tray-corner homography → equal-pitch lanes)           │
│                │                                        │
│                ▼                                        │
│   slot_assigner.py (rectified-space column assignment)   │
│                │                                        │
│                ▼                                        │
│   shelf_analyzer.analyze_shelf()                        │
│   (+ qr_reader.py — optional shelf-id verification)       │
│                │                                        │
│                ▼                                        │
│   { rows, shelf_map, restock_needed, anomalies }        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    DASHBOARD ("IceVision")               │
│                                                         │
│  app.py (entry) ──► Genel Bakış: SQLite KPI'lar          │
│  components/sidebar.py — model/conf seçimi                │
│                                                         │
│  pages/01_shelf_analysis.py  — tekli yükleme + anotasyon  │
│  pages/02_batch_analysis.py  — çoklu yükleme + CSV         │
│  pages/_03_restock_planner.py — dolum planlama             │
│    (dosya adı "_" ile başladığı için nav'da şu an gizli)   │
│                                                         │
│  db.py ──► SQLite (data/analysis_history.db)              │
└─────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### `src/data/`
| Module | Responsibility |
|--------|---------------|
| `scraper.py` | Prototype image collection via DuckDuckGo (rate-limited) |
| `augmentor.py` | Albumentations pipeline; transforms images + YOLO labels together |
| `dataset_builder.py` | Stratified 70/20/10 split; writes `dataset.yaml` |
| `validator.py` | Scans for corrupt images, missing labels, invalid boxes, class imbalance |

### `src/models/`
| Module | Responsibility |
|--------|---------------|
| `trainer.py` | Config-driven YOLOv8 training (`yolov8n`/`yolov8s`/`yolov8m`); experiment naming; resume support |
| `evaluator.py` | mAP, precision/recall, counting accuracy; structured JSON report |
| `predictor.py` | Clean API: image → JSON; demo mode when no model weights are available |
| `model_comparison.py` | Benchmarks speed, size, accuracy across the nano/small/medium variants |

### `src/analysis/`
| Module | Responsibility |
|--------|---------------|
| `orientation.py` | EXIF-based 90° rotation normalization |
| `skew_corrector.py` | Canonical-orientation correction |
| `shelf_grid.py` | Tray-corner homography → rectified shelf grid |
| `product_anchored_grid.py` | Identity-independent, equal-pitch lane detection |
| `slot_assigner.py` | Assigns detections to shelf/column slots in rectified space |
| `shelf_analyzer.py` | Row clustering; per-row stats; restock logic; anomaly detection |
| `metrics.py` | Counting accuracy, MACE, fill rate, threshold sweeps |
| `report_generator.py` | Jinja2 → standalone HTML report |

### `dashboard/`
| Module | Responsibility |
|--------|---------------|
| `app.py` | Streamlit entry point ("Genel Bakış"); SQLite-backed KPIs |
| `db.py` | SQLite persistence for analysis history |
| `theme.py` | Theme/style helpers |
| `components/sidebar.py` | Model selector; confidence slider; session state |
| `components/charts.py` | Reusable Plotly figures |
| `components/widgets.py` | Shared UI widgets (priority pills, shelf-map grid) |
| `components/analyses_list.py` | Recent-analyses list widget |
| `pages/01_shelf_analysis.py` | Single-shelf upload → annotated image + results table |
| `pages/02_batch_analysis.py` | Batch processing with CSV export |
| `pages/_03_restock_planner.py` | Restock priority planning (hidden from nav — filename starts with `_`) |

### Project root
| Module | Responsibility |
|--------|---------------|
| `qr_reader.py` | Layered QR decoder (zxing-cpp → WeChat → cv2.QRCodeDetector) for shelf-id verification |

## Technology Choices

| Technology | Why |
|-----------|-----|
| **YOLOv8** (Ultralytics) | State-of-the-art single-stage detector; Python-native API; supports n/s/m variants for different deployment targets — deployed weight is `yolov8m` |
| **Albumentations** | Fastest Python augmentation library; native YOLO bbox support; domain-appropriate transforms (shadow, noise, jitter) |
| **Streamlit** | Fastest path from Python functions to interactive web UI; no front-end code required |
| **Plotly** | Interactive charts; embeds cleanly in Streamlit; HTML export for reports |
| **scikit-learn** | Stratified splitting; metrics utilities |
| **OpenCV** | Image decode/encode; drawing primitives |
| **Jinja2** | Lightweight HTML templating for standalone reports |
