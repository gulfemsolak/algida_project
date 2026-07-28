# System Architecture

## Overview

The Ice Cream Shelf Detector is a computer-vision pipeline that ingests vending machine shelf photos, runs YOLOv8 object detection, performs shelf-level analysis, and surfaces results through a Streamlit dashboard.

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
│  Image ──► predictor.predict_shelf()                    │
│                │                                        │
│                ▼                                        │
│   { detections, summary, fill_rate, annotated_image }   │
│                │                                        │
│                ▼                                        │
│   shelf_analyzer.analyze_shelf()                        │
│                │                                        │
│                ▼                                        │
│   { rows, shelf_map, restock_needed, anomalies }        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    DASHBOARD                            │
│                                                         │
│  app.py (entry) ──► sidebar (model, conf threshold)     │
│                                                         │
│  Page 1: Shelf Analysis   — upload → annotated image    │
│  Page 2: Model Metrics    — training curves, mAP        │
│  Page 3: Batch Analysis   — multi-image CSV export      │
│  Page 4: Model Comparison — nano/small/medium tradeoff  │
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
| `trainer.py` | Config-driven YOLOv8 training; experiment naming; resume support |
| `evaluator.py` | mAP, precision/recall, counting accuracy; structured JSON report |
| `predictor.py` | Clean API: image → JSON; demo mode when no model available |
| `model_comparison.py` | Benchmarks speed, size, accuracy for all three variants |

### `src/analysis/`
| Module | Responsibility |
|--------|---------------|
| `shelf_analyzer.py` | Row clustering; per-row stats; restock logic; anomaly detection |
| `metrics.py` | Counting accuracy, MACE, fill rate, threshold sweeps |
| `report_generator.py` | Jinja2 → standalone HTML report |

### `dashboard/`
| Module | Responsibility |
|--------|---------------|
| `app.py` | Streamlit entry point; page config |
| `components/sidebar.py` | Model selector; confidence slider; session state |
| `components/charts.py` | Reusable Plotly figures |
| `pages/01_*.py` | Hero page: upload → annotated image + results table |
| `pages/02_*.py` | Training metrics viewer |
| `pages/03_*.py` | Batch processing with CSV export |
| `pages/04_*.py` | Model variant comparison |

## Technology Choices

| Technology | Why |
|-----------|-----|
| **YOLOv8** (Ultralytics) | State-of-the-art single-stage detector; Python-native API; supports n/s/m variants for different deployment targets |
| **Albumentations** | Fastest Python augmentation library; native YOLO bbox support; domain-appropriate transforms (shadow, noise, jitter) |
| **Streamlit** | Fastest path from Python functions to interactive web UI; no front-end code required |
| **Plotly** | Interactive charts; embeds cleanly in Streamlit; HTML export for reports |
| **scikit-learn** | Stratified splitting; metrics utilities |
| **OpenCV** | Image decode/encode; drawing primitives |
| **Jinja2** | Lightweight HTML templating for standalone reports |
