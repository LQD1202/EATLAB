# 🍕 EATLAB - Pizza Detection & Tracking bằng YOLOv8 + DeepSort

Hệ thống này sử dụng YOLOv12 và DeepSort để phát hiện và theo dõi pizza trong video giám sát. Triển khai dễ dàng bằng Docker Compose.

---

## 📂 Cấu trúc thư mục

EATLAB/
│
├── Dockerfile # Docker build YOLO + OpenCV + DeepSort
├── docker-compose.yml # Chạy inference qua container
├── requirements.txt # Các thư viện Python cần cài
├── inference.py # Script chính để detect & track
├── inference.ipynb # Script chính để detect & track
│
├── yolov12m-cam1/ # Thư mục chứa model YOLO huấn luyện (ví dụ)
│ └── yolov12m-cam13/
│ └── weights/
│ └── best.pt
├── yolov12m-cam2/ # Thư mục chứa model YOLO huấn luyện (ví dụ)
│ └── yolov12m-cam26/
│ └── weights/
│ └── best.pt
...
│
├── output/ # (tuỳ chọn) Frame hoặc log output
│ └── frames/
│
└── README.md # Tài liệu này

---

## 🚀 Hướng dẫn chạy

### 1. Build và chạy container

```bash
docker-compose up --build
