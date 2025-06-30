# EATLAB - YOLOv12 + OpenCV + DeepSORT Containerized Inference

This project provides a containerized setup for real-time object detection and tracking using:

- **YOLOv12** for object detection  
- **OpenCV** for video processing  
- **DeepSORT** for object tracking  

## 🗂️ Project Structure

<pre> ```text EATLAB/ │ ├── Dockerfile # Docker configuration for building YOLO + OpenCV + DeepSort 
  ├── docker-compose.yml # Docker Compose file for running inference in a container 
  ├── requirements.txt # Python dependencies for the project 
  ├── inference.py # Main script for detection and tracking 
  ├── inference.ipynb # Jupyter notebook for detection and tracking 
  │ 
  ├── yolov12m-cam1/ # Directory for YOLO model trained for camera 1 
  │ └── weights/ 
  │ └── best.pt # Pre-trained YOLOv12 weights for camera 1 
  ├── yolov12m-cam2/ # Directory for YOLO model trained for camera 2 
  │ └── weights/ 
  │ └── best.pt # Pre-trained YOLOv12 weights for camera 2 
  ...
  │ ├── output/ # (Optional) Directory for saving output frames or logs 
  │ └── frames/ # Directory for output video frames 
  │ └── README.md # Project documentation ``` </pre>

  ## 🚀 Getting Started

Make sure Docker and Docker Compose are installed.

### Build and Run the Container

```bash
docker-compose up --build
```
Link checkpoints: https://drive.google.com/drive/folders/1jCxltMfA4IINkmNYp1rrz10AjiX7tbR5?usp=sharing
