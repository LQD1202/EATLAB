EATLAB/
│
├── Dockerfile              # Docker configuration for building YOLO + OpenCV + DeepSort
├── docker-compose.yml      # Docker Compose file for running inference in a container
├── requirements.txt        # Python dependencies for the project
├── inference.py            # Main script for detection and tracking
├── inference.ipynb         # Jupyter notebook for detection and tracking
│
├── yolov12m-cam1/         # Directory for YOLO model trained for camera 1
│   └── weights/
│       └── best.pt        # Pre-trained YOLOv12 weights for camera 1
├── yolov12m-cam2/         # Directory for YOLO model trained for camera 2
│   └── weights/
│       └── best.pt        # Pre-trained YOLOv12 weights for camera 2
│...
├── output/                # (Optional) Directory for saving output frames or logs
│   └── frames/            # Directory for output video frames
│
└── README.md              # Project documentation

```bash
docker-compose up --build
