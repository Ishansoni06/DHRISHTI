# Drishti: Intelligent Video Understanding System

## Overview

Drishti is an AI-based video analysis system that detects and interprets real-world events from video streams using computer vision.

Currently, it focuses on:

* Object detection using YOLOv8
* Person detection in video frames
* Real-time processing and visualization

This project is the foundation for building advanced systems like:

* Entry / Exit detection
* Interaction detection
* Query-based video search
* Vision-language reasoning (Qwen)

---

## Goal

To build a system where users can input queries like:

"Show when a person enters the room"
"Detect interaction between two people"

And the system intelligently analyzes video to return relevant results.

---

## Tech Stack

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* Git & GitHub

---

## Project Structure

```
DHRISHTI/
│
├── src/
│   └── yolo_detection.py        # Main detection script
│
├── data/                        # Input videos (NOT uploaded to GitHub)
│
├── outputs/                     # Output files (future use)
│
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
└── .gitignore                   # Ignored files
```

---

## Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/Ishansoni06/DHRISHTI.git
cd DHRISHTI
```

---

### 2. Install Dependencies

```
pip install -r requirements.txt
```

OR manually:

```
pip install ultralytics opencv-python numpy
```

---

### 3. Add Input Video

Place your video file inside:

```
data/videos/
```

Example:

```
data/videos/sample.mp4
```

---

### 4. Update Video Path (if needed)

Open `src/yolo_detection.py` and ensure:

```
cap = cv2.VideoCapture("data/videos/sample.mp4")
```

---

### 5. Run the Project

```
python src/yolo_detection.py
```

---

## Output

* A video window will open
* Detected objects will have bounding boxes
* "Person" detections will be highlighted
* Terminal may print detection info

---

## Features (Current)

* Object detection using YOLOv8
* Person detection
* Real-time frame processing
* Bounding box visualization

---

## Features (Planned)

* Entry / Exit detection
* Multi-person tracking
* Interaction detection
* Query-based filtering
* Qwen AI integration

---

## Important Notes

* Large files (videos, models) are ignored using `.gitignore`
* Do NOT upload videos >100MB to GitHub
* First run will download YOLO model automatically
