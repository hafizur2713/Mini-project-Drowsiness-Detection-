# Driver Drowsiness Detection Using Computer Vision

## Overview

Driver fatigue is one of the major causes of road accidents worldwide. This project presents a real-time Driver Drowsiness Detection System that monitors a driver's eye movements using Computer Vision techniques and alerts the driver when signs of drowsiness are detected.

The system uses MediaPipe Face Mesh for facial landmark detection, OpenCV for video processing, and the Eye Aspect Ratio (EAR) algorithm to determine whether the driver's eyes are open or closed. When the EAR remains below a predefined threshold for a certain number of consecutive frames, an alarm is triggered to alert the driver.

---

## Features

* Real-time webcam monitoring
* Face and eye landmark detection using MediaPipe Face Mesh
* Eye Aspect Ratio (EAR) calculation
* Drowsiness detection based on eye closure duration
* Audio alarm system using Pygame
* Interactive GUI built with Tkinter
* Real-time status updates
* EAR trend visualization
* Accuracy and confusion matrix monitoring

---

## Technologies Used

### Programming Language

* Python 3.8+

### Libraries and Frameworks

* OpenCV
* MediaPipe Face Mesh
* Tkinter
* NumPy
* Matplotlib
* Pillow (PIL)
* Pygame

---

## System Architecture

1. Capture live video from webcam
2. Detect driver's face
3. Extract eye landmarks
4. Calculate Eye Aspect Ratio (EAR)
5. Compare EAR with threshold value
6. Classify driver as Awake or Drowsy
7. Trigger alarm if drowsiness is detected

---

## Eye Aspect Ratio (EAR)

The Eye Aspect Ratio is used to determine eye openness.

EAR decreases when the eyes are closed and remains higher when the eyes are open.

Threshold used in this project:

* EAR Threshold: 0.21
* Alert Frame Threshold: 30 Frames

If EAR remains below the threshold for more than 30 consecutive frames, the system classifies the driver as drowsy and activates the alarm.

---

## Installation

### Clone Repository

```bash
https://github.com/hafizur2713/Mini-project-Drowsiness-Detection-.git
```

### Install Dependencies

```bash
pip install opencv-python mediapipe matplotlib pillow pygame numpy
```

### Run the Project

```bash
python main.py
```

---

## Project Structure

```text
Driver-Drowsiness-Detection/
│
├── drasines.py
├── audio/
│   └── alert.wav
├── README.md
└── requirements.txt
```

---

## GUI Features

* Live webcam feed
* Current EAR value display
* Awake/Drowsy prediction status
* Start Detection button
* Stop Detection button
* Confusion Matrix
* Accuracy Monitoring
* EAR Trend Graph

---

## Applications

* Smart vehicle safety systems
* Driver monitoring systems
* Transportation industry
* Fleet management
* Road safety research

---

## Future Enhancements

* Yawning detection
* Head pose estimation
* Mobile application support
* Deep Learning-based classification
* Cloud monitoring dashboard
* Multi-face monitoring

---

## Results

The system successfully detects prolonged eye closure and provides real-time alerts to help prevent accidents caused by driver fatigue.

---

## Author

Hafizur Rahman

B.Tech Project – Driver Drowsiness Detection Using Computer Vision
