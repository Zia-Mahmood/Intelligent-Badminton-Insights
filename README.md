# Intelligent Badminton Insights:  Tracking, Detection, and Shot Classification in Singles Badminton

Intelligent Badminton Insights is a computer vision analysis system for badminton matches. It uses five key modules – Shuttle Tracking, Player Detection, Court Detection, Homography, and Shot Classification – to automatically extract insights such as shuttlecock trajectory, player positions, court layout, and shot types from match footage. By integrating deep learning and computer vision techniques, this pipeline enables detailed tactical analysis without manual annotation.

## Modules

### Shuttle Tracking
This module tracks the shuttlecock in each video frame to extract its trajectory and velocity. It uses frame-by-frame detection and filters the detections to form continuous shuttle trajectories.
Dataset: [link] ()
Pretrained Model: [link] (https://drive.google.com/file/d/1aU7hZcsDHjKtJXmQ1eTLHpMNvXqzYvzn/view?usp=sharing)


### Player Detection
This module identifies and localizes each player on the court in every frame using object detection (e.g., YOLOv8). It helps analyze player movements and positions during rallies.
Dataset: [link]
Pretrained Model: [link] (https://drive.google.com/file/d/1pV6VueUnxgr6IRAX0nMpgdWdi3EVimTM/view?usp=drive_link)

### Court Detection
This module detects the badminton court boundaries (lines and corners) in video frames. It uses line and corner detection to find the court geometry, which is essential for mapping coordinates and trajectories.
Dataset: [link] (https://drive.google.com/drive/folders/1iVRcfPzjBlvucUg1qjutB14w5dnmHFLk?usp=sharing)
Pretrained Model: [link](https://drive.google.com/file/d/1ZdcudWXFW0veARuj_iUXlAlNPmA-IzUM/view?usp=sharing)

### Homography
This module computes a perspective transform (homography) to map the detected court view to a standardized top-down layout. It uses classical computer vision techniques (Hough line detection, corner detection) and does not require model training.
Dataset: [link](https://drive.google.com/drive/folders/1iVRcfPzjBlvucUg1qjutB14w5dnmHFLk?usp=sharing)

### Shot Classification
This module classifies the type of badminton shot (e.g., smash, clear, drop, net shot) from the video. It uses deep learning models trained on labeled rallies to recognize shot types based on player movement and shuttle behavior.
Dataset: [[link](https://drive.google.com/file/d/1vvvb-QqAwtLx70e_6_TxcPPwTJk6gsOm/view?usp=sharing)]
Pretrained Model: [link] (https://drive.google.com/file/d/10VXpT1P2-6FON_r2s2pquNOa0nuYtOjU/view?usp=sharing)


## Installation
Ensure you have Python 3.x installed. Install the required Python libraries with pip:
```bash
pip install numpy opencv-python torch torchvision ultralytics scipy mediapipe matplotlib
```
numpy, opencv-python: for numerical operations and image processing.
torch, torchvision: for deep learning model implementation.
ultralytics: for YOLOv8 object detection models.
scipy, matplotlib: for scientific computations and plotting (if needed).
mediapipe: for pose estimation (if used in shot classification).


## Usage
Training: Train all modules (shuttle tracking, player detection, court detection, shot classification) by running:
```bash
python train.py
```
This script will train the models on the provided datasets. (The homography module uses classical CV and does not require training.)
Evaluation: After training, evaluate the models using:
```bash
python eval.py
```
Inference: Perform inference on a single image to detect players, shuttle, and classify the shot:

```bash
python infer.py /path/to/image.jpg
```