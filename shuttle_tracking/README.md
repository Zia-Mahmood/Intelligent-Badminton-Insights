# shuttle Tracking using Computer Vision

## Introduction
This project utilizes computer vision techniques to detect shuttlecocks in badminton games. By leveraging advanced computer vision algorithms, aim to provide accurate detection of shuttlecocks to enhance the analysis of badminton matches.

## Result

Drive Link: https://drive.google.com/file/d/1jM94wHFcorLMqHXbQvx_NmiBxuQHYJGp/view?usp=sharing

## Installation

1. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Download pre-trained models:
    - Download the necessary pre-trained models for player and shuttlecock detection and place them in the `models/` directory.
    link: https://drive.google.com/file/d/1aU7hZcsDHjKtJXmQ1eTLHpMNvXqzYvzn/view?usp=sharing

## Usage
1. Prepare your dataset:
    - Ensure your dataset contains videos or images of badminton matches with annotated player and shuttlecock positions.
    - Organize the dataset into appropriate directories, such as 'train' and 'val' for training and validation data.

2. Train the model:
    - Modify the training configuration in the train.py file according to your dataset.

3. Analyze new videos or images:
    - Use the trained model to detect players and shuttlecocks in badminton matches

4. Result

results.png : https://drive.google.com/file/d/1jEbJgBIJGCQwKIPKo-9p6h5LETMlBCjG/view?usp=sharing
confusion_matrix_normalized.png: https://drive.google.com/file/d/1cCRnIjokXgj6p6zfh-6MN8exoMw6eyIx/view?usp=sharing

## Dataset Preparation
Ensure your dataset contains diverse videos or images of badminton matches, depicting various players, courts, and lighting conditions. Annotated bounding boxes should accurately enclose the players and shuttlecock in the frames.


