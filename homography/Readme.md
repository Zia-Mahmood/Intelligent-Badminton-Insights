# Homography Module

This module performs homography-based court detection and transformation for badminton court analysis.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

Install the required Python libraries:

```bash
pip install numpy opencv-python matplotlib
```

How to Run
Place your input badminton court images in the input_images directory

Run the script: 
```bash
python homography.py
```

The script will:
Detect court lines and corners
Apply homography transformation
Generate visualization outputs
Outputs Generated
The script generates several output images in the output_images directory:

detected_lines.png - Shows the detected court lines
corners.png - Displays the identified court corners
transformed_court.png - Shows the bird's-eye view after homography

Sample Input/Output
Input Image: Input Court

Output with Detected Lines: Detected Lines https://drive.google.com/file/d/1btxcPWqpdTtDXjlTSIZChGEZdr-MYkx0/view?usp=sharing

Homography points: https://drive.google.com/file/d/10YI797PyiYQDzj-T17v2l9zW4WChj_oy/view?usp=sharing