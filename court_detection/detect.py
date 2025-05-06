import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import combinations

def compute_intersection(l1, l2):
    """Intersect two lines given as (x1,y1,x2,y2). Returns (x,y) or None."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return int(px), int(py)

def detect_lines_and_corners(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found.")

    # Convert to HSV and mask green
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 50, 50])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Edge detection
    edges = cv2.Canny(mask, 50, 150)

    # Line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=10)
    if lines is None:
        raise RuntimeError("No lines found.")

    # Convert to list of tuples
    lines = [tuple(l[0]) for l in lines]

    # Draw lines on a copy of the image
    line_img = image.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Compute intersections
    intersections = []
    h, w = image.shape[:2]
    for l1, l2 in combinations(lines, 2):
        pt = compute_intersection(l1, l2)
        if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
            intersections.append(pt)

    if len(intersections) < 4:
        raise RuntimeError(f"Only {len(intersections)} intersections found; need ≥4.")

    intersections = np.array(intersections)

    # Identify 4 extreme corners
    s = intersections[:, 0] + intersections[:, 1]
    d = intersections[:, 0] - intersections[:, 1]
    tl = tuple(intersections[np.argmin(s)])
    br = tuple(intersections[np.argmax(s)])
    tr = tuple(intersections[np.argmax(d)])
    bl = tuple(intersections[np.argmin(d)])
    corners = [tl, tr, br, bl]
    labels = ['TL', 'TR', 'BR', 'BL']

    # Print corners
    for (x, y), lab in zip(corners, labels):
        print(f"{lab} corner: ({x}, {y})")

    # Display line image
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Lines and Corners (not marked)")
    plt.axis('off')
    plt.savefig('output_detect.png')
    plt.show()

# Example usage
detect_lines_and_corners("./input.jpg")
