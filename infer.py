import os
import argparse
import logging

import cv2
import numpy as np

# Attempt to import ultralytics YOLO for object detection
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Attempt to import TensorFlow for shot classification
try:
    import tensorflow as tf
except ImportError:
    tf = None

def load_models(player_model_path, shuttle_model_path, shot_model_path):
    """Load the YOLO and TensorFlow models from the given paths."""
    if YOLO is None:
        raise ImportError("Ultralytics YOLO package is not installed.")
    player_model = YOLO(player_model_path)
    shuttle_model = YOLO(shuttle_model_path)
    if tf is None:
        raise ImportError("TensorFlow is not installed.")
    shot_model = tf.keras.models.load_model(shot_model_path)
    return player_model, shuttle_model, shot_model

def run_detection(model, image_path):
    """
    Run detection on an image using a YOLO model.
    Returns the first Results object for the image.
    """
    results = model.predict(source=image_path, save=False)
    # The results is a list (one per image). Return the first result.
    if isinstance(results, list):
        return results[0]
    return results

def draw_bboxes_and_labels(image, result, class_names, box_color=(0,255,0)):
    """
    Draw bounding boxes and labels on the image for all detections in result.
    Returns a list of detection info (label, conf, x1, y1, x2, y2) and the annotated image.
    """
    detections = []
    annotated = image.copy()
    if hasattr(result, 'boxes'):
        try:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
        except Exception:
            # Fallback in case attributes need list conversion
            coords = result.boxes.xyxy.tolist()
            boxes = np.array(coords).astype(int)
            classes = np.array(result.boxes.cls.tolist()).astype(int)
            confs = np.array(result.boxes.conf.tolist()).astype(float)

        for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = class_names[cls] if class_names and cls < len(class_names) else str(cls)
            detections.append((label, float(conf), x1, y1, x2, y2))
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            # Prepare label text with a colored background for readability
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - int(1.5 * th)), (x1 + tw, y1), box_color, -1)
            cv2.putText(annotated, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return detections, annotated

def run_shot_classification(model, image, target_size=(224, 224)):
    """
    Run shot classification on the given image array.
    Returns the predicted class index and the raw probability vector.
    """
    # Resize and normalize image
    resized = cv2.resize(image, target_size)
    img_norm = resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)
    preds = model.predict(img_batch)
    # Handle output shape
    if preds.ndim == 2:
        probs = preds[0]
    else:
        probs = preds
    class_idx = int(np.argmax(probs))
    return class_idx, probs

def main():
    parser = argparse.ArgumentParser(
        description="Perform player & shuttle detection and shot classification on an input image."
    )
    parser.add_argument("image_path", help="Path to the input image file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    image_path = args.image_path
    if not os.path.isfile(image_path):
        logging.error(f"Input image {image_path} does not exist.")
        return

    # Define model weight file paths
    player_model_path = os.path.join("player_detection", "weights", "player_detection.pt")
    shuttle_model_path = os.path.join("shuttle_tracking", "weights", "shuttle_tracking.pt")
    shot_model_path = os.path.join("shot_classification", "weights", "shot_classifier.h5")

    # Verify that weight files exist
    if not os.path.isfile(player_model_path):
        logging.error(f"Player detection model not found at {player_model_path}")
        return
    if not os.path.isfile(shuttle_model_path):
        logging.error(f"Shuttle tracking model not found at {shuttle_model_path}")
        return
    if not os.path.isfile(shot_model_path):
        logging.error(f"Shot classification model not found at {shot_model_path}")
        return

    logging.info("Loading models...")
    try:
        player_model, shuttle_model, shot_model = load_models(
            player_model_path, shuttle_model_path, shot_model_path)
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        return
    logging.info("Models loaded successfully.")

    # Load the input image once with OpenCV (BGR format)
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not read image {image_path}")
        return

    # Perform player detection
    logging.info("Detecting players...")
    result_player = run_detection(player_model, image_path)
    player_dets, img_players = draw_bboxes_and_labels(
        image, result_player, player_model.names, box_color=(0, 255, 0))
    if player_dets:
        logging.info("Player Detections:")
        for label, conf, x1, y1, x2, y2 in player_dets:
            logging.info(f"  {label}: confidence={conf:.2f}, bbox=[{x1}, {y1}, {x2}, {y2}]")
    else:
        logging.info("No players detected.")

    # Perform shuttle detection on the image annotated with players
    logging.info("Detecting shuttlecock...")
    result_shuttle = run_detection(shuttle_model, image_path)
    shuttle_dets, img_all = draw_bboxes_and_labels(
        img_players, result_shuttle, shuttle_model.names, box_color=(0, 0, 255))
    if shuttle_dets:
        logging.info("Shuttle Detections:")
        for label, conf, x1, y1, x2, y2 in shuttle_dets:
            logging.info(f"  {label}: confidence={conf:.2f}, bbox=[{x1}, {y1}, {x2}, {y2}]")
    else:
        logging.info("No shuttlecock detected.")

    # Perform shot classification on the original image
    logging.info("Classifying shot type...")
    shot_idx, shot_probs = run_shot_classification(shot_model, image)
    # Define shot class names (adjust based on your model's training)
    shot_classes = ["Drop Shot", "Net Shot", "Clear Shot", "Serve", "Smash"]
    shot_type = shot_classes[shot_idx] if shot_idx < len(shot_classes) else str(shot_idx)
    logging.info(f"Predicted shot type: {shot_type} (class {shot_idx})")

    # Overlay the shot type label on the final image (with detections)
    annotated_image = img_all.copy()
    label_text = f"Shot: {shot_type}"
    # Draw black outline for better contrast
    cv2.putText(annotated_image, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    cv2.putText(annotated_image, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Ensure outputs directory exists and save the result image
    os.makedirs("outputs", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join("outputs", f"{base_name}_inference.jpg")
    cv2.imwrite(output_path, annotated_image)
    logging.info(f"Annotated result saved to {output_path}")

if __name__ == "__main__":
    main()
