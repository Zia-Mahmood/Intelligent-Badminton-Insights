#!/usr/bin/env python3
"""
eval.py: Evaluation script for the Intelligent-Badminton-Insights project.

This script evaluates:
1. YOLOv8 shuttle tracking model on its validation set.
2. YOLOv8 player detection model on its validation set.
3. TensorFlow shot classification model on its validation set.

It prints evaluation metrics for each (mAP for YOLO models, accuracy and confusion matrix for shot classification).
"""
import os
import logging
from pathlib import Path

# YOLOv8 object detection evaluation
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# TensorFlow for shot classification
try:
    import tensorflow as tf
except ImportError:
    tf = None

import numpy as np

# For confusion matrix (optional, fallback implemented if not available)
try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None

def evaluate_yolo_model(weights_path, val_dir, model_name):
    """
    Load a YOLOv8 model from weights and evaluate on the given validation directory.
    weights_path: path to .pt weights file
    val_dir: path to validation images directory
    model_name: string for logging
    """
    weights_path = Path(weights_path)
    val_dir = Path(val_dir)
    # Check if weight and data paths exist
    if not weights_path.exists():
        logging.warning(f"{model_name} weights not found at {weights_path}. Skipping {model_name}.")
        return
    if not val_dir.exists():
        logging.warning(f"{model_name} validation directory not found at {val_dir}. Skipping {model_name}.")
        return
    if YOLO is None:
        logging.error(f"Ultralytics YOLO is not installed. Cannot evaluate {model_name}.")
        return
    logging.info(f"Loading {model_name} model from {weights_path}...")
    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        logging.error(f"Failed to load {model_name} model: {e}")
        return

    # Attempt to detect a data.yaml in the dataset directories
    data_yaml = None
    for parent in [val_dir.parent, val_dir.parent.parent]:
        yaml_path = parent / "data.yaml"
        if yaml_path.exists():
            data_yaml = str(yaml_path)
            logging.info(f"Using data config at {data_yaml} for {model_name}.")
            break

    try:
        if data_yaml:
            # Use data.yaml if found
            metrics = model.val(data=data_yaml)
        else:
            # If no yaml found, attempt to validate directly on images directory (may require manually defining 'data')
            metrics = model.val(data=str(val_dir))
        logging.info(f"{model_name} validation complete.")
        # Print key metrics: e.g. mAP
        try:
            box_metrics = metrics.box
            mAP50_95 = box_metrics.map  # mAP@50-95
            mAP50 = box_metrics.map50
            mAP75 = box_metrics.map75
            print(f"{model_name} mAP50-95: {mAP50_95:.4f}")
            print(f"{model_name} mAP50: {mAP50:.4f}")
            print(f"{model_name} mAP75: {mAP75:.4f}")
        except Exception:
            # Fallback: print full metrics object if attributes not as expected
            print(f"{model_name} evaluation metrics: {metrics}")
    except Exception as e:
        logging.error(f"Evaluation of {model_name} failed: {e}")

def evaluate_shot_classification_model(model_path, val_dir):
    """
    Load a TensorFlow shot classification model and evaluate on validation data.
    Prints overall accuracy and confusion matrix.
    """
    model_path = Path(model_path)
    val_dir = Path(val_dir)
    if not model_path.exists():
        logging.warning(f"Shot classification model not found at {model_path}. Skipping shot classification.")
        return
    if not val_dir.exists():
        logging.warning(f"Shot classification validation data not found at {val_dir}. Skipping shot classification.")
        return
    if tf is None:
        logging.error("TensorFlow is not installed. Cannot evaluate shot classification model.")
        return

    logging.info(f"Loading shot classification model from {model_path}...")
    try:
        model = tf.keras.models.load_model(str(model_path))
    except Exception as e:
        logging.error(f"Failed to load shot classification model: {e}")
        return

    logging.info(f"Loading validation images from {val_dir}...")
    # Assume images organized in subdirectories for each class
    try:
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            str(val_dir),
            labels='inferred',
            label_mode='int',
            image_size=(224, 224),  # Adjust if model expects a different size
            batch_size=32,
            shuffle=False
        )
    except Exception as e:
        logging.error(f"Failed to load validation data: {e}")
        return

    # Evaluate accuracy
    logging.info("Evaluating shot classification model on validation dataset...")
    try:
        loss, accuracy = model.evaluate(val_ds, verbose=0)
        print(f"Shot Classification Accuracy: {accuracy:.4f}")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        accuracy = None

    # Compute confusion matrix
    logging.info("Computing predictions for confusion matrix...")
    try:
        # Gather all true labels
        y_true = []
        for _, labels in val_ds:
            y_true.extend(labels.numpy())
        y_true = np.array(y_true)

        # Gather predicted labels
        y_pred_probs = model.predict(val_ds)
        # Determine predicted classes
        if y_pred_probs.ndim > 1 and y_pred_probs.shape[1] > 1:
            # Multi-class probabilities or logits
            y_pred = np.argmax(y_pred_probs, axis=1)
        else:
            # Binary classification (single output)
            y_pred = (y_pred_probs.ravel() > 0.5).astype(int)

        # Compute confusion matrix
        if confusion_matrix is not None:
            cm = confusion_matrix(y_true, y_pred)
        else:
            # Fallback if sklearn not available
            classes = np.unique(y_true)
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for true_label, pred_label in zip(y_true, y_pred):
                cm[true_label, pred_label] += 1

        print("Confusion Matrix:")
        print(cm)
    except Exception as e:
        logging.error(f"Failed to compute confusion matrix: {e}")

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    base_dir = Path(__file__).resolve().parent

    # Evaluate Shuttle Tracking YOLOv8 model
    shuttle_weights = base_dir / "shuttle_tracking" / "weights" / "shuttle_tracking.pt"
    shuttle_val = base_dir / "shuttle_tracking" / "data_yolo" / "shuttle" / "valid"
    evaluate_yolo_model(shuttle_weights, shuttle_val, "Shuttle Tracking YOLOv8")

    # Evaluate Player Detection YOLOv8 model
    player_weights = base_dir / "player_detection" / "weights" / "player_detection.pt"
    player_val = base_dir / "player_detection" / "data_yolo" / "player" / "valid"
    evaluate_yolo_model(player_weights, player_val, "Player Detection YOLOv8")

    # Evaluate Shot Classification model
    shot_model = base_dir / "shot_classification" / "weights" / "shot_classifier.h5"
    shot_val_dir = base_dir / "shot_classification" / "data" / "val"
    evaluate_shot_classification_model(shot_model, shot_val_dir)

if __name__ == "__main__":
    main()
