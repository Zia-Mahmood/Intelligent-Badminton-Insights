#!/usr/bin/env python3
import os
import sys
import logging
import shutil
import glob

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def train_shuttle_tracking():
    logging.info("=== Shuttle Tracking: Preparing dataset ===")
    # Dynamically import prepare_dataset from shuttle_tracking module
    shuttle_module_path = os.path.join(os.getcwd(), "shuttle_tracking")
    dataset_path = os.path.join(shuttle_module_path, "dataset.py")
    spec = __import__('importlib').util.spec_from_file_location("shuttle_dataset", dataset_path)
    shuttle_ds = __import__('importlib').util.module_from_spec(spec)
    spec.loader.exec_module(shuttle_ds)
    prepare_dataset = shuttle_ds.prepare_dataset

    # Define paths for dataset
    root_dir = os.path.join(shuttle_module_path, 'data_co', 'shuttle')
    output_dir = os.path.join(shuttle_module_path, 'data_yolo', 'shuttle')
    # Prepare YOLO-format dataset
    prepare_dataset(root_dir=root_dir, output_dir=output_dir, train=True, rm_files=True)
    prepare_dataset(root_dir=root_dir, output_dir=output_dir, train=False, rm_files=False)
    logging.info("Dataset prepared at '%s'", output_dir)

    logging.info("=== Shuttle Tracking: Training YOLOv8 model ===")
    try:
        from ultralytics import YOLO
    except ImportError as e:
        logging.error("Failed to import YOLO from ultralytics: %s", e)
        sys.exit(1)

    # Create data YAML for YOLO
    data_yaml = os.path.join(output_dir, "data.yaml")
    train_images = os.path.abspath(os.path.join(output_dir, "train", "images"))
    val_images = os.path.abspath(os.path.join(output_dir, "valid", "images"))
    os.makedirs(output_dir, exist_ok=True)
    with open(data_yaml, 'w') as f:
        f.write(f"train: {train_images}\n")
        f.write(f"val: {val_images}\n")
        f.write("nc: 1\n")
        f.write("names: ['shuttle']\n")
    # Instantiate YOLOv8 model (nano version)
    model = YOLO("yolov8n.pt")
    # Train model
    model.train(data=data_yaml, epochs=50, imgsz=640, batch=16, name="shuttle_tracking", project="runs/train")
    # After training, copy best weights to module directory
    run_dirs = glob.glob(os.path.join("runs", "train", "shuttle_tracking*"))
    if run_dirs:
        last_run = sorted(run_dirs, key=os.path.getctime)[-1]
        weights_src = os.path.join(last_run, "weights", "best.pt")
        if os.path.isfile(weights_src):
            weights_dest_dir = os.path.join(shuttle_module_path, "weights")
            os.makedirs(weights_dest_dir, exist_ok=True)
            weights_dest = os.path.join(weights_dest_dir, "shuttle_tracking.pt")
            shutil.copy(weights_src, weights_dest)
            logging.info("Shuttle tracking weights saved to '%s'", weights_dest)
        else:
            logging.error("Best weights not found at '%s'", weights_src)
    else:
        logging.error("Run directory for shuttle tracking not found.")

def train_player_detection():
    logging.info("=== Player Detection: Preparing dataset ===")
    # Dynamically import prepare_dataset from player_detection module
    player_module_path = os.path.join(os.getcwd(), "player_detection")
    dataset_path = os.path.join(player_module_path, "dataset.py")
    spec = __import__('importlib').util.spec_from_file_location("player_dataset", dataset_path)
    player_ds = __import__('importlib').util.module_from_spec(spec)
    spec.loader.exec_module(player_ds)
    prepare_dataset = player_ds.prepare_dataset

    # Define paths for dataset
    root_dir = os.path.join(player_module_path, 'data_coco', 'player')
    output_dir = os.path.join(player_module_path, 'data_yolo', 'player')
    # Prepare YOLO-format dataset
    prepare_dataset(root_dir=root_dir, output_dir=output_dir, train=True, rm_files=True)
    prepare_dataset(root_dir=root_dir, output_dir=output_dir, train=False, rm_files=False)
    logging.info("Dataset prepared at '%s'", output_dir)

    logging.info("=== Player Detection: Training YOLOv8 model ===")
    try:
        from ultralytics import YOLO
    except ImportError as e:
        logging.error("Failed to import YOLO from ultralytics: %s", e)
        sys.exit(1)

    # Create data YAML for YOLO
    data_yaml = os.path.join(output_dir, "data.yaml")
    train_images = os.path.abspath(os.path.join(output_dir, "train", "images"))
    val_images = os.path.abspath(os.path.join(output_dir, "valid", "images"))
    with open(data_yaml, 'w') as f:
        f.write(f"train: {train_images}\n")
        f.write(f"val: {val_images}\n")
        f.write("nc: 1\n")
        f.write("names: ['player']\n")
    model = YOLO("yolov8n.pt")
    model.train(data=data_yaml, epochs=50, imgsz=640, batch=16, name="player_detection", project="runs/train")
    # Copy best weights
    run_dirs = glob.glob(os.path.join("runs", "train", "player_detection*"))
    if run_dirs:
        last_run = sorted(run_dirs, key=os.path.getctime)[-1]
        weights_src = os.path.join(last_run, "weights", "best.pt")
        if os.path.isfile(weights_src):
            weights_dest_dir = os.path.join(player_module_path, "weights")
            os.makedirs(weights_dest_dir, exist_ok=True)
            weights_dest = os.path.join(weights_dest_dir, "player_detection.pt")
            shutil.copy(weights_src, weights_dest)
            logging.info("Player detection weights saved to '%s'", weights_dest)
        else:
            logging.error("Best weights not found at '%s'", weights_src)
    else:
        logging.error("Run directory for player detection not found.")

def train_shot_classification():
    logging.info("=== Shot Classification: Training model ===")
    try:
        import tensorflow as tf
    except ImportError as e:
        logging.error("Failed to import TensorFlow: %s", e)
        sys.exit(1)
    # Define dataset directories
    data_dir = os.path.join(os.getcwd(), "shot_classification", "data")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        logging.error("Shot classification data not found at '%s' or '%s'", train_dir, val_dir)
        return
    img_size = (224, 224)
    batch_size = 32
    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, labels="inferred", label_mode="int",
        image_size=img_size, batch_size=batch_size, shuffle=True)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, labels="inferred", label_mode="int",
        image_size=img_size, batch_size=batch_size, shuffle=False)
    class_names = train_ds.class_names
    num_classes = len(class_names)
    logging.info("Detected %d classes: %s", num_classes, class_names)
    # Build a simple CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=img_size + (3,)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    logging.info("Starting training for shot classification...")
    model.fit(train_ds, validation_data=val_ds, epochs=20)
    # Save the trained model
    weights_dir = os.path.join(os.getcwd(), "shot_classification", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    model_path = os.path.join(weights_dir, "shot_classifier.h5")
    model.save(model_path)
    logging.info("Shot classification model saved to '%s'", model_path)

def main():
    setup_logging()
    train_shuttle_tracking()
    train_player_detection()
    logging.info("=== Court Detection: no training required, skipping. ===")
    logging.info("=== Homography: no training required, skipping. ===")
    train_shot_classification()
    logging.info("Training complete.")

if __name__ == "__main__":
    main()
