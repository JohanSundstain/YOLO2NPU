import sys
import os
import numpy as np
import cv2

def preprocess_yolov8_detection(image_path, target_size=(640, 640), padding_color = (114, 114, 114)) -> np.ndarray:
    """
    Preprocess an image for YOLOv8 object detection model.
    Steps:
    1. Load image (BGR from cv2, but convert to RGB).
    2. Letterbox resize with padding to target size (keeping aspect ratio).
    3. Normalize pixel values to [0, 1] by dividing by 255.0.
    4. Output shape: (1, H, W, C) - NHWC format, float32.

    Args:
        image_path (str): Path to the input image.
        target_size (Tuple[int, int]): Target size (height, width). Default (640, 640).
        padding_color (Tuple[int, int, int]): Color for padding (RGB). Default (114,114,114).

    Returns:
        np.ndarray: Preprocessed image with shape (1, H, W, C), dtype float32, values in [0,1].
    """
    # Load image with OpenCV (BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert BGR to RGB (YOLOv8 expects RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    orig_h, orig_w = img.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scale and padding (letterbox)
    scale = min(target_h / orig_h, target_w / orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=padding_color
    )
    
    # Normalize to [0, 1]
    normalized = padded.astype(np.float32) / 255.0
    
    # Add batch dimension -> shape (1, H, W, C)
    batch = np.expand_dims(normalized, axis=0)
    
    return batch