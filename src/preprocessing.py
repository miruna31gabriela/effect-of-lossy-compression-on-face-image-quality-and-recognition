import os
from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

# Set up paths
DATASET_PATH = Path(r"C:\Users\Miruna\Desktop\Disertatie\Proiect\.ProjMihai\FERET")
REFERENCE_PATH = DATASET_PATH / "reference"
PROBE_PATH = DATASET_PATH / "probe"
REFERENCE_ROI_PATH = DATASET_PATH / "reference_roi"
PROBE_ROI_PATH = DATASET_PATH / "probe_roi"

# Create output directories if they don't exist
REFERENCE_ROI_PATH.mkdir(exist_ok=True)
PROBE_ROI_PATH.mkdir(exist_ok=True)

# Initialize InsightFace RetinaFace-50
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


def crop_and_save(img_path, save_path, roi_size=250, scale=1.05):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Could not read {img_path}")
        return False
    faces = app.get(img)
    if len(faces) == 0:
        print(f"No face detected in {img_path}")
        return False
    # Use the largest face (in case of multiple detections)
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    x1, y1, x2, y2 = map(int, face.bbox)
    w_box = x2 - x1
    h_box = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    # Expand the bounding box by the scale factor
    new_size = int(max(w_box, h_box) * scale)
    half = new_size // 2
    h, w = img.shape[:2]
    left = max(cx - half, 0)
    top = max(cy - half, 0)
    right = min(cx + half, w)
    bottom = min(cy + half, h)
    cropped = img[top:bottom, left:right]
    # If the crop is not exactly new_size, pad it
    if cropped.shape[0] != new_size or cropped.shape[1] != new_size:
        cropped = cv2.copyMakeBorder(
            cropped,
            top=max(0, half - (cy - top)),
            bottom=max(0, (cy + half) - bottom),
            left=max(0, half - (cx - left)),
            right=max(0, (cx + half) - right),
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
    # Resize to roi_size x roi_size
    cropped = cv2.resize(cropped, (roi_size, roi_size))
    cv2.imwrite(str(save_path), cropped)
    return True


def process_folder(input_folder, output_folder):
    images = list(input_folder.glob("*.png"))
    for img_path in tqdm(images, desc=f"Processing {input_folder.name}"):
        save_path = output_folder / img_path.name
        crop_and_save(img_path, save_path)

if __name__ == "__main__":
    print("Processing reference images...")
    process_folder(REFERENCE_PATH, REFERENCE_ROI_PATH)
    print("Processing probe images...")
    process_folder(PROBE_PATH, PROBE_ROI_PATH)
    print("Done! ROI images saved in:")
    print(f"- {REFERENCE_ROI_PATH}")
    print(f"- {PROBE_ROI_PATH}")
