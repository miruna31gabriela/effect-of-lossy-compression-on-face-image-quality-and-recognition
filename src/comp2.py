import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deepface import DeepFace
from collections import defaultdict
import cv2

# --- STEP 1: CONFIGURATION & PATHS ---

# Path to the folder containing 'probe' and 'reference' subfolders
BASE_PATH = r"C:\Users\Miruna\Desktop\Disertatie\Proiect\.ProjMihai\FERET\png-resized"

# Define the final paths
PROBE_PATH = os.path.join(BASE_PATH, "probe")
REFERENCE_PATH = os.path.join(BASE_PATH, "reference")

# Define the categories for the X-axis
TARGET_SIZE_CATEGORIES = ['5KB', '4.5KB', '4KB', '3.5KB', '3KB', '2.5KB', '2.2KB']

# Define the Face Recognition model for utility metrics
FR_MODEL = "ArcFace"

# --- DEFINE ALL METRICS TO BE TESTED ---
# All FIQA models should be called via DeepFace.analyze for quality.
FIQA_MODELS_TO_TEST = [
    # Removing FIQA models as they are not supported in this deepface version
    # 'CR-FIQA(S)',
    # 'CR-FIQA(L)',
    # 'MagFace',
    # 'SER-FIQ',
    # 'FaceQnet-v0',
    # 'FaceQnet-v1'
]

# Combine all metrics into a single list for easier processing
ALL_METRICS = [
    'Mated-self',     # Utility metric
    'Mated-other',    # Utility metric
    'Sharpness-1',    # Basic quality (full image)
    'Sharpness-2'     # Basic quality (face region)
] + FIQA_MODELS_TO_TEST

# --- PRE-LOAD MODELS (for older deepface versions) ---
# In this version, we may need to pre-load models and pass them in a dictionary
# Although we only need quality, the analyze function might expect a dictionary
# of all possible models. We will try loading just the one we need first.
# This is left as a placeholder for now as we will call analyze directly.
# The logic below will be adapted if direct calling fails.

# --- STEP 2: HELPER FUNCTIONS & SETUP ---

def calculate_sharpness(image_array_gray):
    """Computes the variance of the Laplacian, a measure of sharpness."""
    if image_array_gray is None or image_array_gray.size == 0:
        return 0
    return cv2.Laplacian(image_array_gray, cv2.CV_64F).var()

def parse_probe_filename(filename):
    """Parses a probe filename."""
    match = re.match(r'(.*)_(\d+\.\d+kB)\.png$', filename)
    if not match:
        match = re.match(r'(.*)_(\dkB)\.png$', filename)
        if not match:
           return None, None, None
    base_name = match.group(1) + ".png"
    target_size_str = match.group(2)
    subject_id = base_name.split('_')[0]
    return base_name, subject_id, target_size_str

size_to_category_map = {
    '5.0kB': '5KB', '4.5kB': '4.5KB', '4.0kB': '4KB', '3.5kB': '3.5KB',
    '3.0kB': '3KB', '2.5kB': '2.5KB', '2.1kB': '2.2KB', '2.2kB': '2.2KB',
}

refs_by_subject = defaultdict(list)
for ref_filename in os.listdir(REFERENCE_PATH):
    subject_id = ref_filename.split('_')[0]
    refs_by_subject[subject_id].append(ref_filename)

# --- STEP 3: CALCULATE SCORES FOR ALL METRICS ---

grouped_scores = defaultdict(lambda: {metric: [] for metric in ALL_METRICS})
all_probe_files = os.listdir(PROBE_PATH)
print(f"Found {len(all_probe_files)} total probe files. Starting analysis...")

for i, probe_filename in enumerate(all_probe_files):
    base_ref_name, subject_id, size_str = parse_probe_filename(probe_filename)
    if not base_ref_name: continue
    category = size_to_category_map.get(size_str)
    if not category: continue
    
    probe_path = os.path.join(PROBE_PATH, probe_filename)
    
    # We wrap the whole file processing in a try/except to be safe
    try:
        # --- Utility and Sharpness Calculations ---
        # Mated-self
        mated_self_ref_filename = next((r for r in refs_by_subject.get(subject_id, []) if size_str in r), None)
        if mated_self_ref_filename:
            mated_self_ref_path = os.path.join(REFERENCE_PATH, mated_self_ref_filename)
            res_self = DeepFace.verify(img1_path=probe_path, img2_path=mated_self_ref_path, model_name=FR_MODEL, enforce_detection=False)
            grouped_scores[category]['Mated-self'].append(1 - res_self['distance'])

        # Mated-other
        other_refs = [r for r in refs_by_subject.get(subject_id, []) if r != base_ref_name]
        if other_refs:
            mated_other_ref_path = os.path.join(REFERENCE_PATH, other_refs[0])
            res_other = DeepFace.verify(img1_path=probe_path, img2_path=mated_other_ref_path, model_name=FR_MODEL, enforce_detection=False)
            grouped_scores[category]['Mated-other'].append(1 - res_other['distance'])

        # Sharpness
        full_image = cv2.imread(probe_path)
        if full_image is not None:
            gray_full = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
            grouped_scores[category]['Sharpness-1'].append(calculate_sharpness(gray_full))
            face_objs = DeepFace.extract_faces(img_path=probe_path, enforce_detection=False, detector_backend='opencv')
            if face_objs and face_objs[0]['confidence'] > 0:
                face_region = face_objs[0]['face']
                face_gray = cv2.cvtColor((face_region * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                grouped_scores[category]['Sharpness-2'].append(calculate_sharpness(face_gray))
            else:
                grouped_scores[category]['Sharpness-2'].append(0)

        # ---------------------------------------------------------------- #
        # --- FIQA MODEL CALCULATIONS REMOVED ---
        # The deepface version does not support quality assessment through analyze()
        # Only utility and sharpness metrics are calculated above
        # ---------------------------------------------------------------- #
        # Note: FIQA models would be calculated here if supported
    
    except Exception as e:
        print(f"CRITICAL ERROR processing file {probe_filename}: {e}")
    
    if (i + 1) % 50 == 0:
        print(f"  ...processed {i+1} / {len(all_probe_files)} files.")

# --- STEP 4 & 5: Averaging, Normalizing, and Plotting (No changes needed here) ---

# --- STEP 4: AVERAGE, NORMALIZE, AND CREATE DATAFRAME ---
results = []
print("\nAveraging scores for each category...")

for category in TARGET_SIZE_CATEGORIES:
    row_data = {'target_size': category}
    for metric in ALL_METRICS:
        scores = grouped_scores[category][metric]
        mean_score = np.mean(scores) if scores else 0
        row_data[f'{metric}_score'] = mean_score
    results.append(row_data)

df = pd.DataFrame(results)
df['target_size'] = pd.Categorical(df['target_size'], categories=TARGET_SIZE_CATEGORIES, ordered=True)
df = df.sort_values('target_size')

if not df.empty:
    print("\nNormalizing scores...")
    for metric in ALL_METRICS:
        score_col = f'{metric}_score'
        norm_col = f'norm_{metric}'
        if score_col in df.columns:
            max_val = df[score_col].max()
            df[norm_col] = (df[score_col] / max_val) if max_val > 0 else 0
    
    print("\nFinal DataFrame:")
    display_cols = ['target_size'] + [f'norm_{metric}' for metric in ALL_METRICS]
    print(df[display_cols])
else:
    print("\nNo results were calculated. Exiting.")
    exit()

# --- STEP 5: PLOTTING ---
print("\nGenerating final comparison plot...")
sns.set_style("whitegrid")
plt.figure(figsize=(12, 7))
palette = sns.color_palette("tab10", len(ALL_METRICS))
ax = plt.gca()

for i, metric in enumerate(ALL_METRICS):
    sns.lineplot(
        data=df, x='target_size', y=f'norm_{metric}',
        marker='o', label=metric, color=palette[i], ax=ax
    )

ax.set_title("Comparison of FIQA Models and Utility Metrics", loc='left', fontsize=16, fontweight='bold')
ax.set_xlabel("Target Image Size", fontsize=12)
ax.set_ylabel("Normalized Mean Score", fontsize=12)
ax.set_ylim(-0.05, 1.05) # Start y-axis slightly below 0 to see flat lines clearly
ax.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('.ProjMihai/plots/full_fiqa_comparison_corrected.png', bbox_inches='tight')
plt.show()