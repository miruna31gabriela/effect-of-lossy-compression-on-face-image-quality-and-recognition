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

# Define the categories for the X-axis (from your original graph)
# We will map the parsed filenames to these categories.
TARGET_SIZE_CATEGORIES = ['5KB', '4.5KB', '4KB', '3.5KB', '3KB', '2.5KB', '2.2KB']

# Define the Face Recognition model. ArcFace is an excellent choice.
FR_MODEL = "ArcFace"

# --- STEP 2: HELPER FUNCTIONS & SETUP ---

def calculate_sharpness(image_array_gray):
    """Computes the variance of the Laplacian, a measure of sharpness."""
    if image_array_gray is None or image_array_gray.size == 0:
        return 0
    return cv2.Laplacian(image_array_gray, cv2.CV_64F).var()

def parse_probe_filename(filename):
    """
    Parses a probe filename like '00002_940422_fa_2.1kB.png'
    Returns: (base_reference_name, subject_id, target_size_str)
    e.g., ('00002_940422_fa.png', '00002', '2.1kB')
    Returns (None, None, None) if the format is wrong.
    """
    # Regex to find the base name and the size tag (e.g., _2.1kB)
    match = re.match(r'(.*)_(\d+\.\d+kB)\.png$', filename)
    if not match:
        match = re.match(r'(.*)_(\dkB)\.png$', filename) # For integer sizes like _3kB
        if not match:
           return None, None, None

    base_name = match.group(1) + ".png"
    target_size_str = match.group(2)
    subject_id = base_name.split('_')[0]
    
    return base_name, subject_id, target_size_str

# Create a mapping from filename sizes to our desired categories
# This makes the grouping flexible. Adjust if your filenames differ.
size_to_category_map = {
    '5.0kB': '5KB',
    '4.5kB': '4.5KB',
    '4.0kB': '4KB',
    '3.5kB': '3.5KB',
    '3.0kB': '3KB',
    '2.5kB': '2.5KB',
    '2.1kB': '2.2KB', # Mapping your 2.1kB file to the 2.2KB category
    '2.2kB': '2.2KB',
}

# Pre-load a map of all original reference images by subject ID for faster "mated-other" lookups
refs_by_subject = defaultdict(list)
for ref_filename in os.listdir(REFERENCE_PATH):
    subject_id = ref_filename.split('_')[0]
    refs_by_subject[subject_id].append(ref_filename)


# --- STEP 3: CALCULATE COMPARISON SCORES ---

# Use a dictionary to group scores by category before averaging
# e.g., {'5KB': {'self': [0.9, 0.92], 'other': [0.8, 0.81]}, ...}
grouped_scores = defaultdict(lambda: {
    'mated_self': [], 
    'mated_other': [],
    'sharpness_1': [],
    'sharpness_2': []
})

all_probe_files = os.listdir(PROBE_PATH)
print(f"\nFound {len(all_probe_files)} total probe files. Starting comparison...")

for i, probe_filename in enumerate(all_probe_files):
    # Parse the filename to get necessary info
    base_ref_name, subject_id, size_str = parse_probe_filename(probe_filename)
    
    if not base_ref_name:
        # print(f"Warning: Could not parse filename '{probe_filename}'. Skipping.")
        continue

    # Map the parsed size to our plot category
    category = size_to_category_map.get(size_str)
    if not category:
        # print(f"Warning: No category mapping for size '{size_str}'. Skipping.")
        continue
        
    probe_path = os.path.join(PROBE_PATH, probe_filename)
    
    try:
        # --- Sharpness Calculations ---
        full_image = cv2.imread(probe_path)
        if full_image is not None:
            gray_full = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
            # Sharpness-1 (Full Image)
            sharpness_1_score = calculate_sharpness(gray_full)
            grouped_scores[category]['sharpness_1'].append(sharpness_1_score)

            # Sharpness-2 (Face Region)
            sharpness_2_score = 0
            try:
                face_objs = DeepFace.extract_faces(img_path=probe_path, enforce_detection=False, detector_backend='opencv')
                if face_objs and face_objs[0]['confidence'] > 0:
                    face_region = face_objs[0]['face']
                    face_gray = cv2.cvtColor((face_region * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    sharpness_2_score = calculate_sharpness(face_gray)
            except: # Ignore errors if face detection fails
                pass
            grouped_scores[category]['sharpness_2'].append(sharpness_2_score)

        # --- Mated-self Calculation ---
        # Find the reference file for the same subject with the same compressed size.
        candidate_refs = refs_by_subject.get(subject_id, [])
        mated_self_ref_filename = None
        for ref_filename in candidate_refs:
            if size_str in ref_filename:
                mated_self_ref_filename = ref_filename
                break # Found the corresponding reference

        if mated_self_ref_filename:
            mated_self_ref_path = os.path.join(REFERENCE_PATH, mated_self_ref_filename)
            res_self = DeepFace.verify(img1_path=probe_path, img2_path=mated_self_ref_path, model_name=FR_MODEL, enforce_detection=False)
            # Convert distance to similarity (1 - distance for cosine)
            grouped_scores[category]['mated_self'].append(1 - res_self['distance'])

        # --- Mated-other Calculation ---
        if subject_id in refs_by_subject:
            # Find a different reference photo for the same person
            other_refs = [r for r in refs_by_subject[subject_id] if r != base_ref_name]
            if other_refs:
                # Just pick the first available "other" reference
                mated_other_ref_path = os.path.join(REFERENCE_PATH, other_refs[0])
                res_other = DeepFace.verify(img1_path=probe_path, img2_path=mated_other_ref_path, model_name=FR_MODEL, enforce_detection=False)
                grouped_scores[category]['mated_other'].append(1 - res_other['distance'])

    except Exception as e:
        # This can happen if a face isn't detected. It's common with very compressed images.
        # print(f"Could not process pair for {probe_filename}: {e}")
        pass
    
    if (i + 1) % 100 == 0:
        print(f"  ...processed {i+1} / {len(all_probe_files)} files.")


# --- STEP 4: AVERAGE, NORMALIZE, AND CREATE DATAFRAME ---

results = []
print("\nAveraging scores for each category...")
for category in TARGET_SIZE_CATEGORIES:
    self_scores = grouped_scores[category]['mated_self']
    other_scores = grouped_scores[category]['mated_other']
    sharpness_1_scores = grouped_scores[category]['sharpness_1']
    sharpness_2_scores = grouped_scores[category]['sharpness_2']
    
    mean_self = np.mean(self_scores) if self_scores else 0
    mean_other = np.mean(other_scores) if other_scores else 0
    mean_sharpness_1 = np.mean(sharpness_1_scores) if sharpness_1_scores else 0
    mean_sharpness_2 = np.mean(sharpness_2_scores) if sharpness_2_scores else 0
    
    results.append({
        'target_size': category,
        'mated_self_score': mean_self,
        'mated_other_score': mean_other,
        'sharpness_1_score': mean_sharpness_1,
        'sharpness_2_score': mean_sharpness_2
    })
    print(f"-> {category}: Mean Self = {mean_self:.4f} ({len(self_scores)} images), Mean Other = {mean_other:.4f} ({len(other_scores)} images)")

df = pd.DataFrame(results)

# Set the order of the x-axis
df['target_size'] = pd.Categorical(df['target_size'], categories=TARGET_SIZE_CATEGORIES, ordered=True)
df = df.sort_values('target_size')

if not df.empty:
    # Normalize each column to [0, 1] by dividing by its own max value
    df['norm_mated_self'] = df['mated_self_score'] / df['mated_self_score'].max()
    df['norm_mated_other'] = df['mated_other_score'] / df['mated_other_score'].max()
    if df['sharpness_1_score'].max() > 0:
        df['norm_sharpness_1'] = df['sharpness_1_score'] / df['sharpness_1_score'].max()
    else:
        df['norm_sharpness_1'] = 0
    if df['sharpness_2_score'].max() > 0:
        df['norm_sharpness_2'] = df['sharpness_2_score'] / df['sharpness_2_score'].max()
    else:
        df['norm_sharpness_2'] = 0

    print("\nFinal DataFrame with Normalized Scores:")
    print(df[['target_size', 'norm_mated_self', 'norm_mated_other', 'norm_sharpness_1', 'norm_sharpness_2']])
else:
    print("\nNo results were calculated. Exiting.")
    exit()

# --- STEP 5: PLOTTING ---
print("\nGenerating plot...")
sns.set_style("whitegrid")
plt.figure(figsize=(7, 5))

# Plot Mated-self
ax = sns.lineplot(
    data=df, x='target_size', y='norm_mated_self',
    marker='o', label='Mated-self', color='skyblue'
)

# Plot Mated-other on the SAME axes
sns.lineplot(
    data=df, x='target_size', y='norm_mated_other',
    marker='o', label='Mated-other', color='orange', ax=ax
)

# Plot Sharpness-1
sns.lineplot(
    data=df, x='target_size', y='norm_sharpness_1',
    marker='o', label='Sharpness-1', color='hotpink', ax=ax
)

# Plot Sharpness-2
sns.lineplot(
    data=df, x='target_size', y='norm_sharpness_2',
    marker='o', label='Sharpness-2', color='red', ax=ax
)

# Customize the plot
ax.set_title("PNG-resized", loc='left', fontsize=14, fontweight='bold')
ax.set_xlabel("Target size", fontsize=12)
ax.set_ylabel("Normalized mean score", fontsize=12)
ax.set_ylim(0, 1.05)
ax.legend()

plt.tight_layout()
plt.savefig('.ProjMihai/plots/png_resized_comparison.png')
plt.show()