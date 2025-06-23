import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
import re
from pathlib import Path

def analyze_feret_dataset(dataset_path):
    """
    Analyze the FERET dataset to understand pose distribution and display sample images.
    """
    dataset_path = Path(dataset_path)
    reference_path = dataset_path / "reference"
    probe_path = dataset_path / "probe"
    
    # Analyze reference images
    reference_files = list(reference_path.glob("*.png"))
    print(f"Reference images: {len(reference_files)}")
    
    # Analyze probe images
    probe_files = list(probe_path.glob("*.png"))
    print(f"Probe images: {len(probe_files)}")
    
    # Extract pose information from filenames
    def extract_pose_info(filename):
        # Pattern: subject_id_date_pose.png
        # Example: 01208_940128_fa_a.png
        match = re.match(r'(\d+)_(\d+)_([a-z]+)(?:_([a-z]))?\.png', filename.name)
        if match:
            subject_id = match.group(1)
            date = match.group(2)
            pose = match.group(3)
            variant = match.group(4) if match.group(4) else ""
            return subject_id, date, pose, variant
        return None, None, None, None
    
    # Analyze poses in reference images
    reference_poses = []
    for file in reference_files:
        subject_id, date, pose, variant = extract_pose_info(file)
        if pose:
            reference_poses.append(pose)
    
    # Analyze poses in probe images
    probe_poses = []
    probe_subjects = set()
    for file in probe_files:
        subject_id, date, pose, variant = extract_pose_info(file)
        if pose:
            probe_poses.append(pose)
            if subject_id:
                probe_subjects.add(subject_id)
    
    # Count pose distributions
    reference_pose_counts = Counter(reference_poses)
    probe_pose_counts = Counter(probe_poses)
    
    print("\n=== POSE DISTRIBUTION ANALYSIS ===")
    print(f"Number of unique subjects in probe: {len(probe_subjects)}")
    
    print("\nReference images pose distribution:")
    for pose, count in reference_pose_counts.most_common():
        print(f"  {pose}: {count}")
    
    print("\nProbe images pose distribution:")
    for pose, count in probe_pose_counts.most_common():
        print(f"  {pose}: {count}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot reference pose distribution
    poses_ref = list(reference_pose_counts.keys())
    counts_ref = list(reference_pose_counts.values())
    ax1.bar(poses_ref, counts_ref, color='skyblue')
    ax1.set_title('Reference Images - Pose Distribution')
    ax1.set_xlabel('Pose')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot probe pose distribution
    poses_probe = list(probe_pose_counts.keys())
    counts_probe = list(probe_pose_counts.values())
    ax2.bar(poses_probe, counts_probe, color='lightcoral')
    ax2.set_title('Probe Images - Pose Distribution')
    ax2.set_xlabel('Pose')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('feret_pose_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return reference_files, probe_files, reference_pose_counts, probe_pose_counts

def display_sample_images(dataset_path, num_samples=4):
    """
    Display sample images from the FERET dataset.
    """
    dataset_path = Path(dataset_path)
    reference_path = dataset_path / "reference"
    probe_path = dataset_path / "probe"
    
    # Get sample images
    reference_files = list(reference_path.glob("*.png"))[:num_samples//2]
    probe_files = list(probe_path.glob("*.png"))[:num_samples//2]
    
    fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 8))
    
    # Display reference images
    for i, file in enumerate(reference_files):
        img = mpimg.imread(file)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Reference: {file.name}')
        axes[0, i].axis('off')
    
    # Display probe images
    for i, file in enumerate(probe_files):
        img = mpimg.imread(file)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Probe: {file.name}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('feret_sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()

def explain_pose_codes():
    """
    Explain the FERET pose codes.
    """
    print("\n=== FERET POSE CODES EXPLANATION ===")
    pose_codes = {
        'fa': 'Frontal (neutral expression)',
        'fb': 'Frontal (alternative expression)',
        'fc': 'Frontal (alternative expression)',
        'ra': 'Right profile (0°)',
        'rb': 'Right profile (15°)',
        'rc': 'Right profile (30°)',
        'rd': 'Right profile (45°)',
        're': 'Right profile (60°)',
        'rf': 'Right profile (75°)',
        'rg': 'Right profile (90°)',
        'la': 'Left profile (0°)',
        'lb': 'Left profile (15°)',
        'lc': 'Left profile (30°)',
        'ld': 'Left profile (45°)',
        'le': 'Left profile (60°)',
        'lf': 'Left profile (75°)',
        'lg': 'Left profile (90°)',
        'ql': 'Quarter left (67.5°)',
        'qr': 'Quarter right (67.5°)',
        'hl': 'Half left (45°)',
        'hr': 'Half right (45°)',
        'pl': 'Profile left (90°)',
        'pr': 'Profile right (90°)'
    }
    
    for code, description in pose_codes.items():
        print(f"  {code}: {description}")

if __name__ == "__main__":
    # Path to the FERET dataset
    dataset_path = r"C:\Users\Miruna\Desktop\Disertatie\Proiect\.ProjMihai\FERET"
    
    # Analyze the dataset
    print("Analyzing FERET dataset...")
    reference_files, probe_files, ref_poses, probe_poses = analyze_feret_dataset(dataset_path)
    
    # Explain pose codes
    explain_pose_codes()
    
    # Display sample images
    print("\nDisplaying sample images...")
    display_sample_images(dataset_path, num_samples=6)
    
    print("\nAnalysis complete! Check the generated plots:")
    print("- feret_pose_distribution.png")
    print("- feret_sample_images.png")
