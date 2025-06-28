import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from keras_facenet import FaceNet
import imagecodecs
from tqdm import tqdm
import io
import re

# Constants
DATASET_PATH = '.ProjMihai/FERET'
OUTPUT_DIR = '.ProjMihai/evaluation_results_feret'
# The compression formats are now directories in the dataset path
# The target sizes are also part of the structure, we will parse them
COMPRESSION_FORMATS = ['jpeg-xr-resized', 'jpeg-2000-resized', 'jpeg-resized', 'png-resized']

def load_facenet_model():
    """Loads the FaceNet model."""
    return FaceNet()

def get_embedding(model, image):
    """Gets the face embedding for a single image."""
    image = image.convert('RGB')
    pixels = np.asarray(image)
    faces = model.extract(pixels)
    if not faces:
        return None

    # Manually extract the face using the bounding box
    x1, y1, width, height = faces[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face_pixels = pixels[y1:y2, x1:x2]

    # Resize face to the required input size for the model (typically 160x160)
    face_image = Image.fromarray(face_pixels)
    face_image = face_image.resize((160, 160))
    face_array = np.asarray(face_image)

    # The embeddings method expects a list of face arrays.
    embedding = model.embeddings([face_array])[0]
    return embedding


def get_embedding_from_path(model, image_path):
    """Gets the face embedding for a single image from a file path."""
    file_ext = os.path.splitext(image_path)[1].lower()

    try:
        # For special formats, we must read the file's bytes first
        if file_ext == '.jxl':
            with open(image_path, 'rb') as f:
                encoded_data = f.read() # Read file content into a bytes object
            pixels = imagecodecs.jxl_decode(encoded_data) # Pass the bytes to the decoder
            image = Image.fromarray(pixels)
            
        elif file_ext in ['.jp2', '.j2k', '.jpx']:
            with open(image_path, 'rb') as f:
                encoded_data = f.read() # Read file content into a bytes object
            pixels = imagecodecs.j2k_decode(encoded_data) # Pass the bytes to the decoder
            image = Image.fromarray(pixels)
            
        elif file_ext == '.jxr':
            with open(image_path, 'rb') as f:
                encoded_data = f.read()
            pixels = imagecodecs.jxr_decode(encoded_data)
            image = Image.fromarray(pixels)
            
        else:
            # For standard formats like PNG, PIL.Image.open() is fine
            image = Image.open(image_path)
            
    except Exception as e:
        print(f"Error opening or decoding image {image_path}: {e}")
        return None

    # The rest of the function remains the same
    return get_embedding(model, image)


def compress_image(image, target_size_kb, format='JPEG'):
    """Compresses an image to a target size using a specific format."""
    target_size_bytes = target_size_kb * 1024
    buffer = io.BytesIO()
    
    if format == 'JPEG':
        quality = 95
        while quality > 0:
            buffer.seek(0)
            buffer.truncate(0)
            image.save(buffer, format='JPEG', quality=quality)
            if buffer.tell() <= target_size_bytes:
                break
            quality -= 5
    elif format == 'PNG-resized':
        scale = 1.0
        while scale > 0.1:
            buffer.seek(0)
            buffer.truncate(0)
            new_size = (int(image.width * scale), int(image.height * scale))
            if new_size[0] == 0 or new_size[1] == 0:
                break
            resized_image = image.resize(new_size, Image.LANCZOS)
            resized_image.save(buffer, format='PNG')
            if buffer.tell() <= target_size_bytes:
                break
            scale -= 0.05
    elif format == 'JPEG 2000':
        quality = 100
        pixels = np.array(image.convert('RGB'))
        while quality > 0:
            buffer.seek(0)
            buffer.truncate(0)
            encoded = imagecodecs.j2k_encode(pixels, level=quality)
            buffer.write(encoded)
            if buffer.tell() <= target_size_bytes:
                break
            quality -= 5
    elif format == 'JPEG XL':
        quality = 100
        pixels = np.array(image.convert('RGB'))
        while quality > 0:
            buffer.seek(0)
            buffer.truncate(0)
            encoded = imagecodecs.jxl_encode(pixels, level=quality)
            buffer.write(encoded)
            if buffer.tell() <= target_size_bytes:
                break
            quality -= 5
            
    buffer.seek(0)
    return Image.open(buffer)

def evaluate_mated_self(model, reference_images):
    """Performs the Mated-self evaluation."""
    results = {fmt: {} for fmt in COMPRESSION_FORMATS}
    
    ref_path = os.path.join(DATASET_PATH, 'reference_roi')
    
    for fmt in COMPRESSION_FORMATS:
        probe_base_path = os.path.join(DATASET_PATH, fmt, 'probe')
        
        if not os.path.exists(probe_base_path):
            continue
            
        for probe_filename in tqdm(os.listdir(probe_base_path), desc=f"Mated-self {fmt}"):
            match = re.match(r'(.+)_(\d+\.?\d*kB)\..+', probe_filename)
            if not match:
                continue
            
            ref_filename_base = match.group(1)
            size_str = match.group(2)
            size_kb = float(size_str.replace('kB', ''))

            ref_filename = ref_filename_base + '.png'
            
            ref_image_path = os.path.join(ref_path, ref_filename)
            probe_image_path = os.path.join(probe_base_path, probe_filename)

            if not os.path.exists(ref_image_path):
                continue
            
            probe_embedding = get_embedding_from_path(model, probe_image_path)
            ref_embedding = get_embedding_from_path(model, ref_image_path)

            if probe_embedding is not None and ref_embedding is not None:
                score = 1 - cosine(probe_embedding, ref_embedding)
                
                if size_kb not in results[fmt]:
                    results[fmt][size_kb] = []
                results[fmt][size_kb].append(score)

    return results

def evaluate_mated_other(model, reference_images):
    """Performs the Mated-other evaluation."""
    results = {fmt: {} for fmt in COMPRESSION_FORMATS}
    
    ref_path = os.path.join(DATASET_PATH, 'reference_roi')
    
    for fmt in COMPRESSION_FORMATS:
        probe_base_path = os.path.join(DATASET_PATH, fmt, 'probe')
        
        if not os.path.exists(probe_base_path):
            continue
            
        probe_files = os.listdir(probe_base_path)
        for i, probe_filename in enumerate(tqdm(probe_files, desc=f"Mated-other {fmt}")):
            
            match = re.match(r'(.+)_(\d+\.?\d*kB)\..+', probe_filename)
            if not match:
                continue
            
            probe_subject_id = probe_filename.split('_')[0]
            size_str = match.group(2)
            size_kb = float(size_str.replace('kB', ''))

            probe_image_path = os.path.join(probe_base_path, probe_filename)
            
            other_ref_index = (i + 1) % len(reference_images)
            other_ref_filename = reference_images[other_ref_index]
            other_ref_subject_id = other_ref_filename.split('_')[0]

            while probe_subject_id == other_ref_subject_id:
                other_ref_index = (other_ref_index + 1) % len(reference_images)
                other_ref_filename = reference_images[other_ref_index]
                other_ref_subject_id = other_ref_filename.split('_')[0]

            ref_image_path = os.path.join(ref_path, other_ref_filename)

            if not os.path.exists(ref_image_path):
                continue

            probe_embedding = get_embedding_from_path(model, probe_image_path)
            ref_embedding = get_embedding_from_path(model, ref_image_path)

            if probe_embedding is not None and ref_embedding is not None:
                score = 1 - cosine(probe_embedding, ref_embedding)
                
                if size_kb not in results[fmt]:
                    results[fmt][size_kb] = []
                results[fmt][size_kb].append(score)
                
    return results

def plot_results(results, title):
    """Plots the evaluation results."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    for fmt in results:
        if not results[fmt]:
            continue
            
        # Sort sizes in descending order for the plot
        sorted_sizes = sorted(results[fmt].keys(), reverse=True)
        
        mean_scores = []
        std_devs = []
        plot_sizes = []

        for size in sorted_sizes:
            scores = results[fmt][size]
            if scores:
                mean_scores.append(np.mean(scores))
                std_devs.append(np.std(scores))
                plot_sizes.append(size)

        if not plot_sizes:
            continue

        mean_scores = np.array(mean_scores)
        std_devs = np.array(std_devs)
        
        ax.plot(plot_sizes, mean_scores, marker='o', linestyle='-', label=fmt.replace('-resized', '').upper())
        ax.fill_between(plot_sizes, mean_scores - std_devs, mean_scores + std_devs, alpha=0.2)

    ax.set_xlabel('Target size (kB)')
    ax.set_ylabel('Comparison score')
    ax.set_title(title)
    # Invert x-axis to show largest sizes first
    ax.invert_xaxis()
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    plt.savefig(os.path.join(OUTPUT_DIR, f'{title}.png'))
    print(f"Plot saved to {os.path.join(OUTPUT_DIR, f'{title}.png')}")


def plot_results_combined(mated_self_results, mated_other_results):
    """
    Plots combined Mated-self and Mated-other results exactly as described
    in the research paper. This includes deterministic offsets for each format,
    a "smear" of all data points, and opaque min/max markers.
    """
    plt.style.use('seaborn-v0_8-white')
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 11), sharex=True)

    # Vibrant color palette to match the target image
    colors = {
        'JPEG XL': '#4A4AFF',
        'JPEG 2000': '#FFA54A',
        'JPEG': '#29B3A9',
        'PNG-RESIZED': '#FF6B6B',
        'PNG': '#FF6B6B'
    }

    # --- NEW: Create deterministic horizontal offsets for each format ---
    # This separates the "streaks" of data for each compression type.
    num_formats = len(COMPRESSION_FORMATS)
    # Create evenly spaced offsets around the center (0.0)
    offsets = np.linspace(-0.2, 0.2, num_formats) 
    format_offsets = {fmt: offset for fmt, offset in zip(COMPRESSION_FORMATS, offsets)}

    titles = ['Mated-other:', 'Mated-self:']
    results_all = [mated_other_results, mated_self_results]
    
    handles = []
    labels = []
    all_plot_sizes = set()

    for ax, title, results in zip(axes, titles, results_all):
        ax.grid(False)

        for fmt in COMPRESSION_FORMATS:
            if not results.get(fmt):
                continue

            sorted_sizes = sorted(results[fmt].keys(), reverse=True)
            mean_scores = []
            plot_sizes_for_fmt = []
            
            clean_label = fmt.replace('-resized', '').upper()
            color = colors.get(clean_label, 'gray')
            offset = format_offsets.get(fmt, 0) # Get the specific offset for this format

            for size in sorted_sizes:
                scores = results[fmt][size]
                if scores:
                    mean_scores.append(np.mean(scores))
                    plot_sizes_for_fmt.append(size)
                    all_plot_sizes.add(size)
                    
                    x_pos_base = size + offset

                    # --- PLOT 1: The "smeared" individual data points ---
                    # Add a very small random jitter to the base offset for the smear effect
                    jittered_x = np.random.normal(x_pos_base, 0.03, len(scores))
                    ax.scatter(jittered_x, scores, 
                               alpha=0.1,          # Very transparent for the "smear"
                               color=color,        
                               marker='d',         
                               s=40,               
                               ec='none',          
                               zorder=1)           # Draw in the background

                    # --- PLOT 2: The opaque min/max markers ---
                    min_score = np.min(scores)
                    max_score = np.max(scores)
                    ax.scatter([x_pos_base, x_pos_base], [min_score, max_score],
                               marker='d',         # Rhombus shape
                               color=color,
                               s=45,               # Slightly larger to be visible
                               alpha=1.0,          # Opaque
                               zorder=3)           # Draw on top of everything

            if not plot_sizes_for_fmt:
                continue

            # --- PLOT 3: The mean score line ---
            # The x-coordinates for the line must also be offset
            mean_line_x_coords = [s + offset for s in plot_sizes_for_fmt]
            line, = ax.plot(mean_line_x_coords, mean_scores, 
                            linestyle='-', 
                            color=color, 
                            linewidth=2.5,  
                            zorder=2)       # Draw on top of smear, below min/max

            if clean_label not in labels:
                handles.append(line)
                labels.append(clean_label)

        ax.set_title(title, loc='left', fontsize=16, fontweight='bold')
        ax.set_ylabel('Comparison score', fontsize=14)
        ax.set_ylim(0.15, 1.05)

    # Set custom x-ticks based on all data found
    sorted_ticks = sorted(list(all_plot_sizes), reverse=True)
    axes[1].set_xticks(sorted_ticks)
    axes[1].set_xticklabels([f'{s}kB' for s in sorted_ticks], fontsize=12)
    axes[1].set_xlabel('Target size', fontsize=14)
    
    # Create a single shared legend at the bottom
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(labels), fontsize=14)
    
    fig.tight_layout(rect=[0, 0.06, 1, 1]) # Adjust for legend

    out_path = os.path.join(OUTPUT_DIR, 'Final_Report_Figure.png')
    plt.savefig(out_path, dpi=200)
    print(f"Final report figure saved to {out_path}")
    plt.show()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ref_path = os.path.join(DATASET_PATH, 'reference_roi')
    reference_images = [f for f in os.listdir(ref_path) if f.endswith('.png')]

    model = load_facenet_model()

    print("Running Mated-self evaluation...")
    mated_self_results = evaluate_mated_self(model, reference_images)

    print("Running Mated-other evaluation...")
    mated_other_results = evaluate_mated_other(model, reference_images)

    plot_results_combined(mated_self_results, mated_other_results)


if __name__ == '__main__':
    main()
