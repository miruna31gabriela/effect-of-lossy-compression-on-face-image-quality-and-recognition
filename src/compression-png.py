import os
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm

# Target sizes in bytes
TARGET_SIZES = [
    5 * 1024,
    int(4.5 * 1024),
    4 * 1024,
    int(3.5 * 1024),
    3 * 1024,
    int(2.5 * 1024),
    int(2.2 * 1024)
]

# Input folders
REFERENCE_ROI = Path(r"C:\Users\Miruna\Desktop\Disertatie\Proiect\.ProjMihai\FERET\reference_roi")
PROBE_ROI = Path(r"C:\Users\Miruna\Desktop\Disertatie\Proiect\.ProjMihai\FERET\probe_roi")

# Output folders
OUT_BASE = Path(r"C:\Users\Miruna\Desktop\Disertatie\Proiect\.ProjMihai\FERET\png-resized")
OUT_REF = OUT_BASE / "reference"
OUT_PROBE = OUT_BASE / "probe"
OUT_REF.mkdir(parents=True, exist_ok=True)
OUT_PROBE.mkdir(parents=True, exist_ok=True)

def png_resized_compress(img_path, out_folder, target_sizes):
    img = Image.open(img_path)
    basename = img_path.name
    for target in target_sizes:
        # Start with original size
        scale = 1.0
        min_side = min(img.size)
        found = False
        while min_side > 10:  # Don't go below 10px
            new_size = (int(img.width * scale), int(img.height * scale))
            resized = img.resize(new_size, Image.LANCZOS)
            # Save to buffer
            buf = io.BytesIO()
            resized.save(buf, format='PNG', optimize=True)
            size = buf.tell()
            if size <= target:
                # Save to disk
                out_name = f"{basename[:-4]}_{int(target/1024*10)/10:.1f}kB.png"
                out_path = out_folder / out_name
                resized.save(out_path, format='PNG', optimize=True)
                found = True
                break
            scale -= 0.02  # Reduce scale by 2%
            min_side = min(int(img.width * scale), int(img.height * scale))
        if not found:
            print(f"{basename}: Could not reach target {target} bytes (min size {new_size})")

def process_folder(in_folder, out_folder, target_sizes):
    images = list(in_folder.glob("*.png"))
    for img_path in tqdm(images, desc=f"Processing {in_folder.name}"):
        png_resized_compress(img_path, out_folder, target_sizes)

if __name__ == "__main__":
    print("Processing reference ROI images...")
    process_folder(REFERENCE_ROI, OUT_REF, TARGET_SIZES)
    print("Processing probe ROI images...")
    process_folder(PROBE_ROI, OUT_PROBE, TARGET_SIZES)
    print("Done! PNG-resized images saved in:")
    print(f"- {OUT_REF}")
    print(f"- {OUT_PROBE}")
