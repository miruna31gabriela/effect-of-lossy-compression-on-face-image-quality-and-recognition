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
OUT_BASE = Path(r"C:\Users\Miruna\Desktop\Disertatie\Proiect\.ProjMihai\FERET\jpeg-2000-resized")
OUT_REF = OUT_BASE / "reference"
OUT_PROBE = OUT_BASE / "probe"
OUT_REF.mkdir(parents=True, exist_ok=True)
OUT_PROBE.mkdir(parents=True, exist_ok=True)

def jpeg2000_compress(img_path, out_folder, target_sizes):
    img = Image.open(img_path).convert('RGB')
    basename = img_path.name
    for target in target_sizes:
        # Binary search for best quality (quality is 0-100, but Pillow uses 0.0-100.0 for JPEG 2000)
        low, high = 0.0, 100.0
        best_quality = None
        best_buf = None
        while high - low > 0.5:
            mid = (low + high) / 2
            buf = io.BytesIO()
            img.save(buf, format='JPEG2000', quality_mode='rates', quality_layers=[mid])
            size = buf.tell()
            if size <= target:
                best_quality = mid
                best_buf = buf.getvalue()
                high = mid - 0.5
            else:
                low = mid + 0.5
        if best_quality is not None:
            out_name = f"{basename[:-4]}_{int(target/1024*10)/10:.1f}kB.jp2"
            out_path = out_folder / out_name
            with open(out_path, 'wb') as f:
                f.write(best_buf)
        else:
            print(f"{basename}: Could not reach target {target} bytes (min quality 0.0)")

def process_folder(in_folder, out_folder, target_sizes):
    images = list(in_folder.glob("*.png"))
    for img_path in tqdm(images, desc=f"Processing {in_folder.name}"):
        jpeg2000_compress(img_path, out_folder, target_sizes)

if __name__ == "__main__":
    print("Processing reference ROI images...")
    process_folder(REFERENCE_ROI, OUT_REF, TARGET_SIZES)
    print("Processing probe ROI images...")
    process_folder(PROBE_ROI, OUT_PROBE, TARGET_SIZES)
    print("Done! JPEG2000-resized images saved in:")
    print(f"- {OUT_REF}")
    print(f"- {OUT_PROBE}")
