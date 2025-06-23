import os
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm
import pillow_jxl

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
REFERENCE_ROI = Path(r"/zhome/dd/9/202544/Documents/test/effect-of-lossy-compression-on-face-image-quality-and-recognition/FERET/reference_roi")
PROBE_ROI = Path(r"/zhome/dd/9/202544/Documents/test/effect-of-lossy-compression-on-face-image-quality-and-recognition/FERET/probe_roi")

# Output folders
OUT_BASE = Path(r"/zhome/dd/9/202544/Documents/test/effect-of-lossy-compression-on-face-image-quality-and-recognition/FERET/jpeg-xl-resized")
OUT_REF = OUT_BASE / "reference"
OUT_PROBE = OUT_BASE / "probe"
OUT_REF.mkdir(parents=True, exist_ok=True)
OUT_PROBE.mkdir(parents=True, exist_ok=True)

def jxl_compress(img_path, out_folder, target_sizes):
    """Compress image using JPEG XL format to target sizes"""
    try:
        img = Image.open(img_path).convert('RGB')
        basename = img_path.stem  # Get filename without extension
        
        for target in target_sizes:
            # Binary search for best quality (1-100)
            low, high = 1, 100
            best_quality = None
            best_buf = None
            
            while low <= high:
                mid = (low + high) // 2
                buf = io.BytesIO()
                
                try:
                    img.save(buf, format='JXL', quality=mid)
                    size = buf.tell()
                    
                    if size <= target:
                        best_quality = mid
                        best_buf = buf.getvalue()
                        low = mid + 1  # Try higher quality
                    else:
                        high = mid - 1  # Try lower quality
                except Exception as e:
                    print(f"Error saving with quality {mid}: {e}")
                    high = mid - 1
            
            if best_quality is not None:
                out_name = f"{basename}_{int(target/1024*10)/10:.1f}kB.jxl"
                out_path = out_folder / out_name
                with open(out_path, 'wb') as f:
                    f.write(best_buf)
                #print(f"Saved {out_name} with quality {best_quality} ({len(best_buf)} bytes)")
            else:
                print(f"{basename}: Could not reach target {target} bytes (min quality 1)")
                
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def process_folder(in_folder, out_folder, target_sizes):
    """Process all images in a folder"""
    # Look for common image formats
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    images = []
    
    for ext in image_extensions:
        images.extend(list(in_folder.glob(ext)))
        images.extend(list(in_folder.glob(ext.upper())))
    
    if not images:
        print(f"No images found in {in_folder}")
        return
    
    print(f"Found {len(images)} images in {in_folder.name}")
    
    for img_path in tqdm(images, desc=f"Processing {in_folder.name}"):
        jxl_compress(img_path, out_folder, target_sizes)

if __name__ == "__main__":
    # Check if input folders exist
    if not REFERENCE_ROI.exists():
        print(f"Error: Reference ROI folder not found: {REFERENCE_ROI}")
        exit(1)
    
    if not PROBE_ROI.exists():
        print(f"Error: Probe ROI folder not found: {PROBE_ROI}")
        exit(1)
    
    print("Processing reference ROI images...")
    process_folder(REFERENCE_ROI, OUT_REF, TARGET_SIZES)
    print("Processing probe ROI images...")
    process_folder(PROBE_ROI, OUT_PROBE, TARGET_SIZES)
    print("Done! JPEG XL-resized images saved in:")
    print(f"- {OUT_REF}")
    print(f"- {OUT_PROBE}")
