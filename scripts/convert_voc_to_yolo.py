"""
Convert Pascal VOC XML annotations to YOLO format and split into train/val/test sets.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "PCB_DATASET"
OUTPUT_DIR = BASE_DIR / "data" / "yolo_dataset"

# Class mapping
CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


def parse_voc_annotation(xml_path: Path) -> tuple[str, int, int, list[dict]]:
    """Parse a Pascal VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find("filename").text
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.lower()
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        
        objects.append({
            "name": name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        })
    
    return filename, width, height, objects


def convert_to_yolo_format(obj: dict, img_width: int, img_height: int) -> str:
    """Convert a single bounding box to YOLO format."""
    # Get class index
    class_name = obj["name"].replace(" ", "_")
    class_idx = CLASS_TO_IDX.get(class_name, -1)
    
    if class_idx == -1:
        print(f"Warning: Unknown class '{class_name}'")
        return None
    
    # Convert to YOLO format (center_x, center_y, width, height) normalized
    x_center = (obj["xmin"] + obj["xmax"]) / 2 / img_width
    y_center = (obj["ymin"] + obj["ymax"]) / 2 / img_height
    width = (obj["xmax"] - obj["xmin"]) / img_width
    height = (obj["ymax"] - obj["ymin"]) / img_height
    
    return f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def collect_all_samples() -> list[tuple[Path, Path]]:
    """Collect all image-annotation pairs from the dataset."""
    samples = []
    
    annotations_dir = DATA_DIR / "Annotations"
    images_dir = DATA_DIR / "images"
    
    for class_dir in annotations_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        for xml_file in class_dir.glob("*.xml"):
            # Find corresponding image
            img_name = xml_file.stem + ".jpg"
            img_path = images_dir / class_name / img_name
            
            if img_path.exists():
                samples.append((img_path, xml_file))
            else:
                print(f"Warning: Image not found for {xml_file}")
    
    return samples


def process_and_split_dataset():
    """Process all annotations and split into train/val/test."""
    print("Collecting samples...")
    samples = collect_all_samples()
    print(f"Found {len(samples)} samples")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(samples)
    
    n_train = int(len(samples) * TRAIN_RATIO)
    n_val = int(len(samples) * VAL_RATIO)
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    print(f"Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    
    # Process each split
    splits = [
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples)
    ]
    
    stats = {cls: 0 for cls in CLASSES}
    
    for split_name, split_samples in splits:
        print(f"\nProcessing {split_name} split...")
        
        images_out = OUTPUT_DIR / "images" / split_name
        labels_out = OUTPUT_DIR / "labels" / split_name
        
        for img_path, xml_path in split_samples:
            # Parse annotation
            filename, width, height, objects = parse_voc_annotation(xml_path)
            
            # Convert to YOLO format
            yolo_lines = []
            for obj in objects:
                line = convert_to_yolo_format(obj, width, height)
                if line:
                    yolo_lines.append(line)
                    stats[obj["name"].replace(" ", "_")] += 1
            
            # Copy image
            new_img_name = f"{xml_path.parent.name}_{xml_path.stem}.jpg"
            shutil.copy(img_path, images_out / new_img_name)
            
            # Write label file
            label_file = labels_out / f"{xml_path.parent.name}_{xml_path.stem}.txt"
            with open(label_file, "w") as f:
                f.write("\n".join(yolo_lines))
    
    print("\n" + "="*50)
    print("Class distribution:")
    for cls, count in stats.items():
        print(f"  {cls}: {count}")
    print("="*50)
    
    return stats


def create_yaml_config():
    """Create the YOLO dataset configuration file."""
    yaml_content = f"""# PCB Defects Dataset Configuration
# Auto-generated by convert_voc_to_yolo.py

path: {OUTPUT_DIR.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper

# Number of classes
nc: 6
"""
    
    yaml_path = OUTPUT_DIR / "pcb_defects.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print(f"\nCreated dataset config: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    print("="*50)
    print("VOC to YOLO Conversion Script")
    print("="*50)
    
    # Process dataset
    stats = process_and_split_dataset()
    
    # Create YAML config
    create_yaml_config()
    
    print("\n✅ Conversion complete!")

