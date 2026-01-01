"""
Train YOLOv8 on PCB Defects Dataset
Optimized for Apple Silicon (M1/M2) with MPS acceleration
"""

from pathlib import Path
import torch
from ultralytics import YOLO

# Paths
BASE_DIR = Path(__file__).parent
DATA_CONFIG = BASE_DIR / "data" / "yolo_dataset" / "pcb_defects.yaml"
MODELS_DIR = BASE_DIR / "models"

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

# Detect best available device
def get_device():
    """Get the best available device for training."""
    if torch.backends.mps.is_available():
        print("✅ Using MPS (Apple Silicon GPU)")
        return "mps"
    elif torch.cuda.is_available():
        print("✅ Using CUDA GPU")
        return 0  # First CUDA device
    else:
        print("⚠️ Using CPU (slower)")
        return "cpu"


def train():
    """Train YOLOv8 model on PCB defects dataset."""
    
    device = get_device()
    
    # Load pretrained YOLOv8 small model (faster, good accuracy)
    model = YOLO("yolov8s.pt")
    
    # Training configuration - optimized for Mac
    results = model.train(
        data=str(DATA_CONFIG),
        epochs=50,  # Reduced for faster training
        imgsz=640,
        batch=8,  # Smaller batch for MPS memory
        patience=15,  # Early stopping patience
        device=device,
        save=True,
        project=str(MODELS_DIR),
        name="pcb_defects_yolov8s",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        verbose=True,
        seed=42,
        deterministic=True,
        plots=True,
        workers=4,  # Reduced workers for Mac
        # Augmentation settings for small object detection
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
    )
    
    return results


def validate(model_path: str = None):
    """Validate the trained model."""
    if model_path is None:
        model_path = MODELS_DIR / "pcb_defects_yolov8s" / "weights" / "best.pt"
    
    device = get_device()
    model = YOLO(model_path)
    
    # Validate on test set
    metrics = model.val(
        data=str(DATA_CONFIG),
        split="test",
        device=device,
        plots=True,
        save_json=True,
    )
    
    print("\n" + "="*50)
    print("Validation Results on Test Set")
    print("="*50)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("="*50)
    
    return metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        # Run validation only
        model_path = sys.argv[2] if len(sys.argv) > 2 else None
        validate(model_path)
    else:
        # Train the model
        print("="*50)
        print("YOLOv8s Training - PCB Defects Dataset")
        print("="*50)
        print(f"Dataset config: {DATA_CONFIG}")
        print(f"Output directory: {MODELS_DIR}")
        print()
        
        results = train()
        
        print("\n✅ Training complete!")
        print(f"Best model saved to: {MODELS_DIR}/pcb_defects_yolov8s/weights/best.pt")
        
        # Run validation on test set
        print("\nRunning validation on test set...")
        validate()
