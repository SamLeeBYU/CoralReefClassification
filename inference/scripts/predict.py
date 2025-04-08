import os
import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from typing import List, Dict
from sam2 import sam_model_registry, SamPredictor
from PIL import Image

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path: str, device: str) -> SamPredictor:
    """Load trained SAM2.1 model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint. This should be your fine-tuned model
                        from the training process, not the original SAM2.1 weights.
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        SamPredictor: Loaded model ready for inference
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Make sure you have trained the model and the checkpoint exists in training/checkpoints/"
        )
    
    print(f"Loading fine-tuned model from {checkpoint_path}")
    # Load model configuration
    model_type = "vit_l"  # (large) or "vit_h" (huge) or "vit_b" (base) depending on desired trade-off between speed and accuracy
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    # Create predictor
    predictor = SamPredictor(sam)
    return predictor

def process_image(image_path: str, predictor: SamPredictor, device: str) -> np.ndarray:
    """Process a single image and generate mask using SAM2.1."""
    # Read and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set image for predictor
    predictor.set_image(image)
    
    # Generate mask (using automatic mask generation)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        multimask_output=True,
    )
    
    # Select the best mask based on score
    best_mask_idx = np.argmax(scores)
    mask = masks[best_mask_idx]
    
    return mask.astype(np.uint8) * 255

def save_mask(mask: np.ndarray, output_path: str):
    """Save generated mask to file."""
    cv2.imwrite(output_path, mask)

def main():
    parser = argparse.ArgumentParser(description='Generate masks for unlabeled coral images using SAM2.1')
    parser.add_argument('--config', type=str, required=True, help='Path to inference config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Set up device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print("Loading SAM2.1 model...")
    predictor = load_model(args.checkpoint, device)
    print("Model loaded successfully")

    # Process images
    input_dir = Path(config['data']['input_dir'])
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files (supporting multiple formats)
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(ext))
    
    print(f"Found {len(image_files)} images to process")

    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Generate mask
            mask = process_image(str(img_path), predictor, device)
            
            # Save mask
            output_path = output_dir / f"{img_path.stem}_mask.png"
            save_mask(mask, str(output_path))
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 